import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set

import torch
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import (
    ArgModelBase,
    FuncMetadata,
)
from pydantic import BaseModel, create_model

from .fm_base_tool import TorchModuleTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AcceleratorInfo:
    """Metadata describing an available accelerator device."""

    device: torch.device
    kind: str
    index: int | None
    name: str | None


@dataclass(slots=True)
class DeviceSlot:
    """Tracks which tool currently occupies a device."""

    device: torch.device
    current_tool: str | None = None
    busy: bool = False
    last_used: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class ToolRequest:
    """Represents a pending tool invocation."""

    input: BaseModel
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class ToolState:
    """Queue and metadata for a registered tool."""

    tool: TorchModuleTool
    queue: Deque[ToolRequest] = field(default_factory=deque)
    enqueued: bool = False
    batch_size: int = 1


def _detect_accelerators() -> list[AcceleratorInfo]:
    """Discover accelerators on the current host.

    Preference order is CUDA, Apple MPS, then CPU fallback.
    """
    accelerators: list[AcceleratorInfo] = []

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{idx}")
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:  # pragma: no cover - defensive
                name = None
            accelerators.append(
                AcceleratorInfo(
                    device=device,
                    kind="cuda",
                    index=idx,
                    name=name,
                )
            )

    else:
        mps_backend = getattr(torch.backends, "mps", None)
        if (
            mps_backend is not None
            and getattr(mps_backend, "is_available", lambda: False)()
        ):
            device = torch.device("mps")
            accelerators.append(
                AcceleratorInfo(
                    device=device,
                    kind="mps",
                    index=0,
                    name="mps",
                )
            )

    if not accelerators:
        cpu_device = torch.device("cpu")
        accelerators.append(
            AcceleratorInfo(
                device=cpu_device,
                kind="cpu",
                index=None,
                name="cpu",
            )
        )

    return accelerators


class TorchModelToolManager:
    """Coordinate execution of multiple ``TorchModuleTool`` instances across devices."""

    def __init__(
        self,
        *,
        idle_seconds: float = 300.0,
        device_provider: Callable[[], list[torch.device]] | None = None,
        max_pending_per_tool: int | None = None,
    ) -> None:
        """
        Args:
            idle_seconds: Time after which idle tools may be offloaded from a device.
            device_provider: Optional function returning the devices to manage.
            max_pending_per_tool: Optional limit on queued requests per tool.
        """
        self.idle_seconds = idle_seconds
        self._max_pending_per_tool = max_pending_per_tool
        threshold = float(idle_seconds) if idle_seconds is not None else None
        self._idle_threshold = (
            None if threshold is None else max(threshold, 0.0)
        )
        self._janitor_interval = (
            None
            if self._idle_threshold is None
            else (
                0.1
                if self._idle_threshold == 0
                else max(0.1, self._idle_threshold / 2)
            )
        )

        if device_provider is not None:
            provided_devices = list(device_provider())
            if not provided_devices:
                raise ValueError(
                    "device_provider must return at least one device"
                )
            self._accelerators = [
                AcceleratorInfo(
                    device=torch.device(dev),
                    kind=torch.device(dev).type,
                    index=torch.device(dev).index,
                    name=str(dev),
                )
                for dev in provided_devices
            ]
        else:
            self._accelerators = _detect_accelerators()

        self._device_slots: list[DeviceSlot] = [
            DeviceSlot(device=info.device) for info in self._accelerators
        ]
        if not self._device_slots:
            raise RuntimeError("No devices detected for TorchModelToolManager")

        self._tools: Dict[str, TorchModuleTool] = {}
        self._tool_states: Dict[str, ToolState] = {}
        self._tool_devices: Dict[str, Optional[torch.device]] = {}

        self._condition = asyncio.Condition()
        self._pending_tools: Deque[str] = deque()
        self._rr_index: int = 0

        self._started = False
        self._closed = False
        self._tasks: Set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def devices(self) -> list[torch.device]:
        """Devices managed by this instance."""
        return [info.device for info in self._accelerators]

    @property
    def accelerator_info(self) -> list[AcceleratorInfo]:
        """Detailed accelerator metadata."""
        return list(self._accelerators)

    def register_tool(self, name: str, tool: TorchModuleTool) -> None:
        """Register a tool for managed execution."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = tool
        effective_batch = max(1, tool.batch_size or 1)
        tool.batch_size = effective_batch
        self._tool_states[name] = ToolState(
            tool=tool, batch_size=effective_batch
        )
        self._tool_devices[name] = None
        self._offload_tool(tool)

    def add_to_fastmcp(self, server: FastMCP) -> dict[str, FastMCPTool]:
        """Register all managed tools with a FastMCP server."""
        fast_tools: dict[str, FastMCPTool] = {}
        for name, state in self._tool_states.items():
            fast_tool = self._build_fastmcp_tool(name, state.tool)

            server_tools = server._tool_manager._tools
            if fast_tool.name not in server_tools:
                server_tools[fast_tool.name] = fast_tool
            elif server._tool_manager.warn_on_duplicate_tools:
                logger.warning("Tool already exists: %s", fast_tool.name)

            fast_tools[name] = fast_tool

        return fast_tools

    async def call_tool(
        self,
        name: str,
        input: BaseModel | dict[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> BaseModel:
        """Queue a call to ``name`` and await its result."""
        if self._closed:
            raise RuntimeError("TorchModelToolManager is closed")

        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'")

        tool_state = self._tool_states[name]
        request_input = self._normalize_input(tool_state.tool, input, **kwargs)

        loop = asyncio.get_running_loop()
        self._ensure_started(loop)
        future: asyncio.Future = loop.create_future()
        request = ToolRequest(input=request_input, future=future)

        await self._enqueue_request(name, request)

        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await self._cancel_request(name, request)
            raise

    async def aclose(self) -> None:
        """Cancel background tasks and drain queues."""
        if self._closed:
            return

        self._closed = True
        async with self._condition:
            self._condition.notify_all()

        for task in list(self._tasks):
            task.cancel()

        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._tasks.clear()

    async def __aenter__(self) -> "TorchModelToolManager":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_started(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._started:
            return

        dispatch_task = loop.create_task(self._dispatch_loop())
        self._track_task(dispatch_task)
        if self._janitor_interval is not None:
            janitor_task = loop.create_task(self._janitor_loop())
            self._track_task(janitor_task)
        self._started = True

    def _track_task(self, task: asyncio.Task) -> None:
        self._tasks.add(task)

        def _discard(_):
            self._tasks.discard(task)

        task.add_done_callback(_discard)

    def _normalize_input(
        self,
        tool: TorchModuleTool,
        input: BaseModel | dict[str, Any] | None,
        **kwargs: Any,
    ) -> BaseModel:
        if input is not None:
            if isinstance(input, tool.args_schema):
                return input
            if isinstance(input, BaseModel):
                return input
            if isinstance(input, dict):
                return tool.args_schema(**input)
        if kwargs:
            return tool.args_schema(**kwargs)
        raise ValueError("No input provided for tool execution")

    async def _enqueue_request(
        self, tool_name: str, request: ToolRequest
    ) -> None:
        async with self._condition:
            state = self._tool_states[tool_name]

            if self._max_pending_per_tool is not None:
                active_pending = sum(
                    1
                    for pending in state.queue
                    if not pending.future.cancelled()
                )
                if active_pending >= self._max_pending_per_tool:
                    raise RuntimeError(
                        f"Too many pending requests for tool '{tool_name}'"
                    )

            state.queue.append(request)
            if not state.enqueued:
                state.enqueued = True
                self._pending_tools.append(tool_name)
            self._condition.notify_all()

    async def _cancel_request(
        self, tool_name: str, request: ToolRequest
    ) -> None:
        async with self._condition:
            state = self._tool_states[tool_name]
            try:
                state.queue.remove(request)
            except ValueError:
                return

            if not state.queue:
                state.enqueued = False
            self._condition.notify_all()

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                async with self._condition:
                    while True:
                        if self._closed:
                            return

                        device_index = self._next_available_device_index()
                        tool_name = self._pop_next_ready_tool_locked()

                        if device_index is not None and tool_name is not None:
                            slot = self._device_slots[device_index]
                            slot.busy = True
                            self._advance_round_robin(device_index)
                            break

                        await self._condition.wait()

                await self._execute_request(tool_name, device_index)
        except asyncio.CancelledError:
            raise

    async def _janitor_loop(self) -> None:
        if self._janitor_interval is None:
            return

        try:
            while not self._closed:
                await asyncio.sleep(self._janitor_interval)
                await self._evict_idle_tools()
        except asyncio.CancelledError:
            raise

    def _next_available_device_index(self) -> int | None:
        total = len(self._device_slots)
        for offset in range(total):
            idx = (self._rr_index + offset) % total
            slot = self._device_slots[idx]
            if not slot.busy:
                return idx
        return None

    def _advance_round_robin(self, last_index: int) -> None:
        self._rr_index = (last_index + 1) % len(self._device_slots)

    def _pop_next_ready_tool_locked(self) -> str | None:
        while self._pending_tools:
            tool_name = self._pending_tools.popleft()
            state = self._tool_states[tool_name]
            state.enqueued = False
            if state.queue:
                return tool_name
        return None

    async def _dequeue_batch(
        self, tool_name: str, max_batch: int
    ) -> list[ToolRequest]:
        async with self._condition:
            state = self._tool_states[tool_name]
            batch: list[ToolRequest] = []

            while len(batch) < max_batch:
                request = self._pop_request_locked(state)
                if request is None:
                    break
                batch.append(request)

            if state.queue and not state.enqueued:
                state.enqueued = True
                self._pending_tools.append(tool_name)
                self._condition.notify_all()

            return batch

    async def _requeue_requests_front(
        self, tool_name: str, requests: list[ToolRequest]
    ) -> None:
        if not requests:
            return

        async with self._condition:
            state = self._tool_states[tool_name]
            for request in reversed(requests):
                state.queue.appendleft(request)
            if not state.enqueued:
                state.enqueued = True
                self._pending_tools.appendleft(tool_name)
            self._condition.notify_all()

    def _pop_request_locked(self, state: ToolState) -> ToolRequest | None:
        while state.queue:
            request = state.queue.popleft()
            if request.future.cancelled():
                continue
            return request
        return None

    def _current_batch_size(self, state: ToolState) -> int:
        return max(1, state.batch_size or state.tool.batch_size or 1)

    def _reduce_batch_size(self, state: ToolState) -> int:
        current = self._current_batch_size(state)
        if current <= 1:
            return 0

        new_size = max(1, current // 2)
        if new_size == current:
            new_size = current - 1
        new_size = max(1, new_size)

        logger.warning(
            "Reducing batch size for tool '%s' from %s to %s after OOM",
            state.tool.name,
            current,
            new_size,
        )

        state.batch_size = new_size
        state.tool.batch_size = new_size
        return new_size

    async def _evict_idle_tools(self, *, now: float | None = None) -> None:
        if self._idle_threshold is None:
            return

        moment = time.monotonic() if now is None else now
        evicted_any = False
        async with self._condition:
            for slot in self._device_slots:
                if slot.current_tool is None or slot.busy:
                    continue
                if moment - slot.last_used >= self._idle_threshold:
                    self._evict_slot_locked(slot, moment)
                    evicted_any = True
            if evicted_any:
                self._condition.notify_all()

    def _evict_slot_locked(self, slot: DeviceSlot, now: float) -> None:
        tool_name = slot.current_tool
        if tool_name is None:
            return

        tool = self._tools[tool_name]
        self._offload_tool(tool)
        self._tool_devices[tool_name] = None
        slot.current_tool = None
        slot.last_used = now

        if slot.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - defensive
                logger.debug(
                    "torch.cuda.empty_cache() failed during eviction",
                    exc_info=True,
                )

    async def _execute_request(self, tool_name: str, device_index: int) -> None:
        slot = self._device_slots[device_index]
        await self._evict_idle_tools()
        requests: list[ToolRequest] = []
        state = self._tool_states[tool_name]
        tool = state.tool
        try:
            while True:
                self._ensure_tool_on_slot(tool_name, slot)
                batch_size = self._current_batch_size(state)
                requests = await self._dequeue_batch(tool_name, batch_size)
                if not requests:
                    return

                try:
                    results = await asyncio.to_thread(
                        self._run_batch, state, requests, batch_size
                    )
                except Exception as exc:
                    if self._is_out_of_memory(exc):
                        new_size = self._reduce_batch_size(state)
                        if new_size == 0:
                            for request in requests:
                                if not request.future.done():
                                    request.future.set_exception(exc)
                            raise
                        await self._requeue_requests_front(tool_name, requests)
                        requests = []
                        continue
                    raise

                for request, output in zip(requests, results):
                    if not request.future.cancelled():
                        request.future.set_result(output)
                break
        except Exception as exc:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(exc)
            logger.exception(
                "Error while executing tool '%s' on %s",
                tool_name,
                slot.device,
                exc_info=exc,
            )
        finally:
            await self._mark_slot_free(device_index)

    def _run_batch(
        self,
        state: ToolState,
        requests: list[ToolRequest],
        batch_size: int,
    ) -> list[BaseModel]:
        inputs = [request.input for request in requests]
        outputs = list(
            state.tool.batch_as_completed(
                inputs,
                max_concurency=batch_size,
            )
        )
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"Tool '{state.tool.name}' returned {len(outputs)} outputs for "
                f"{len(requests)} requests"
            )
        return outputs

    def _build_fastmcp_tool(
        self,
        name: str,
        tool: TorchModuleTool,
    ) -> FastMCPTool:
        field_definitions = {
            field: (field_info.annotation, field_info)
            for field, field_info in tool.args_schema.model_fields.items()
        }
        arg_model = create_model(
            f"{tool.name}Arguments",
            **field_definitions,
            __base__=ArgModelBase,
        )
        fn_metadata = FuncMetadata(
            arg_model=arg_model,
            output_model=tool.output_schema,
            output_schema=tool.output_schema.model_json_schema(),
        )

        async def fn(**input_data: Any) -> BaseModel:
            args = tool.args_schema(**input_data)
            return await self.call_tool(name, args)

        return FastMCPTool(
            fn=fn,
            name=tool.name,
            description=tool.description,
            parameters=tool.args_schema.model_json_schema(),
            fn_metadata=fn_metadata,
            is_async=True,
        )

    def _is_out_of_memory(self, exc: Exception) -> bool:
        cuda_oom = getattr(torch.cuda, "OutOfMemoryError", ())
        if cuda_oom and isinstance(exc, cuda_oom):
            return True
        if (
            isinstance(exc, RuntimeError)
            and "out of memory" in str(exc).lower()
        ):
            return True
        return False

    async def _mark_slot_free(self, device_index: int) -> None:
        async with self._condition:
            slot = self._device_slots[device_index]
            slot.busy = False
            slot.last_used = time.monotonic()
            self._condition.notify_all()

    def _ensure_tool_on_slot(self, tool_name: str, slot: DeviceSlot) -> None:
        if slot.current_tool == tool_name:
            return

        if slot.current_tool is not None:
            previous_tool = self._tools[slot.current_tool]
            self._offload_tool(previous_tool)
            self._tool_devices[slot.current_tool] = None
            slot.current_tool = None

        tool = self._tools[tool_name]
        self._load_tool(tool, slot.device)
        slot.current_tool = tool_name
        self._tool_devices[tool_name] = slot.device
        slot.last_used = time.monotonic()

    def _load_tool(self, tool: TorchModuleTool, device: torch.device) -> None:
        try:
            tool.fm = tool.fm.to(device)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Failed to move tool '%s' to %s", tool.name, device
            )
            raise
        tool.device = device

    def _offload_tool(self, tool: TorchModuleTool) -> None:
        tool.fm = tool.fm.to(torch.device("cpu"))
        tool.device = torch.device("cpu")
