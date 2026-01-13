import asyncio
import time
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

torch = pytest.importorskip("torch")

from mcp.server.fastmcp import FastMCP

from ursa.tools.fm_base_tool import TorchModuleTool, default_device
from ursa.tools.torch_tool_manager import TorchModelToolManager


class DummyModule(torch.nn.Module):
    def __init__(
        self,
        increment: int = 1,
        delay: float = 0.0,
        label: str | None = None,
        oom_threshold: int | None = None,
    ) -> None:
        super().__init__()
        self.increment = increment
        self.delay = delay
        self.label = label or "module"
        self.last_device = torch.device("cpu")
        self.forward_batch_sizes: list[int] = []
        self.execution_devices: list[torch.device] = []
        self.forward_start_times: list[float] = []
        self.oom_threshold = oom_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        self.forward_batch_sizes.append(batch_size)
        if self.oom_threshold is not None and batch_size > self.oom_threshold:
            raise RuntimeError("CUDA out of memory")
        self.execution_devices.append(self.last_device)
        self.forward_start_times.append(time.perf_counter())
        if self.delay:
            time.sleep(self.delay)
        return x + self.increment

    def to(self, device, **kwargs):
        self.last_device = torch.device(device)
        return self


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    value: int


class DummyTool(
    TorchModuleTool[DummyInput, DummyOutput, torch.Tensor, torch.Tensor]
):
    args_schema: type[DummyInput] = DummyInput
    output_schema: type[DummyOutput] = DummyOutput
    description: str = "Add one to the provided value."

    def preprocess(self, input: Sequence[DummyInput]) -> torch.Tensor:
        values = [item.value for item in input]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    def postprocess(self, model_output: torch.Tensor):
        for value in model_output.squeeze(-1):
            yield DummyOutput(value=int(value.item()))


@pytest.fixture()
def dummy_tool_factory():
    def _factory(
        *,
        increment: int = 1,
        delay: float = 0.0,
        label: str = "tool",
        batch_size: int = 4,
        oom_threshold: int | None = None,
    ) -> DummyTool:
        module = DummyModule(
            increment=increment,
            delay=delay,
            label=label,
            oom_threshold=oom_threshold,
        )
        return DummyTool(
            description=f"{label} tool",
            fm=module,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )

    return _factory


def make_accelerator_device(index: int = 0) -> torch.device:
    base = default_device()
    dtype = base.type
    if dtype in {"cuda", "xpu"}:
        return torch.device(f"{dtype}:{index}")
    if dtype == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def make_accelerator_devices(count: int) -> list[torch.device]:
    base = default_device()
    dtype = base.type
    if dtype in {"cuda", "xpu"}:
        return [make_accelerator_device(i) for i in range(count)]
    if dtype == "mps":
        devices = [torch.device("mps")]
        devices.extend(torch.device("cpu") for _ in range(count - 1))
        return devices
    return [torch.device("cpu") for _ in range(count)]


@pytest.fixture()
def dummy_tool(dummy_tool_factory) -> DummyTool:
    return dummy_tool_factory(label="dummy")


def test_default_device_detection():
    manager = TorchModelToolManager()
    devices = manager.devices
    assert devices, "At least one device should be detected"
    for device in devices:
        assert isinstance(device, torch.device)


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_device_provider_injection(dummy_tool: DummyTool):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(device_provider=lambda: fake_devices)
    assert manager.devices == fake_devices

    manager.register_tool("dummy", dummy_tool)

    result = await manager.call_tool("dummy", {"value": 1})
    assert isinstance(result, DummyOutput)
    assert result.value == 2

    fm = dummy_tool.fm
    assert isinstance(fm, DummyModule)
    assert fm.last_device == fake_devices[0]
    assert fm.execution_devices == [fake_devices[0]]
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_round_robin_assignment_and_eviction(dummy_tool_factory):
    fake_devices = make_accelerator_devices(2)
    manager = TorchModelToolManager(device_provider=lambda: fake_devices)

    tool_a = dummy_tool_factory(label="A", increment=1)
    tool_b = dummy_tool_factory(label="B", increment=2)
    tool_c = dummy_tool_factory(label="C", increment=3)

    manager.register_tool("A", tool_a)
    manager.register_tool("B", tool_b)
    manager.register_tool("C", tool_c)

    out_a = await manager.call_tool("A", {"value": 1})
    out_b = await manager.call_tool("B", {"value": 2})
    out_c = await manager.call_tool("C", {"value": 3})

    assert [out_a.value, out_b.value, out_c.value] == [2, 4, 6]

    assert tool_a.fm.execution_devices == [fake_devices[0]]
    assert tool_b.fm.execution_devices == [fake_devices[1]]
    assert tool_c.fm.execution_devices == [fake_devices[0]]

    assert tool_a.fm.last_device == torch.device("cpu")
    assert tool_b.fm.last_device == fake_devices[1]
    assert tool_c.fm.last_device == fake_devices[0]
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_queue_single_device_serializes_calls(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])

    slow_tool = dummy_tool_factory(label="slow", delay=0.05)
    fast_tool = dummy_tool_factory(label="fast")

    manager.register_tool("slow", slow_tool)
    manager.register_tool("fast", fast_tool)

    slow_task = asyncio.create_task(manager.call_tool("slow", {"value": 0}))
    await asyncio.sleep(0)
    fast_task = asyncio.create_task(manager.call_tool("fast", {"value": 1}))

    slow_result, fast_result = await asyncio.gather(slow_task, fast_task)
    assert slow_result.value == 1
    assert fast_result.value == 2

    assert slow_tool.fm.execution_devices == [device]
    assert fast_tool.fm.execution_devices == [device]

    slow_start = slow_tool.fm.forward_start_times[0]
    fast_start = fast_tool.fm.forward_start_times[0]
    assert fast_start >= slow_start
    assert fast_start - slow_start >= slow_tool.fm.delay * 0.8

    assert slow_tool.fm.last_device == torch.device("cpu")
    assert fast_tool.fm.last_device == device
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_batching_respects_tool_batch_size(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="batch", batch_size=4)
    manager.register_tool("batch", tool)

    tasks = [
        asyncio.create_task(manager.call_tool("batch", {"value": idx}))
        for idx in range(10)
    ]

    results = await asyncio.gather(*tasks)
    assert [result.value for result in results] == [idx + 1 for idx in range(10)]

    assert tool.fm.forward_batch_sizes == [4, 4, 2]
    assert len(tool.fm.execution_devices) == 3
    await manager.aclose()


@pytest.mark.asyncio()
async def test_idle_offloading(monkeypatch, dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(
        device_provider=lambda: [device],
        idle_seconds=0.2,
    )
    tool = dummy_tool_factory(label="idle")
    manager.register_tool("idle", tool)

    clock = {"value": 1_000.0}

    def fake_monotonic():
        return clock["value"]

    monkeypatch.setattr(
        "ursa.tools.torch_tool_manager.time.monotonic", fake_monotonic
    )

    await manager.call_tool("idle", {"value": 10})
    assert tool.fm.last_device == device

    # Advance time but not enough to trigger eviction
    clock["value"] += 0.05
    await manager._evict_idle_tools(now=clock["value"])
    assert tool.fm.last_device == device

    # Advance past idle threshold and evict
    clock["value"] += 0.3
    await manager._evict_idle_tools(now=clock["value"])
    assert tool.fm.last_device == torch.device("cpu")
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_oom_backoff_reduces_batch_size(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="oom", batch_size=8, oom_threshold=2)
    manager.register_tool("oom", tool)

    tasks = [
        asyncio.create_task(manager.call_tool("oom", {"value": idx}))
        for idx in range(8)
    ]

    results = await asyncio.gather(*tasks)
    assert [result.value for result in results] == [idx + 1 for idx in range(8)]

    state = manager._tool_states["oom"]
    assert state.batch_size <= 2
    assert tool.batch_size == state.batch_size
    assert tool.fm.forward_batch_sizes[:2] == [8, 4]
    assert all(size <= 2 for size in tool.fm.forward_batch_sizes[2:])

    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_oom_at_batch_size_one_raises(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="oom1", batch_size=1, oom_threshold=0)
    manager.register_tool("oom1", tool)

    with pytest.raises(RuntimeError) as excinfo:
        await manager.call_tool("oom1", {"value": 42})

    assert "out of memory" in str(excinfo.value).lower()
    assert tool.batch_size == 1
    await manager.aclose()


@pytest.mark.asyncio()
@pytest.mark.gpu
async def test_fastmcp_integration(dummy_tool_factory):
    device = make_accelerator_device(0)
    manager = TorchModelToolManager(device_provider=lambda: [device])
    tool = dummy_tool_factory(label="fast")
    manager.register_tool("fast", tool)

    server = FastMCP()
    fast_tools = manager.add_to_fastmcp(server)
    assert "fast" in fast_tools

    tools = await server.list_tools()
    assert any(t.name == tool.name for t in tools)

    fast_tool = fast_tools["fast"]
    result = await fast_tool.fn(value=7)
    assert isinstance(result, DummyOutput)
    assert result.value == 8

    await manager.aclose()
