import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from fastmcp.client import Client

from ursa.cli.config import Settings
from ursa.cli.hitl import HITL


@pytest.fixture()
def default_hitl(tmp_path):
    hitl = HITL.from_settings(
        Settings(
            workspace=tmp_path,
            llm_model_name="openai:gpt-5-mini",
            emb_model_name="openai:text-embedding-3-small",
        )
    )
    return hitl


@pytest.mark.parametrize(
    "agent",
    [
        "chatter",
        "executor",
        "hypothesizer",
        "planner",
        "websearcher",
        "rememberer",
        "arxiv",
    ],
)
async def test_agents(default_hitl, agent):
    runner = getattr(default_hitl, f"run_{agent}")
    result = await runner("hello")
    print(result)


async def test_mcp_tools(tmp_path):
    hitl = HITL.from_settings(
        Settings(
            workspace=tmp_path,
            llm_model_name="openai:gpt-5-mini",
            emb_model_name="openai:text-embedding-3-small",
            mcp_servers={
                "demo": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        str(
                            Path(__file__).parent.joinpath(
                                "dummy_mcp_server.py"
                            )
                        )
                    ],
                )
            },
        )
    )
    chatter = await hitl.chatter
    assert len(chatter.tools) > 0
    out = await hitl.run_chatter("What is 12342143 + 124242?")
    assert out is not None


async def test_mcp_server(tmp_path):
    hitl = HITL.from_settings(
        Settings(
            workspace=tmp_path,
            llm_model_name="openai:gpt-5-mini",
            emb_model_name="openai:text-embedding-3-small",
        )
    )
    mcp_server = hitl.as_mcp_server()
    async with Client(mcp_server) as mcp_client:
        tools = await mcp_client.list_tools()
        assert len(tools) > 0
        out = await mcp_client.call_tool(
            "run_chatter", {"prompt": "What is your purpose?"}
        )
        assert out is not None
