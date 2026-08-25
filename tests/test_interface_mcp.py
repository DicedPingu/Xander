import asyncio
from pathlib import Path

from mcp.shared.memory import create_connected_server_and_client_session

from xander_agent.mcp_server import create_server


def test_mcp_lists_structured_sidecar_tools_and_calls_doctor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    async def scenario() -> None:
        async with create_connected_server_and_client_session(create_server()) as session:
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {
                "xander_doctor",
                "xander_inspect",
                "xander_research",
                "xander_plan",
                "xander_propose_patch",
                "xander_triage_tests",
                "xander_tasks_list",
                "xander_task_show",
                "xander_resume",
            } == names
            result = await session.call_tool("xander_doctor", {"workspace": str(tmp_path)})
            assert not result.isError
            assert result.structuredContent["event"] == "doctor"
            assert result.structuredContent["mcp_server_version"] == "1.0.0"

    asyncio.run(scenario())
