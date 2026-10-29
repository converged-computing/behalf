"""Tests for the Claude (Agent SDK) runner that need no API key or CLI.

They are skipped when claude-agent-sdk is not installed (pip install behalf[claude]).
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("claude_agent_sdk")

import behalf.runner.sdk as sdk  # noqa: E402
from behalf import Task, ToolSpec  # noqa: E402
from behalf.runner.sdk import SDKRunner, _finalize_tool, _to_sdk_tool  # noqa: E402


def _text(obj) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj)}]}


class SetupTask(Task):
    name = "setup"

    def manifest_schema(self):
        return {"goal": str, "sites": list}

    def setup_system_prompt(self):
        return "Plan it."

    def execute_system_prompt(self, manifest):
        return "run"


def test_sdk_tool_carries_spec_and_gates_actions():
    acted = []

    async def act(args):
        acted.append(args)
        return _text("acted")

    spec = ToolSpec("act", "do it", {"n": int}, act, kind="action", confirm=True)
    denied = _to_sdk_tool(spec, lambda n, a: False)
    assert (denied.name, denied.description, denied.input_schema) == (
        "act",
        "do it",
        {"n": int},
    )
    assert "cancelled by user" in json.dumps(asyncio.run(denied.handler({"n": 1})))
    assert acted == []
    asyncio.run(_to_sdk_tool(spec, lambda n, a: True).handler({"n": 2}))
    assert acted == [{"n": 2}]


def test_finalize_tool_rejects_incomplete_manifest():
    captured = {}
    finalize = _finalize_tool(SetupTask(), captured)
    out = asyncio.run(finalize.handler({"manifest": {"goal": "g"}}))
    assert "missing fields: sites" in json.dumps(out) and captured == {}
    asyncio.run(finalize.handler({"manifest": {"goal": "g", "sites": ["a"]}}))
    assert captured == {"manifest": {"goal": "g", "sites": ["a"]}}


def test_run_agent_passes_prompt_and_allowed_tools(monkeypatch):
    seen = {}

    async def fake_query(prompt, options):
        seen["prompt"], seen["options"] = prompt, options
        yield "message"

    monkeypatch.setattr(sdk, "query", fake_query)
    spec = ToolSpec("look", "look", {"q": str}, lambda a: None)
    result = asyncio.run(
        SDKRunner(verbose=False).run_agent("system", "go", [spec], lambda n, a: True)
    )
    assert result is None and seen["prompt"] == "go"
    assert seen["options"].system_prompt == "system"
    assert seen["options"].allowed_tools == ["mcp__behalf__look"]


def test_converse_uses_schema_prompt_and_retries_incomplete(monkeypatch):
    tools = []
    real_finalize = sdk._finalize_tool

    def spy(task, captured):
        tools.append(real_finalize(task, captured))
        return tools[-1]

    class StubClient:
        """Stands in for ClaudeSDKClient: each reply turn calls finalize_setup once."""

        attempts = iter([{"goal": "g"}, {"goal": "g", "sites": ["a"]}])
        queries = []

        def __init__(self, options):
            StubClient.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, text):
            StubClient.queries.append(text)

        async def receive_response(self):
            await tools[0].handler({"manifest": next(StubClient.attempts)})
            yield "message"

    monkeypatch.setattr(sdk, "_finalize_tool", spy)
    monkeypatch.setattr(sdk, "ClaudeSDKClient", StubClient)
    monkeypatch.setattr("builtins.input", lambda prompt="": "site a")
    manifest = asyncio.run(SDKRunner(verbose=False).converse(SetupTask()))
    assert manifest == {"goal": "g", "sites": ["a"]}
    assert "- sites: list" in StubClient.options.system_prompt
    assert StubClient.queries[1:] == ["site a"]


def test_make_runner_claude():
    from behalf import make_runner

    assert make_runner("claude", "claude-test").model == "claude-test"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
