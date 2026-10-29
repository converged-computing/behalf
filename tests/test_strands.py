"""Tests for the AWS (Strands) runner that need no AWS credentials.

They are skipped when strands-agents is not installed (pip install behalf[aws]).
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("strands")

from strands.models import Model  # noqa: E402

from behalf import Task, ToolSpec  # noqa: E402
from behalf.runner.strands import StrandsRunner, _to_strands_tool  # noqa: E402


def _text(obj) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj)}]}


class ScriptedModel(Model):
    """Streams the next scripted step: a (tool, args) call or plain text."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def get_config(self):
        return {}

    def update_config(self, **model_config):
        pass

    async def structured_output(self, output_model, prompt, system_prompt=None, **kw):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls.append(
            {
                "messages": json.loads(json.dumps(messages, default=str)),
                "tools": [t["name"] for t in tool_specs or []],
                "system_prompt": system_prompt,
            }
        )
        step = self.script.pop(0)
        yield {"messageStart": {"role": "assistant"}}
        if isinstance(step, tuple):
            name, args = step
            tool_use = {"toolUseId": f"t{len(self.calls)}", "name": name}
            yield {"contentBlockStart": {"start": {"toolUse": tool_use}}}
            yield {
                "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": step}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def _runner(model) -> StrandsRunner:
    runner = StrandsRunner(verbose=False)
    runner._model = lambda: model
    return runner


def _tool_results(call):
    return [
        block["toolResult"]
        for message in call["messages"]
        for block in message["content"]
        if "toolResult" in block
    ]


def test_tool_spec_carries_every_argument_and_type():
    spec = ToolSpec(
        "run",
        "run something",
        {
            "name": str,
            "n": int,
            "ratio": float,
            "flag": bool,
            "items": list,
            "opts": dict,
        },
        lambda args: None,
    )
    schema = _to_strands_tool(spec, lambda n, a: True).tool_spec
    props = schema["inputSchema"]["json"]["properties"]
    assert schema["name"] == "run" and schema["description"] == "run something"
    assert {k: v["type"] for k, v in props.items()} == {
        "name": "string",
        "n": "integer",
        "ratio": "number",
        "flag": "boolean",
        "items": "array",
        "opts": "object",
    }


def test_run_agent_with_real_strands_agent():
    log = []

    async def look(args):
        log.append(("look", args))
        return _text({"saw": args["q"]})

    async def act(args):
        log.append(("act", args))
        return _text("acted")

    tools = [
        ToolSpec("look", "look", {"q": str}, look),
        ToolSpec("act", "act", {"n": int}, act, kind="action", confirm=True),
    ]
    model = ScriptedModel([("look", {"q": "hi"}), ("act", {"n": 3}), "done"])
    denied = []

    def deny(name, args):
        denied.append((name, args))
        return False

    assert asyncio.run(_runner(model).run_agent("system", "go", tools, deny)) is None
    assert log == [("look", {"q": "hi"})]
    assert denied == [("act", {"n": 3})]
    first, *_, last = model.calls
    assert first["system_prompt"] == "system" and first["tools"] == ["look", "act"]
    results = json.dumps(_tool_results(last))
    assert "hi" in results and "cancelled by user" in results


class SetupTask(Task):
    name = "setup"

    def manifest_schema(self):
        return {"goal": str, "sites": list}

    def setup_system_prompt(self):
        return "Plan it."

    def execute_system_prompt(self, manifest):
        return "run"


def test_converse_with_real_strands_agent(monkeypatch):
    model = ScriptedModel(
        [
            "Which sites?",
            ("finalize_setup", {"manifest": {"goal": "g"}}),
            ("finalize_setup", {"manifest": {"goal": "g", "sites": ["a"]}}),
            "ok",
        ]
    )
    replies = iter(["site a"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    manifest = asyncio.run(_runner(model).converse(SetupTask()))
    assert manifest == {"goal": "g", "sites": ["a"]}

    first, *_, last = model.calls
    assert "- sites: list" in first["system_prompt"]
    history = json.dumps(last["messages"])
    assert "Which sites?" in history and "site a" in history
    assert "missing fields: sites" in json.dumps(_tool_results(last))


def test_make_runner_aws():
    from behalf import make_runner

    runner = make_runner("aws", "us.anthropic.claude-test")
    assert (
        isinstance(runner, StrandsRunner) and runner.model == "us.anthropic.claude-test"
    )
    assert runner._model().get_config()["model_id"] == "us.anthropic.claude-test"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
