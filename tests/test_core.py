import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json

import pytest

from behalf import (
    RunOutcome,
    Task,
    ToolSpec,
    accept_manifest,
    missing_manifest_fields,
    record_tool,
    render_setup_prompt,
    run_task,
)


class DemoTask(Task):
    name = "demo"

    def manifest_schema(self):
        return {"goal": str}

    def build_tools(self, manifest):
        async def look(args):
            return {"content": [{"type": "text", "text": "looked"}]}

        async def act(args):
            self.acted.append(args)
            return {"content": [{"type": "text", "text": "acted"}]}

        self.acted = []
        return [
            ToolSpec("look", "read something", {"path": str}, look),
            ToolSpec(
                "act", "do something", {"x": int}, act, kind="action", confirm=True
            ),
        ]

    def execute_system_prompt(self, manifest):
        return "do the demo"


class FakeRunner:
    """Applies the same read/action confirm gate the real runners do, without
    any SDK — so the core contract is testable with no key."""

    def __init__(self, manifest):
        self.manifest = manifest

    async def converse(self, task):
        return self.manifest

    async def run_agent(self, system_prompt, user_prompt, tools, confirm_fn):
        by = {t.name: t for t in tools}
        await by["look"].handler({"path": "/x"})
        act = by["act"]
        if act.kind == "action" and act.confirm and not confirm_fn(act.name, {"x": 1}):
            return "blocked"
        await act.handler({"x": 1})
        return "ran"


def test_manifest_passthrough_runs_execute():
    task = DemoTask()
    outcome = asyncio.run(
        run_task(
            task,
            FakeRunner({"goal": "g"}),
            manifest={"goal": "g"},
            confirm_fn=lambda n, a: True,
        )
    )
    assert (
        isinstance(outcome, RunOutcome)
        and outcome.result == "ran"
        and task.acted == [{"x": 1}]
    )
    print("OK manifest passthrough -> execute runs, action fired on approval")


def test_action_gate_blocks_on_denial():
    task = DemoTask()
    outcome = asyncio.run(
        run_task(
            task,
            FakeRunner({"goal": "g"}),
            manifest={"goal": "g"},
            confirm_fn=lambda n, a: False,
        )
    )
    assert (
        outcome.result == "blocked" and task.acted == []
    ), "denied action must not run"
    print("OK action tool gates on confirmation (denial blocks)")


def test_setup_approval_can_abort():
    task = DemoTask()
    task.requires_setup_approval = True
    outcome = asyncio.run(
        run_task(
            task,
            FakeRunner({"goal": "g"}),
            manifest={"goal": "g"},
            approve_fn=lambda m: False,
        )
    )
    assert outcome.approved is False and outcome.result is None
    print("OK setup approval can abort before execution")


class SchemaTask(DemoTask):
    name = "schema"

    def manifest_schema(self):
        return {"goal": str, "sites": list, "budget": float}

    def setup_system_prompt(self):
        return "Plan an experiment."


def test_setup_prompt_carries_schema():
    prompt = render_setup_prompt(SchemaTask())
    assert prompt.startswith("Plan an experiment.")
    for line in ("- goal: str", "- sites: list", "- budget: float", "finalize_setup"):
        assert line in prompt


def test_missing_manifest_fields():
    task = SchemaTask()
    assert missing_manifest_fields(task, {"goal": "g"}) == ["sites", "budget"]
    assert missing_manifest_fields(task, {"goal": 1, "sites": [], "budget": 0}) == []
    assert missing_manifest_fields(task, "not a dict") == ["goal", "sites", "budget"]


def test_accept_manifest_only_captures_complete_manifests():
    task, captured = SchemaTask(), {}
    message = accept_manifest(task, {"goal": "g"}, captured)
    assert "sites, budget" in message and captured == {}
    full = {"goal": "g", "sites": ["a"], "budget": 1.0}
    assert accept_manifest(task, full, captured) == "setup finalized"
    assert captured == {"manifest": full}


def test_run_task_rejects_incomplete_manifest():
    task = SchemaTask()
    with pytest.raises(ValueError, match="sites"):
        asyncio.run(
            run_task(
                task,
                FakeRunner({}),
                manifest={"goal": "g"},
                confirm_fn=lambda n, a: True,
            )
        )
    assert not hasattr(task, "acted"), "execute must not run"


def test_run_task_rejects_incomplete_conversed_manifest():
    with pytest.raises(ValueError, match="goal"):
        asyncio.run(run_task(DemoTask(), FakeRunner({}), confirm_fn=lambda n, a: True))


def test_record_tool_appends_to_sink():
    sink = []
    spec = record_tool(sink, {"job": str, "site": str}, name="record_placement")
    assert (spec.name, spec.kind, spec.confirm) == ("record_placement", "action", False)
    assert spec.input_schema == {"job": str, "site": str}
    out = asyncio.run(spec.handler({"job": "j1", "site": "a"}))
    asyncio.run(spec.handler({"job": "j2", "site": "b"}))
    assert json.loads(out["content"][0]["text"]) == {"recorded": 1}
    assert sink == [{"job": "j1", "site": "a"}, {"job": "j2", "site": "b"}]


if __name__ == "__main__":
    test_manifest_passthrough_runs_execute()
    test_action_gate_blocks_on_denial()
    test_setup_approval_can_abort()
    print("\nall behalf core tests passed")
