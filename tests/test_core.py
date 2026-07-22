import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from behalf import RunOutcome, Task, ToolSpec, run_task


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


if __name__ == "__main__":
    test_manifest_passthrough_runs_execute()
    test_action_gate_blocks_on_denial()
    test_setup_approval_can_abort()
    print("\nall behalf core tests passed")
