"""A tiny framework for a recurring pattern: give an agent a SCOPED task that
first gathers intent from the user, freezes it into a reproducible manifest,
then executes toward one specific outcome with a fixed, typed toolset.

    elicit (adaptive conversation, seeded)  ->  manifest (frozen, reviewable)
                                            ->  execute (agent + fixed tools)

The core knows nothing about any particular domain. A Task supplies the three
things that vary: what to ask the user, which tools the agent may use, and what
a valid result is. Read-only tasks, tasks with an action tool that gates on user
confirmation, and pipelines of tasks are all just Tasks.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol


@dataclass
class ToolSpec:
    """A tool the agent can call. `kind` distinguishes read (call freely) from
    action (mutates something -> may require confirming the exact arguments)."""

    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Awaitable[dict]]
    kind: str = "read"  # "read" | "action"
    confirm: bool = False  # if action: pause for approval before running


# confirm_fn(tool_name, args) -> bool. Supplied by the runtime (CLI prompt, test
# stub, auto-approve). Only ever called for action tools with confirm=True.
ConfirmFn = Callable[[str, dict], bool]


class Task(ABC):
    """A scoped capability. Subclasses fill in the three varying pieces."""

    name: str = "task"

    # Whether the frozen setup manifest must be approved before execution.
    # (Action tools have their OWN per-call confirmation regardless.)
    requires_setup_approval: bool = False

    def setup_system_prompt(self) -> str:
        """Seeds elicitation: what the agent already knows (schemas, fixed
        metadata) and what it must learn from the user."""
        return "Gather what you need from the user, then finalize a setup manifest."

    @abstractmethod
    def manifest_schema(self) -> dict:
        """The schema the elicitation must produce and freeze."""

    def build_tools(self, manifest: dict) -> list[ToolSpec]:
        """Fixed toolset for the default single-run execute(). Iterating tasks
        override execute() and build tools per item, leaving this as []."""
        return []

    @abstractmethod
    def execute_system_prompt(self, manifest: dict) -> str:
        """Instruction for the execution agent."""

    async def execute(
        self, runner: "AgentRunner", manifest: dict, confirm_fn: "ConfirmFn"
    ) -> Any:
        """Run the task. Default: a single agent run over the built tools."""
        tools = self.build_tools(manifest)
        return await runner.run_agent(
            system_prompt=self.execute_system_prompt(manifest),
            user_prompt=manifest.get("goal", "Complete the task."),
            tools=tools,
            confirm_fn=confirm_fn,
        )

    def validate_result(self, result: Any) -> None:
        """Raise if the produced result is unacceptable. Override per task."""
        return None


class AgentRunner(Protocol):
    """How the framework talks to a model. Real ones wrap an SDK; tests inject a
    fake. This seam is what lets the whole flow be tested without a key."""

    async def converse(self, task: "Task") -> dict:
        """Run the seeded setup conversation; return the manifest."""
        ...

    async def run_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        confirm_fn: "ConfirmFn",
    ) -> Any:
        """One agent run with a fixed toolset; action tools gate on confirm_fn."""
        ...


@dataclass
class RunOutcome:
    manifest: dict
    result: Any
    approved: bool = True


def default_confirm(tool_name: str, args: dict) -> bool:
    """CLI confirmation: show the exact call and ask. Safe default for actions."""
    print(f"\n[confirm] about to run action '{tool_name}' with:")
    print(json.dumps(args, indent=2))
    return input("proceed? [y/N] ").strip().lower() in ("y", "yes")


async def run_task(
    task: Task,
    runner: AgentRunner,
    manifest: Optional[dict] = None,
    confirm_fn: ConfirmFn = default_confirm,
    approve_fn: Optional[Callable[[dict], bool]] = None,
) -> RunOutcome:
    """The one entrypoint every task flows through. manifest given -> skip the
    conversation (reproducible/batch); None -> converse to produce it. Then
    optional setup approval, then execute; action tools gate on confirm_fn."""
    if manifest is None:
        manifest = await runner.converse(task)

    approved = True
    if task.requires_setup_approval:
        approver = approve_fn or (lambda m: default_confirm("finalize-setup", m))
        approved = approver(manifest)
        if not approved:
            return RunOutcome(manifest=manifest, result=None, approved=False)

    result = await task.execute(runner, manifest, confirm_fn)
    task.validate_result(result)
    return RunOutcome(manifest=manifest, result=result, approved=approved)
