"""behalf — a small framework for scoped agents.

Give an agent a brief, a fixed set of tools (read, or action-with-confirmation),
and a pluggable model backend; it works within that behalf and returns a result.
The core is dependency-free; a backend extra (claude/gemini/aws) adds a runner.
"""

from .core import (
    AgentRunner,
    ConfirmFn,
    RunOutcome,
    Task,
    ToolSpec,
    default_confirm,
    run_task,
)

__all__ = [
    "AgentRunner",
    "ConfirmFn",
    "RunOutcome",
    "Task",
    "ToolSpec",
    "default_confirm",
    "run_task",
    "make_runner",
]


def make_runner(backend: str, model: str | None = None):
    """Resolve a runner by name, importing only that backend's SDK.

    backend: "claude" | "gemini" | "aws". Raises SystemExit with an install hint
    if the backend's package isn't installed.
    """
    if backend == "claude":
        try:
            from .runner.sdk import SDKRunner
        except ImportError as e:
            raise SystemExit(
                f"backend 'claude' needs the Claude Agent SDK: pip install behalf[claude] ({e})"
            )
        return SDKRunner(model) if model else SDKRunner()
    if backend == "gemini":
        try:
            from .runner.adk import ADKRunner
        except ImportError as e:
            raise SystemExit(
                f"backend 'gemini' needs Google ADK: pip install behalf[gemini] ({e})"
            )
        return ADKRunner(model=model) if model else ADKRunner()
    if backend == "aws":
        try:
            from .runner.strands import StrandsRunner
        except ImportError as e:
            raise SystemExit(
                f"backend 'aws' needs Strands: pip install behalf[aws] ({e})"
            )
        return StrandsRunner(model=model) if model else StrandsRunner()
    raise SystemExit(f"unknown backend {backend!r} (choose: claude, gemini, aws)")
