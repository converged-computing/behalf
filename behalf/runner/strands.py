"""An AWS backend via the Strands Agents SDK, running against Amazon Bedrock.

Handy when Bedrock (IAM/AWS credentials) is easier to reach than an Anthropic or
Google key. Authenticates with your normal AWS credential chain (profile / env /
role) and a region; the model is a Bedrock model id, which can still be Claude
(a Bedrock Anthropic inference profile), just reached through Bedrock.

Strands tools are plain Python functions, so ToolSpecs convert directly (with
the read/action confirm gate). Tested end to end with a scripted Model in place
of Bedrock (tests/test_strands.py). Install with `pip install behalf[aws]`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from strands import Agent
from strands.models import BedrockModel

from ..core import (
    AgentRunner,
    ConfirmFn,
    Task,
    ToolSpec,
    accept_manifest,
    render_setup_prompt,
)

_PY_TYPE = {str: str, int: int, float: float, bool: bool, list: list, dict: dict}


def _op_detail(name: str, kwargs: dict) -> str:
    """A short, domain-agnostic one-liner for the trace: the scalar arguments,
    space-joined. (A caller that wants prettier per-tool formatting can wrap the
    runner; the library stays generic.)"""
    scalars = [
        str(v)
        for v in kwargs.values()
        if isinstance(v, (str, int, float, bool)) and str(v) != ""
    ]
    detail = " ".join(scalars) if scalars else json.dumps(kwargs, default=str)
    return detail[:97] + "..." if len(detail) > 100 else detail


def _to_strands_tool(ts: ToolSpec, confirm_fn: ConfirmFn, on_call=None):
    """ToolSpec -> Strands tool. Strands reads the function signature + docstring
    for the schema, so we synthesize typed params. Our async handler is bridged
    to sync, the confirm gate is applied for action tools, and the
    {"content":[...]} return is adapted to a plain value. on_call, if given, is
    invoked with (name, kwargs) when the tool actually runs — that's where the
    real-time trace line comes from, since the arguments are complete by then."""
    from strands import tool

    def _invoke(**kwargs) -> Any:
        if on_call is not None:
            on_call(ts.name, kwargs)
        if ts.kind == "action" and ts.confirm and not confirm_fn(ts.name, kwargs):
            return "cancelled by user"
        result = asyncio.run(ts.handler(kwargs))
        try:
            text = result["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return result
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text

    _invoke.__name__ = ts.name
    _invoke.__doc__ = ts.description
    _invoke.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                n, inspect.Parameter.KEYWORD_ONLY, annotation=_PY_TYPE.get(t, str)
            )
            for n, t in ts.input_schema.items()
        ]
    )
    _invoke.__annotations__ = {
        n: _PY_TYPE.get(t, str) for n, t in ts.input_schema.items()
    }
    return tool(_invoke)


class StrandsRunner(AgentRunner):
    """AgentRunner backed by Strands + Bedrock. Same interface as SDKRunner."""

    def __init__(
        self, model: str | None = None, region: str = "us-east-1", verbose: bool = True
    ):
        self.model = model  # None -> Strands' default (an Anthropic model on Bedrock)
        self.region = region
        self.verbose = verbose

    def _model(self):
        if self.model is None:
            return None
        return BedrockModel(model_id=self.model, region_name=self.region)

    def _render(self, event) -> None:
        """Stream assistant text (dimmed). Tool lines print from the wrapper."""
        if self.verbose and isinstance(event, dict) and event.get("data"):
            from .. import console

            console.think(event["data"])
            self._mid_line = not event["data"].endswith("\n")

    def _trace(self, name: str, kwargs: dict) -> None:
        if not self.verbose:
            return
        from .. import console

        if getattr(self, "_mid_line", False):
            print()  # close a partial line of streamed text before the op line
            self._mid_line = False
        console.op(name, _op_detail(name, kwargs))

    async def run_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        confirm_fn: ConfirmFn,
    ) -> Any:
        self._mid_line = False
        kwargs = {
            "system_prompt": system_prompt,
            "tools": [
                _to_strands_tool(t, confirm_fn, on_call=self._trace) for t in tools
            ],
            # silence Strands' own stdout printer so it doesn't double with ours
            "callback_handler": None,
        }
        model = self._model()
        if model is not None:
            kwargs["model"] = model
        agent = Agent(**kwargs)

        stream = getattr(agent, "stream_async", None)
        if stream is None:
            # Older Strands without stream_async.
            result = await asyncio.to_thread(agent, user_prompt)
            if self.verbose:
                print(result)
        else:
            async for event in stream(user_prompt):
                self._render(event)
            if self.verbose and self._mid_line:
                print()  # newline after trailing streamed text
        return None  # results collected via tool side effects (e.g. a record_tool sink)

    async def converse(self, task: Task) -> dict:
        from strands import tool

        captured: dict = {}

        def finalize_setup(manifest: dict) -> str:
            """Finalize the setup manifest (matching the task's schema) and end setup."""
            return accept_manifest(task, manifest, captured)

        kwargs = {
            "system_prompt": render_setup_prompt(task),
            "tools": [tool(finalize_setup)],
        }
        model = self._model()
        if model is not None:
            kwargs["model"] = model
        agent = Agent(**kwargs)

        prompt = "Let's set up this task. Ask me what you need."
        while "manifest" not in captured:
            out = await asyncio.to_thread(agent, prompt)
            if self.verbose:
                print(out)
            if "manifest" in captured:
                break
            prompt = input("\nyou> ")
        return captured["manifest"]
