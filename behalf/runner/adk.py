"""A Gemini backend via Google's Agent Development Kit (ADK), as an alternative
AgentRunner to SDKRunner.

Converts ToolSpecs to ADK function tools (carrying the same read/action
confirmation gate) and drives ADK's Runner. Tested end to end with a scripted
BaseLlm in place of Gemini (tests/test_adk.py). Install with `pip install
behalf[gemini]`.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

# Top-level imports so a missing ADK fails here, at import, not mid-run. This
# module is only imported when you explicitly want the Gemini path
# (`from behalf.runner.adk import ADKRunner`), so it never burdens the
# Claude/SDK path or `import behalf`.
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from ..core import (
    AgentRunner,
    ConfirmFn,
    Task,
    ToolSpec,
    accept_manifest,
    render_setup_prompt,
)

_PY_TYPE = {str: str, int: int, float: float, bool: bool, list: list, dict: dict}


async def _create_session(service, **kwargs):
    """Create an ADK session; create_session is a coroutine in ADK >= 1.0 and sync before."""
    session = service.create_session(**kwargs)
    if inspect.isawaitable(session):
        session = await session
    return session


def _to_adk_tool(ts: ToolSpec, confirm_fn: ConfirmFn) -> FunctionTool:
    """Wrap one ToolSpec as an ADK FunctionTool.

    ADK builds a tool's schema from the wrapped function's SIGNATURE (typed
    params + docstring), whereas our ToolSpec carries an input_schema dict. So we
    synthesize a function with matching parameters, then let ADK introspect it.

    The wrapper also applies the read/action confirmation gate (same semantics as
    SDKRunner) and adapts our handler's Anthropic-style {"content":[...]} return
    into the plain value ADK expects a tool to return.
    """

    async def _invoke(**kwargs) -> Any:
        if ts.kind == "action" and ts.confirm and not confirm_fn(ts.name, kwargs):
            return "cancelled by user"
        result = await ts.handler(kwargs)
        # our handlers return {"content": [{"type": "text", "text": "..."}]}
        try:
            text = result["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return result
        try:
            return json.loads(text)  # most of our tools emit JSON
        except (ValueError, TypeError):
            return text

    # give ADK a real signature + docstring to introspect
    params = [
        inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, annotation=_PY_TYPE.get(typ, str)
        )
        for name, typ in ts.input_schema.items()
    ]
    _invoke.__name__ = ts.name
    _invoke.__doc__ = ts.description
    _invoke.__signature__ = inspect.Signature(params)
    _invoke.__annotations__ = {
        n: _PY_TYPE.get(t, str) for n, t in ts.input_schema.items()
    }
    return FunctionTool(_invoke)


class ADKRunner(AgentRunner):
    """AgentRunner backed by Google ADK / Gemini. Same interface as SDKRunner."""

    def __init__(
        self,
        model: Any = "gemini-2.5-flash",
        app_name: str = "behalf",
        verbose: bool = True,
    ):
        self.model = model
        self.app_name = app_name
        self.verbose = verbose

    async def _run_once(
        self, instruction: str, user_prompt: str, adk_tools: list
    ) -> list[str]:
        agent = Agent(
            name="behalf", model=self.model, instruction=instruction, tools=adk_tools
        )
        session_service = InMemorySessionService()
        session = await _create_session(
            session_service, app_name=self.app_name, user_id="local", session_id="s1"
        )
        runner = Runner(
            agent=agent, app_name=self.app_name, session_service=session_service
        )

        msg = types.Content(role="user", parts=[types.Part(text=user_prompt)])
        texts: list[str] = []
        async for event in runner.run_async(
            user_id="local", session_id=session.id, new_message=msg
        ):
            if self.verbose:
                print(event)
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    if getattr(part, "text", None):
                        texts.append(part.text)
        return texts

    async def run_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        confirm_fn: ConfirmFn,
    ) -> Any:
        adk_tools = [_to_adk_tool(t, confirm_fn) for t in tools]
        await self._run_once(system_prompt, user_prompt, adk_tools)
        # Results are collected through tool side effects (e.g. a record_tool
        # sink), same as SDKRunner.
        return None

    async def converse(self, task: Task) -> dict:
        """Interactive setup over one ADK session; loops until finalize_setup accepts a manifest."""
        captured: dict = {}

        async def finalize_setup(manifest: dict) -> str:
            """Finalize the setup manifest (matching the task's schema) and end setup."""
            return accept_manifest(task, manifest, captured)

        finalize = FunctionTool(finalize_setup)
        agent = Agent(
            name="setup",
            model=self.model,
            instruction=render_setup_prompt(task),
            tools=[finalize],
        )
        session_service = InMemorySessionService()
        session = await _create_session(
            session_service, app_name=self.app_name, user_id="local", session_id="setup"
        )
        runner = Runner(
            agent=agent, app_name=self.app_name, session_service=session_service
        )

        prompt = "Let's set up this task. Ask me what you need."
        while "manifest" not in captured:
            msg = types.Content(role="user", parts=[types.Part(text=prompt)])
            async for event in runner.run_async(
                user_id="local", session_id=session.id, new_message=msg
            ):
                if self.verbose:
                    print(event)
            if "manifest" in captured:
                break
            prompt = input("\nyou> ")
        return captured["manifest"]
