"""Tests for the Gemini (Google ADK) runner that need no API key.

They are skipped when google-adk is not installed (pip install behalf[gemini]).
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("google.adk")

from google.adk.sessions import InMemorySessionService  # noqa: E402

import behalf.runner.adk as adk  # noqa: E402
from behalf import ToolSpec  # noqa: E402
from behalf.runner.adk import ADKRunner, _create_session, _to_adk_tool  # noqa: E402


def _text(obj) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj)}]}


def _schema(decl) -> dict:
    """Parameter schema of a declaration, for ADK versions with and without JSON schema."""
    if getattr(decl, "parameters_json_schema", None):
        return decl.parameters_json_schema
    params = decl.parameters
    if params is None:
        return {}
    return {
        "properties": {
            k: {"type": v.type.name.lower()} for k, v in params.properties.items()
        },
        "required": list(params.required or []),
    }


# _create_session: the bug was calling create_session synchronously; in ADK >= 1.0
# it is a coroutine, so the "session" had no .id and every run failed.


def test_create_session_with_installed_adk():
    session = asyncio.run(
        _create_session(
            InMemorySessionService(), app_name="t", user_id="u", session_id="s1"
        )
    )
    assert session.id == "s1"


def test_create_session_accepts_sync_and_async_services():
    class Sync:
        def create_session(self, **kw):
            return {"id": kw["session_id"]}

    class Async:
        async def create_session(self, **kw):
            return {"id": kw["session_id"]}

    for service in (Sync(), Async()):
        session = asyncio.run(_create_session(service, session_id="x"))
        assert session == {"id": "x"}, type(service).__name__


# _to_adk_tool: ADK derives the schema from a synthesized signature; every
# argument and type in the ToolSpec must survive, and the confirm gate must hold.


def test_tool_declaration_carries_every_argument_and_type():
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
    decl = _to_adk_tool(spec, lambda n, a: True)._get_declaration()
    assert decl.name == "run" and decl.description == "run something"
    schema = _schema(decl)
    types = {k: v["type"] for k, v in schema["properties"].items()}
    assert types == {
        "name": "string",
        "n": "integer",
        "ratio": "number",
        "flag": "boolean",
        "items": "array",
        "opts": "object",
    }
    assert set(schema["required"]) == set(spec.input_schema)


def test_tool_with_no_arguments():
    spec = ToolSpec("ping", "no args", {}, lambda args: None)
    decl = _to_adk_tool(spec, lambda n, a: True)._get_declaration()
    assert not _schema(decl).get("properties")


def test_tool_invoke_unwraps_json_and_text_results():
    calls = []

    async def handler(args):
        calls.append(args)
        return _text({"ok": True, "got": args})

    tool = _to_adk_tool(
        ToolSpec("look", "d", {"path": str}, handler), lambda n, a: True
    )
    assert asyncio.run(tool.func(path="/x")) == {"ok": True, "got": {"path": "/x"}}
    assert calls == [{"path": "/x"}]

    async def plain(args):
        return {"content": [{"type": "text", "text": "not json"}]}

    tool = _to_adk_tool(ToolSpec("say", "d", {}, plain), lambda n, a: True)
    assert asyncio.run(tool.func()) == "not json"

    async def raw(args):
        return {"value": 1}

    tool = _to_adk_tool(ToolSpec("raw", "d", {}, raw), lambda n, a: True)
    assert asyncio.run(tool.func()) == {"value": 1}


def test_action_tool_gates_on_confirmation():
    acted = []

    async def act(args):
        acted.append(args)
        return _text("done")

    seen = []

    def deny(name, args):
        seen.append((name, args))
        return False

    spec = ToolSpec("act", "d", {"x": int}, act, kind="action", confirm=True)
    assert asyncio.run(_to_adk_tool(spec, deny).func(x=1)) == "cancelled by user"
    assert acted == [] and seen == [("act", {"x": 1})]
    assert asyncio.run(_to_adk_tool(spec, lambda n, a: True).func(x=2)) == "done"
    assert acted == [{"x": 2}]

    # read tools and unconfirmed actions never consult confirm_fn
    read = ToolSpec("look", "d", {}, act)
    unconfirmed = ToolSpec("act2", "d", {}, act, kind="action", confirm=False)
    for spec in (read, unconfirmed):
        asyncio.run(_to_adk_tool(spec, deny).func())
    assert len(seen) == 1


# run_agent: drive ADKRunner with the real session service and a stub in place
# of ADK's Runner, so the whole path runs without a model.


class StubRunner:
    """Stands in for google.adk.runners.Runner: calls every tool once, then
    emits one final text event. Records what it was given."""

    created = []

    def __init__(self, agent, app_name, session_service):
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service
        StubRunner.created.append(self)
        self.calls = []

    async def run_async(self, user_id, session_id, new_message):
        self.calls.append((user_id, session_id, new_message.parts[0].text))
        # The session must exist in the service: proves create_session was awaited.
        session = await self.session_service.get_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )
        assert session is not None and session.id == session_id
        results = {}
        for tool in self.agent.tools:
            results[tool.name] = await tool.func(
                **{"q": "hi"} if tool.name == "look" else {}
            )

        class Part:
            text = json.dumps(results)

        class Content:
            parts = [Part()]

        class Event:
            content = Content()

        yield Event()


def test_run_agent_end_to_end_with_stubbed_adk_runner(monkeypatch):
    monkeypatch.setattr(adk, "Runner", StubRunner)
    StubRunner.created.clear()
    log = []

    async def look(args):
        log.append(("look", args))
        return _text({"saw": args["q"]})

    async def act(args):
        log.append(("act", args))
        return _text("acted")

    tools = [
        ToolSpec("look", "look", {"q": str}, look),
        ToolSpec("act", "act", {}, act, kind="action", confirm=True),
    ]
    runner = ADKRunner(model="gemini-test", app_name="app", verbose=False)
    result = asyncio.run(
        runner.run_agent("system", "user prompt", tools, lambda n, a: True)
    )
    assert result is None  # results flow through tool side effects, like SDKRunner
    assert log == [("look", {"q": "hi"}), ("act", {})]

    (stub,) = StubRunner.created
    assert stub.app_name == "app" and stub.agent.model == "gemini-test"
    assert stub.agent.instruction == "system"
    assert [t.name for t in stub.agent.tools] == ["look", "act"]
    assert stub.calls == [("local", "s1", "user prompt")]


def test_run_agent_denied_action_is_cancelled(monkeypatch):
    monkeypatch.setattr(adk, "Runner", StubRunner)
    acted = []

    async def act(args):
        acted.append(args)
        return _text("acted")

    async def look(args):
        return _text("ok")

    tools = [
        ToolSpec("look", "look", {"q": str}, look),
        ToolSpec("act", "act", {}, act, kind="action", confirm=True),
    ]
    runner = ADKRunner(verbose=False)
    asyncio.run(runner.run_agent("s", "u", tools, lambda n, a: n != "act"))
    assert acted == []


def test_make_runner_gemini():
    from behalf import make_runner

    runner = make_runner("gemini", "gemini-2.5-pro")
    assert isinstance(runner, ADKRunner) and runner.model == "gemini-2.5-pro"
    assert make_runner("gemini").model == "gemini-2.5-flash"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
