"""Model backends. Each submodule imports its own SDK at module load, so nothing
here is imported until you ask for a specific runner — installing one backend's
extra doesn't drag in the others.

`from behalf.runner import SDKRunner` resolves lazily; importing the submodule
directly (`behalf.runner.sdk`) is equivalent.
"""

_RUNNERS = {
    "SDKRunner": ".sdk",  # Claude, via claude-agent-sdk
    "ADKRunner": ".adk",  # Gemini, via google-adk
    "StrandsRunner": ".strands",  # AWS Bedrock, via strands-agents
}


def __getattr__(name):
    module = _RUNNERS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(_RUNNERS)
