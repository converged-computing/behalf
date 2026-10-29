# behalf

[![DOI](https://zenodo.org/badge/1309288270.svg)](https://doi.org/10.5281/zenodo.21745822)

A small framework for the recurring pattern of a **scoped agent**: hand it a
brief, a fixed set of tools, and bounded authority, and it does one job.

## Abstractions

- **Task** what to elicit from the user, which tools the agent gets, and what a valid result is.
- **ToolSpec** can be a single tool or action.
- **runner backend** is the model like the Claude Agent SDK, Google ADK, or Strands on Bedrock.


```python
from behalf import run_task, make_runner
outcome = await run_task(MyTask(), make_runner("aws"))
```

### Manifests

`manifest_schema()` is sent to the model during setup (see `render_setup_prompt`),
and `finalize_setup` only accepts a manifest that has every schema field. `run_task`
raises `ValueError` if a manifest, conversed or passed in, is missing a field.

### Results

Runners return `None` from `run_agent`; results come from tool side effects.
`record_tool(sink, fields)` returns a tool that appends each call's arguments to `sink`:

```python
from behalf import record_tool

class MyTask(Task):
    def build_tools(self, manifest):
        self.results = []
        return [record_tool(self.results, {"job": str, "site": str})]
```


## Install

```bash
pip install behalf            # core only (no backend)
pip install behalf[claude]    # + Claude Agent SDK
pip install behalf[aws]       # + Strands / Bedrock
pip install behalf[gemini]    # + Google ADK
```

## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614
