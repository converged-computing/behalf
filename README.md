# behalf

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
