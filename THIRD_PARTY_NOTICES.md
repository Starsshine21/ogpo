# Third-Party Notices

This repository vendors a modified copy of
[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)
under `third_party/openpi` so the PI0.5 PyTorch/JAX actor adapters can be
reproduced without depending on a moving upstream checkout.

The original OpenPI license, Gemma license, and notice are preserved in:

- `third_party/openpi/LICENSE`
- `third_party/openpi/LICENSE_GEMMA.txt`
- `third_party/openpi/NOTICE`

RoboTwin is an external runtime dependency and is not vendored. Install or
clone RoboTwin separately, then set `ROBOTWIN_ROOT` before collecting or
evaluating episodes.

