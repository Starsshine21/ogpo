# OGPO: current RoboTwin training recipe

This release contains the current three-camera, 10-task OGPO critic and actor
training paths. Historical experiment configs, protocol dumps, checkpoints,
replays and rollout videos are intentionally excluded.

## Method and status

- Critic: five-member distributional DIVL V, two **scalar** Q heads per member
  (10 raw Q heads), task/outcome-balanced sampling, and optional full RankQ.
  The three configs reproduce the running `tau=0.1, max_gap=1.0`, no-RankQ,
  and `tau=1` ablations. RankQ uses same-task permutation and shared-direction
  1×/2× noise with `sigma=0.04`.
- Actor: frozen-critic, raw10 conservative advantage + ChiPO, Flash-GRPO,
  head-only PI0.5, group size 8, no reference-KL update guard, and success-only
  flow-matching BC whose weighted gradient is capped at 20% of the combined
  gradient. `configs/ogpo/threecam_flash_actor.yaml` is the **next three-camera
  run template**; it has not yet produced a three-camera actor result.
- Cameras: `image_base`, `image_left_wrist`, `image_right_wrist`. PI0.5 samples
  50-step action chunks in the model's 32-dimensional action space; the
  environment executes the configured prefix.

No measured success rate is claimed by this source release.

## Layout

`src/ogpo/` is the training implementation; `third_party/openpi/` is the
modified PI0.5 dependency. `scripts/` holds the current replay preparation and
training entry points. `configs/ogpo/` holds four standalone recipes. Some
legacy-compatible source modules remain because the shared trainer imports
them; no legacy experiment launcher or config is published.

## Installation and external assets

Use Python 3.11 and a CUDA-capable environment. Install the local OpenPI copy
and OGPO package, then run the installation check:

```bash
python -m pip install -e ./third_party/openpi
python -m pip install -e ./third_party/openpi/packages/openpi-client
python -m pip install -e '.[dev]'
python scripts/check_install.py
```

Large assets are not in Git. Edit the YAML paths if yours differ:

```text
checkpoints/models/gemma-3-270m/
checkpoints/models/siglip2-so400m-patch14-224-fixed/
checkpoints/pi05/model_clean50/
data/base_threecam90/{model_init_8.pt,heldout.pt,train_rank00.pt..train_rank03.pt}
data/clean50_threecam/demo_rank00.pt..demo_rank03.pt
```

The 1,400-episode critic training set is 900 newly collected base rollouts
plus 500 official clean demos. The remaining 100 base episodes are held out.
The two training sources stay in separate file-backed shards; **do not merge
their image tensors into one replay**. A replay stores action chunks,
execution masks, trajectory outcomes, returns, task/episode IDs, language,
proprioception and three image streams. The data builder checks the camera
schema. Replays, model weights and simulator assets must be obtained separately.

Convert each downloaded RoboTwin2.0 clean50 task into dense episodes first
(the converter requires `h5py`):

```bash
python -m pip install h5py
python scripts/convert_robotwin2_raw_to_dense.py --input /path/to/task.zip --output /path/to/clean50/dense --task adjust_bottle --limit 50
```

Repeat for all ten tasks. To prepare base rollout data from dense RoboTwin
episodes, first supply a split manifest
with top-level `train` and `heldout` lists, each entry containing a `source`
episode directory. Preserve episode IDs and the 900/100 split:

```bash
python scripts/build_threecam_base90_shard.py --split-manifest split.json --split train --rank 0 --world-size 4 --output data/base_threecam90/train_rank00.pt
python scripts/build_threecam_base90_shard.py --split-manifest split.json --split heldout --rank 0 --world-size 1 --output data/base_threecam90/heldout.pt
python scripts/build_threecam_demo_shard.py --dense-root /path/to/clean50/dense --rank 0 --world-size 4 --output data/clean50_threecam/demo_rank00.pt
python scripts/prepare_threecam_critic_init.py --source data/base_threecam90/train_rank00.pt --output data/base_threecam90/model_init_8.pt
```

Repeat the train/demo shard commands for ranks 1–3. The raw dense episode
format expected by `build_robotwin_critic_replay.py` is described in its
`_episode_rows` function.

## Train

Run from the repository root. Four critic processes each mmap their own
base/demo shards. Select one of `threecam_scalar_guarded.yaml`,
`threecam_scalar_norank.yaml`, or `threecam_scalar_tau1.yaml`:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_udivl_critic.py --config configs/ogpo/threecam_scalar_guarded.yaml
```

Choose a trained critic checkpoint explicitly before actor training. The
actor YAML's 8k checkpoint path is a **placeholder example**, not an automatic
checkpoint selection; replace it with the validated checkpoint you intend to
use. The actor loads prepared shards file-backed and samples only selected
rows, avoiding a full image-tensor merge:

```bash
python scripts/train_taskbalanced_flash_ogpo.py --config configs/ogpo/threecam_flash_actor.yaml --critic-checkpoint /path/to/selected-critic.pt
```

The actor requires a PI0.5 clean50 base checkpoint with matching normalization
metadata and the vendored `pi05_robotwin2_clean50_full` transform. For
cross-machine reproduction, paths to all external assets and the exact split
manifest must be retained alongside the run's resolved-config snapshot.

## Validation

```bash
python -m pytest tests/ogpo -q
python -m py_compile scripts/*.py src/ogpo/*.py
```

See `THIRD_PARTY_NOTICES.md` for vendored dependency information.
