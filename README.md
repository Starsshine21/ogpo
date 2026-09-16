# OGPO: Offline Generative Policy Optimization for PI0.5

This repository contains the standalone OGPO implementation used for our
RoboTwin PI0.5 experiments. It includes:

- multi-head U-DIVL, scalar-Q, and categorical-Q critic training;
- task-balanced, outcome-balanced, and episode-bootstrap critic sampling;
- frozen-critic OGPO/PPO actor training for PyTorch and JAX PI0.5;
- OGPO critic readiness and CARL policy-alignment evaluation;
- fixed-manifest RoboTwin evaluation protocols;
- Best-of-N and Q-guided flow (QGF) test-time inference baselines;
- a vendored, reproducible copy of the modified OpenPI source.

The package import name is `ogpo`. The original experimental monorepo is not
required.

## What Is Not Included

Large local artifacts are intentionally excluded:

- raw RoboTwin trajectories (`frames.npz`, videos, images);
- prepared replay `.pt` files;
- model checkpoints, optimizer states, and tensorboard/event logs;
- cluster-specific Slurm output;
- local Conda environments and caches.

The fixed mixed-10 protocol metadata is included under
`protocols/robotwin_mixed1000/`. It records manifests, split assignments,
bootstrap masks, evaluation seeds, prompts, and policy-noise rules, but not the
large raw episodes needed to rebuild the training replay.

## Repository Layout

```text
src/ogpo/                  Core critic, actor, replay, and evaluator package
scripts/                   Training, data-building, evaluation, and inference CLIs
configs/ogpo/              Final mixed-10 recipes and generic templates
protocols/                 Fixed RoboTwin manifests and evaluation protocol
third_party/openpi/        Vendored modified OpenPI source
tests/                     Unit and protocol tests
docs/                      Algorithm notes and experiment references
```

## Requirements

- Linux with NVIDIA GPU for full training and RoboTwin evaluation;
- Python 3.11;
- CUDA 12-compatible PyTorch/JAX installation;
- PI0.5 base checkpoint;
- local Gemma and SigLIP model directories for the multimodal critic;
- RoboTwin only if you collect or evaluate new simulation episodes.

Training from an already prepared replay does not require RoboTwin. Online
RoboTwin collection, strict-100 evaluation, Best-of-N, and QGF do require a
RoboTwin checkout and assets.

## Installation

```bash
git clone https://github.com/Starsshine21/ogpo.git
cd ogpo
conda env create -f environment.yml
conda activate ogpo
python scripts/check_install.py
```

If you prefer an existing environment:

```bash
python -m pip install -e .
python -m pip install -e ./third_party/openpi
python -m pip install -e ./third_party/openpi/packages/openpi-client
python -m pip install pytest
python scripts/check_install.py
```

The OpenPI copy is vendored because OGPO uses local PI0.5 compatibility
changes. See `THIRD_PARTY_NOTICES.md`.

## External Assets

Create local directories for large assets. The defaults in the configs are:

```text
checkpoints/models/gemma-3-270m/
checkpoints/models/siglip2-so400m-patch14-224-fixed/
checkpoints/pi05/model_clean50/
```

Expected asset roles:

- Gemma directory: critic language backbone;
- SigLIP directory: critic vision backbone;
- PI0.5 directory: base actor checkpoint and normalization/config assets.

You can keep these directories anywhere and edit the YAML paths instead. Do not
commit them.

For RoboTwin rollout/evaluation:

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
export PI05_ROOT=/path/to/pi05
export PI05_CHECKPOINT_DIR=/path/to/model_clean50
```

## Quick Smoke Tests

Run a synthetic critic smoke test:

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/critic_udivl.yaml \
  --smoke --synthetic-smoke
```

Run a short actor smoke test after you have a critic checkpoint:

```bash
python scripts/train_full_ogpo.py \
  --config configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml \
  --critic-checkpoint /path/to/critic.pt \
  --actor-steps 10 --smoke
```

The actor smoke test still needs the prepared replay paths from the config and
a compatible PI0.5 base checkpoint.

## Data Preparation

OGPO trains on chunk transitions, not single-step actions. A prepared replay
contains fields such as:

- `observations`, `proprioceptions`;
- `action_chunks` with shape `[N, horizon, action_dim]`;
- `execution_masks`, `executed_lengths`;
- `chunk_returns`, `discounts`, `dones`, `successes`;
- `episode_ids`, `timesteps`, `task_ids`, `languages`;
- optional image dictionaries and cached critic features.

To build a RoboTwin replay from dense episode directories:

```bash
python scripts/build_robotwin_multitask_streaming_replay.py \
  --input-root /path/to/dense_rollouts \
  --config configs/ogpo/robotwin_multitask10_divl_vmean_headonly_taskoutcomebalanced_20k.yaml \
  --output-prefix outputs/ogpo/replays/my_mixed_replay \
  --expected-episodes 1000 \
  --num-train-shards 4
```

For the frozen mixed-1000/bootstrap recipe, copy the raw episodes described by
`protocols/robotwin_mixed1000/data_transfer_manifest.json`, preserve their
global episode IDs and split assignments, then run:

```bash
python scripts/prepare_mixed1000_bootstrap.py --build
```

That recipe writes:

```text
outputs/ogpo/replays/mixed1000_bootstrap_v1/train_rank00.pt
outputs/ogpo/replays/mixed1000_bootstrap_v1/train_rank01.pt
outputs/ogpo/replays/mixed1000_bootstrap_v1/train_rank02.pt
outputs/ogpo/replays/mixed1000_bootstrap_v1/train_rank03.pt
outputs/ogpo/replays/mixed1000_bootstrap_v1/validation_80.pt
outputs/ogpo/replays/mixed1000_bootstrap_v1/model_init_8.pt
outputs/ogpo/replays/mixed1000_bootstrap_v1/bootstrap_masks.json
```

If you use your own dataset, update `data.*` in a copied config rather than
changing the fixed recipe in place.

## Critic Training

Final categorical-Q bootstrap recipe:

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml
```

Final scalar-Q bootstrap recipe:

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_scalar_shared32_20k.yaml
```

For four-process distributed critic training with pre-sharded replay files:

```bash
torchrun --nproc_per_node=4 scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml
```

Useful overrides:

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --resume outputs/ogpo/checkpoints/.../latest.pt

python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --critic-steps 2000
```

The default outputs are under `outputs/ogpo/`, which is ignored by Git.

## Critic Evaluation

### OGPO critic evaluator v2

The evaluator compares one or more checkpoints on the same fixed candidate
cache:

```bash
python scripts/evaluate_ogpo_critic_v2.py \
  --candidate-cache /path/to/candidate_cache.pt \
  --output-dir outputs/ogpo/evaluations/my_eval \
  --critic-spec CatQ::/path/to/catq.pt::configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --critic-spec Scalar::/path/to/scalar.pt::configs/ogpo/robotwin_mixed1000_bootstrap_scalar_shared32_20k.yaml
```

It reports value fidelity, candidate reliability, advantage density, and
ensemble diagnostics. Candidate caches are data artifacts and are not committed.

### CARL v2

First freeze a candidate bank:

```bash
python scripts/prepare_carl.py \
  --candidate-cache /path/to/candidate_cache.pt \
  --output-dir outputs/ogpo/carl/bank_v1 \
  --states-per-task 10 \
  --terminal-replay /path/to/terminal_replay.pt \
  --logged-replay /path/to/logged_replay.pt
```

Then evaluate checkpoints against that bank:

```bash
python scripts/evaluate_carl.py \
  --root outputs/ogpo/carl/bank_v1 \
  --report-dir outputs/ogpo/carl/report_v1 \
  --critic-spec CatQ::/path/to/catq.pt::configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --critic-spec Scalar::/path/to/scalar.pt::configs/ogpo/robotwin_mixed1000_bootstrap_scalar_shared32_20k.yaml
```

CARL v2 performs Sparse Reward Policy Alignment. It does not execute candidate
actions in the simulator.

## Actor Training

Train the CatQ-supervised mixed-10 OGPO actor:

```bash
python scripts/train_full_ogpo.py \
  --config configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml \
  --critic-checkpoint /path/to/catq_critic.pt
```

Resume from an actor checkpoint:

```bash
python scripts/train_full_ogpo.py \
  --config configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml \
  --resume outputs/ogpo/checkpoints/.../latest.pt
```

The actor config uses:

- `G=4` candidate groups;
- frozen critic;
- task-balanced actor sampling;
- conservative advantage by default;
- periodic checkpointing under `outputs/ogpo/checkpoints/`.

Before a long run, verify `training.actor_steps`, `training.batch_size`,
`actor.gradient_microbatch_size`, and all replay/checkpoint paths.

## Test-Time Inference Baselines

Best-of-N and QGF are implemented for RoboTwin online evaluation through
`scripts/collect_robotwin_dense_rollouts.py`.

Best-of-4:

```bash
python scripts/collect_robotwin_dense_rollouts.py \
  --task-name click_bell \
  --task-config demo_clean \
  --train-config-name pi05_robotwin2_clean50_full \
  --model-name model_clean50 \
  --checkpoint-id 20000 \
  --pi05-checkpoint-dir "$PI05_CHECKPOINT_DIR" \
  --episode-manifest /path/to/click_bell.json \
  --trust-episode-manifest \
  --seed 0 --num-episodes 13 \
  --output-dir outputs/robotwin_eval/bestof4/click_bell/shard_00 \
  --bestofn-critic-checkpoint /path/to/catq_critic.pt \
  --bestofn-critic-config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --bestofn-group-size 4
```

QGF:

```bash
python scripts/collect_robotwin_dense_rollouts.py \
  --task-name click_bell \
  --task-config demo_clean \
  --train-config-name pi05_robotwin2_clean50_full \
  --model-name model_clean50 \
  --checkpoint-id 20000 \
  --pi05-checkpoint-dir "$PI05_CHECKPOINT_DIR" \
  --episode-manifest /path/to/click_bell.json \
  --trust-episode-manifest \
  --seed 0 --num-episodes 13 \
  --output-dir outputs/robotwin_eval/qgf/click_bell/shard_00 \
  --qgf-critic-checkpoint /path/to/catq_critic.pt \
  --qgf-critic-config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --qgf-strength 1.0
```

Both modes score complete action chunks with the mean of the 10 raw Q heads.
They do not modify the base actor.

## Fixed Strict-100 Evaluation

The mixed-10 evaluation bundle lives at:

```text
protocols/robotwin_mixed1000/evaluation/
```

It contains the 10 task manifests, fixed prompts, environment seeds, policy
noise seeds, historical Base metadata, and comparison utilities.

Example:

```bash
cd protocols/robotwin_mixed1000/evaluation
python scripts/bundle_tools.py validate --bundle-root "$PWD"

export OGPO_ROOT=/path/to/ogpo
export PI05_ROOT=/path/to/pi05
export ROBOTWIN_ROOT=/path/to/RoboTwin
export PI05_CHECKPOINT_DIR=/path/to/model_clean50
export PYTHON_BIN=/path/to/python
export OUTPUT_ROOT=/path/to/results/base
export POLICY_LABEL=base
bash scripts/run_policy_strict100.sh
```

For an actor checkpoint:

```bash
export OUTPUT_ROOT=/path/to/results/actor
export POLICY_LABEL=ogpo_actor
export ACTOR_CHECKPOINT=/path/to/actor.pt
bash scripts/run_policy_strict100.sh
```

The script supports 1-8 visible GPUs. Keep the same manifest and noise protocol
for every compared model.

## Tests

Run the core unit tests:

```bash
PYTHONPATH=src:third_party/openpi/src:third_party/openpi/packages/openpi-client/src \
  pytest tests/ogpo tests/test_carl_metrics.py tests/test_evaluate_ogpo_critic_v2.py -q
```

Some tests need OpenPI/JAX or mocked model components; simulation rollout tests
additionally need RoboTwin and external checkpoints.

## Documentation

- `docs/QUICKSTART_ZH.md`: Chinese end-to-end quickstart;
- `docs/ogpo_implementation_zh.md`: implementation details;
- `docs/ogpo_paper_derivation_zh.md`: algorithm derivation;
- `docs/ogpo_critic_evaluation_v2_2026-09-14_zh.md`: critic evaluator v2;
- `docs/CARL_sparse_reward_policy_alignment_v2_zh.md`: CARL v2 protocol;
- `docs/BestOf4_inference_v1_zh.md`: Best-of-4 inference;
- `docs/QGF_inference_v1_zh.md`: QGF inference;
- `protocols/robotwin_mixed1000/README.md`: data/split/noise protocol;
- `protocols/robotwin_mixed1000/evaluation/README.md`: strict-100 evaluation.

## License

OGPO code in this repository is released under the MIT License; see `LICENSE`.
Third-party notices are in `THIRD_PARTY_NOTICES.md`.

