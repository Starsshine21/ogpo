# OGPO Configuration Guide

The YAML files in this directory are runnable templates, not references to
files that exist in a fresh clone. Copy a recipe, then update three groups of
paths before launching:

1. `data.*`: prepared replay `.pt` files.
2. `critic.backbone.*`: local Gemma/SigLIP model directories.
3. `flow.*` and `training.critic_checkpoint`: PI0.5 base and trained critic
   checkpoints.

## Final RoboTwin mixed-10 recipes

- Scalar bootstrap critic:
  `robotwin_mixed1000_bootstrap_scalar_shared32_20k.yaml`
- Categorical-Q bootstrap critic:
  `robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml`
- CatQ-supervised OGPO actor:
  `robotwin_mixed1000_catq9k_actor_2k.yaml`

The critic configs use task-balanced + outcome-balanced sampling, frozen
Gemma/SigLIP backbones, and fixed bootstrap masks. The actor config freezes the
critic and trains the PI0.5 actor with OGPO/PPO.

## Generic smoke recipes

- `critic_udivl.yaml`: small synthetic critic/actor configuration.
- `pi05_gemma_udivl_critic.yaml`: Gemma+SigLIP critic template.
- `pi05_jax_flash_ogpo_100ep.yaml`: JAX PI0.5 actor template.

## Important path placeholders

The checked-in configs intentionally point at ignored local directories:

- `checkpoints/models/gemma-3-270m`
- `checkpoints/models/siglip2-so400m-patch14-224-fixed`
- `checkpoints/pi05/model_clean50`
- `outputs/ogpo/replays/...`
- `outputs/ogpo/checkpoints/...`

These paths are examples. Keep large assets outside Git or replace the values
with absolute paths on your machine.

