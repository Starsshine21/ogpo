#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "openpi" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "openpi" / "packages" / "openpi-client" / "src"))

from ogpo.inference_policy import create_pi05_ogpo_inference_policy


def _changed_backend_leaf(current, reference) -> tuple[str, float]:
    current_with_paths = jax.tree_util.tree_flatten_with_path(current.to_pure_dict())[0]
    reference_leaves = jax.tree_util.tree_leaves(reference.to_pure_dict())
    for (path, current_leaf), reference_leaf in zip(
        current_with_paths,
        reference_leaves,
        strict=True,
    ):
        if not path or getattr(path[0], "key", None) != "backend":
            continue
        # Keep the full-leaf comparison in its native dtype so XLA can fuse the
        # comparison and reduction. Casting a multi-billion-element parameter
        # leaf to float32 creates a 4.5 GiB temporary on A100-40G.
        if not bool(jax.device_get(jnp.any(current_leaf != reference_leaf))):
            continue
        flat_current = current_leaf.reshape(-1)
        flat_reference = reference_leaf.reshape(-1)
        sample_count = min(4096, flat_current.size)
        sample_indices = jnp.linspace(
            0,
            flat_current.size - 1,
            sample_count,
            dtype=jnp.int32,
        )
        sample_difference = float(
            jax.device_get(
                jnp.max(
                    jnp.abs(
                        flat_current[sample_indices].astype(jnp.float32)
                        - flat_reference[sample_indices].astype(jnp.float32)
                    )
                )
            )
        )
        return "/".join(str(entry) for entry in path), sample_difference
    raise RuntimeError("restored current actor has no changed PI0.5 backend parameters")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--ogpo-checkpoint", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = create_pi05_ogpo_inference_policy(
        pi05_checkpoint_dir=args.pi05_checkpoint,
        train_config_name=args.train_config,
        ogpo_checkpoint=args.ogpo_checkpoint,
        device=args.device,
    )
    if policy.reference_flow_policy is None:
        raise RuntimeError("inference policy did not construct the frozen SFT reference")
    changed_leaf, sample_max_difference = _changed_backend_leaf(
        policy.flow_policy.actor_state,
        policy.reference_flow_policy.actor_state,
    )
    print(
        f"checkpoint_restore_ok changed_backend_leaf={changed_leaf} "
        f"sample_max_abs_difference={sample_max_difference:.8g}"
    )

    replay = torch.load(args.replay, map_location="cpu", weights_only=False)
    base_image = replay["images"]["image_base"][0].numpy()
    wrist_image = replay["images"]["image_wrist"][0].numpy()
    observation = {
        "state": replay["proprioceptions"][0].numpy(),
        "base": base_image,
        "wrist": wrist_image,
        "prompt": replay["languages"][0],
    }
    noise = np.zeros(
        (
            policy.flow_policy.model_horizon,
            policy.flow_policy.model_action_dim,
        ),
        dtype=np.float32,
    )
    output = policy.infer(observation, noise=noise)
    actions = np.asarray(output["actions"])
    if actions.shape != (
        policy.flow_policy.model_horizon,
        policy.flow_policy.environment_action_dim,
    ):
        raise RuntimeError(f"unexpected inferred action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("inference returned non-finite actions")
    print(
        f"inference_ok action_shape={actions.shape} "
        f"reference_divergence={output['policy_reference_action_divergence']:.8g} "
        f"predicted_q={output['predicted_q']:.8g}"
    )


if __name__ == "__main__":
    main()
