#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ogpo.replay import (  # noqa: E402
    add_monte_carlo_returns,
    save_replay,
    save_replay_metadata,
    split_replay,
    split_success_buffers,
)
from ogpo.zarr_replay import rows_to_chunk_batch  # noqa: E402


def validate_dense_transition_gamma(
    *,
    episode_dir: Path,
    gamma: float,
    executed_lengths: np.ndarray,
    chunk_returns: np.ndarray,
    discounts: np.ndarray,
) -> None:
    """Validate preserved dense transition values against their source gamma.

    Dense rollouts already encode their true executed-prefix return semantics,
    so the builder must not recompute them.  Current RoboTwin dense data stores
    ``gamma ** executed_length`` for every transition, including terminal
    transitions; terminal bootstrapping is disabled separately by ``done``.
    """
    if not np.isfinite(chunk_returns).all():
        raise ValueError(f"{episode_dir}: dense chunk_return contains NaN or infinity")
    if not np.isfinite(discounts).all():
        raise ValueError(f"{episode_dir}: dense discount contains NaN or infinity")
    if np.any(executed_lengths <= 0):
        index = int(np.flatnonzero(executed_lengths <= 0)[0])
        raise ValueError(
            f"{episode_dir}: dense executed_length must be positive; "
            f"timestep={index}, executed_length={int(executed_lengths[index])}"
        )
    expected_discounts = np.power(
        float(gamma), executed_lengths.astype(np.float64)
    )
    matches = np.isclose(
        discounts.astype(np.float64),
        expected_discounts,
        rtol=1e-5,
        atol=1e-7,
    )
    if not bool(matches.all()):
        mismatch_indices = np.flatnonzero(~matches)
        first = int(mismatch_indices[0])
        absolute_errors = np.abs(
            discounts.astype(np.float64) - expected_discounts
        )
        raise ValueError(
            f"{episode_dir}: Dense replay discount is inconsistent with "
            f"--gamma={float(gamma)}. The builder does not rebase dense replay "
            "gamma. Regenerate the rollout/replay with the correct source gamma "
            "or pass the actual source gamma. "
            f"max_absolute_error={float(absolute_errors.max()):.9g}, "
            f"first_mismatched_timestep={first}, "
            f"actual_discount={float(discounts[first]):.9g}, "
            f"expected_discount={float(expected_discounts[first]):.9g}, "
            f"executed_length={int(executed_lengths[first])}, "
            f"gamma={float(gamma)}"
        )


def _episode_rows(
    episode_dir: Path,
    *,
    episode_id: int,
    gamma: float,
    behavior_policy: str,
) -> list[dict]:
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "frames.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    required = {
        "image",
        "wrist_image",
        "state",
        "actions",
        "next_image",
        "next_wrist_image",
        "next_state",
        "timestamp",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"{episode_dir}: missing arrays {sorted(missing)}")

    actions = np.asarray(arrays["actions"], dtype=np.float32)
    states = np.asarray(arrays["state"], dtype=np.float32)
    next_states = np.asarray(arrays["next_state"], dtype=np.float32)
    if actions.ndim != 3 or actions.shape[2] != states.shape[1]:
        raise ValueError(
            f"{episode_dir}: expected actions [T,H,{states.shape[1]}], got {actions.shape}"
        )
    length, horizon, _ = actions.shape
    if length != int(meta["num_frames"]):
        raise ValueError(f"{episode_dir}: frame count disagrees with metadata")
    if states.shape != next_states.shape or states.shape[0] != length:
        raise ValueError(f"{episode_dir}: invalid state/next_state shapes")
    if float(np.abs(actions).max()) == 0.0:
        raise ValueError(f"{episode_dir}: all-zero action chunk")

    success = bool(meta["success"])
    dense_fields = {
        "execution_mask",
        "executed_lengths",
        "chunk_return",
        "discount",
        "done",
    }
    present_dense_fields = dense_fields.intersection(arrays)
    if present_dense_fields and present_dense_fields != dense_fields:
        raise ValueError(
            f"{episode_dir}: incomplete dense transition fields; "
            f"missing {sorted(dense_fields.difference(arrays))}"
        )
    is_dense = present_dense_fields == dense_fields
    if is_dense:
        execution_masks = np.asarray(arrays["execution_mask"], dtype=np.bool_)
        executed_lengths = np.asarray(arrays["executed_lengths"], dtype=np.int64)
        chunk_returns = np.asarray(arrays["chunk_return"], dtype=np.float32)
        discounts = np.asarray(arrays["discount"], dtype=np.float32)
        dones = np.asarray(arrays["done"], dtype=np.float32)
        expected_shapes = {
            "execution_mask": (length, horizon),
            "executed_lengths": (length,),
            "chunk_return": (length,),
            "discount": (length,),
            "done": (length,),
        }
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(
                    f"{episode_dir}: {key} shape {arrays[key].shape}, expected {shape}"
                )
        validate_dense_transition_gamma(
            episode_dir=episode_dir,
            gamma=gamma,
            executed_lengths=executed_lengths,
            chunk_returns=chunk_returns,
            discounts=discounts,
        )
    else:
        execution_masks = np.ones((length, horizon), dtype=np.bool_)
        executed_lengths = np.full((length,), horizon, dtype=np.int64)
        chunk_returns = np.zeros((length,), dtype=np.float32)
        if success:
            chunk_returns[-1] = float(gamma ** (horizon - 1))
        discounts = np.full((length,), float(gamma**horizon), dtype=np.float32)
        dones = np.zeros((length,), dtype=np.float32)
        dones[-1] = 1.0
    rows = []
    for index in range(length):
        rows.append(
            {
                "observation": states[index],
                "proprioception": states[index],
                "action_chunk": actions[index],
                "execution_mask": execution_masks[index],
                "executed_length": int(executed_lengths[index]),
                "chunk_return": float(chunk_returns[index]),
                "discount": float(discounts[index]),
                "next_observation": next_states[index],
                "next_proprioception": next_states[index],
                "images": {
                    "image_base": arrays["image"][index],
                    "image_wrist": arrays["wrist_image"][index],
                },
                "next_images": {
                    "image_base": arrays["next_image"][index],
                    "image_wrist": arrays["next_wrist_image"][index],
                },
                "done": float(dones[index]),
                "success": float(success),
                "episode_id": episode_id,
                "timestep": index,
                "task_id": str(meta["task_name"]),
                "language": str(meta["prompt"]),
                "behavior_metadata": {
                    "behavior_policy": behavior_policy,
                    "source_path": str(episode_dir),
                    "source_seed": int(meta["seed"]),
                    "ogpo_checkpoint": meta.get("ogpo_checkpoint"),
                    "action_semantics": (
                        "dense_sliding_executed_pi05_absolute_joint_chunk"
                        if is_dense
                        else "executed_pi05_absolute_joint_chunk"
                    ),
                    "transition_success": bool(
                        success and float(chunk_returns[index]) > 0.0
                    ),
                    "task_config": str(meta["task_config"]),
                    "train_config": str(meta["train_config"]),
                    "model_name": str(meta["model_name"]),
                },
            }
        )
    return rows


def _split_summary(batch) -> dict:
    episode_ids = torch.unique(batch.episode_ids)
    success_episodes = 0
    for episode_id in episode_ids.tolist():
        mask = batch.episode_ids == int(episode_id)
        success_episodes += int(bool(batch.successes[mask].max().item()))
    return {
        "transitions": batch.batch_size,
        "episodes": int(episode_ids.numel()),
        "success_episodes": success_episodes,
        "failure_episodes": int(episode_ids.numel()) - success_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument(
        "--episode-limit",
        type=int,
        help="Deterministically keep the first N sorted episodes before validation.",
    )
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument(
        "--behavior-policy",
        default="pi05_jax_robotwin",
        help="Behavior-policy provenance stored on every replay transition.",
    )
    parser.add_argument(
        "--allow-single-outcome",
        action="store_true",
        help="Permit one-outcome transition splits for a one-episode smoke replay.",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Save train/validation/heldout only, omitting duplicated all/outcome buffers.",
    )
    args = parser.parse_args()

    episode_dirs = sorted(
        args.input_root.glob("shards/shard_*/raw_rollouts/episode_*")
    )
    if not episode_dirs:
        episode_dirs = sorted(
            args.input_root.glob("shard_*/raw_rollouts/episode_*")
        )
    if not episode_dirs:
        # A multitask rollout root stores each task under its own directory,
        # e.g. <root>/<task>/shard_00/raw_rollouts/episode_*.  Keep this
        # fallback after the single-task layouts so existing replay builds
        # preserve their ordering and behavior.
        episode_dirs = sorted(
            args.input_root.glob("*/shard_*/raw_rollouts/episode_*")
        )
    if not episode_dirs:
        episode_dirs = sorted((args.input_root / "raw_rollouts").glob("episode_*"))
    if args.episode_limit is not None:
        if args.episode_limit <= 0:
            raise ValueError("--episode-limit must be positive")
        if len(episode_dirs) < args.episode_limit:
            raise ValueError(
                f"episode limit requests {args.episode_limit}, found only {len(episode_dirs)}"
            )
        episode_dirs = episode_dirs[: args.episode_limit]
    if len(episode_dirs) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} episodes, found {len(episode_dirs)}"
        )
    rows = []
    for episode_id, episode_dir in enumerate(episode_dirs):
        rows.extend(
            _episode_rows(
                episode_dir,
                episode_id=episode_id,
                gamma=float(args.gamma),
                behavior_policy=str(args.behavior_policy),
            )
        )
    batch = add_monte_carlo_returns(rows_to_chunk_batch(rows))
    splits = split_replay(
        batch,
        train_ratio=0.8,
        validation_ratio=0.1,
        seed=int(args.seed),
    )
    for name, split in splits.items():
        summary = _split_summary(split)
        if (
            not args.allow_single_outcome
            and (summary["success_episodes"] == 0 or summary["failure_episodes"] == 0)
        ):
            raise ValueError(f"split {name} lacks both outcomes: {summary}")

    if len(episode_dirs) >= 3:
        episode_sets = {
            name: set(split.episode_ids.tolist()) for name, split in splits.items()
        }
        names = list(episode_sets)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                overlap = episode_sets[left_name].intersection(episode_sets[right_name])
                if overlap:
                    raise ValueError(
                        f"episode leakage between {left_name} and {right_name}: "
                        f"{sorted(overlap)}"
                    )

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not args.splits_only:
        save_replay(batch, output)
    for name, split in splits.items():
        save_replay(split, output.with_name(f"{output.stem}_{name}{output.suffix}"))
    if not args.splits_only:
        for name, subset in split_success_buffers(batch).items():
            if name != "all":
                save_replay(subset, output.with_name(f"{output.stem}_{name}{output.suffix}"))

    metadata = {
        "source": "robotwin_dense_actionfix_v2_shards",
        "source_root": str(args.input_root),
        "behavior_policy": str(args.behavior_policy),
        "gamma": float(args.gamma),
        "reward_semantics": "unit success reward on final micro-action",
        "action_semantics": "dense sliding executed PI0.5 absolute-joint chunk",
        "generated_horizon": batch.generated_horizon,
        "action_dim": batch.action_dim,
        "all": _split_summary(batch),
        "splits": {name: _split_summary(split) for name, split in splits.items()},
        "max_abs_action": float(batch.action_chunks.abs().max().item()),
        "episode_split_isolated": len(episode_dirs) >= 3,
        "splits_only": bool(args.splits_only),
        "replay_build_passed": True,
    }
    save_replay_metadata(output, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"output={output}")


if __name__ == "__main__":
    main()
