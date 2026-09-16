#!/usr/bin/env python3
"""Build episode-split replay outputs one bounded group at a time."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from build_robotwin_critic_replay import _episode_rows
from ogpo.replay import (
    add_monte_carlo_returns,
    prepare_replay_from_config,
    save_replay,
)
from ogpo.zarr_replay import rows_to_chunk_batch
from train_udivl_critic import load_config


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_group(
    episode_dirs: list[Path],
    episode_ids: list[int],
    *,
    gamma: float,
    behavior_policy: str,
):
    rows = []
    for episode_id in episode_ids:
        rows.extend(
            _episode_rows(
                episode_dirs[episode_id],
                episode_id=episode_id,
                gamma=gamma,
                behavior_policy=behavior_policy,
            )
        )
    batch = add_monte_carlo_returns(rows_to_chunk_batch(rows))
    del rows
    gc.collect()
    return batch


def split_summary(batch) -> dict:
    success = batch.successes.bool()
    episodes = torch.unique(batch.episode_ids)
    task_counts = {}
    for task in sorted(set(str(value) for value in batch.task_ids)):
        indices = [index for index, value in enumerate(batch.task_ids) if str(value) == task]
        selected = torch.tensor(indices, dtype=torch.long)
        episode_ids = torch.unique(batch.episode_ids.index_select(0, selected))
        success_episodes = 0
        for episode_id in episode_ids.tolist():
            mask = batch.episode_ids == int(episode_id)
            success_episodes += int(bool(batch.successes[mask][0]))
        task_counts[task] = {
            "episodes": int(episode_ids.numel()),
            "success_episodes": success_episodes,
            "failure_episodes": int(episode_ids.numel()) - success_episodes,
            "transitions": int(selected.numel()),
            "success_transitions": int(success.index_select(0, selected).sum()),
            "failure_transitions": int((~success.index_select(0, selected)).sum()),
        }
    return {
        "episodes": int(episodes.numel()),
        "transitions": batch.batch_size,
        "task_counts": task_counts,
    }


def metadata_group_summary(episode_dirs: list[Path], episode_ids: list[int]) -> dict:
    task_counts = {}
    for episode_id in episode_ids:
        metadata = json.loads(
            (episode_dirs[episode_id] / "meta.json").read_text(encoding="utf-8")
        )
        task = str(metadata["task_name"])
        success = bool(metadata["success"])
        transitions = int(metadata["num_frames"])
        row = task_counts.setdefault(
            task,
            {
                "episodes": 0,
                "success_episodes": 0,
                "failure_episodes": 0,
                "transitions": 0,
                "success_transitions": 0,
                "failure_transitions": 0,
            },
        )
        row["episodes"] += 1
        row["success_episodes" if success else "failure_episodes"] += 1
        row["transitions"] += transitions
        row["success_transitions" if success else "failure_transitions"] += transitions
    return {
        "episodes": len(episode_ids),
        "transitions": sum(value["transitions"] for value in task_counts.values()),
        "task_counts": task_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--num-train-shards", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--behavior-policy", default="pi05_pytorch_clean50_base")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    episode_dirs = sorted(args.input_root.glob("*/shard_*/raw_rollouts/episode_*"))
    if len(episode_dirs) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} staged episodes, found {len(episode_dirs)}"
        )
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    generator = torch.Generator().manual_seed(args.seed)
    permuted = torch.randperm(len(episode_dirs), generator=generator).tolist()
    train_count = int(len(episode_dirs) * 0.8)
    validation_count = int(len(episode_dirs) * 0.1)
    train_ids = sorted(permuted[:train_count])
    validation_ids = sorted(permuted[train_count : train_count + validation_count])
    heldout_ids = sorted(permuted[train_count + validation_count :])
    outputs = []
    summary = {"schema_version": 1, "splits": {}, "train_shards": {}}
    for rank in range(args.num_train_shards):
        selected = train_ids[rank :: args.num_train_shards]
        prepared_path = args.output_prefix.with_name(
            f"{args.output_prefix.name}_g995_n2_train_rank{rank:02d}.pt"
        )
        outputs.append(prepared_path)
        shard_summary = metadata_group_summary(episode_dirs, selected)
        if prepared_path.exists():
            if args.resume and prepared_path.stat().st_size > 1_000_000_000:
                summary["train_shards"][str(rank)] = shard_summary
                print(
                    f"train rank={rank} resume_skip=1 episodes={len(selected)} "
                    f"transitions={shard_summary['transitions']} prepared={prepared_path}",
                    flush=True,
                )
                continue
            raise FileExistsError(f"refusing existing train shard output for rank {rank}")
        batch = build_group(
            episode_dirs,
            selected,
            gamma=args.gamma,
            behavior_policy=args.behavior_policy,
        )
        prepared = prepare_replay_from_config(batch, config["data"])
        save_replay(prepared, prepared_path)
        summary["train_shards"][str(rank)] = shard_summary
        print(
            f"train rank={rank} episodes={len(selected)} transitions={batch.batch_size} "
            f"prepared={prepared_path}",
            flush=True,
        )
        del batch, prepared
        gc.collect()
    for split, selected in (("validation", validation_ids), ("heldout", heldout_ids)):
        summary["splits"][split] = metadata_group_summary(episode_dirs, selected)
        print(
            f"split={split} metadata_only=1 episodes={len(selected)} "
            f"transitions={summary['splits'][split]['transitions']}",
            flush=True,
        )
    summary.update(
        {
            "input_root": str(args.input_root.resolve()),
            "expected_episodes": args.expected_episodes,
            "train_episodes": len(train_ids),
            "validation_episodes": len(validation_ids),
            "heldout_episodes": len(heldout_ids),
            "split_seed": args.seed,
            "gamma_source": args.gamma,
            "prepared_gamma": float(config["data"]["gamma"]),
            "prepared_n_step": int(config["data"]["n_step"]),
            "outputs": [str(path.resolve()) for path in outputs],
        }
    )
    atomic_json(
        args.output_prefix.with_name(f"{args.output_prefix.name}_BUILD_SUMMARY.json"),
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
