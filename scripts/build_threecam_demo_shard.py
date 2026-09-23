#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ogpo.replay import add_monte_carlo_returns, make_n_step_replay  # noqa: E402
from ogpo.zarr_replay import rows_to_chunk_batch  # noqa: E402
from build_robotwin_critic_replay import _episode_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    episodes = sorted(args.dense_root.glob("*/shard_*/raw_rollouts/episode_*"))
    if len(episodes) != 500:
        raise ValueError(f"expected 500 demo episodes, found {len(episodes)}")
    selected = [
        episode for episode in episodes
        if int(episode.parents[1].name.removeprefix("shard_")) == args.rank
    ]
    rows = []
    for episode_id, episode in enumerate(selected):
        rows.extend(_episode_rows(
            episode,
            episode_id=episode_id,
            gamma=0.995,
            behavior_policy="robotwin2_aloha_agilex_clean50_demo_threecam",
        ))
    replay = make_n_step_replay(add_monte_carlo_returns(rows_to_chunk_batch(rows)), n_step=2)
    if set(replay.images or {}) != {"image_base", "image_left_wrist", "image_right_wrist"}:
        raise ValueError(f"unexpected camera keys: {sorted(replay.images or {})}")
    if not bool(replay.successes.bool().all()):
        raise ValueError("clean50 demo shard contains a failure episode")
    report = {
        "rank": args.rank,
        "world_size": args.world_size,
        "episodes": len(selected),
        "transitions": replay.batch_size,
        "tasks": sorted(set(replay.task_ids)),
        "camera_keys": sorted(replay.images),
        "gamma": 0.995,
        "n_step": 2,
    }
    del rows
    gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {field.name: getattr(replay, field.name) for field in fields(replay)}
    torch.save(payload, args.output)
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
