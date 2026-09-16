#!/usr/bin/env python3
"""Prepare and save episode-complete distributed shards without raw shard files."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ogpo.replay import load_replay, prepare_replay_from_config, save_replay
from train_udivl_critic import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-pattern", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    outputs = [Path(args.output_pattern.format(rank=rank)) for rank in range(args.num_shards)]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prepared shards: {existing}")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    config = load_config(config_path)
    batch = load_replay(input_path)
    episode_ids = torch.unique(batch.episode_ids, sorted=True)
    for rank, output in enumerate(outputs):
        selected = episode_ids[rank :: args.num_shards]
        indices = torch.nonzero(torch.isin(batch.episode_ids, selected), as_tuple=False).flatten()
        shard = batch.index_select(indices)
        prepared = prepare_replay_from_config(shard, config["data"])
        save_replay(prepared, output)
        print(
            f"rank={rank} episodes={selected.numel()} transitions={prepared.batch_size} output={output}",
            flush=True,
        )
        del shard, prepared, indices
        gc.collect()


if __name__ == "__main__":
    main()
