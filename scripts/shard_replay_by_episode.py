#!/usr/bin/env python3
"""Split one replay into deterministic, episode-complete distributed shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ogpo.replay import load_replay, save_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-pattern", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    outputs = [Path(args.output_pattern.format(rank=rank)) for rank in range(args.num_shards)]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite replay shards: {existing}")

    batch = load_replay(args.input)
    episode_ids = torch.unique(batch.episode_ids, sorted=True)
    for rank, output in enumerate(outputs):
        selected = episode_ids[rank :: args.num_shards]
        mask = torch.isin(batch.episode_ids, selected)
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        shard = batch.index_select(indices)
        save_replay(shard, output)
        print(
            f"rank={rank} episodes={selected.numel()} transitions={shard.batch_size} output={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()

