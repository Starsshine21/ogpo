#!/usr/bin/env python3
"""Apply configured replay transforms to rank shards one at a time."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ogpo.replay import load_replay, prepare_replay_from_config, save_replay
from train_udivl_critic import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-pattern", required=True)
    parser.add_argument("--output-pattern", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    for rank in range(args.num_shards):
        source = Path(args.input_pattern.format(rank=rank))
        destination = Path(args.output_pattern.format(rank=rank))
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        batch = load_replay(source)
        prepared = prepare_replay_from_config(batch, config["data"])
        save_replay(prepared, destination)
        print(
            f"rank={rank} transitions={prepared.batch_size} output={destination}",
            flush=True,
        )


if __name__ == "__main__":
    main()

