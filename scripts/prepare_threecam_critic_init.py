#!/usr/bin/env python3
"""Write a tiny three-camera replay for critic schema/model initialization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ogpo.replay import load_replay, save_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = load_replay(args.source, mmap=True)
    expected = {"image_base", "image_left_wrist", "image_right_wrist"}
    if set(batch.images or {}) != expected:
        raise ValueError(f"expected three cameras: {expected}")
    if batch.batch_size < 8:
        raise ValueError("model init source has fewer than eight transitions")
    save_replay(batch.index_select(torch.arange(8)), args.output)
    print(f"three-camera model init: {args.output}", flush=True)


if __name__ == "__main__":
    main()
