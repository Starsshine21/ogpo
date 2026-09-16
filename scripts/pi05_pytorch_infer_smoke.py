#!/usr/bin/env python3
"""Load a converted DexJoCo PI0.5 checkpoint and run one real inference."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from openpi.policies import policy_config
from openpi.training import config as train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="click_mouse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prompt", default="click the mouse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    config = train_config.get_config(args.config)
    load_start = time.monotonic()
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_steps},
    )
    load_seconds = time.monotonic() - load_start

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = {
        "base": image,
        "wrist": image.copy(),
        "state": np.zeros(23, dtype=np.float32),
        "prompt": args.prompt,
    }
    noise = np.zeros(
        (config.model.action_horizon, config.model.action_dim), dtype=np.float32
    )
    if args.repeats < 1:
        raise ValueError("repeats must be at least one")
    infer_ms = []
    result = None
    for _ in range(args.repeats):
        result = policy.infer(observation, noise=noise)
        infer_ms.append(float(result["policy_timing"]["infer_ms"]))
    assert result is not None
    actions = np.asarray(result["actions"])

    expected_shape = (config.model.action_horizon, 22)
    if actions.shape != expected_shape:
        raise AssertionError(f"expected actions {expected_shape}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise AssertionError("inference produced NaN or Inf actions")

    report = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "load_seconds": load_seconds,
        "infer_ms": infer_ms,
        "warm_infer_ms": infer_ms[-1],
        "num_steps": args.num_steps,
        "actions_shape": list(actions.shape),
        "actions_min": float(actions.min()),
        "actions_max": float(actions.max()),
        "actions_mean": float(actions.mean()),
        "actions_finite": bool(np.isfinite(actions).all()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
