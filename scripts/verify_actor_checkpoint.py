#!/usr/bin/env python3
"""Fail-fast structural verification for a resumable OGPO actor checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-actor-step", type=int, required=True)
    parser.add_argument("--expected-critic-step", type=int, required=True)
    parser.add_argument("--expected-checkpoint-interval", type=int, required=True)
    parser.add_argument("--expected-lambda-success", type=float, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"missing actor checkpoint: {args.checkpoint}")
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(payload.get("actor_step", -1)) != args.expected_actor_step:
        raise ValueError(
            f"actor_step mismatch: expected {args.expected_actor_step}, "
            f"got {payload.get('actor_step')}"
        )
    if int(payload.get("training_step", -1)) != args.expected_critic_step:
        raise ValueError(
            f"critic step mismatch: expected {args.expected_critic_step}, "
            f"got {payload.get('training_step')}"
        )
    metadata = payload.get("critic_metadata", {})
    if int(metadata.get("num_pairs", -1)) != 5:
        raise ValueError("checkpoint does not contain five DIVL pairs")
    if int(metadata.get("num_raw_q_heads", -1)) != 10:
        raise ValueError("checkpoint does not contain ten raw Q heads")
    for key in ("policy", "old_policy", "slow_policy", "actor_optimizer"):
        if not payload.get(key):
            raise ValueError(f"checkpoint is missing resumable state: {key}")
    training = payload.get("config", {}).get("training", {})
    if int(training.get("checkpoint_interval", -1)) != args.expected_checkpoint_interval:
        raise ValueError("checkpoint interval metadata does not match the preflight")
    lambda_success = float(
        payload.get("config", {}).get("regularization", {}).get("lambda_success", -1.0)
    )
    if abs(lambda_success - args.expected_lambda_success) > 1e-12:
        raise ValueError(
            f"lambda_success mismatch: expected {args.expected_lambda_success}, "
            f"got {lambda_success}"
        )
    print(
        "checkpoint_verified "
        f"path={args.checkpoint.resolve()} "
        f"size={args.checkpoint.stat().st_size} "
        f"actor_step={payload['actor_step']} "
        f"accepted={payload.get('accepted_actor_updates')} "
        f"critic_step={payload['training_step']} "
        f"lambda_success={lambda_success} "
        "pairs=5 raw_q=10 current=1 old=1 slow=1 optimizer=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
