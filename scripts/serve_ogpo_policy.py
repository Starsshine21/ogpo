#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "openpi" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "openpi" / "packages" / "openpi-client" / "src"))

from ogpo.inference_policy import create_pi05_ogpo_inference_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--ogpo-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-reference-policy", action="store_true")
    parser.add_argument("--skip-critic", action="store_true")
    parser.add_argument(
        "--inference-dynamics",
        choices=("native_ode", "training_sde_drift"),
    )
    parser.add_argument("--inference-num-steps", type=int)
    parser.add_argument("--require-model-flow-dim", action="store_true")
    args = parser.parse_args()

    policy = create_pi05_ogpo_inference_policy(
        pi05_checkpoint_dir=args.pi05_checkpoint,
        train_config_name=args.train_config,
        ogpo_checkpoint=args.ogpo_checkpoint,
        device=args.device,
        include_reference_policy=not args.skip_reference_policy,
        include_critic=not args.skip_critic,
        inference_dynamics=args.inference_dynamics,
        inference_num_steps=args.inference_num_steps,
        require_model_flow_dim=args.require_model_flow_dim,
    )
    hostname = socket.gethostname()
    logging.info("Creating OGPO server on %s:%d", hostname, args.port)
    WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
