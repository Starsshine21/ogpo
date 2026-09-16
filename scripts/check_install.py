#!/usr/bin/env python3
"""Small installation check for the standalone OGPO repository."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"repo_root={ROOT}")

    import torch

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_devices={torch.cuda.device_count()}")

    import ogpo

    print(f"ogpo_import=ok ({ogpo.__name__})")

    optional_failures = []
    for module_name in ("jax", "flax", "optax", "orbax", "openpi", "openpi_client"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report all optional deps
            optional_failures.append((module_name, exc))
        else:
            version = getattr(module, "__version__", "unknown")
            print(f"{module_name}=ok ({version})")

    for relative in (
        "configs/ogpo/critic_udivl.yaml",
        "configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml",
        "configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml",
        "protocols/robotwin_mixed1000/evaluation/protocol.json",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"found={relative}")

    if optional_failures:
        for module_name, exc in optional_failures:
            print(f"{module_name}=missing_or_failed: {exc}")
        raise SystemExit(2)
    print("OGPO_INSTALL_CHECK_PASS")


if __name__ == "__main__":
    main()

