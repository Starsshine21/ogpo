from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


COLLECTOR = Path(__file__).resolve().parents[1] / "evaluator" / "collect_robotwin_dense_rollouts.py"


def test_collector_uses_explicit_portable_dependency_roots(monkeypatch) -> None:
    monkeypatch.setenv("ROBOTWIN_ROOT", "/portable/pi05/external/RoboTwin")
    monkeypatch.setenv("EVO_RL_ROOT", "/portable/evo-RL")
    spec = importlib.util.spec_from_file_location("portable_collector", COLLECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.ROBOTWIN_ROOT == Path("/portable/pi05/external/RoboTwin")
    assert module.DEXJOCO_SOURCE_ROOT == Path("/portable/evo-RL/dexjoco")
