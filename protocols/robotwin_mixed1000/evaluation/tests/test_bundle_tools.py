from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


BUNDLE = Path(__file__).resolve().parents[1]
SCRIPT = BUNDLE / "scripts" / "bundle_tools.py"
SPEC = importlib.util.spec_from_file_location("strict100_bundle_tools", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLS)


def test_canonical_bundle_has_ten_valid_fixed_manifests() -> None:
    report = TOOLS.validate_bundle(BUNDLE)

    assert report["task_count"] == 10
    assert report["episodes_per_task"] == 100
    assert report["total_episodes"] == 1000
    assert report["all_paired_keys_unique"] is True


def test_manifest_validation_rejects_duplicate_paired_key(tmp_path: Path) -> None:
    source = json.loads((BUNDLE / "manifests" / "click_bell.json").read_text())
    source["episodes"][1]["seed"] = source["episodes"][0]["seed"]
    source["episodes"][1]["prompt"] = source["episodes"][0]["prompt"]
    source["episodes"][1]["policy_noise_seed"] = source["episodes"][0][
        "policy_noise_seed"
    ]
    path = tmp_path / "click_bell.json"
    path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="duplicate paired key"):
        TOOLS.validate_manifest(path, "click_bell")


def test_macro_is_equal_weight_per_task() -> None:
    summaries = {
        "pick_dual_bottles": {"success_rate_percent": 90.0},
        "handover_mic": {"success_rate_percent": 10.0},
    }

    assert TOOLS.task_macro(summaries) == 50.0


def test_paired_counts_preserve_direction() -> None:
    base = {("a",): True, ("b",): True, ("c",): False, ("d",): False}
    actor = {("a",): True, ("b",): False, ("c",): True, ("d",): True}

    result = TOOLS.paired_counts(base, actor)

    assert result == {
        "base_success_actor_success": 1,
        "base_success_actor_failure": 1,
        "base_failure_actor_success": 2,
        "base_failure_actor_failure": 0,
        "actor_minus_base_success_count": 1,
    }


def test_result_audit_rejects_noncanonical_action_shape(tmp_path: Path) -> None:
    manifest = json.loads((BUNDLE / "manifests" / "click_bell.json").read_text())
    for row in manifest["episodes"]:
        path = (
            tmp_path
            / f"shard_{row['shard']:02d}"
            / "raw_rollouts"
            / f"episode_{row['episode_index']:06d}"
            / "meta.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "success": False,
            "episode_index": row["episode_index"],
            "seed": row["seed"],
            "prompt": row["prompt"],
            "policy_noise_seed": row["policy_noise_seed"],
            "task_name": "click_bell",
            "task_config": "demo_clean",
            "train_config": "pi05_robotwin2_clean50_full",
            "model_name": "model_clean50",
            "action_horizon": 50,
            "action_dim": 14,
            "transition_semantics": "evaluation metadata only",
            "num_frames": 0,
            "ogpo_checkpoint": None,
            "num_environment_steps": 400,
            "num_policy_chunks": 8,
        }
        path.write_text(json.dumps(meta))
    first = next(tmp_path.glob("shard_*/raw_rollouts/episode_*/meta.json"))
    malformed = json.loads(first.read_text())
    malformed["action_dim"] = 7
    first.write_text(json.dumps(malformed))

    with pytest.raises(Exception, match="mismatched fields"):
        TOOLS._result_map(tmp_path, manifest, "click_bell", None)
