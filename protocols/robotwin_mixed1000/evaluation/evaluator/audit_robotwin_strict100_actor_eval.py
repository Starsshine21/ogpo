#!/usr/bin/env python3
"""Fail-closed audit and paired summary for one strict-100 actor evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class AuditError(RuntimeError):
    """Raised when strict evaluation artifacts do not match the manifest."""


ROBOTWIN_EVAL_STEP_LIMITS = {
    "pick_dual_bottles": 400,
    "handover_mic": 600,
    "place_cans_plasticbox": 800,
    "click_bell": 400,
    "dump_bin_bigbin": 600,
    "move_playingcard_away": 400,
    "open_laptop": 700,
    "click_alarmclock": 400,
    "place_a2b_right": 400,
    "place_shoe": 500,
}


def _resolve_max_environment_steps(task_name: str, override: int | None) -> int:
    if override is not None:
        if override <= 0:
            raise AuditError("max environment steps must be positive")
        return override
    return ROBOTWIN_EVAL_STEP_LIMITS.get(task_name, 400)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise AuditError(f"missing JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON {path}: {exc}") from exc


def _mcnemar_exact_two_sided(base_only: int, actor_only: int) -> float:
    discordant = base_only + actor_only
    if discordant == 0:
        return 1.0
    lower = min(base_only, actor_only)
    numerator = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * numerator / (1 << discordant))


def _meta_path(root: Path, shard: int, episode_index: int) -> Path:
    return (
        root
        / f"shard_{shard:02d}"
        / "raw_rollouts"
        / f"episode_{episode_index:06d}"
        / "meta.json"
    )


def _validate_meta(
    meta: dict[str, Any],
    record: dict[str, Any],
    *,
    source: Path,
    expected_checkpoint: Path | None,
    expected_task_name: str,
    max_environment_steps: int,
) -> None:
    if type(meta.get("success")) is not bool:
        raise AuditError(f"{source}: success must be bool")
    expected_values = {
        "episode_index": int(record["episode_index"]),
        "seed": int(record["seed"]),
        "prompt": str(record["prompt"]),
        "policy_noise_seed": int(record["policy_noise_seed"]),
        "task_name": expected_task_name,
        "task_config": "demo_clean",
        "train_config": "pi05_robotwin2_clean50_full",
        "model_name": "model_clean50",
        "action_horizon": 50,
        "action_dim": 14,
        "transition_semantics": "evaluation metadata only",
        "num_frames": 0,
    }
    if expected_checkpoint is None and meta.get("transition_semantics") == "dense overlapping actual-executed action windows":
        # The three added tasks reuse audited dense Base collection results.
        # Keep original metadata intact; require one frame per environment step.
        expected_values["transition_semantics"] = "dense overlapping actual-executed action windows"
        expected_values["num_frames"] = meta.get("num_environment_steps")
    failures = [
        key for key, expected in expected_values.items() if meta.get(key) != expected
    ]
    if failures:
        raise AuditError(f"{source}: mismatched fields {failures}")
    observed_checkpoint = meta.get("ogpo_checkpoint")
    if expected_checkpoint is None:
        if observed_checkpoint is not None:
            raise AuditError(f"{source}: Base result unexpectedly has an OGPO checkpoint")
    else:
        if not isinstance(observed_checkpoint, str):
            raise AuditError(f"{source}: missing OGPO checkpoint provenance")
        if Path(observed_checkpoint).resolve() != expected_checkpoint.resolve():
            raise AuditError(
                f"{source}: checkpoint {observed_checkpoint!r} != {expected_checkpoint}"
            )
    environment_steps = meta.get("num_environment_steps")
    policy_chunks = meta.get("num_policy_chunks")
    if (
        type(environment_steps) is not int
        or not 1 <= environment_steps <= max_environment_steps
    ):
        raise AuditError(f"{source}: invalid environment step count {environment_steps!r}")
    if type(policy_chunks) is not int or policy_chunks != (environment_steps + 49) // 50:
        raise AuditError(f"{source}: invalid policy chunk count {policy_chunks!r}")


def _read_results(
    root: Path,
    records: list[dict[str, Any]],
    *,
    expected_checkpoint: Path | None,
    expected_task_name: str,
    max_environment_steps: int,
) -> dict[tuple[int, str, int], bool]:
    expected_paths = {
        _meta_path(root, int(record["shard"]), int(record["episode_index"]))
        for record in records
    }
    actual_paths = set(root.glob("shard_*/raw_rollouts/episode_*/meta.json"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise AuditError(f"{root}: result set mismatch missing={missing} extra={extra}")
    partials = sorted(root.glob("shard_*/raw_rollouts/.*.partial"))
    if partials:
        raise AuditError(f"{root}: incomplete episode directories {partials}")

    results: dict[tuple[int, str, int], bool] = {}
    for record in records:
        path = _meta_path(root, int(record["shard"]), int(record["episode_index"]))
        meta = _load_json(path)
        if not isinstance(meta, dict):
            raise AuditError(f"{path}: metadata root must be an object")
        _validate_meta(
            meta,
            record,
            source=path,
            expected_checkpoint=expected_checkpoint,
            expected_task_name=expected_task_name,
            max_environment_steps=max_environment_steps,
        )
        key = (int(record["seed"]), str(record["prompt"]), int(record["policy_noise_seed"]))
        if key in results:
            raise AuditError(f"{path}: duplicate paired episode key")
        results[key] = bool(meta["success"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--actor-root", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--task-name", default="pick_dual_bottles")
    parser.add_argument("--max-environment-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    max_environment_steps = _resolve_max_environment_steps(
        args.task_name, args.max_environment_steps
    )

    manifest = _load_json(args.manifest)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise AuditError("strict manifest must be a version-1 object")
    if manifest.get("task_name") != args.task_name:
        raise AuditError(
            f"manifest task {manifest.get('task_name')!r} != {args.task_name!r}"
        )
    records = manifest.get("episodes")
    if not isinstance(records, list) or len(records) != 100:
        raise AuditError("strict manifest must contain exactly 100 episodes")
    expected_counts = {shard: (13 if shard < 7 else 9) for shard in range(8)}
    observed_counts = {
        shard: sum(int(record.get("shard", -1)) == shard for record in records)
        for shard in range(8)
    }
    if observed_counts != expected_counts:
        raise AuditError(f"manifest shard counts {observed_counts} != {expected_counts}")

    actor = _read_results(
        args.actor_root,
        records,
        expected_checkpoint=args.actor_checkpoint,
        expected_task_name=args.task_name,
        max_environment_steps=max_environment_steps,
    )
    if args.base_root is None:
        actor_successes = sum(actor.values())
        summary = {
            "schema_version": 1,
            "audit_passed": True,
            "manifest": str(args.manifest.resolve()),
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "task_name": args.task_name,
            "episodes": len(records),
            "actor": {
                "successes": actor_successes,
                "failures": len(records) - actor_successes,
                "success_rate_percent": 100.0 * actor_successes / len(records),
            },
            "paired_base_available": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"strict100_audit_passed=true output={args.output.resolve()}")
        return
    base = _read_results(
        args.base_root,
        records,
        expected_checkpoint=None,
        expected_task_name=args.task_name,
        max_environment_steps=max_environment_steps,
    )
    if set(actor) != set(base):
        raise AuditError("Base and actor paired key sets differ")

    both_success = base_only = actor_only = both_failure = 0
    for key in base:
        if base[key] and actor[key]:
            both_success += 1
        elif base[key]:
            base_only += 1
        elif actor[key]:
            actor_only += 1
        else:
            both_failure += 1
    actor_successes = sum(actor.values())
    base_successes = sum(base.values())
    difference = (actor_successes - base_successes) / len(records)
    summary = {
        "schema_version": 1,
        "audit_passed": True,
        "manifest": str(args.manifest.resolve()),
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "episodes": len(records),
        "actor": {
            "successes": actor_successes,
            "failures": len(records) - actor_successes,
            "success_rate_percent": 100.0 * actor_successes / len(records),
        },
        "base": {
            "successes": base_successes,
            "failures": len(records) - base_successes,
            "success_rate_percent": 100.0 * base_successes / len(records),
        },
        "paired": {
            "base_success_actor_success": both_success,
            "base_success_actor_failure": base_only,
            "base_failure_actor_success": actor_only,
            "base_failure_actor_failure": both_failure,
            "discordant_pairs": base_only + actor_only,
            "actor_minus_base_success_count": actor_successes - base_successes,
            "actor_minus_base_percentage_points": 100.0 * difference,
            "mcnemar_exact_two_sided_p_value": _mcnemar_exact_two_sided(
                base_only, actor_only
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"strict100_audit_passed=true output={args.output.resolve()}")


if __name__ == "__main__":
    main()
