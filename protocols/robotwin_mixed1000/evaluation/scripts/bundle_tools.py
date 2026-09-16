#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_EPISODE_KEYS = {
    "shard",
    "episode_index",
    "seed",
    "prompt",
    "policy_noise_seed",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: Path, expected_task: str) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("version") != 1:
        raise ValueError(f"{path}: expected manifest version 1")
    if payload.get("task_name") != expected_task:
        raise ValueError(f"{path}: task_name does not match {expected_task}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 100:
        raise ValueError(f"{path}: expected exactly 100 episodes")
    expected_counts = [13, 13, 13, 13, 13, 13, 13, 9]
    counts = [0] * 8
    paired_keys: set[tuple[int, str, int]] = set()
    noise_seeds: set[int] = set()
    per_shard_indices: dict[int, list[int]] = {index: [] for index in range(8)}
    for row in episodes:
        missing = REQUIRED_EPISODE_KEYS - set(row)
        if missing:
            raise ValueError(f"{path}: episode missing keys {sorted(missing)}")
        shard = int(row["shard"])
        if shard not in range(8):
            raise ValueError(f"{path}: invalid shard {shard}")
        counts[shard] += 1
        per_shard_indices[shard].append(int(row["episode_index"]))
        key = (int(row["seed"]), str(row["prompt"]), int(row["policy_noise_seed"]))
        if key in paired_keys:
            raise ValueError(f"{path}: duplicate paired key {key}")
        paired_keys.add(key)
        noise_seed = int(row["policy_noise_seed"])
        if noise_seed in noise_seeds:
            raise ValueError(f"{path}: duplicate policy_noise_seed {noise_seed}")
        noise_seeds.add(noise_seed)
    if counts != expected_counts:
        raise ValueError(f"{path}: shard counts {counts} != {expected_counts}")
    for shard, expected_count in enumerate(expected_counts):
        expected_indices = list(range(expected_count))
        if sorted(per_shard_indices[shard]) != expected_indices:
            raise ValueError(f"{path}: shard {shard} episode indices are not canonical")
    return {
        "task": expected_task,
        "episodes": len(episodes),
        "paired_keys_unique": True,
        "policy_noise_seeds_unique": True,
        "shard_counts": counts,
    }


def validate_bundle(bundle_root: Path) -> dict[str, Any]:
    protocol = _load(bundle_root / "protocol.json")
    if protocol.get("schema_version") != 1:
        raise ValueError("protocol schema_version must be 1")
    tasks = protocol.get("tasks", [])
    if len(tasks) != 10:
        raise ValueError("protocol must contain exactly 10 tasks")
    reports = [
        validate_manifest(bundle_root / row["manifest"], row["name"])
        for row in tasks
    ]
    return {
        "protocol_name": protocol["protocol_name"],
        "task_count": len(reports),
        "episodes_per_task": 100,
        "total_episodes": sum(row["episodes"] for row in reports),
        "all_paired_keys_unique": all(row["paired_keys_unique"] for row in reports),
        "tasks": reports,
    }


def task_macro(summaries: dict[str, dict[str, float]]) -> float:
    if not summaries:
        raise ValueError("cannot compute macro over no tasks")
    return sum(row["success_rate_percent"] for row in summaries.values()) / len(
        summaries
    )


def paired_counts(base: dict, actor: dict) -> dict[str, int]:
    if set(base) != set(actor):
        raise ValueError("paired result keys differ")
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
    return {
        "base_success_actor_success": both_success,
        "base_success_actor_failure": base_only,
        "base_failure_actor_success": actor_only,
        "base_failure_actor_failure": both_failure,
        "actor_minus_base_success_count": actor_only - base_only,
    }


def _mcnemar_exact(base_only: int, actor_only: int) -> float:
    discordant = base_only + actor_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(base_only, actor_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _result_map(result_root: Path, manifest: dict, task: str, checkpoint: Path | None) -> dict:
    audit_path = (
        Path(__file__).resolve().parents[1]
        / "evaluator"
        / "audit_robotwin_strict100_actor_eval.py"
    )
    spec = importlib.util.spec_from_file_location("strict100_bundle_auditor", audit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bundled auditor {audit_path}")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    return auditor._read_results(
        result_root,
        manifest["episodes"],
        expected_checkpoint=checkpoint,
        expected_task_name=task,
        max_environment_steps=auditor._resolve_max_environment_steps(task, None),
    )


def audit_task(
    manifest_path: Path,
    result_root: Path,
    task: str,
    label: str,
    checkpoint: Path | None,
    output: Path,
) -> dict[str, Any]:
    validate_manifest(manifest_path, task)
    manifest = _load(manifest_path)
    results = _result_map(result_root, manifest, task, checkpoint)
    successes = sum(results.values())
    summary = {
        "schema_version": 1,
        "audit_passed": True,
        "manifest": str(manifest_path.resolve()),
        "task_name": task,
        "episodes": 100,
        "policy": {
            "label": label,
            "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
            "successes": successes,
            "failures": 100 - successes,
            "success_rate_percent": float(successes),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def summarize_policy(bundle_root: Path, result_root: Path, output: Path) -> dict[str, Any]:
    protocol = _load(bundle_root / "protocol.json")
    rows = {}
    label = None
    for task_row in protocol["tasks"]:
        task = task_row["name"]
        summary = _load(result_root / task / "strict100_summary.json")
        if not summary.get("audit_passed") or summary.get("task_name") != task:
            raise ValueError(f"invalid task summary for {task}")
        policy = summary["policy"]
        label = label or policy["label"]
        if policy["label"] != label:
            raise ValueError("policy labels differ across tasks")
        rows[task] = {"success_rate_percent": float(policy["success_rate_percent"])}
    payload = {
        "schema_version": 1,
        "policy_label": label,
        "task_count": 10,
        "episodes_per_task": 100,
        "macro_success_rate_percent": task_macro(rows),
        "tasks": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def compare_policies(
    bundle_root: Path,
    base_root: Path,
    actor_root: Path,
    actor_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    protocol = _load(bundle_root / "protocol.json")
    task_rows = {}
    for task_row in protocol["tasks"]:
        task = task_row["name"]
        manifest = _load(bundle_root / task_row["manifest"])
        base = _result_map(base_root / task, manifest, task, None)
        actor = _result_map(actor_root / task, manifest, task, actor_checkpoint)
        paired = paired_counts(base, actor)
        base_successes = sum(base.values())
        actor_successes = sum(actor.values())
        paired["mcnemar_exact_two_sided_p_value"] = _mcnemar_exact(
            paired["base_success_actor_failure"], paired["base_failure_actor_success"]
        )
        task_rows[task] = {
            "base_success_rate_percent": float(base_successes),
            "actor_success_rate_percent": float(actor_successes),
            "actor_minus_base_percentage_points": float(actor_successes - base_successes),
            "paired": paired,
        }
    payload = {
        "schema_version": 1,
        "task_count": 10,
        "episodes_per_task": 100,
        "base_macro_success_rate_percent": sum(
            row["base_success_rate_percent"] for row in task_rows.values()
        )
        / 10,
        "actor_macro_success_rate_percent": sum(
            row["actor_success_rate_percent"] for row in task_rows.values()
        )
        / 10,
        "tasks": task_rows,
    }
    payload["actor_minus_base_macro_percentage_points"] = (
        payload["actor_macro_success_rate_percent"]
        - payload["base_macro_success_rate_percent"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--bundle-root", type=Path, required=True)
    audit = sub.add_parser("audit-task")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--result-root", type=Path, required=True)
    audit.add_argument("--task", required=True)
    audit.add_argument("--label", required=True)
    audit.add_argument("--checkpoint", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    summarize = sub.add_parser("summarize-policy")
    summarize.add_argument("--bundle-root", type=Path, required=True)
    summarize.add_argument("--result-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--bundle-root", type=Path, required=True)
    compare.add_argument("--base-root", type=Path, required=True)
    compare.add_argument("--actor-root", type=Path, required=True)
    compare.add_argument("--actor-checkpoint", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_bundle(args.bundle_root)
    elif args.command == "audit-task":
        result = audit_task(
            args.manifest,
            args.result_root,
            args.task,
            args.label,
            args.checkpoint,
            args.output,
        )
    elif args.command == "summarize-policy":
        result = summarize_policy(args.bundle_root, args.result_root, args.output)
    else:
        result = compare_policies(
            args.bundle_root,
            args.base_root,
            args.actor_root,
            args.actor_checkpoint,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
