#!/usr/bin/env python3
"""Continuously summarize sharded RoboTwin evaluation logs into one live report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import re
import time


EPISODE_RE = re.compile(r"^episode=(\d+) success=(True|False)\b")


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_shard(log_path: Path) -> dict[str, int | bool]:
    successes = 0
    failures = 0
    complete = False
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = EPISODE_RE.match(line)
            if match:
                if match.group(2) == "True":
                    successes += 1
                else:
                    failures += 1
            elif line.startswith("shard_complete episodes="):
                complete = True
    return {
        "completed": successes + failures,
        "successes": successes,
        "failures": failures,
        "complete": complete,
    }


def snapshot(output_root: Path, shard_count: int, total_episodes: int) -> dict:
    shards = {
        f"{index:02d}": read_shard(output_root / "logs" / f"shard_{index:02d}.log")
        for index in range(shard_count)
    }
    completed = sum(int(row["completed"]) for row in shards.values())
    successes = sum(int(row["successes"]) for row in shards.values())
    failures = sum(int(row["failures"]) for row in shards.values())
    final_summary = output_root / "strict100_summary.json"
    if final_summary.is_file():
        status = "complete"
    elif completed == 0:
        status = "warming_up"
    elif completed < total_episodes:
        status = "running"
    else:
        status = "finalizing"
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "completed": completed,
        "total": total_episodes,
        "successes": successes,
        "failures": failures,
        "success_rate_completed": successes / completed if completed else None,
        "shards": shards,
        "final_summary": str(final_summary),
    }


def markdown(report: dict) -> str:
    rate = report["success_rate_completed"]
    rate_text = "n/a" if rate is None else f"{100.0 * rate:.1f}%"
    lines = [
        "# Live RoboTwin evaluation progress",
        "",
        f"- Updated: `{report['updated_at']}`",
        f"- Status: `{report['status']}`",
        f"- Progress: **{report['completed']}/{report['total']}**",
        f"- Success / failure: **{report['successes']} / {report['failures']}**",
        f"- Running success rate: **{rate_text}** (completed episodes only)",
        "",
        "| Shard | Completed | Success | Failure | Done |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for shard, row in report["shards"].items():
        done = "yes" if row["complete"] else "no"
        lines.append(
            f"| {shard} | {row['completed']} | {row['successes']} | "
            f"{row['failures']} | {done} |"
        )
    note = (
        "Evaluation is complete; this is the final audited episode count."
        if report["status"] == "complete"
        else "This file is refreshed automatically. The success rate is provisional until all episodes finish."
    )
    lines.extend(["", note, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--total-episodes", type=int, default=100)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    while True:
        report = snapshot(args.output_root, args.shard_count, args.total_episodes)
        atomic_write(
            args.output_root / "live_progress.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        atomic_write(args.output_root / "LIVE_PROGRESS.md", markdown(report))
        if report["status"] == "complete":
            break
        time.sleep(max(args.interval_seconds, 1.0))


if __name__ == "__main__":
    main()
