"""Shared helpers for the RoboTwin mixed-1000 protocol bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "protocols" / "robotwin_mixed1000"
SPLIT_MANIFEST_PATH = PROTOCOL_ROOT / "split_manifest.json"


def _load_split_manifest() -> dict[str, Any]:
    return json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))


TASKS = tuple(str(task) for task in _load_split_manifest()["tasks"])


def resolve_protocol_path(source: str | Path) -> Path:
    """Resolve a protocol source path on the current machine.

    Raw trajectories are intentionally not shipped in Git.  Users can either
    keep an evo-RL-compatible tree and set ``EVO_RL_ROOT``, or copy the files
    according to ``data_transfer_manifest.json`` and set
    ``ROBOTWIN_MIXED1000_DATA_ROOT`` to the directory containing
    ``<task>/<shard>/raw_rollouts/...``.
    """
    text = str(source)
    prefix = "<EVO_RL_ROOT>/"
    if not text.startswith(prefix):
        return Path(text)

    relative = Path(text.removeprefix(prefix))
    data_root = os.environ.get("ROBOTWIN_MIXED1000_DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser()
        episode_relative = Path(*relative.parts[-4:])
        for candidate in (root / episode_relative, root / relative):
            if candidate.exists():
                return candidate

    evo_root = os.environ.get("EVO_RL_ROOT")
    if evo_root:
        return Path(evo_root).expanduser() / relative

    raise FileNotFoundError(
        f"Cannot resolve {source!r}. Set ROBOTWIN_MIXED1000_DATA_ROOT to the "
        "directory containing task/shard/raw_rollouts, or set EVO_RL_ROOT to "
        "an evo-RL-compatible source tree."
    )


def mixed1000_episodes(split: str) -> list[dict[str, Any]]:
    """Return protocol episodes with metadata loaded from local trajectories."""
    selected = []
    for item in _load_split_manifest()["episodes"]:
        if str(item["split"]) != split:
            continue
        path = resolve_protocol_path(item["source"])
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        selected.append({
            "episode_id": int(item["episode_id"]),
            "source_episode_index": int(meta["episode_index"]),
            "task": str(item["task"]),
            "success": bool(item["success"]),
            "length": int(meta["num_frames"]),
            "seed": int(meta["seed"]),
            "path": str(path),
        })
    return selected


def heldout_episodes() -> list[dict[str, Any]]:
    """Return the fixed 100-episode mixed-1000 heldout split."""
    selected = mixed1000_episodes("heldout")
    if len(selected) != 100:
        raise ValueError(f"expected 100 heldout episodes, found {len(selected)}")
    for task in TASKS:
        outcomes = {bool(row["success"]) for row in selected if row["task"] == task}
        if outcomes != {False, True}:
            raise ValueError(f"task {task!r} does not contain both heldout outcomes")
    return selected
