#!/usr/bin/env python3
"""Visualize ten held-out/validation success trajectories with key frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ogpo.replay import load_replay, prepare_replay_from_config
from ogpo.trainer import build_train_state, load_critic_checkpoint
from train_udivl_critic import load_config
from visualize_critic_episode_curves import (
    HEAD_COLORS,
    _configure_plot_style,
    _episode_arrays,
    episode_catalog,
    score_validation_raw10,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def representative_indices(count: int, selected_count: int) -> list[int]:
    if count < selected_count or selected_count <= 0:
        raise ValueError("representative selection requires count >= selected_count > 0")
    if selected_count == 1:
        return [count // 2]
    indices = np.rint(np.linspace(0, count - 1, selected_count)).astype(int).tolist()
    if len(set(indices)) != selected_count:
        raise AssertionError("representative selection produced duplicate indices")
    return indices


def key_frame_local_indices(length: int) -> list[int]:
    if length <= 0:
        raise ValueError("episode length must be positive")
    return [
        int(np.rint(progress * (length - 1)))
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]


def select_episodes_from_manifest(
    batches: dict[str, object], manifest_path: Path
) -> list[tuple[int, str, int, object]]:
    """Resolve the exact split/episode sequence recorded by an earlier plot."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = payload.get("episodes")
    if not isinstance(requested, list) or not requested:
        raise ValueError("selection manifest must contain a non-empty episodes list")
    catalogs = {
        split: {episode.episode_id: episode for episode in episode_catalog(batch)}
        for split, batch in batches.items()
    }
    selected = []
    for record in requested:
        split = str(record["split"])
        episode_id = int(record["episode_id"])
        if split not in catalogs or episode_id not in catalogs[split]:
            raise ValueError(
                f"selection manifest episode is absent: {split} episode {episode_id}"
            )
        episode = catalogs[split][episode_id]
        if not episode.success:
            raise ValueError(
                f"selection manifest episode is not successful: {split} episode {episode_id}"
            )
        expected_length = int(record.get("length", episode.length))
        if episode.length != expected_length:
            raise ValueError(
                f"selection manifest length mismatch for {split} episode {episode_id}: "
                f"expected {expected_length}, got {episode.length}"
            )
        selected.append((episode.length, split, episode_id, episode))
    return selected


def _plot_curve_grid(records: list[dict], output: Path, checkpoint_label: str) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), squeeze=False)
    for ax, record in zip(axes.flat, records, strict=True):
        arrays = record["arrays"]
        progress = arrays["progress"]
        raw_q = arrays["raw_q"]
        mean = arrays["q_mean"]
        std = arrays["q_std"]
        for head in range(10):
            ax.plot(progress, raw_q[head], color=HEAD_COLORS[head], alpha=0.30, linewidth=0.7)
        ax.fill_between(progress, mean - std, mean + std, color="#34495E", alpha=0.17)
        ax.plot(progress, mean, color="#17202A", linewidth=2.2, label="10Q mean")
        ax.plot(
            progress,
            arrays["mc_returns"],
            color="#D35400",
            linewidth=1.4,
            linestyle="--",
            label="MC return",
        )
        ax.set_title(
            f"{record['split']} episode {record['episode'].episode_id} | "
            f"T={record['episode'].length} | terminal Q={mean[-1]:.3f}"
        )
        ax.set_xlabel("Normalized progress")
        ax.set_ylabel("Raw 10Q / MC return")
        ax.legend(loc="lower right")
    fig.suptitle(
        f"click_bell — 10 successful trajectories | pair-mean high-tau critic {checkpoint_label}",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_key_frames(record: dict, output: Path, checkpoint_label: str) -> list[dict]:
    episode = record["episode"]
    batch = record["batch"]
    arrays = record["arrays"]
    if batch.images is None:
        raise ValueError("replay has no images for key-frame visualization")
    required = ("image_base", "image_wrist")
    missing = [key for key in required if key not in batch.images]
    if missing:
        raise KeyError(f"replay images missing cameras: {missing}")
    local_indices = key_frame_local_indices(episode.length)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), squeeze=False)
    metadata = []
    for column, local_index in enumerate(local_indices):
        global_index = int(episode.indices[local_index])
        timestep = int(batch.timesteps[global_index])
        progress = float(arrays["progress"][local_index])
        q_mean = float(arrays["q_mean"][local_index])
        mc_return = float(arrays["mc_returns"][local_index])
        metadata.append(
            {
                "progress": progress,
                "timestep": timestep,
                "q_mean": q_mean,
                "mc_return": mc_return,
            }
        )
        for row, camera in enumerate(required):
            image = batch.images[camera][global_index].cpu().numpy()
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(
                    f"p={progress:.0%}  t={timestep}\nQ={q_mean:.3f}  G={mc_return:.3f}",
                    fontsize=10,
                )
        axes[0, 0].set_ylabel("head", fontsize=12)
        axes[1, 0].set_ylabel("wrist", fontsize=12)
    fig.suptitle(
        f"{record['split']} success episode {episode.episode_id} key frames | critic={checkpoint_label}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-replay", required=True)
    parser.add_argument("--heldout-replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--checkpoint-label")
    parser.add_argument("--selection-manifest")
    args = parser.parse_args()
    _configure_plot_style()
    config_path = _resolve(args.critic_config)
    checkpoint = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    config = load_config(config_path)
    config["training"]["device"] = args.device
    split_paths = {
        "validation": _resolve(args.validation_replay),
        "heldout": _resolve(args.heldout_replay),
    }
    batches = {
        split: prepare_replay_from_config(load_replay(path), config["data"])
        for split, path in split_paths.items()
    }
    candidates = []
    for split, batch in batches.items():
        for episode in episode_catalog(batch):
            if episode.success:
                candidates.append((episode.length, split, episode.episode_id, episode))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    selection_manifest = (
        None if args.selection_manifest is None else _resolve(args.selection_manifest)
    )
    selected = (
        select_episodes_from_manifest(batches, selection_manifest)
        if selection_manifest is not None
        else [
            candidates[index]
            for index in representative_indices(len(candidates), args.count)
        ]
    )
    print(
        "[success10] selected="
        + json.dumps(
            [
                {"split": split, "episode_id": episode_id, "length": length}
                for length, split, episode_id, _ in selected
            ]
        ),
        flush=True,
    )

    state = build_train_state(config, batches["validation"], device=args.device)
    payload = load_critic_checkpoint(checkpoint, state, load_optimizer=False)
    checkpoint_step = int(payload.get("training_step", -1))
    if args.expected_step is not None and checkpoint_step != args.expected_step:
        raise ValueError(
            f"expected selected checkpoint step {args.expected_step}, got {checkpoint_step}"
        )
    checkpoint_label = args.checkpoint_label or f"step {checkpoint_step}"
    state.critic.eval()
    raw_by_split = {
        split: score_validation_raw10(
            state.critic,
            batch,
            config,
            inference_batch_size=args.inference_batch_size,
        )
        for split, batch in batches.items()
    }
    records = []
    for length, split, episode_id, episode in selected:
        records.append(
            {
                "length": length,
                "split": split,
                "episode_id": episode_id,
                "episode": episode,
                "batch": batches[split],
                "arrays": _episode_arrays(batches[split], raw_by_split[split], episode),
            }
        )
    output_dir.mkdir(parents=True)
    grid_path = output_dir / "success10_raw10_q_progress_grid.png"
    _plot_curve_grid(records, grid_path, checkpoint_label)
    manifest_records = []
    for record in records:
        episode = record["episode"]
        frame_path = output_dir / (
            f"{record['split']}_episode_{episode.episode_id:03d}_keyframes.png"
        )
        key_frames = _plot_key_frames(record, frame_path, checkpoint_label)
        arrays = record["arrays"]
        manifest_records.append(
            {
                "split": record["split"],
                "episode_id": episode.episode_id,
                "length": episode.length,
                "q_mean_start": float(arrays["q_mean"][0]),
                "q_mean_end": float(arrays["q_mean"][-1]),
                "mc_return_start": float(arrays["mc_returns"][0]),
                "mc_return_end": float(arrays["mc_returns"][-1]),
                "key_frames": key_frames,
                "key_frame_figure": str(frame_path.resolve()),
            }
        )
    manifest = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "checkpoint_label": checkpoint_label,
        "critic_config": str(config_path.resolve()),
        "selection_role": "visualization_only_not_checkpoint_selection",
        "selection": (
            f"exact episode sequence reused from {selection_manifest.resolve()}"
            if selection_manifest is not None
            else f"{args.count} evenly spaced order statistics of success episode length"
        ),
        "selection_manifest": (
            None if selection_manifest is None else str(selection_manifest.resolve())
        ),
        "candidate_success_episode_count": len(candidates),
        "validation_replay": str(split_paths["validation"].resolve()),
        "heldout_replay": str(split_paths["heldout"].resolve()),
        "raw_q_order": [f"Q{pair}{head}" for pair in range(1, 6) for head in range(1, 3)],
        "pair_min_used": False,
        "curve_grid": str(grid_path.resolve()),
        "episodes": manifest_records,
    }
    manifest_path = output_dir / "success10_visualization_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(f"[success10] curve_grid={grid_path}", flush=True)
    print(f"[success10] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
