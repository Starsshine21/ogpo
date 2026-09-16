#!/usr/bin/env python3
"""Render only the episode-summary panel for click_bell critic milestones."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ogpo.replay import load_replay, prepare_replay_from_config
from ogpo.trainer import build_train_state, load_critic_checkpoint
from train_udivl_critic import load_config
from visualize_critic_episode_curves import (
    Episode,
    _configure_plot_style,
    _episode_arrays,
    _resolve,
    episode_catalog,
    plot_summary,
    score_validation_raw10,
)


def choose_new_episode(
    episodes: list[Episode],
    *,
    success: bool,
    excluded_id: int,
) -> Episode:
    """Choose nearest-to-median length while excluding the prior figure's ID."""
    outcome = [episode for episode in episodes if episode.success == success]
    candidates = [episode for episode in outcome if episode.episode_id != excluded_id]
    if not candidates:
        raise ValueError(
            f"no {'success' if success else 'failure'} episode remains after "
            f"excluding episode {excluded_id}"
        )
    lengths = torch.tensor([episode.length for episode in outcome], dtype=torch.float64)
    median_length = float(lengths.median().item())
    # torch.median is the lower middle for even counts; use the conventional
    # midpoint median so this matches the original deterministic selector.
    if lengths.numel() % 2 == 0:
        sorted_lengths = lengths.sort().values
        middle = lengths.numel() // 2
        median_length = float(
            (sorted_lengths[middle - 1] + sorted_lengths[middle]).item() / 2.0
        )
    return min(
        candidates,
        key=lambda episode: (
            abs(episode.length - median_length),
            episode.episode_id,
        ),
    )


def _parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not steps or any(step <= 0 for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("--steps must contain unique positive integers")
    return steps


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-config", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--steps", default="1000,2000,3000,4000,5000,6000,7000,8000")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--exclude-success-episode-id", type=int, default=70)
    parser.add_argument("--exclude-failure-episode-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    _configure_plot_style()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = _resolve(args.critic_config)
    replay_path = _resolve(args.replay)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    config = load_config(config_path)
    configured_validation = _resolve(config["data"]["validation_path"]).resolve()
    if replay_path.resolve() != configured_validation:
        raise ValueError(
            "summary sweep is validation-only; expected "
            f"{configured_validation}, got {replay_path.resolve()}"
        )
    config.setdefault("training", {})["device"] = args.device
    validation = prepare_replay_from_config(
        load_replay(replay_path), config.get("data", {})
    )
    episodes = episode_catalog(validation)
    success_episode = choose_new_episode(
        episodes,
        success=True,
        excluded_id=args.exclude_success_episode_id,
    )
    failure_episode = choose_new_episode(
        episodes,
        success=False,
        excluded_id=args.exclude_failure_episode_id,
    )
    print(
        "[summary-sweep] selected "
        f"success_episode_id={success_episode.episode_id} "
        f"success_length={success_episode.length} success_label=1 "
        f"failure_episode_id={failure_episode.episode_id} "
        f"failure_length={failure_episode.length} failure_label=0",
        flush=True,
    )

    combined_indices = torch.cat(
        [success_episode.indices, failure_episode.indices], dim=0
    )
    selected = validation.index_select(combined_indices)
    success_local = Episode(
        episode_id=success_episode.episode_id,
        indices=torch.arange(success_episode.length),
        timesteps=success_episode.timesteps,
        success=True,
    )
    failure_local = Episode(
        episode_id=failure_episode.episode_id,
        indices=torch.arange(
            success_episode.length,
            success_episode.length + failure_episode.length,
        ),
        timesteps=failure_episode.timesteps,
        success=False,
    )

    steps = _parse_steps(args.steps)
    checkpoints = {
        step: _resolve(args.checkpoint_pattern.format(step=step)) for step in steps
    }
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    output_dir.mkdir(parents=True)
    state = build_train_state(config, selected, device=args.device)
    figures: list[str] = []
    for step in steps:
        checkpoint = checkpoints[step]
        payload = load_critic_checkpoint(checkpoint, state, load_optimizer=False)
        actual_step = int(payload.get("training_step", -1))
        if actual_step != step:
            raise ValueError(
                f"checkpoint step mismatch for {checkpoint}: expected {step}, got {actual_step}"
            )
        state.critic.eval()
        raw_q = score_validation_raw10(
            state.critic,
            selected,
            config,
            inference_batch_size=args.inference_batch_size,
        )
        success_arrays = _episode_arrays(selected, raw_q, success_local)
        failure_arrays = _episode_arrays(selected, raw_q, failure_local)
        label = f"{step // 1000}k" if step % 1000 == 0 else str(step)
        figure_path = output_dir / f"critic_{label}_episode_summary.png"
        plot_summary(
            success_arrays,
            failure_arrays,
            success_local,
            failure_local,
            checkpoint_label=label,
            output=figure_path,
        )
        figures.append(str(figure_path.resolve()))
        print(
            f"[summary-sweep] rendered step={step} figure={figure_path}",
            flush=True,
        )
        del raw_q
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "critic_config": str(config_path.resolve()),
        "validation_replay": str(replay_path.resolve()),
        "steps": steps,
        "checkpoints": {str(step): str(path.resolve()) for step, path in checkpoints.items()},
        "success_episode_id": success_episode.episode_id,
        "success_length": success_episode.length,
        "failure_episode_id": failure_episode.episode_id,
        "failure_length": failure_episode.length,
        "excluded_previous_success_episode_id": args.exclude_success_episode_id,
        "excluded_previous_failure_episode_id": args.exclude_failure_episode_id,
        "pair_min_used": False,
        "figures": figures,
    }
    manifest_path = output_dir / "critic_episode_summary_sweep_manifest.json"
    _atomic_json(manifest, manifest_path)
    print(f"[summary-sweep] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
