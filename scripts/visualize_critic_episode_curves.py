#!/usr/bin/env python3
"""Visualize a frozen 5k click_bell critic on validation episodes only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ogpo.categorical_q import ranking_action_negatives
from ogpo.critic_raw10_evaluator import spearman_correlation
from ogpo.multimodal_critic import MultiHeadUdivlCritic
from ogpo.replay import load_replay, prepare_replay_from_config
from ogpo.trainer import (
    build_train_state,
    load_critic_checkpoint,
)
from train_udivl_critic import load_config


RAW_Q_HEADS = 10
PROGRESS_POINTS = {
    "t020": 0.20,
    "t050": 0.50,
    "t080": 0.80,
    "t095": 0.95,
}
HEAD_COLORS = plt.get_cmap("tab10").colors
SUCCESS_COLOR = "#148F77"
FAILURE_COLOR = "#C0392B"


@dataclass(frozen=True)
class Episode:
    episode_id: int
    indices: torch.Tensor
    timesteps: np.ndarray
    success: bool

    @property
    def length(self) -> int:
        return int(self.indices.numel())


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def episode_catalog(batch) -> list[Episode]:
    """Return deterministic, timestep-sorted episodes with validated labels."""
    episodes: list[Episode] = []
    for episode_id in sorted(int(value) for value in torch.unique(batch.episode_ids)):
        indices = torch.nonzero(
            batch.episode_ids == episode_id,
            as_tuple=False,
        ).flatten()
        order = torch.argsort(batch.timesteps.index_select(0, indices), stable=True)
        indices = indices.index_select(0, order)
        timesteps_t = batch.timesteps.index_select(0, indices)
        if int(torch.unique(timesteps_t).numel()) != int(indices.numel()):
            raise ValueError(f"episode {episode_id} contains duplicate timesteps")
        success_values = torch.unique(batch.successes.index_select(0, indices))
        if int(success_values.numel()) != 1:
            raise ValueError(
                f"episode {episode_id} has inconsistent success labels: "
                f"{success_values.tolist()}"
            )
        episodes.append(
            Episode(
                episode_id=episode_id,
                indices=indices,
                timesteps=timesteps_t.cpu().numpy().astype(np.int64),
                success=bool(success_values.item()),
            )
        )
    return episodes


def choose_episode(
    episodes: list[Episode],
    *,
    success: bool,
    requested_id: int | None,
) -> Episode:
    candidates = [episode for episode in episodes if episode.success == success]
    if not candidates:
        raise ValueError(
            "validation replay contains no "
            + ("successful" if success else "failed")
            + " episode"
        )
    if requested_id is not None:
        matching = [episode for episode in episodes if episode.episode_id == requested_id]
        if not matching:
            raise ValueError(f"episode {requested_id} is absent from validation replay")
        selected = matching[0]
        if selected.success != success:
            expected = "success" if success else "failure"
            raise ValueError(f"episode {requested_id} is not labelled {expected}")
        return selected
    median_length = float(np.median([episode.length for episode in candidates]))
    return min(
        candidates,
        key=lambda episode: (
            abs(episode.length - median_length),
            episode.episode_id,
        ),
    )


def _validate_episode_extraction(batch, episode: Episode) -> dict[str, float]:
    if batch.mc_returns is None:
        raise ValueError("prepared validation replay does not contain mc_returns")
    selected = batch.index_select(episode.indices)
    if not torch.equal(selected.episode_ids, batch.episode_ids[episode.indices]):
        raise AssertionError("episode extraction changed episode IDs")
    if not torch.equal(selected.timesteps, batch.timesteps[episode.indices]):
        raise AssertionError("episode extraction changed timesteps")
    if not torch.equal(selected.successes, batch.successes[episode.indices]):
        raise AssertionError("episode extraction changed success labels")
    for name in ("chunk_returns", "mc_returns"):
        values = getattr(selected, name)
        if values is None or not bool(torch.isfinite(values).all()):
            raise ValueError(f"episode {episode.episode_id} has invalid {name}")
    return {
        "chunk_return_min": float(selected.chunk_returns.min().item()),
        "chunk_return_max": float(selected.chunk_returns.max().item()),
        "mc_return_min": float(selected.mc_returns.min().item()),
        "mc_return_max": float(selected.mc_returns.max().item()),
    }


def _critic_mask(batch, config: dict[str, Any]) -> torch.Tensor:
    if bool(config.get("data", {}).get("use_execution_mask", True)):
        return batch.execution_masks
    return torch.ones_like(batch.execution_masks, dtype=torch.bool)


@torch.no_grad()
def score_validation_raw10(
    critic: MultiHeadUdivlCritic,
    validation,
    config: dict[str, Any],
    *,
    inference_batch_size: int,
) -> torch.Tensor:
    """Score dataset actions via the production raw-10-Q interface."""
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    device = next(critic.parameters()).device
    outputs: list[torch.Tensor] = []
    for start in range(0, validation.batch_size, inference_batch_size):
        stop = min(start + inference_batch_size, validation.batch_size)
        indices = torch.arange(start, stop)
        sample = validation.index_select(indices).to(device)
        features = critic.encode_state(sample)
        raw_q = critic.raw_q_ensemble_from_features(
            features,
            sample.action_chunks,
            _critic_mask(sample, config),
        )
        expected = (RAW_Q_HEADS, stop - start)
        if tuple(raw_q.shape) != expected:
            raise ValueError(f"expected raw Q shape {expected}, got {tuple(raw_q.shape)}")
        if not bool(torch.isfinite(raw_q).all()):
            raise ValueError("critic produced non-finite raw Q values")
        outputs.append(raw_q.detach().float().cpu())
        print(f"[episode-viz] scored {stop}/{validation.batch_size}", flush=True)
    result = torch.cat(outputs, dim=1)
    if tuple(result.shape) != (RAW_Q_HEADS, validation.batch_size):
        raise AssertionError("raw10 concatenation changed the expected shape")
    return result


def _episode_arrays(
    validation,
    raw_q: torch.Tensor,
    episode: Episode,
) -> dict[str, np.ndarray]:
    indices = episode.indices
    values = raw_q.index_select(1, indices).numpy()
    return {
        "timesteps": episode.timesteps.astype(np.float64),
        "progress": np.linspace(0.0, 1.0, episode.length),
        "raw_q": values,
        "q_mean": values.mean(axis=0),
        "q_std": values.std(axis=0),
        "chunk_returns": validation.chunk_returns.index_select(0, indices)
        .float()
        .cpu()
        .numpy(),
        "mc_returns": validation.mc_returns.index_select(0, indices)
        .float()
        .cpu()
        .numpy(),
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _draw_raw10(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    *,
    title: str,
    compact: bool = False,
) -> None:
    x = arrays["timesteps"]
    raw_q = arrays["raw_q"]
    mean = arrays["q_mean"]
    std = arrays["q_std"]
    for head in range(RAW_Q_HEADS):
        ax.plot(
            x,
            raw_q[head],
            color=HEAD_COLORS[head],
            alpha=0.48,
            linewidth=0.8,
            label=f"Q{head + 1}",
        )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color="#34495E",
        alpha=0.17,
        label="mean ± std",
    )
    ax.plot(x, mean, color="#17202A", linewidth=2.4, label="10Q mean")
    ax.set_title(title)
    ax.set_xlabel("Replay timestep")
    ax.set_ylabel("Q(s, dataset action)")
    if not compact:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)


def plot_raw10_episode(
    arrays: dict[str, np.ndarray],
    episode: Episode,
    *,
    checkpoint_label: str,
    output: Path,
) -> None:
    outcome = "Success" if episode.success else "Failure"
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _draw_raw10(
        ax,
        arrays,
        title=(
            f"{outcome} Episode — raw 10Q over time\n"
            f"episode_id={episode.episode_id} | checkpoint={checkpoint_label}"
        ),
    )
    _save_figure(fig, output)


def plot_disagreement(
    success: dict[str, np.ndarray],
    failure: dict[str, np.ndarray],
    *,
    checkpoint_label: str,
    normalized: bool,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x_key = "progress" if normalized else "timesteps"
    ax.plot(
        success[x_key],
        success["q_std"],
        color=SUCCESS_COLOR,
        linewidth=2.0,
        label="Success episode",
    )
    ax.plot(
        failure[x_key],
        failure["q_std"],
        color=FAILURE_COLOR,
        linewidth=2.0,
        label="Failure episode",
    )
    ax.set_title(f"Raw 10Q ensemble disagreement | checkpoint={checkpoint_label}")
    ax.set_xlabel("Normalized episode progress" if normalized else "Replay timestep")
    ax.set_ylabel("Std(Q1, …, Q10)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save_figure(fig, output)


def plot_q_vs_mc_return(
    arrays: dict[str, np.ndarray],
    episode: Episode,
    *,
    checkpoint_label: str,
    output: Path,
) -> None:
    outcome = "Success" if episode.success else "Failure"
    x = arrays["timesteps"]
    fig, left = plt.subplots(figsize=(9.5, 4.8))
    right = left.twinx()
    q_line = left.plot(
        x,
        arrays["q_mean"],
        color="#2471A3",
        linewidth=2.2,
        label="10Q mean",
    )[0]
    return_line = right.plot(
        x,
        arrays["mc_returns"],
        color="#D35400",
        linewidth=1.8,
        linestyle="--",
        label="MC return",
    )[0]
    left.set_title(
        f"{outcome} Episode — Q mean vs raw MC return\n"
        f"episode_id={episode.episode_id} | checkpoint={checkpoint_label}"
    )
    left.set_xlabel("Replay timestep")
    left.set_ylabel("10Q mean", color="#2471A3")
    right.set_ylabel("MC return", color="#D35400")
    left.tick_params(axis="y", labelcolor="#2471A3")
    right.tick_params(axis="y", labelcolor="#D35400")
    left.legend(
        [q_line, return_line],
        [q_line.get_label(), return_line.get_label()],
        loc="upper left",
        bbox_to_anchor=(1.10, 1.0),
    )
    _save_figure(fig, output)


def nearest_progress_indices(episode: Episode) -> dict[str, int]:
    first = float(episode.timesteps[0])
    span = float(episode.timesteps[-1] - episode.timesteps[0])
    selected: dict[str, int] = {}
    for label, progress in PROGRESS_POINTS.items():
        target = first + progress * span
        local_index = int(np.abs(episode.timesteps.astype(float) - target).argmin())
        selected[label] = int(episode.indices[local_index].item())
    return selected


@torch.no_grad()
def score_action_ranking(
    critic: MultiHeadUdivlCritic,
    validation,
    config: dict[str, Any],
    *,
    global_index: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | bool]]]:
    cpu_sample = validation.index_select(torch.tensor([global_index]))
    action_pool = critic.core.action_pool
    sigma = float(
        config.get("critic", {}).get(
            "rank_noise_sigma",
            config.get("critic", {}).get("rankq_noise_sigma", 0.15),
        )
    )
    generator = torch.Generator().manual_seed(int(seed))
    mild, _ = ranking_action_negatives(
        cpu_sample.action_chunks,
        cpu_sample.execution_masks,
        action_mean=action_pool.action_mean.cpu(),
        action_std=action_pool.action_std.cpu(),
        action_min=action_pool.action_min.cpu(),
        action_max=action_pool.action_max.cpu(),
        noise_sigma=0.5 * sigma,
        generator=generator,
    )
    strong, random = ranking_action_negatives(
        cpu_sample.action_chunks,
        cpu_sample.execution_masks,
        action_mean=action_pool.action_mean.cpu(),
        action_std=action_pool.action_std.cpu(),
        action_min=action_pool.action_min.cpu(),
        action_max=action_pool.action_max.cpu(),
        noise_sigma=sigma,
        generator=generator,
    )
    device = next(critic.parameters()).device
    sample = cpu_sample.to(device)
    features = critic.encode_state(sample)
    mask = _critic_mask(sample, config)
    dataset_q = critic.raw_q_ensemble_from_features(
        features,
        sample.action_chunks,
        mask,
    )
    if tuple(dataset_q.shape) != (RAW_Q_HEADS, 1):
        raise ValueError(f"unexpected dataset raw-Q shape: {tuple(dataset_q.shape)}")
    margins: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float | bool]] = {}
    for name, negative in (("mild", mild), ("strong", strong), ("random", random)):
        negative_q = critic.raw_q_ensemble_from_features(
            features,
            negative.to(device),
            mask,
        )
        delta = (dataset_q - negative_q).flatten().float().cpu().numpy()
        margins[name] = delta
        metrics[name] = {
            "unanimous": bool(np.all(delta > 0.0)),
            "positive_head_ratio": float(np.mean(delta > 0.0)),
            "min_margin": float(np.min(delta)),
            "mean_margin": float(np.mean(delta)),
        }
    return margins, metrics


def plot_action_ranking(
    margins: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float | bool]],
    episode: Episode,
    *,
    timestep: int,
    progress_label: str,
    checkpoint_label: str,
    output: Path,
) -> None:
    x = np.arange(RAW_Q_HEADS)
    width = 0.25
    styles = (
        ("mild", -width, "#5DADE2"),
        ("strong", 0.0, "#AF7AC5"),
        ("random", width, "#F5B041"),
    )
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    for name, offset, color in styles:
        ax.bar(x + offset, margins[name], width=width, label=name, color=color)
    ax.axhline(0.0, color="#17202A", linewidth=1.1)
    ax.set_xticks(x, [f"Q{index}" for index in range(1, RAW_Q_HEADS + 1)])
    ax.set_ylabel("ΔQ = Q(dataset action) − Q(perturbation)")
    if episode.success:
        heading = "Success Episode — dataset action ranking"
    else:
        heading = "Failure trajectory diagnostic — dataset action vs perturbations"
    ax.set_title(
        f"{heading}\n"
        f"episode_id={episode.episode_id} | timestep={timestep} "
        f"({progress_label[1:]}% target) | checkpoint={checkpoint_label}"
    )
    annotation = []
    for name in ("mild", "strong", "random"):
        row = metrics[name]
        annotation.append(
            f"{name}: 10/10={str(row['unanimous']).lower()} | "
            f"positive-head ratio={row['positive_head_ratio']:.2f} | "
            f"min Δ={row['min_margin']:.4f}"
        )
    ax.text(
        1.01,
        0.98,
        "\n".join(annotation),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.48))
    _save_figure(fig, output)


def interpolate_episode(
    arrays: dict[str, np.ndarray],
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    progress = arrays["progress"]
    return (
        np.interp(bins, progress, arrays["q_mean"]),
        np.interp(bins, progress, arrays["q_std"]),
    )


def aggregate_progress(
    episodes: list[Episode],
    validation,
    raw_q: torch.Tensor,
    *,
    max_per_class: int,
    bins: int = 100,
) -> dict[str, Any]:
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    progress_bins = np.linspace(0.0, 1.0, bins)
    result: dict[str, Any] = {"progress": progress_bins}
    for success, name in ((True, "success"), (False, "failure")):
        selected = sorted(
            (episode for episode in episodes if episode.success == success),
            key=lambda episode: episode.episode_id,
        )[:max_per_class]
        if not selected:
            raise ValueError(f"no {name} episodes available for aggregation")
        q_curves = []
        disagreement_curves = []
        for episode in selected:
            arrays = _episode_arrays(validation, raw_q, episode)
            q_curve, disagreement_curve = interpolate_episode(arrays, progress_bins)
            q_curves.append(q_curve)
            disagreement_curves.append(disagreement_curve)
        q_stack = np.stack(q_curves)
        disagreement_stack = np.stack(disagreement_curves)
        result[name] = {
            "episode_ids": [episode.episode_id for episode in selected],
            "q_mean": q_stack.mean(axis=0),
            "q_episode_std": q_stack.std(axis=0),
            "disagreement_mean": disagreement_stack.mean(axis=0),
            "disagreement_episode_std": disagreement_stack.std(axis=0),
        }
    return result


def plot_aggregate(
    aggregate: dict[str, Any],
    *,
    checkpoint_label: str,
    metric: str,
    output: Path,
) -> None:
    if metric == "q":
        mean_key = "q_mean"
        std_key = "q_episode_std"
        ylabel = "Episode 10Q mean"
        title = "Validation episode aggregate Q over normalized progress"
    elif metric == "disagreement":
        mean_key = "disagreement_mean"
        std_key = "disagreement_episode_std"
        ylabel = "Within-state raw 10Q std"
        title = "Validation episode aggregate disagreement over normalized progress"
    else:
        raise ValueError(f"unsupported aggregate metric: {metric}")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    progress = aggregate["progress"]
    for name, color in (("success", SUCCESS_COLOR), ("failure", FAILURE_COLOR)):
        mean = aggregate[name][mean_key]
        std = aggregate[name][std_key]
        label = f"{name.capitalize()} (n={len(aggregate[name]['episode_ids'])})"
        ax.plot(progress, mean, color=color, linewidth=2.2, label=label)
        ax.fill_between(progress, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_title(f"{title}\ncheckpoint={checkpoint_label}")
    ax.set_xlabel("Normalized episode progress")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save_figure(fig, output)


def plot_summary(
    success: dict[str, np.ndarray],
    failure: dict[str, np.ndarray],
    success_episode: Episode,
    failure_episode: Episode,
    *,
    checkpoint_label: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    _draw_raw10(
        axes[0, 0],
        success,
        title=f"Success raw 10Q | episode {success_episode.episode_id}",
        compact=True,
    )
    _draw_raw10(
        axes[0, 1],
        failure,
        title=f"Failure raw 10Q | episode {failure_episode.episode_id}",
        compact=True,
    )
    axes[1, 0].plot(
        success["progress"], success["q_std"], color=SUCCESS_COLOR, linewidth=2, label="Success"
    )
    axes[1, 0].plot(
        failure["progress"], failure["q_std"], color=FAILURE_COLOR, linewidth=2, label="Failure"
    )
    axes[1, 0].set_title("Raw 10Q disagreement")
    axes[1, 0].set_xlabel("Normalized episode progress")
    axes[1, 0].set_ylabel("10Q std")
    axes[1, 0].legend()
    axes[1, 1].plot(
        success["progress"], success["q_mean"], color=SUCCESS_COLOR, linewidth=2, label="Success"
    )
    axes[1, 1].plot(
        failure["progress"], failure["q_mean"], color=FAILURE_COLOR, linewidth=2, label="Failure"
    )
    axes[1, 1].set_title("Ensemble Q mean")
    axes[1, 1].set_xlabel("Normalized episode progress")
    axes[1, 1].set_ylabel("10Q mean")
    axes[1, 1].legend()
    fig.suptitle(
        f"RoboTwin click_bell critic episode summary | checkpoint={checkpoint_label}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _save_figure(fig, output)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.ptp(x)) <= 0.0:
        return 0.0
    return float(np.polyfit(x.astype(float), y.astype(float), 1)[0])


def _episode_summary(
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    raw_q = torch.from_numpy(arrays["raw_q"])
    mc_return = torch.from_numpy(arrays["mc_returns"])
    head_slopes = [
        _linear_slope(arrays["progress"], arrays["raw_q"][head])
        for head in range(RAW_Q_HEADS)
    ]
    return {
        "q_mean_start": float(arrays["q_mean"][0]),
        "q_mean_end": float(arrays["q_mean"][-1]),
        "q_mean_slope": _linear_slope(arrays["progress"], arrays["q_mean"]),
        "q_mean_slope_axis": "normalized_progress_0_to_1",
        "q_mean_slope_per_timestep": _linear_slope(
            arrays["timesteps"], arrays["q_mean"]
        ),
        "q_std_mean": float(arrays["q_std"].mean()),
        "q_std_max": float(arrays["q_std"].max()),
        "q_mc_spearman": spearman_correlation(raw_q.mean(dim=0), mc_return),
        "head_progress_slopes": head_slopes,
    }


def _ranking_summary_flat(
    metrics: dict[str, dict[str, float | bool]],
) -> dict[str, float | bool]:
    flattened: dict[str, float | bool] = {}
    for name in ("mild", "strong", "random"):
        flattened[f"{name}_unanimous"] = bool(metrics[name]["unanimous"])
        flattened[f"{name}_positive_head_ratio"] = float(
            metrics[name]["positive_head_ratio"]
        )
        flattened[f"{name}_min_margin"] = float(metrics[name]["min_margin"])
        flattened[f"{name}_mean_margin"] = float(metrics[name]["mean_margin"])
    return flattened


def _trace_json(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: arrays[key].tolist()
        for key in (
            "timesteps",
            "progress",
            "q_mean",
            "q_std",
            "chunk_returns",
            "mc_returns",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--success-episode-id", type=int)
    parser.add_argument("--failure-episode-id", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--max-episodes-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    _configure_plot_style()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = _resolve(args.critic_config)
    checkpoint_path = _resolve(args.checkpoint)
    replay_path = _resolve(args.replay)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    for name, path in (("critic config", config_path), ("checkpoint", checkpoint_path), ("replay", replay_path)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing {name}: {path}")

    config = load_config(config_path)
    configured_validation = _resolve(config["data"]["validation_path"]).resolve()
    if replay_path.resolve() != configured_validation:
        raise ValueError(
            "episode visualization is validation-only; expected replay "
            f"{configured_validation}, got {replay_path.resolve()}"
        )
    config.setdefault("training", {})["device"] = args.device
    validation = prepare_replay_from_config(
        load_replay(replay_path),
        config.get("data", {}),
    )
    if validation.mc_returns is None:
        raise ValueError("validation replay preprocessing did not produce mc_returns")

    episodes = episode_catalog(validation)
    success_episode = choose_episode(
        episodes,
        success=True,
        requested_id=args.success_episode_id,
    )
    failure_episode = choose_episode(
        episodes,
        success=False,
        requested_id=args.failure_episode_id,
    )
    for source, episode in (("requested" if args.success_episode_id is not None else "automatic", success_episode), ("requested" if args.failure_episode_id is not None else "automatic", failure_episode)):
        print(
            f"[episode-viz] {source} selection episode_id={episode.episode_id} "
            f"length={episode.length} success={int(episode.success)}",
            flush=True,
        )
    success_validation = _validate_episode_extraction(validation, success_episode)
    failure_validation = _validate_episode_extraction(validation, failure_episode)
    print(
        f"[episode-viz] success reward/mc={success_validation} "
        f"failure reward/mc={failure_validation}",
        flush=True,
    )

    state = build_train_state(config, validation, device=args.device)
    payload = load_critic_checkpoint(checkpoint_path, state, load_optimizer=False)
    checkpoint_step = int(payload.get("training_step", -1))
    if checkpoint_step <= 0:
        raise ValueError(f"checkpoint has invalid critic step {checkpoint_step}")
    critic = state.critic
    if not isinstance(critic, MultiHeadUdivlCritic):
        raise TypeError("episode raw10 visualization requires MultiHeadUdivlCritic")
    if critic.num_raw_q_heads != RAW_Q_HEADS:
        raise ValueError(f"expected 10 raw Q heads, got {critic.num_raw_q_heads}")
    critic.eval()
    checkpoint_label = (
        f"{checkpoint_step // 1000}k"
        if checkpoint_step % 1000 == 0
        else str(checkpoint_step)
    )
    output_dir.mkdir(parents=True)

    raw_q = score_validation_raw10(
        critic,
        validation,
        config,
        inference_batch_size=args.inference_batch_size,
    )
    success_arrays = _episode_arrays(validation, raw_q, success_episode)
    failure_arrays = _episode_arrays(validation, raw_q, failure_episode)

    outputs: list[Path] = []
    success_raw_path = output_dir / "success_episode_raw10_q.png"
    failure_raw_path = output_dir / "failure_episode_raw10_q.png"
    plot_raw10_episode(success_arrays, success_episode, checkpoint_label=checkpoint_label, output=success_raw_path)
    plot_raw10_episode(failure_arrays, failure_episode, checkpoint_label=checkpoint_label, output=failure_raw_path)
    outputs.extend([success_raw_path, failure_raw_path])

    disagreement_path = output_dir / "success_failure_q_disagreement.png"
    disagreement_normalized_path = output_dir / "success_failure_q_disagreement_normalized.png"
    plot_disagreement(success_arrays, failure_arrays, checkpoint_label=checkpoint_label, normalized=False, output=disagreement_path)
    plot_disagreement(success_arrays, failure_arrays, checkpoint_label=checkpoint_label, normalized=True, output=disagreement_normalized_path)
    outputs.extend([disagreement_path, disagreement_normalized_path])

    success_mc_path = output_dir / "success_episode_q_vs_mc_return.png"
    failure_mc_path = output_dir / "failure_episode_q_vs_mc_return.png"
    plot_q_vs_mc_return(success_arrays, success_episode, checkpoint_label=checkpoint_label, output=success_mc_path)
    plot_q_vs_mc_return(failure_arrays, failure_episode, checkpoint_label=checkpoint_label, output=failure_mc_path)
    outputs.extend([success_mc_path, failure_mc_path])

    ranking_summary: dict[str, dict[str, Any]] = {"success": {}, "failure": {}}
    for outcome, episode in (("success", success_episode), ("failure", failure_episode)):
        for order, (progress_label, global_index) in enumerate(
            nearest_progress_indices(episode).items()
        ):
            timestep = int(validation.timesteps[global_index].item())
            margins, metrics = score_action_ranking(
                critic,
                validation,
                config,
                global_index=global_index,
                seed=args.seed + episode.episode_id * 10_007 + order * 101,
            )
            ranking_path = output_dir / f"{outcome}_{progress_label}_action_ranking.png"
            plot_action_ranking(
                margins,
                metrics,
                episode,
                timestep=timestep,
                progress_label=progress_label,
                checkpoint_label=checkpoint_label,
                output=ranking_path,
            )
            outputs.append(ranking_path)
            ranking_summary[outcome][progress_label] = {
                "timestep": timestep,
                **_ranking_summary_flat(metrics),
                "head_margins": {name: values.tolist() for name, values in margins.items()},
            }

    aggregate = aggregate_progress(
        episodes,
        validation,
        raw_q,
        max_per_class=args.max_episodes_per_class,
    )
    aggregate_q_path = output_dir / "success_failure_aggregate_q_progress.png"
    aggregate_disagreement_path = output_dir / "success_failure_aggregate_disagreement.png"
    plot_aggregate(aggregate, checkpoint_label=checkpoint_label, metric="q", output=aggregate_q_path)
    plot_aggregate(
        aggregate,
        checkpoint_label=checkpoint_label,
        metric="disagreement",
        output=aggregate_disagreement_path,
    )
    outputs.extend([aggregate_q_path, aggregate_disagreement_path])

    summary_figure_path = output_dir / f"critic_{checkpoint_label}_episode_summary.png"
    plot_summary(
        success_arrays,
        failure_arrays,
        success_episode,
        failure_episode,
        checkpoint_label=checkpoint_label,
        output=summary_figure_path,
    )
    outputs.append(summary_figure_path)

    success_stats = _episode_summary(success_arrays)
    failure_stats = _episode_summary(failure_arrays)
    progress = aggregate["progress"]
    aggregate_summary: dict[str, Any] = {}
    for outcome in ("success", "failure"):
        values = aggregate[outcome]
        aggregate_summary[outcome] = {
            "episode_ids": values["episode_ids"],
            "q_mean_start": float(values["q_mean"][0]),
            "q_mean_end": float(values["q_mean"][-1]),
            "q_mean_slope": _linear_slope(progress, values["q_mean"]),
            "disagreement_start": float(values["disagreement_mean"][0]),
            "disagreement_end": float(values["disagreement_mean"][-1]),
            "disagreement_mean": float(values["disagreement_mean"].mean()),
            "disagreement_max": float(values["disagreement_mean"].max()),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "critic_config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint_step,
        "validation_replay": str(replay_path.resolve()),
        "validation_transition_count": validation.batch_size,
        "validation_episode_count": len(episodes),
        "seed": args.seed,
        "raw_q_order": [f"Q{pair}{head}" for pair in range(1, 6) for head in range(1, 3)],
        "raw_q_shape": list(raw_q.shape),
        "pair_min_used": False,
        "success_episode_id": success_episode.episode_id,
        "failure_episode_id": failure_episode.episode_id,
        "success_length": success_episode.length,
        "failure_length": failure_episode.length,
        "success_q_mean_start": success_stats["q_mean_start"],
        "success_q_mean_end": success_stats["q_mean_end"],
        "success_q_mean_slope": success_stats["q_mean_slope"],
        "success_q_std_mean": success_stats["q_std_mean"],
        "success_q_std_max": success_stats["q_std_max"],
        "failure_q_mean_start": failure_stats["q_mean_start"],
        "failure_q_mean_end": failure_stats["q_mean_end"],
        "failure_q_mean_slope": failure_stats["q_mean_slope"],
        "failure_q_std_mean": failure_stats["q_std_mean"],
        "failure_q_std_max": failure_stats["q_std_max"],
        "success_q_mc_spearman": success_stats["q_mc_spearman"],
        "failure_q_mc_spearman": failure_stats["q_mc_spearman"],
        "slope_axis": "normalized_progress_0_to_1",
        "selected_episode_validation": {
            "success": success_validation,
            "failure": failure_validation,
        },
        "per_episode": {
            "success": success_stats,
            "failure": failure_stats,
        },
        "action_ranking": ranking_summary,
        "aggregate": aggregate_summary,
        "episode_traces": {
            "success": _trace_json(success_arrays),
            "failure": _trace_json(failure_arrays),
        },
        "output_files": [str(path.resolve()) for path in outputs],
    }
    summary_path = (
        output_dir
        / f"critic_{checkpoint_label}_episode_visualization_summary.json"
    )
    _atomic_json(summary, summary_path)
    print(f"[episode-viz] raw_q_shape={tuple(raw_q.shape)} pair_min_used=0", flush=True)
    print(f"[episode-viz] summary={summary_path}", flush=True)
    for output in outputs:
        print(f"[episode-viz] figure={output}", flush=True)


if __name__ == "__main__":
    main()
