#!/usr/bin/env python3
"""OGPO-facing evaluation of the final multi-task plain-DIVL critic.

Return and perturbation metrics use every transition in the MT1000 actual
heldout episodes.  Actor candidates use a fixed, task-balanced cache because
generating flow-policy candidates is substantially more expensive.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from build_robotwin_critic_replay import _episode_rows
from ogpo.chi2_regularization import apply_chipo_to_ca_advantage
from ogpo.conservative_advantage import group_relative_conservative_advantage
from ogpo.critic_raw10_evaluator import spearman_correlation
from ogpo.pi05_pytorch_adapter import load_pi05_pytorch_flow_policy
from ogpo.rankq import make_same_state_rankq_actions
from ogpo.replay import add_monte_carlo_returns, load_replay, prepare_replay_from_config
from ogpo.trainer import build_train_state, load_critic_checkpoint
from ogpo.types import ChunkBatch
from ogpo.value_critic_protocol import StateFeatures
from ogpo.zarr_replay import concat_chunk_batches, rows_to_chunk_batch
from robotwin_mixed_protocol import TASKS, heldout_episodes
from train_udivl_critic import load_config


C_RUN = "robotwin_multitask10_divl_vmean_headonly_taskoutcomebalanced_20k"
A_RUN = "robotwin_multitask10_divl_vmean_fullft_taskoutcomebalanced_20k"
B_RUN = "robotwin_click_bell_divl_vmean_headonly_uniform_8k"


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checkpoint(run: str, step: int) -> Path:
    return ROOT / "outputs/ogpo/checkpoints" / run / "milestones" / f"critic_step_{step:05d}.pt"


def build_task_batch(task: str, episodes: list[dict[str, Any]], config: dict[str, Any]) -> ChunkBatch:
    batches = []
    for item in episodes:
        if item["task"] != task:
            continue
        rows = _episode_rows(
            Path(item["path"]),
            episode_id=int(item["episode_id"]),
            gamma=0.999,
            behavior_policy="pi05_pytorch_clean50_base",
        )
        batches.append(
            prepare_replay_from_config(
                add_monte_carlo_returns(rows_to_chunk_batch(rows)), config["data"]
            )
        )
    if not batches:
        raise ValueError(f"no heldout episodes for task {task}")
    return concat_chunk_batches(batches)


def fixed_task_indices(batch: ChunkBatch, count: int, seed: int) -> torch.Tensor:
    """Task-local deterministic sample with episode coverage before random fill."""
    generator = torch.Generator().manual_seed(int(seed))
    chosen: list[int] = []
    for episode_id in sorted(int(x) for x in torch.unique(batch.episode_ids)):
        pool = torch.nonzero(batch.episode_ids == episode_id, as_tuple=False).flatten()
        chosen.append(int(pool[torch.randint(pool.numel(), (1,), generator=generator)].item()))
    remaining = [i for i in torch.randperm(batch.batch_size, generator=generator).tolist() if i not in set(chosen)]
    chosen.extend(remaining[: max(0, min(count, batch.batch_size) - len(chosen))])
    return torch.tensor(chosen[: min(count, batch.batch_size)], dtype=torch.long)


@torch.no_grad()
def create_candidate_cache(
    task_batches: dict[str, ChunkBatch],
    actor_config: dict[str, Any],
    cache_path: Path,
    *,
    count_per_task: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    selected_parts = []
    manifest = []
    for task_index, task in enumerate(TASKS):
        batch = task_batches[task]
        indices = fixed_task_indices(batch, count_per_task, seed + task_index)
        selected = batch.index_select(indices)
        selected_parts.append(selected)
        for local_index, source_index in enumerate(indices.tolist()):
            manifest.append({
                "task": task,
                "candidate_index_within_task": local_index,
                "source_transition_index": source_index,
                "episode_id": int(batch.episode_ids[source_index]),
                "timestep": int(batch.timesteps[source_index]),
                "success": bool(batch.successes[source_index]),
            })
    selected = concat_chunk_batches(selected_parts)
    flow_cfg = actor_config["flow"]
    actor_cfg = actor_config["actor"]
    policy = load_pi05_pytorch_flow_policy(
        checkpoint_dir=Path(flow_cfg["checkpoint_dir"]),
        train_config_name=str(flow_cfg["train_config"]),
        image_mapping=dict(flow_cfg.get("image_mapping_override", flow_cfg.get("image_mapping", {}))),
        image_container_key=flow_cfg.get("image_container_key"),
        transpose_images_to_chw=bool(flow_cfg.get("transpose_images_to_chw", False)),
        environment_action_dim=selected.action_dim,
        num_steps=int(flow_cfg.get("num_steps", 10)),
        stochastic_variance=float(flow_cfg.get("stochastic_variance", 0.04)),
        sde_mode=str(flow_cfg.get("sde_mode", "gaussian_adapter")),
        constant_noise_std=float(flow_cfg.get("constant_noise_std", 0.005)),
        learn_sde_std=bool(flow_cfg.get("learn_sde_std", True)),
        randn_clip_value=float(flow_cfg.get("randn_clip_value", 3.0)),
        residual_hidden_dim=int(actor_cfg.get("hidden_dim", 128)),
        residual_enabled=bool(actor_cfg.get("residual_enabled", True)),
        backend_train_mode="none",
        device=device,
    )
    policy.eval()
    generator = torch.Generator(device=torch.device(device)).manual_seed(int(seed))
    candidate_parts = []
    for index in range(selected.batch_size):
        sample = selected.index_select(torch.tensor([index]))
        condition = policy.condition_from_batch(sample)
        rollout = policy.rollout(condition, group_size=4, generator=generator)
        condition_g = policy.repeat_condition(condition, 4)
        actions = policy.flat_actions_to_environment(rollout.endpoint, condition_g)
        candidate_parts.append(actions.reshape(1, 4, selected.generated_horizon, selected.action_dim).cpu())
        if (index + 1) % 16 == 0 or index + 1 == selected.batch_size:
            print(f"candidate_generation={index + 1}/{selected.batch_size}", flush=True)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "group_size": 4,
        "count_per_task": count_per_task,
        "task_order": list(TASKS),
        "manifest": manifest,
        "batch": asdict(selected),
        "candidate_actions": torch.cat(candidate_parts, dim=0),
        "base_actor_checkpoint": str((Path(flow_cfg["checkpoint_dir"]) / "model.safetensors").resolve()),
    }
    atomic_torch(cache_path, payload)
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return payload


@torch.no_grad()
def raw_candidate_q(critic, batch: ChunkBatch, candidates: torch.Tensor, config: dict[str, Any], size: int) -> torch.Tensor:
    outputs = []
    device = next(critic.parameters()).device
    group = candidates.shape[1]
    for start in range(0, batch.batch_size, size):
        stop = min(start + size, batch.batch_size)
        sample = batch.index_select(torch.arange(start, stop)).to(device)
        features = critic.encode_state(sample)
        grouped = StateFeatures(readout=features.readout.repeat_interleave(group, dim=0))
        mask = sample.execution_masks if config.get("data", {}).get("use_execution_mask", True) else torch.ones_like(sample.execution_masks)
        masks = mask[:, None].expand(-1, group, -1).reshape((stop - start) * group, -1)
        actions = candidates[start:stop].to(device).reshape((stop - start) * group, sample.generated_horizon, sample.action_dim)
        outputs.append(critic.raw_q_ensemble_from_features(grouped, actions, masks).reshape(10, stop - start, group).cpu())
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def score_task(critic, batch: ChunkBatch, actions: torch.Tensor, config: dict[str, Any], size: int) -> torch.Tensor:
    """Return [logged/mild/strong/random, 10, N]."""
    outputs = [[], [], [], []]
    device = next(critic.parameters()).device
    for start in range(0, batch.batch_size, size):
        stop = min(start + size, batch.batch_size)
        sample = batch.index_select(torch.arange(start, stop)).to(device)
        features = critic.encode_state(sample)
        mask = sample.execution_masks if config.get("data", {}).get("use_execution_mask", True) else torch.ones_like(sample.execution_masks)
        for action_index in range(4):
            outputs[action_index].append(
                critic.raw_q_ensemble_from_features(features, actions[action_index, start:stop].to(device), mask).cpu()
            )
    return torch.stack([torch.cat(parts, dim=1) for parts in outputs])


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q).item()) if values.numel() else float("nan")


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten(); y = y.float().flatten()
    x = x - x.mean(); y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum().div(denom).item()) if float(denom) > 1e-12 else 0.0


def same_episode_rank_metrics(raw: torch.Tensor, target: torch.Tensor, episode_ids: torch.Tensor) -> dict[str, Any]:
    correct = torch.zeros(10, dtype=torch.float64)
    mean_correct = 0
    total = 0
    episode_rhos = []
    for episode_id in sorted(int(x) for x in torch.unique(episode_ids)):
        mask = episode_ids == episode_id
        local_target = target[mask]
        if local_target.numel() < 2 or int(torch.unique(local_target).numel()) < 2:
            continue
        local_raw = raw[:, mask]
        episode_rhos.append(spearman_correlation(local_raw.mean(0), local_target))
        pairs = torch.triu_indices(local_target.numel(), local_target.numel(), offset=1)
        delta = local_target[pairs[0]] - local_target[pairs[1]]
        valid = delta != 0
        if not bool(valid.any()):
            continue
        prediction_delta = local_raw[:, pairs[0]] - local_raw[:, pairs[1]]
        correct += (torch.sign(prediction_delta[:, valid]) == torch.sign(delta[valid])).sum(1)
        mean_delta = local_raw.mean(0)[pairs[0]] - local_raw.mean(0)[pairs[1]]
        mean_correct += int(
            (torch.sign(mean_delta[valid]) == torch.sign(delta[valid])).sum()
        )
        total += int(valid.sum())
    member_accuracy = correct / max(total, 1)
    rho = torch.tensor(episode_rhos, dtype=torch.float32)
    return {
        "pairwise_accuracy": float(mean_correct / max(total, 1)),
        "mean_member_pairwise_accuracy": float(member_accuracy.mean()),
        "worst_member_pairwise_accuracy": float(member_accuracy.min()),
        "member_pairwise_accuracy": member_accuracy.tolist(),
        "pair_count": total,
        "episode_spearman_count": len(episode_rhos),
        "episode_spearman_median": percentile(rho, 0.5),
        "episode_spearman_p10": percentile(rho, 0.1),
        "episode_spearman_min": float(rho.min()) if rho.numel() else float("nan"),
    }


def relation_metrics(preferred: torch.Tensor, inferior: torch.Tensor, prefix: str) -> dict[str, Any]:
    margins = preferred - inferior
    mean_margin = margins.mean(0)
    member_accuracy = (margins > 0).float().mean(1)
    return {
        f"{prefix}_accuracy": float((mean_margin > 0).float().mean()),
        f"{prefix}_unanimous_accuracy": float((margins > 0).all(0).float().mean()),
        f"{prefix}_mean_margin": float(mean_margin.mean()),
        f"{prefix}_median_margin": percentile(mean_margin, 0.5),
        f"{prefix}_p10_margin": percentile(mean_margin, 0.1),
        f"{prefix}_member_accuracy": member_accuracy.tolist(),
    }


def return_and_action_metrics(scores: torch.Tensor, batch: ChunkBatch) -> dict[str, Any]:
    raw = scores[0].float()
    mean_q = raw.mean(0)
    target = batch.mc_returns.float()
    error = mean_q - target
    disagreement = raw.std(0, unbiased=False)
    member_rho = [spearman_correlation(head, target) for head in raw]
    centered = raw - raw.mean(1, keepdim=True)
    norms = centered.square().sum(1).sqrt().clamp_min(1e-12)
    correlation = (centered / norms[:, None]) @ (centered / norms[:, None]).T
    corr_values = correlation[~torch.eye(10, dtype=torch.bool)]
    metrics: dict[str, Any] = {
        "transition_count": batch.batch_size,
        "success_transition_count": int(batch.successes.bool().sum()),
        "mc_spearman": spearman_correlation(mean_q, target),
        "member_mc_spearman": member_rho,
        "worst_member_mc_spearman": min(member_rho),
        "rmse": float(error.square().mean().sqrt()),
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
        "q_head_corr_mean": float(corr_values.mean()),
        "q_head_corr_median": percentile(corr_values, 0.5),
        "q_head_corr_min": float(corr_values.min()),
        "q_head_corr_max": float(corr_values.max()),
        "ensemble_std_mean": float(disagreement.mean()),
        "ensemble_std_median": percentile(disagreement, 0.5),
        "ensemble_std_p90": percentile(disagreement, 0.9),
        "error_disagreement_pearson": pearson(error.abs(), disagreement),
        "error_disagreement_spearman": spearman_correlation(error.abs(), disagreement),
    }
    metrics.update(same_episode_rank_metrics(raw, target, batch.episode_ids))
    high = target > 0.8
    high_error = error[high]
    metrics.update({
        "high_return_count": int(high.sum()),
        "high_return_mae": float(high_error.abs().mean()),
        "high_return_bias": float(high_error.mean()),
    })
    terminal = batch.successes.bool() & batch.dones.bool() & torch.isclose(target, torch.ones_like(target), atol=1e-5)
    terminal_q = mean_q[terminal]
    terminal_error = (terminal_q - 1).abs()
    metrics.update({
        "terminal_success_count": int(terminal.sum()),
        "terminal_error_mean": float(terminal_error.mean()),
        "terminal_error_median": percentile(terminal_error, 0.5),
        "terminal_q_mean": float(terminal_q.mean()),
        "terminal_q_p10": percentile(terminal_q, 0.1),
        "terminal_q_p90": percentile(terminal_q, 0.9),
    })
    success = batch.successes.bool()
    logged, mild, strong, random = (x[:, success] for x in scores)
    metrics.update(relation_metrics(logged, mild, "lm"))
    metrics.update(relation_metrics(mild, strong, "ms"))
    metrics.update(relation_metrics(strong, random, "sr"))
    all_margins = torch.cat((logged - mild, mild - strong, strong - random), dim=1)
    metrics["overall_action_accuracy"] = float((all_margins.mean(0) > 0).float().mean())
    metrics["worst_member_action_accuracy"] = float((all_margins > 0).float().mean(1).min())
    metrics["member_overall_action_accuracy"] = (all_margins > 0).float().mean(1).tolist()
    return metrics


def actor_metrics(candidate_q: torch.Tensor, actor_config: dict[str, Any]) -> dict[str, Any]:
    ca, member_advantage, ca_stats = group_relative_conservative_advantage(candidate_q)
    consensus_ratio = member_advantage.mean(0).abs() / (
        member_advantage.std(0, unbiased=False) + 1.0e-8
    )
    positive = (member_advantage > 0).all(0)
    negative = (member_advantage < 0).all(0)
    mixed = ~(positive | negative)
    nonzero = ca != 0
    magnitudes = ca[nonzero].abs()
    chi2 = actor_config.get("actor", {}).get("chi2", {})
    final, chipo_stats = apply_chipo_to_ca_advantage(
        ca,
        candidate_q,
        torch.ones_like(ca),
        beta_base=float(chi2.get("beta_base", 0.1)),
        q_std_target=float(chi2.get("q_std_target", 1.0)),
        ensemble_alpha=float(chi2.get("ensemble_alpha", 5.0)),
        r_max=float(chi2.get("r_max", 10.0)),
        normalize_group=bool(chi2.get("normalize_group", False)),
    )
    final_nonzero = final != 0
    return {
        "candidate_state_count": int(candidate_q.shape[1]),
        "candidate_group_size": int(candidate_q.shape[2]),
        "ca_nonzero_fraction": float(nonzero.float().mean()),
        "candidate_zero_advantage_fraction": float((~nonzero).float().mean()),
        "group_all_zero_advantage_fraction": float((~nonzero).all(1).float().mean()),
        "ca_nonzero_abs_median": percentile(magnitudes, 0.5),
        "ca_nonzero_abs_mean": float(magnitudes.mean()) if magnitudes.numel() else 0.0,
        "ca_nonzero_abs_p10": percentile(magnitudes, 0.1),
        "ca_nonzero_abs_p90": percentile(magnitudes, 0.9),
        "ca_advantage_mean": float(ca.mean()),
        "ca_advantage_std": float(ca.std(unbiased=False)),
        "all_positive_fraction": float(positive.float().mean()),
        "all_negative_fraction": float(negative.float().mean()),
        "mixed_sign_fraction": float(mixed.float().mean()),
        "ca_chipo_nonzero_fraction_at_ratio1": float(final_nonzero.float().mean()),
        "chipo_extra_filtering_fraction_at_ratio1": float((nonzero & ~final_nonzero).float().mean()),
        "ca_chipo_abs_mean_at_ratio1": float(final.abs().mean()),
        "chipo_beta_at_ratio1": float(chipo_stats.beta),
        "ca_sign_agreement_fraction": float(ca_stats.sign_agreement_ratio),
        "candidate_consensus_ratio_median": percentile(consensus_ratio, 0.5),
        "candidate_consensus_ratio_p10": percentile(consensus_ratio, 0.1),
        "candidate_consensus_ratio_p90": percentile(consensus_ratio, 0.9),
        "candidate_consensus_ratio_gt1_fraction": float(
            (consensus_ratio > 1.0).float().mean()
        ),
        "candidate_consensus_ratio_gt2_fraction": float(
            (consensus_ratio > 2.0).float().mean()
        ),
    }


def macro(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    skip = {"run", "task", "member_mc_spearman", "member_pairwise_accuracy", "lm_member_accuracy", "ms_member_accuracy", "sr_member_accuracy", "member_overall_action_accuracy"}
    result: dict[str, Any] = {"run": label, "task": "Macro average", "task_count": len(rows)}
    for key in rows[0]:
        if key in skip or isinstance(rows[0][key], (list, dict, str)):
            continue
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        if values:
            result[key] = float(np.mean(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--candidate-states-per-task", type=int, default=64)
    parser.add_argument("--c-steps", default="12000,20000")
    parser.add_argument("--a-steps", default="12000")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--candidate-cache", default=None)
    parser.add_argument("--skip-b-reference", action="store_true")
    parser.add_argument(
        "--critic-spec",
        action="append",
        default=[],
        metavar="LABEL=CHECKPOINT",
        help="evaluate an explicit checkpoint in addition to --c-steps/--a-steps",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    cache_path = output / "task_balanced_g4_candidate_cache.pt"

    c_steps = [int(value) for value in args.c_steps.split(",") if value.strip()]
    if len(c_steps) != len(set(c_steps)):
        raise ValueError("--c-steps must contain unique integer steps")
    a_steps = [int(value) for value in args.a_steps.split(",") if value.strip()]
    if len(a_steps) != len(set(a_steps)):
        raise ValueError("--a-steps must contain unique integer steps")
    c_paths = {step: checkpoint(C_RUN, step) for step in c_steps}
    a_paths = {step: checkpoint(A_RUN, step) for step in a_steps}
    explicit_specs = []
    for value in args.critic_spec:
        if "=" not in value:
            raise ValueError(f"--critic-spec must be LABEL=CHECKPOINT, got {value!r}")
        label, checkpoint_value = value.split("=", 1)
        explicit_path = Path(checkpoint_value)
        if not explicit_path.is_absolute():
            explicit_path = ROOT / explicit_path
        explicit_specs.append((label, -1, explicit_path))
    critic_specs = [
        *((f"C_balanced_{step // 1000}k", step, path) for step, path in c_paths.items()),
        *((f"A_balanced_{step // 1000}k", step, path) for step, path in a_paths.items()),
        *explicit_specs,
    ]
    if not critic_specs:
        raise ValueError("at least one checkpoint must be requested")
    labels = [label for label, _, _ in critic_specs]
    if len(labels) != len(set(labels)):
        raise ValueError("critic labels must be unique")
    b_path = checkpoint(B_RUN, 8000)
    required_paths = [path for _, _, path in critic_specs]
    if not args.skip_b_reference:
        required_paths.append(b_path)
    for path in required_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    config = load_config(ROOT / "configs/ogpo/robotwin_multitask10_divl_vmean_headonly_taskoutcomebalanced_20k.yaml")
    actor_config = load_config(ROOT / "configs/ogpo/robotwin_click_bell_C8k_mt_headonly_actor_bs4_lr2e6_4k.yaml")
    episodes = heldout_episodes()
    task_batches = {task: build_task_batch(task, episodes, config) for task in TASKS}
    if args.candidate_cache:
        source_cache = Path(args.candidate_cache)
        if not source_cache.is_absolute():
            source_cache = ROOT / source_cache
        candidate_cache = torch.load(source_cache, map_location="cpu", weights_only=False)
        if int(candidate_cache["seed"]) != args.seed:
            raise ValueError(f"candidate cache seed mismatch: {candidate_cache['seed']} != {args.seed}")
        if int(candidate_cache["count_per_task"]) != args.candidate_states_per_task:
            raise ValueError("candidate cache count_per_task mismatch")
        atomic_torch(cache_path, candidate_cache)
        print(f"reused_candidate_cache={source_cache}", flush=True)
    else:
        candidate_cache = create_candidate_cache(
            task_batches, actor_config, cache_path,
            count_per_task=args.candidate_states_per_task, seed=args.seed, device=args.device,
        )
    candidate_batch = ChunkBatch(**candidate_cache["batch"])
    candidate_actions = candidate_cache["candidate_actions"]

    init = load_replay(ROOT / config["data"]["distributed_model_init_path"])
    state = build_train_state(config, init, device=args.device)
    # The action-pool statistics are fixed training-data buffers, so use the
    # first requested C checkpoint to materialize one perturbation set shared
    # by every C checkpoint and the paired B8 click_bell reference.
    load_critic_checkpoint(critic_specs[0][2], state, load_optimizer=False)
    c_critic = state.critic.eval()

    # Freeze one exact perturbation set from C12's training action statistics.
    task_actions: dict[str, torch.Tensor] = {}
    for task_index, task in enumerate(TASKS):
        batch = task_batches[task]
        pool = c_critic.core.action_pool
        generated = make_same_state_rankq_actions(
            batch.action_chunks, batch.execution_masks,
            action_mean=pool.action_mean, action_std=pool.action_std,
            action_min=pool.action_min, action_max=pool.action_max,
            mild_sigma=0.02, strong_sigma=0.05, use_random_negative=True,
            generator=torch.Generator().manual_seed(args.seed + 100 + task_index),
        )
        assert generated.random is not None
        task_actions[task] = torch.stack((generated.logged, generated.mild, generated.strong, generated.random))

    per_task = []
    prediction_rows = []
    candidate_rows = []
    ten_task_rows = []
    for run_label, eval_step, eval_path in critic_specs:
        load_critic_checkpoint(eval_path, state, load_optimizer=False)
        c_critic = state.critic.eval()
        candidate_q = raw_candidate_q(c_critic, candidate_batch, candidate_actions, actor_config, args.inference_batch_size)
        step_rows = []
        offset = 0
        for task in TASKS:
            started = time.monotonic()
            batch = task_batches[task]
            scores = score_task(c_critic, batch, task_actions[task], config, args.inference_batch_size)
            row = {"run": run_label, "task": task, **return_and_action_metrics(scores, batch)}
            count = args.candidate_states_per_task
            task_candidate_q = candidate_q[:, offset:offset + count]
            row.update(actor_metrics(task_candidate_q, actor_config))
            per_task.append(row)
            step_rows.append(row)
            for index in range(batch.batch_size):
                item = {
                    "run": run_label, "task": task,
                    "episode_id": int(batch.episode_ids[index]), "timestep": int(batch.timesteps[index]),
                    "success": bool(batch.successes[index]), "mc_return": float(batch.mc_returns[index]),
                }
                for head in range(10):
                    item[f"logged_q{head}"] = float(scores[0, head, index])
                    item[f"mild_q{head}"] = float(scores[1, head, index])
                    item[f"strong_q{head}"] = float(scores[2, head, index])
                    item[f"random_q{head}"] = float(scores[3, head, index])
                prediction_rows.append(item)
            for local in range(task_candidate_q.shape[1]):
                for group in range(4):
                    item = {"run": run_label, "task": task, "state_index": local, "candidate_index": group}
                    for head in range(10):
                        item[f"q{head}"] = float(task_candidate_q[head, local, group])
                    candidate_rows.append(item)
            offset += count
            print(f"{run_label}_task_complete={task} elapsed={time.monotonic()-started:.1f}s", flush=True)
            write_csv(output / "ten_task_metrics.partial.csv", per_task)
        ten_task_rows.extend(step_rows)
        ten_task_rows.append(macro(step_rows, run_label))

    # Strict paired click_bell B8 reference: identical batch, perturbations and candidates.
    b_bell = None
    if not args.skip_b_reference:
        load_critic_checkpoint(b_path, state, load_optimizer=False)
        b_scores = score_task(state.critic.eval(), task_batches["click_bell"], task_actions["click_bell"], config, args.inference_batch_size)
        b_candidate_q = raw_candidate_q(
            state.critic, candidate_batch.index_select(torch.arange(TASKS.index("click_bell") * args.candidate_states_per_task, (TASKS.index("click_bell") + 1) * args.candidate_states_per_task)),
            candidate_actions[TASKS.index("click_bell") * args.candidate_states_per_task:(TASKS.index("click_bell") + 1) * args.candidate_states_per_task],
            actor_config, args.inference_batch_size,
        )
        b_bell = {"run": "B_8k", "task": "click_bell", **return_and_action_metrics(b_scores, task_batches["click_bell"])}
        b_bell.update(actor_metrics(b_candidate_q, actor_config))
    multitask_bells = [row for row in per_task if row["task"] == "click_bell"]

    summary_keys = ["mc_spearman", "pairwise_accuracy", "rmse", "terminal_error_mean", "lm_accuracy", "ms_accuracy", "sr_accuracy", "worst_member_action_accuracy", "ca_nonzero_fraction", "q_head_corr_mean"]
    final_summary = [{key: row.get(key) for key in ("run", "task", *summary_keys)} for row in ten_task_rows]
    bell_keys = [
        "mc_spearman", "pairwise_accuracy", "rmse", "high_return_mae", "terminal_error_mean",
        "lm_accuracy", "lm_unanimous_accuracy", "ms_accuracy", "ms_unanimous_accuracy", "sr_accuracy",
        "overall_action_accuracy", "worst_member_action_accuracy", "ca_nonzero_fraction", "ca_nonzero_abs_median",
        "q_head_corr_mean", "ensemble_std_mean", "error_disagreement_pearson",
    ]
    bell_rows = [*multitask_bells] if b_bell is None else [b_bell, *multitask_bells]
    bell_comparison = [{"run": row["run"], **{key: row[key] for key in bell_keys}} for row in bell_rows]

    write_csv(output / "ten_task_full_metrics.csv", ten_task_rows)
    write_csv(output / "ten_task_final_summary.csv", final_summary)
    write_csv(output / "click_bell_B8_vs_C12.csv", bell_comparison)
    write_csv(output / "transition_raw10_predictions.csv", prediction_rows)
    write_csv(output / "candidate_raw10_predictions.csv", candidate_rows)
    atomic_json(output / "candidate_manifest.json", {
        "seed": args.seed, "group_size": 4, "states_per_task": args.candidate_states_per_task,
        "base_actor_checkpoint": candidate_cache["base_actor_checkpoint"], "samples": candidate_cache["manifest"],
    })
    atomic_json(output / "RESULTS.json", {
        "schema_version": 1,
        "protocol": {
            "split": "MT1000 actual heldout (global randperm seed 27)",
            "return_and_perturbation": "all 30120 transitions; action metrics on all 10320 success transitions",
            "mc_return_gamma": 0.999,
            "raw_q": "mean of 10 raw heads; pair-min unused",
            "pairwise": "within-episode pairs with unequal MC return",
            "perturbation": "RankQ normalized-action protocol, shared epsilon, sigma mild=0.02 strong=0.05, C12 action-pool bounds; cached identically for B8",
            "actor_candidates": f"{args.candidate_states_per_task} fixed states/task, G=4 from frozen pi0.5 base actor",
            "ca": "production group-relative 10-head sign-consensus rule",
            "ca_chipo": "diagnostic at initialization ratio=1; not a live actor-policy likelihood-ratio evaluation",
            "macro": "arithmetic mean of the ten per-task metrics",
        },
        "checkpoints": {
            **{f"C_balanced_{step // 1000}k": str(path.resolve()) for step, path in c_paths.items()},
            **{f"A_balanced_{step // 1000}k": str(path.resolve()) for step, path in a_paths.items()},
            **{label: str(path.resolve()) for label, _, path in explicit_specs},
            **({} if args.skip_b_reference else {"B_8k": str(b_path.resolve())}),
        },
        "ten_task": ten_task_rows,
        "click_bell_reference": bell_rows,
    })
    print(f"evaluation_complete={output}", flush=True)


if __name__ == "__main__":
    main()
