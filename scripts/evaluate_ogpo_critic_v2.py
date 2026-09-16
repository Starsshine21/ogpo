#!/usr/bin/env python3
"""Fast policy-candidate-aligned evaluation for OGPO critics."""

from __future__ import annotations

import argparse
import csv
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

from ogpo.conservative_advantage import group_relative_conservative_advantage
from ogpo.critic_raw10_evaluator import spearman_correlation
from ogpo.replay import load_replay
from ogpo.trainer import build_train_state, load_critic_checkpoint
from ogpo.types import ChunkBatch
from ogpo.value_critic_protocol import StateFeatures
from ogpo.zarr_replay import concat_chunk_batches
from evaluate_multitask_ogpo_critic_readiness import TASKS, build_task_batch, heldout_episodes
from train_udivl_critic import load_config


EPSILON = 1.0e-8


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q).item()) if values.numel() else float("nan")


def candidate_ranking_metrics(raw_q: torch.Tensor) -> dict[str, float]:
    """Return ranking metrics for Q shaped [state, candidate, head]."""
    if raw_q.ndim != 3:
        raise ValueError(f"expected [state,candidate,head], got {tuple(raw_q.shape)}")
    states, candidates, heads = raw_q.shape
    if candidates != 4 or heads != 10:
        raise ValueError(f"expected G=4 and 10 heads, got {candidates=} {heads=}")

    choices = raw_q.argmax(dim=1)  # [state, head]
    top1 = []
    for state in range(states):
        top1.append(torch.bincount(choices[state], minlength=candidates).max().float() / heads)
    top1_agreement = torch.stack(top1)

    pair_consistency = []
    for left in range(candidates):
        for right in range(left + 1, candidates):
            delta = raw_q[:, left] - raw_q[:, right]
            positive = (delta > 0).float().mean(dim=1)
            negative = (delta < 0).float().mean(dim=1)
            pair_consistency.append(torch.maximum(positive, negative))
    pairwise = torch.stack(pair_consistency, dim=1).mean(dim=1)

    ensemble_mean = raw_q.mean(dim=2)
    top2 = ensemble_mean.topk(k=2, dim=1).indices
    top_index = top2[:, 0, None, None].expand(-1, 1, heads)
    second_index = top2[:, 1, None, None].expand(-1, 1, heads)
    top_values = raw_q.gather(1, top_index).squeeze(1)
    second_values = raw_q.gather(1, second_index).squeeze(1)
    delta = top_values - second_values
    normalized = delta.mean(dim=1).abs() / (delta.std(dim=1, unbiased=False) + EPSILON)

    candidate_disagreement = raw_q.std(dim=2, unbiased=False)
    return {
        "top1_agreement": float(top1_agreement.mean()),
        "pairwise_ranking_consistency": float(pairwise.mean()),
        "normalized_ranking_margin": float(normalized.mean()),
        "normalized_ranking_margin_median": percentile(normalized, 0.5),
        "normalized_ranking_margin_p10": percentile(normalized, 0.1),
        "normalized_ranking_margin_p90": percentile(normalized, 0.9),
        "candidate_disagreement": float(candidate_disagreement.mean()),
    }


def advantage_metrics(raw_q: torch.Tensor) -> dict[str, float]:
    q_heads_first = raw_q.permute(2, 0, 1).contiguous()
    ca, _, _ = group_relative_conservative_advantage(q_heads_first)
    nonzero = ca != 0
    magnitudes = ca[nonzero].abs()
    return {
        "ca_nonzero_fraction": float(nonzero.float().mean()),
        "median_abs_advantage": percentile(magnitudes, 0.5),
        "mean_abs_advantage": float(ca.abs().mean()),
    }


def value_and_ensemble_metrics(
    logged_q: torch.Tensor,
    targets: torch.Tensor,
    terminal_q: torch.Tensor,
) -> dict[str, float]:
    """Metrics for logged actions; Q inputs are shaped [head, sample]."""
    mean_q = logged_q.float().mean(dim=0)
    targets = targets.float()
    error = mean_q - targets
    centered = logged_q.float() - logged_q.float().mean(dim=1, keepdim=True)
    norms = centered.square().sum(dim=1).sqrt().clamp_min(1.0e-12)
    correlation = (centered / norms[:, None]) @ (centered / norms[:, None]).T
    off_diagonal = correlation[~torch.eye(logged_q.shape[0], dtype=torch.bool)]
    terminal_mean = terminal_q.float().mean(dim=0)
    return {
        "mc_spearman": spearman_correlation(mean_q, targets),
        "rmse": float(error.square().mean().sqrt()),
        "terminal_q_mean": float(terminal_mean.mean()),
        "terminal_error": float((terminal_mean - 1.0).abs().mean()),
        "q_head_correlation": float(off_diagonal.mean()),
        "ensemble_std": float(logged_q.float().std(dim=0, unbiased=False).mean()),
    }


@torch.no_grad()
def raw_q_for_actions(
    critic: torch.nn.Module,
    batch: ChunkBatch,
    actions: torch.Tensor,
    *,
    use_execution_mask: bool,
    inference_batch_size: int,
) -> torch.Tensor:
    """Return [head,state,candidate] for actions [state,candidate,horizon,dim]."""
    outputs = []
    device = next(critic.parameters()).device
    group = actions.shape[1]
    for start in range(0, batch.batch_size, inference_batch_size):
        stop = min(start + inference_batch_size, batch.batch_size)
        sample = batch.index_select(torch.arange(start, stop)).to(device)
        features = critic.encode_state(sample)
        grouped = StateFeatures(readout=features.readout.repeat_interleave(group, dim=0))
        mask = sample.execution_masks if use_execution_mask else torch.ones_like(sample.execution_masks)
        masks = mask[:, None].expand(-1, group, -1).reshape((stop - start) * group, -1)
        chunk = actions[start:stop].to(device).reshape(
            (stop - start) * group, sample.generated_horizon, sample.action_dim
        )
        raw = critic.raw_q_ensemble_from_features(grouped, chunk, masks)
        outputs.append(raw.reshape(10, stop - start, group).cpu())
    return torch.cat(outputs, dim=1)


def parse_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split("::")
    if len(parts) != 3 or not all(parts):
        raise ValueError("--critic-spec must be LABEL::CHECKPOINT::CONFIG")
    label, checkpoint, config = parts
    checkpoint_path, config_path = Path(checkpoint), Path(config)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return label, checkpoint_path, config_path


def select_candidate_cache(path: Path, states_per_task: int, seed: int) -> tuple[ChunkBatch, torch.Tensor, list[dict[str, Any]]]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if int(cache["seed"]) != seed or int(cache["group_size"]) != 4:
        raise ValueError("candidate cache seed/group mismatch")
    source_count = int(cache["count_per_task"])
    if states_per_task > source_count:
        raise ValueError(f"requested {states_per_task} states from {source_count}-state cache")
    indices = []
    manifest = []
    for task_index, task in enumerate(TASKS):
        for local_index in range(states_per_task):
            source_index = task_index * source_count + local_index
            indices.append(source_index)
            source = cache["manifest"][source_index]
            for candidate_index in range(4):
                manifest.append({
                    "global_state_index": task_index * states_per_task + local_index,
                    "task": task,
                    "episode_id": int(source["episode_id"]),
                    "state_index": int(source["timestep"]),
                    "source_transition_index": int(source["source_transition_index"]),
                    "random_seed": seed,
                    "candidate_index": candidate_index,
                })
    index = torch.tensor(indices, dtype=torch.long)
    batch = ChunkBatch(**cache["batch"]).index_select(index)
    actions = cache["candidate_actions"].index_select(0, index)
    return batch, actions, manifest


def terminal_batch(config: dict[str, Any]) -> ChunkBatch:
    episodes = heldout_episodes()
    parts = []
    for task in TASKS:
        batch = build_task_batch(task, episodes, config)
        terminal = batch.successes.bool() & batch.dones.bool() & torch.isclose(
            batch.mc_returns.float(), torch.ones_like(batch.mc_returns.float()), atol=1.0e-5
        )
        indices = torch.nonzero(terminal, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError(f"no success terminal samples for {task}")
        parts.append(batch.index_select(indices))
    return concat_chunk_batches(parts)


def macro(rows: list[dict[str, Any]], label: str, checkpoint: Path, elapsed: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint": label,
        "checkpoint_path": str(checkpoint.resolve()),
        "task_count": len(rows),
        "states_per_task": int(rows[0]["state_count"]),
        "evaluation_seconds": elapsed,
    }
    skip = {"checkpoint", "task", "state_count", "terminal_count"}
    for key in rows[0]:
        if key in skip:
            continue
        result[key] = float(np.mean([float(row[key]) for row in rows]))
    result["terminal_count"] = sum(int(row["terminal_count"]) for row in rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--critic-spec", action="append", required=True)
    parser.add_argument("--states-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=20)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--terminal-replay", type=Path)
    parser.add_argument("--logged-replay", type=Path)
    args = parser.parse_args()
    global TASKS
    if args.task_manifest:
        TASKS = tuple(json.loads(args.task_manifest.read_text())["tasks"])

    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    candidate_cache = args.candidate_cache if args.candidate_cache.is_absolute() else ROOT / args.candidate_cache
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    specs = [parse_spec(value) for value in args.critic_spec]
    for _, checkpoint, config in specs:
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(checkpoint if not checkpoint.is_file() else config)

    selected, candidate_actions, manifest = select_candidate_cache(
        candidate_cache, args.states_per_task, args.seed
    )
    atomic_json(output / "candidate_manifest.json", {
        "schema_version": 2,
        "candidate_generation_seed": args.seed,
        "states_per_task": args.states_per_task,
        "num_states": selected.batch_size,
        "num_candidates": 4,
        "num_heads": 10,
        "source_candidate_cache": str(candidate_cache.resolve()),
        "records": manifest,
    })

    reference_config = load_config(specs[0][2])
    terminals = load_replay(args.terminal_replay) if args.terminal_replay else terminal_batch(reference_config)
    logged = load_replay(args.logged_replay) if args.logged_replay else selected
    per_task_rows = []
    summary_rows = []
    for label, checkpoint, config_path in specs:
        started = time.monotonic()
        config = load_config(config_path)
        init = load_replay(ROOT / config["data"]["distributed_model_init_path"])
        state = build_train_state(config, init, device=args.device)
        load_critic_checkpoint(checkpoint, state, load_optimizer=False)
        critic = state.critic.eval()
        use_mask = bool(config.get("data", {}).get("use_execution_mask", True))
        candidate_heads_first = raw_q_for_actions(
            critic, selected, candidate_actions,
            use_execution_mask=use_mask, inference_batch_size=args.inference_batch_size,
        )
        logged_heads_first = raw_q_for_actions(
            critic, logged, logged.action_chunks[:, None],
            use_execution_mask=use_mask, inference_batch_size=args.inference_batch_size,
        ).squeeze(2)
        # Alignment must compare logged/candidate actions at exactly the SAME states.
        alignment_heads_first = logged_heads_first if logged is selected else raw_q_for_actions(
            critic, selected, selected.action_chunks[:, None],
            use_execution_mask=use_mask, inference_batch_size=args.inference_batch_size,
        ).squeeze(2)
        terminal_heads_first = raw_q_for_actions(
            critic, terminals, terminals.action_chunks[:, None],
            use_execution_mask=use_mask, inference_batch_size=args.inference_batch_size,
        ).squeeze(2)
        raw_q = candidate_heads_first.permute(1, 2, 0).contiguous()
        checkpoint_dir = output / label
        checkpoint_dir.mkdir()
        np.save(checkpoint_dir / "candidate_raw_q_values.npy", raw_q.numpy())
        from carl_metrics import logged_advantages, policy_alignment
        alignment_q = alignment_heads_first.T.contiguous().numpy()
        outcome_labels = selected.successes.bool().cpu().numpy().reshape(-1)
        np.save(checkpoint_dir / "logged_raw_q_values.npy", alignment_q)
        np.save(checkpoint_dir / "source_episode_success.npy", outcome_labels)
        np.save(checkpoint_dir / "logged_advantages.npy", logged_advantages(raw_q.numpy(), alignment_q))

        rows = []
        for task in TASKS:
            value_indices = torch.tensor([i for i, item in enumerate(selected.task_ids) if item == task])
            logged_indices = torch.tensor([i for i, item in enumerate(logged.task_ids) if item == task])
            terminal_indices = torch.tensor([i for i, item in enumerate(terminals.task_ids) if item == task])
            task_raw_q = raw_q.index_select(0, value_indices)
            row = {
                "checkpoint": label,
                "task": task,
                "state_count": int(value_indices.numel()),
                "terminal_count": int(terminal_indices.numel()),
                **value_and_ensemble_metrics(
                    logged_heads_first.index_select(1, logged_indices),
                    logged.mc_returns.index_select(0, logged_indices),
                    terminal_heads_first.index_select(1, terminal_indices),
                ),
                **candidate_ranking_metrics(task_raw_q),
                **advantage_metrics(task_raw_q),
                **policy_alignment(task_raw_q.numpy(), alignment_q[value_indices.numpy()], outcome_labels[value_indices.numpy()]),
            }
            rows.append(row)
            per_task_rows.append(row)
        elapsed = time.monotonic() - started
        summary_rows.append(macro(rows, label, checkpoint, elapsed))
        print(f"checkpoint_complete={label} seconds={elapsed:.1f}", flush=True)
        del state, critic, candidate_heads_first, logged_heads_first, alignment_heads_first, terminal_heads_first, raw_q
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(output / "checkpoint_summary.csv", summary_rows)
    write_csv(output / "per_task_metrics.csv", per_task_rows)
    atomic_json(output / "RESULTS.json", {
        "schema_version": 2,
        "protocol": {
            "tasks": list(TASKS),
            "states_per_task": args.states_per_task,
            "state_count": len(TASKS) * args.states_per_task,
            "candidate_group_size": 4,
            "raw_q_heads": 10,
            "candidate_source": "fixed frozen PI0.5 base-actor candidate cache",
            "value_domain": str(args.logged_replay) if args.logged_replay else "logged actions on the selected heldout candidate states",
            "terminal_domain": str(args.terminal_replay) if args.terminal_replay else "all success terminal states in the existing MT1000 actual heldout split",
            "macro": "equal arithmetic mean over ten per-task metrics",
            "median_abs_advantage": "median over nonzero production conservative-advantage entries",
            "q_head_correlation_and_ensemble_std": "logged actions; candidate disagreement is separately computed on actor candidates",
        },
        "candidate_manifest": str((output / "candidate_manifest.json").resolve()),
        "checkpoints": summary_rows,
        "per_task": per_task_rows,
    })
    print(f"evaluation_complete={output}", flush=True)


if __name__ == "__main__":
    main()
