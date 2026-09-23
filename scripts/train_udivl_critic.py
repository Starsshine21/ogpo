#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ogpo.metrics import add_run_metadata, create_metrics_writer
from ogpo.evaluator import validation_metrics_for_training
from ogpo.replay import (
    BalancedCriticReplay,
    CompositeChunkBatch,
    OfflineChunkReplay,
    OutcomeBalancedCriticReplay,
    TaskBalancedCriticReplay,
    TaskOutcomeBalancedCriticReplay,
    load_replay,
    make_synthetic_replay,
    prepare_replay_from_config,
    save_replay,
)
from ogpo.origin_cache import load_or_build_origin_feature_cache
from ogpo.training_control import TrainableSnapshot, ValidationEarlyStopper
from ogpo.critic import soft_update
from ogpo.rankq import same_state_rankq_settings
from ogpo.trainer import (
    accumulated_critic_update,
    apply_scheduled_critic_stage,
    build_train_state,
    critic_update,
    initialize_critic_from_checkpoint,
    load_checkpoint,
    maybe_advance_critic_stage,
    save_checkpoint,
)


def _deep_update(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    include = cfg.pop("include", None)
    if include:
        cfg = _deep_update(load_config(ROOT / include), cfg)
    return cfg


def critic_smoke_output_root(config: dict) -> Path:
    """Derive an isolated critic smoke directory from resolved outputs."""
    checkpoint_path = config.get("training", {}).get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("critic smoke mode requires training.checkpoint_path")
    checkpoint = Path(checkpoint_path)
    run_name = checkpoint.parent.name or checkpoint.stem.removesuffix("_final")
    if not run_name:
        raise ValueError("could not derive critic smoke run name")
    return Path("outputs/ogpo/smoke") / run_name


def _episode_shard(batch, *, rank: int, world_size: int):
    episode_ids = torch.unique(batch.episode_ids, sorted=True)
    selected = episode_ids[rank::world_size]
    if selected.numel() == 0:
        raise ValueError(f"distributed rank {rank} received no replay episodes")
    mask = torch.isin(batch.episode_ids, selected)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    return batch.index_select(indices)


def _distributed_mean_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist(), strict=True)}


def _safe_metric_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def distributed_sample_metrics(sample, config: dict, device: torch.device) -> dict[str, float]:
    """Report the actual global sampled task/outcome composition each step."""
    sampling_cfg = config.get("training", {}).get("critic_sampling", {})
    task_names = [str(value) for value in sampling_cfg.get("task_names", [])]
    if not task_names:
        task_names = sorted(set(str(value) for value in sample.task_ids))
    task_to_index = {task: index for index, task in enumerate(task_names)}
    counts = torch.zeros(len(task_names), dtype=torch.float64, device=device)
    for task in sample.task_ids:
        value = str(task)
        if value not in task_to_index:
            raise ValueError(f"sampled unexpected task_id={value!r}")
        counts[task_to_index[value]] += 1.0
    outcome = torch.tensor(
        [
            float(sample.successes.float().sum().item()),
            float(sample.batch_size),
            float(sample.dones.float().sum().item()),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(outcome, op=dist.ReduceOp.SUM)
    total = float(outcome[1].item())
    metrics = {
        "sample_success_fraction": float(outcome[0].item() / total),
        "sample_failure_fraction": float(1.0 - outcome[0].item() / total),
        "sample_done_fraction": float(outcome[2].item() / total),
        "sampled_task_active_count": float((counts > 0).sum().item()),
    }
    for task, count in zip(task_names, counts.tolist(), strict=True):
        key = _safe_metric_name(task)
        metrics[f"sampled_task_count/{key}"] = float(count)
        metrics[f"sampled_task_fraction/{key}"] = float(count / total)
    if bool(sampling_cfg.get("require_multiple_sampled_tasks", False)) and int(
        (counts > 0).sum().item()
    ) < 2:
        raise AssertionError("multi-task smoke batch sampled fewer than two tasks")
    return metrics


def critic_parameter_summary(critic) -> dict:
    groups = {
        "pretrained_vision": [],
        "pretrained_gemma": [],
        "critic_encoder_projection": [],
        "critic_core": [],
    }
    total = trainable = 0
    trainable_names = []
    frozen_names = []
    for name, parameter in critic.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
        else:
            frozen_names.append(name)
        if name.startswith("state_encoder.vision_model."):
            group = "pretrained_vision"
        elif name.startswith("state_encoder.gemma_model."):
            group = "pretrained_gemma"
        elif name.startswith("state_encoder."):
            group = "critic_encoder_projection"
        else:
            group = "critic_core"
        groups[group].append(
            {"name": name, "parameters": count, "trainable": bool(parameter.requires_grad)}
        )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100.0 * trainable / total,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "groups": groups,
    }


def assert_head_only_trainability_and_gradients(state, *, require_gradients: bool) -> None:
    pretrained = []
    critic_specific = []
    for name, parameter in state.critic.named_parameters():
        if name.startswith(("state_encoder.vision_model.", "state_encoder.gemma_model.")):
            pretrained.append((name, parameter))
        else:
            critic_specific.append((name, parameter))
    if not pretrained or not critic_specific:
        raise AssertionError("head-only audit could not resolve parameter groups")
    if any(parameter.requires_grad or parameter.grad is not None for _, parameter in pretrained):
        raise AssertionError("head-only pretrained backbone is trainable or received gradients")
    if not all(parameter.requires_grad for _, parameter in critic_specific):
        frozen = [name for name, parameter in critic_specific if not parameter.requires_grad]
        raise AssertionError(f"head-only critic-specific parameters unexpectedly frozen: {frozen}")
    if require_gradients:
        core_norm = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in critic_specific
            if name.startswith("core.") and parameter.grad is not None
        )
        projection_norm = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in critic_specific
            if name.startswith("state_encoder.") and parameter.grad is not None
        )
        if core_norm <= 0.0 or projection_norm <= 0.0:
            raise AssertionError(
                "head-only critic-specific core/projection gradients must both be nonzero"
            )


def advance_replay_generator_for_exact_resume(
    generator: torch.Generator,
    *,
    completed_steps: int,
    train_replay_size: int,
    train_batch_size: int,
    is_main: bool,
    validation_replay_size: int,
    validation_batch_size: int,
    evaluation_interval: int,
    validation_consumes_generator: bool,
) -> dict[str, int]:
    """Reconstruct the uninterrupted replay RNG position without loading samples."""
    completed_steps = int(completed_steps)
    if completed_steps < 0:
        raise ValueError("completed replay-resume steps must be non-negative")
    train_draws = validation_draws = 0
    for prior_step in range(completed_steps):
        torch.randint(
            int(train_replay_size),
            (int(train_batch_size),),
            generator=generator,
        )
        train_draws += int(train_batch_size)
        evaluation_due = prior_step == 0 or (prior_step + 1) % int(evaluation_interval) == 0
        if is_main and validation_consumes_generator and evaluation_due:
            torch.randint(
                int(validation_replay_size),
                (int(validation_batch_size),),
                generator=generator,
            )
            validation_draws += int(validation_batch_size)
    return {"train_draws": train_draws, "validation_draws": validation_draws}


@torch.no_grad()
def _set_distributed_action_statistics(state, batch) -> None:
    """Restore exact global replay action statistics after schema-first model load."""
    if not dist.is_initialized() or not hasattr(state.critic, "core"):
        return
    device = next(state.critic.parameters()).device
    actions = batch.action_chunks[batch.execution_masks.bool()].to(device=device, dtype=torch.float64)
    count = torch.tensor(float(actions.shape[0]), dtype=torch.float64, device=device)
    total = actions.sum(dim=0)
    squared = actions.square().sum(dim=0)
    minimum = actions.min(dim=0).values
    maximum = actions.max(dim=0).values
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    mean = total / count
    std = (squared / count - mean.square()).clamp_min(0.0).sqrt().clamp_min(1e-6)
    for critic in (state.critic, state.target_critic):
        pool = critic.core.action_pool
        pool.action_mean.copy_(mean.to(pool.action_mean))
        pool.action_std.copy_(std.to(pool.action_std))
        pool.action_min.copy_(minimum.to(pool.action_min))
        pool.action_max.copy_(maximum.to(pool.action_max))


def fixed_validation_batch(
    replay: OfflineChunkReplay,
    batch_size: int,
    *,
    seed: int,
    stratified: bool,
):
    generator = torch.Generator().manual_seed(seed)
    if not stratified or replay.batch.mc_returns is None:
        return replay.sample(batch_size, generator=generator)
    targets = replay.batch.mc_returns
    low = torch.nonzero(targets < targets.max(), as_tuple=False).flatten()
    high = torch.nonzero(targets == targets.max(), as_tuple=False).flatten()
    if low.numel() == 0 or high.numel() == 0:
        return replay.sample(batch_size, generator=generator)
    high_count = min(batch_size // 2, int(high.numel()))
    low_count = min(batch_size - high_count, int(low.numel()))
    indices = torch.cat(
        [
            high[torch.randperm(high.numel(), generator=generator)[:high_count]],
            low[torch.randperm(low.numel(), generator=generator)[:low_count]],
        ]
    )
    if indices.numel() < batch_size:
        extra = torch.randint(
            len(replay),
            (batch_size - indices.numel(),),
            generator=generator,
        )
        indices = torch.cat([indices, extra])
    return replay.batch.index_select(indices)


def critic_selection_score(metrics: dict[str, float], config: dict) -> tuple[float, bool]:
    selection = config.get("evaluation", {}).get("checkpoint_selection", {})
    ranking = float(metrics["validation_pairwise_ranking_accuracy"])
    correlation = float(metrics["validation_q_rank_correlation"])
    rmse = float(metrics["validation_q_rmse"])
    gap = abs(float(metrics["validation_q_exploitation_gap"]))
    eligible = (
        ranking >= float(selection.get("min_pairwise_ranking_accuracy", 0.0))
        and correlation >= float(selection.get("min_q_rank_correlation", -1.0))
        and gap <= float(selection.get("max_abs_q_exploitation_gap", float("inf")))
    )
    score = (
        ranking
        + float(selection.get("rank_correlation_weight", 0.25)) * correlation
        - float(selection.get("rmse_weight", 0.5)) * rmse
        - float(selection.get("exploitation_gap_weight", 0.5)) * gap
    )
    return score, eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/critic_udivl.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--critic-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Explicitly create a synthetic replay when the configured dataset is missing.",
    )
    args = parser.parse_args()
    if args.synthetic_smoke and not args.smoke:
        raise ValueError("--synthetic-smoke requires --smoke")
    cfg = load_config(ROOT / args.config)
    for overlay in args.overlay:
        cfg = _deep_update(cfg, load_config(ROOT / overlay))
    if args.critic_steps is not None:
        if args.critic_steps <= 0:
            raise ValueError("--critic-steps must be positive")
        cfg.setdefault("training", {})["critic_steps"] = args.critic_steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    expected_world_size = int(
        cfg.get("training", {}).get("expected_world_size", world_size)
    )
    if world_size != expected_world_size:
        raise ValueError(
            "critic distributed world-size mismatch: "
            f"expected {expected_world_size}, got {world_size}"
        )
    if distributed:
        torch.cuda.set_device(local_rank)
        free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
        print(
            f"[ddp] rank={rank} local_rank={local_rank} visible={torch.cuda.device_count()} "
            f"free_gib={free_bytes / 2**30:.3f} total_gib={total_bytes / 2**30:.3f}",
            flush=True,
        )
        dist.init_process_group(backend="nccl")
        # NCCL allocates communicator buffers lazily. Initialize them before
        # model/activation memory fills the device so the first gradient
        # all-reduce cannot fail solely from communicator setup.
        nccl_warmup = torch.ones(1, device=f"cuda:{local_rank}")
        dist.all_reduce(nccl_warmup, op=dist.ReduceOp.SUM)
        dist.barrier()
        del nccl_warmup
        torch.cuda.empty_cache()
        cfg.setdefault("training", {})["device"] = f"cuda:{local_rank}"
        if bool(cfg["training"].get("early_stopping", {}).get("enabled", False)):
            raise ValueError("distributed critic training requires early_stopping.enabled=false")
        if bool(cfg.get("data", {}).get("origin_feature_cache", {}).get("enabled", False)):
            raise ValueError("distributed critic training does not support origin_feature_cache")
    is_main = rank == 0
    if args.smoke:
        smoke_root = critic_smoke_output_root(cfg)
        cfg["training"].update(
            {
                "checkpoint_path": str(smoke_root / "critic_smoke.pt"),
                "latest_checkpoint_path": None,
                "best_checkpoint_path": None,
                "milestone_steps": [],
                "checkpoint_interval": 0,
                "metrics_path": str(smoke_root / "metrics.jsonl"),
                "tensorboard_dir": str(smoke_root / "tensorboard"),
                "config_snapshot_path": str(smoke_root / "resolved_config.yaml"),
                "parameter_summary_path": str(smoke_root / "parameter_summary.json"),
            }
        )
    snapshot_path = cfg.get("training", {}).get("config_snapshot_path")
    if snapshot_path and is_main:
        resolved_path = ROOT / snapshot_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
    model_seed = int(cfg.get("training", {}).get("seed", 0))
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    state = None
    if distributed:
        # Load the VLM before the larger rank replay. Transformers mmaps its
        # local safetensors during construction; doing this first stays under
        # the cluster's per-process virtual-memory limit.
        init_path = ROOT / cfg["data"].get(
            "distributed_model_init_path",
            cfg["data"]["validation_path"],
        )
        init_batch = load_replay(init_path)
        state = build_train_state(
            cfg,
            init_batch,
            device=cfg["training"].get("device", "cpu"),
        )
        del init_batch
        gc.collect()
    distributed_paths = cfg.get("data", {}).get("distributed_dataset_paths") if distributed else None
    distributed_pattern = cfg.get("data", {}).get("distributed_dataset_path_pattern")
    if distributed_paths and distributed_pattern:
        raise ValueError("choose either distributed_dataset_paths or distributed_dataset_path_pattern")
    data_path = ROOT / (
        str(distributed_pattern).format(rank=rank)
        if distributed and distributed_pattern
        else cfg["data"]["dataset_path"]
    )
    if not distributed_paths and not data_path.exists():
        if not args.synthetic_smoke:
            raise FileNotFoundError(
                f"configured critic replay does not exist: {data_path}; "
                "synthetic data is allowed only with --smoke --synthetic-smoke"
            )
        if is_main:
            synthetic = make_synthetic_replay()
            save_replay(synthetic, data_path)
            print(f"[critic] explicitly created synthetic smoke dataset at {data_path}")
        if distributed:
            dist.barrier()
    if distributed_paths:
        sources = []
        next_episode_id = 0
        for pattern in distributed_paths:
            source_path = ROOT / str(pattern).format(rank=rank)
            if not source_path.is_file():
                raise FileNotFoundError(f"configured critic replay does not exist: {source_path}")
            source = load_replay(source_path, mmap=True)
            source = replace(source, episode_ids=source.episode_ids + next_episode_id)
            next_episode_id = int(source.episode_ids.max()) + 1
            sources.append(source)
        batch = CompositeChunkBatch(sources)
        print(f"[critic] file-backed replay sources={[str(p).format(rank=rank) for p in distributed_paths]} "
              f"local_transitions={batch.batch_size}", flush=True)
    else:
        # Historical single-file runs keep their original loading semantics.
        batch = load_replay(data_path)
    if distributed and not distributed_pattern and not distributed_paths:
        batch = _episode_shard(batch, rank=rank, world_size=world_size)
    validation_path_value = cfg.get("data", {}).get(
        "distributed_validation_path" if distributed else "validation_path",
        cfg.get("data", {}).get("validation_path"),
    )
    if validation_path_value:
        validation_path = ROOT / validation_path_value
        if not validation_path.exists():
            raise FileNotFoundError(f"configured validation replay does not exist: {validation_path}")
        validation_batch = load_replay(validation_path, mmap=bool(distributed_paths)) if is_main else batch
    else:
        validation_batch = batch
    data_cfg = cfg.get("data", {})
    gamma_rebase_enabled = bool(data_cfg.get("gamma_rebase", {}).get("enabled", False))
    n_step = int(data_cfg.get("n_step", 1))
    if gamma_rebase_enabled or (
        bool(data_cfg.get("apply_n_step_on_load", False)) and n_step > 1
    ):
        distributed_preprocessed = distributed and bool(
            data_cfg.get("distributed_dataset_preprocessed", False)
        )
        if not distributed_preprocessed:
            batch = prepare_replay_from_config(batch, data_cfg)
        if is_main and not distributed_preprocessed:
            validation_batch = prepare_replay_from_config(validation_batch, data_cfg)
        if is_main:
            print(
            f"[critic] prepared replay gamma={float(data_cfg.get('gamma', 1.0))} "
            f"gamma_rebase={gamma_rebase_enabled} n_step={n_step} "
            f"local_train={batch.batch_size} validation={validation_batch.batch_size} "
            f"world_size={world_size}",
            flush=True,
            )
    sampling_cfg = cfg.get("training", {}).get("critic_sampling", {})
    sampling_mode = str(sampling_cfg.get("mode", "")).lower()
    if sampling_mode == "shared_episode_bootstrap":
        from ogpo.episode_bootstrap import SharedEpisodeBootstrapReplay
        replay = SharedEpisodeBootstrapReplay(batch,
            masks=json.loads((ROOT / sampling_cfg["mask_path"]).read_text()),
            task_names=sampling_cfg["task_names"],
            success_probability=float(sampling_cfg.get("success_probability", .5)))
        print('[critic] shared batch with fixed episode loss masks; global batch32 TOTAL, no 5x replication',flush=True)
    elif sampling_mode == "member_episode_bootstrap":
        from ogpo.episode_bootstrap import MemberEpisodeBootstrapReplay
        masks = json.loads((ROOT / sampling_cfg["mask_path"]).read_text())
        replay = MemberEpisodeBootstrapReplay(
            batch, masks=masks, task_names=sampling_cfg["task_names"],
            success_probability=float(sampling_cfg.get("success_probability", 0.5)),
        )
        print(f"[critic] persistent bootstrap local episode coverage={replay.coverage}; batch_size is per member; total samples=5*batch_size", flush=True)
    elif sampling_mode == "task_balanced":
        replay = TaskBalancedCriticReplay(
            batch,
            task_names=tuple(str(value) for value in sampling_cfg.get("task_names", [])),
        )
        print(
            f"[critic] task-balanced sampling tasks={list(replay.task_names)}",
            flush=True,
        )
    elif sampling_mode == "task_outcome_balanced":
        replay = TaskOutcomeBalancedCriticReplay(
            batch,
            task_names=tuple(str(value) for value in sampling_cfg.get("task_names", [])),
            success_probability=float(sampling_cfg.get("success_probability", 0.5)),
        )
        pool_sizes = {
            task: {
                "success": int(replay._indices[task][True].numel()),
                "failure": int(replay._indices[task][False].numel()),
            }
            for task in replay.task_names
        }
        print(
            "[critic] task-outcome-balanced sampling "
            f"tasks={list(replay.task_names)} "
            f"success_probability={replay.success_probability:.6f} "
            f"local_pool_sizes={pool_sizes}",
            flush=True,
        )
    elif sampling_mode == "outcome_balanced":
        replay = OutcomeBalancedCriticReplay(
            batch,
            success_probability=float(sampling_cfg.get("success_probability", 0.5)),
        )
        print(
            "[critic] outcome-balanced sampling "
            f"success_probability={replay.success_probability:.6f} "
            f"success_pool={replay._success.numel()} failure_pool={replay._failure.numel()}",
            flush=True,
        )
    elif sampling_mode not in {"", "uniform", "legacy_balanced"}:
        raise ValueError(f"unsupported training.critic_sampling.mode={sampling_mode!r}")
    elif bool(sampling_cfg.get("enabled", False)):
        replay = BalancedCriticReplay(
            batch,
            uniform_fraction=float(sampling_cfg.get("uniform_fraction", 0.5)),
            success_fraction=float(sampling_cfg.get("success_fraction", 0.25)),
            terminal_success_fraction=float(
                sampling_cfg.get("terminal_success_fraction", 0.125)
            ),
            failure_fraction=float(sampling_cfg.get("failure_fraction", 0.125)),
        )
        print(f"[critic] balanced sampling={replay.fractions}", flush=True)
    else:
        replay = OfflineChunkReplay(batch)
    validation_replay = OfflineChunkReplay(validation_batch)
    if state is None:
        state = build_train_state(cfg, batch, device=cfg["training"].get("device", "cpu"))
    else:
        _set_distributed_action_statistics(state, batch)
    if str(cfg.get("critic", {}).get("stage", "head_td")) == "head_td":
        assert_head_only_trainability_and_gradients(state, require_gradients=False)
    if is_main:
        parameter_summary = critic_parameter_summary(state.critic)
        parameter_summary_path = cfg.get("training", {}).get("parameter_summary_path")
        if parameter_summary_path:
            destination = ROOT / parameter_summary_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(
                json.dumps(parameter_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        else:
            destination = None
        compact_groups = {
            name: {
                "total": sum(int(row["parameters"]) for row in rows),
                "trainable": sum(
                    int(row["parameters"]) for row in rows if bool(row["trainable"])
                ),
                "tensor_count": len(rows),
            }
            for name, rows in parameter_summary["groups"].items()
        }
        print(
            "[critic-parameters] "
            f"total={parameter_summary['total_parameters']} "
            f"trainable={parameter_summary['trainable_parameters']} "
            f"trainable_percent={parameter_summary['trainable_percent']:.8f} "
            f"groups={json.dumps(compact_groups, sort_keys=True)} "
            f"full_names_path={destination}",
            flush=True,
        )
    cache_cfg = cfg.get("data", {}).get("origin_feature_cache", {})
    if bool(cache_cfg.get("enabled", False)):
        batch = load_or_build_origin_feature_cache(
            state,
            batch,
            ROOT / cache_cfg["train_path"],
            inference_batch_size=int(cache_cfg.get("batch_size", 8)),
        )
        validation_batch = load_or_build_origin_feature_cache(
            state,
            validation_batch,
            ROOT / cache_cfg["validation_path"],
            inference_batch_size=int(cache_cfg.get("batch_size", 8)),
        )
        replay = OfflineChunkReplay(batch)
        validation_replay = OfflineChunkReplay(validation_batch)
    resume = args.resume or cfg["training"].get("resume_checkpoint")
    if resume:
        resume_path = ROOT / resume
        resume_payload = load_checkpoint(resume_path, state)
        expected_resume_step = cfg["training"].get("expected_resume_critic_step")
        if expected_resume_step is not None and state.step != int(expected_resume_step):
            raise ValueError(
                "critic resume step mismatch: "
                f"expected={int(expected_resume_step)} loaded={state.step}"
            )
        optimizer_payload = resume_payload.get("critic_optimizer", {})
        optimizer_state_count = len(optimizer_payload.get("state", {}))
        optimizer_steps = [
            int(value["step"].item())
            for value in optimizer_payload.get("state", {}).values()
            if isinstance(value, dict) and isinstance(value.get("step"), torch.Tensor)
        ]
        optimizer_step_min = min(optimizer_steps) if optimizer_steps else -1
        optimizer_step_max = max(optimizer_steps) if optimizer_steps else -1
        optimizer_restored = (
            optimizer_state_count > 0
            and len(state.critic_optimizer.state) == optimizer_state_count
        )
        target_payload = resume_payload.get("target_multimodal_critic")
        target_restored = (
            isinstance(target_payload, dict)
            and len(target_payload) == len(state.target_critic.state_dict())
        )
        source_critic_config = resume_payload.get("config", {}).get("critic", {})
        resume_match_fields = cfg["training"].get("resume_critic_match_fields", [])
        mismatched_resume_fields = [
            field
            for field in resume_match_fields
            if source_critic_config.get(field) != cfg["critic"].get(field)
        ]
        if mismatched_resume_fields:
            raise ValueError(
                "resume checkpoint critic metadata mismatch: "
                + ", ".join(str(field) for field in mismatched_resume_fields)
            )
        if bool(cfg["training"].get("require_critic_optimizer_resume", False)) and not optimizer_restored:
            raise RuntimeError("critic optimizer state was not restored from resume checkpoint")
        if (
            expected_resume_step is not None
            and bool(cfg["training"].get("require_critic_optimizer_resume", False))
            and (
                optimizer_step_min != int(expected_resume_step)
                or optimizer_step_max != int(expected_resume_step)
            )
        ):
            raise RuntimeError(
                "critic optimizer step mismatch after resume: "
                f"min={optimizer_step_min} max={optimizer_step_max} "
                f"expected={int(expected_resume_step)}"
            )
        if bool(cfg["training"].get("require_target_critic_resume", False)) and not target_restored:
            raise RuntimeError("target critic was not restored from resume checkpoint")
        if is_main:
            print(
                "[critic-resume] "
                f"loaded_checkpoint={resume_path.resolve()} "
                f"loaded_critic_step={state.step} "
                f"optimizer_restored={int(optimizer_restored)} "
                f"optimizer_state_count={optimizer_state_count} "
                f"optimizer_step_min={optimizer_step_min} "
                f"optimizer_step_max={optimizer_step_max} "
                f"target_critic_restored={int(target_restored)} "
                f"target_tensor_count={len(target_payload) if isinstance(target_payload, dict) else 0} "
                f"critic_metadata_match={int(not mismatched_resume_fields)}",
                flush=True,
            )
    else:
        initial_critic_checkpoint = cfg["training"].get("initial_critic_checkpoint")
        if initial_critic_checkpoint:
            initialize_critic_from_checkpoint(
                ROOT / initial_critic_checkpoint,
                state,
                affine_rebase_action_normalizer=bool(
                    cfg["training"].get(
                        "affine_rebase_action_normalizer_on_init",
                        False,
                    )
                ),
            )
            print(
                f"[critic] initialized model weights from: {initial_critic_checkpoint}",
                flush=True,
            )
    writer = (
        create_metrics_writer(
            ROOT / cfg["training"].get("metrics_path", "outputs/ogpo/critic_metrics.jsonl"),
            ROOT / cfg["training"]["tensorboard_dir"] if cfg["training"].get("tensorboard_dir") else None,
        )
        if is_main
        else None
    )
    generator = torch.Generator().manual_seed(11 + 100003 * rank)
    critic_steps = int(cfg["training"].get("critic_steps", cfg["critic"].get("warmup_steps", 5)))
    evaluation_interval = max(1, int(cfg.get("evaluation", {}).get("interval", 10)))
    if bool(cfg.get("evaluation", {}).get("full_validation", False)):
        validation_batch_size = len(validation_replay)
    else:
        validation_batch_size = min(
            len(validation_replay),
            int(cfg.get("evaluation", {}).get("validation_batch_size", cfg["training"].get("batch_size", 16))),
        )
    fixed_validation_sample = (
        fixed_validation_batch(
            validation_replay,
            validation_batch_size,
            seed=int(cfg["training"].get("seed", 0)) + 2903,
            stratified=bool(
                cfg.get("evaluation", {}).get("stratified_validation_batch", False)
            ),
        )
        if (
            bool(cfg.get("evaluation", {}).get("fixed_validation_batch", False))
            and not bool(cfg.get("evaluation", {}).get("full_validation", False))
        )
        else None
    )
    early_cfg = cfg.get("training", {}).get("early_stopping", {})
    early_stopper = (
        ValidationEarlyStopper(
            mode=str(early_cfg.get("mode", "min")),
            patience=int(early_cfg.get("patience", 20)),
            min_delta=float(early_cfg.get("min_delta", 0.0)),
        )
        if bool(early_cfg.get("enabled", False))
        else None
    )
    early_metric = str(early_cfg.get("metric", "validation_q_huber"))
    early_start_stage = early_cfg.get("start_stage")
    global_batch_size = int(cfg["training"].get("batch_size", 16))
    if global_batch_size % world_size:
        raise ValueError(
            f"training.batch_size={global_batch_size} must be divisible by world_size={world_size}"
        )
    effective_batch_size = global_batch_size // world_size
    microbatch_size = min(
        int(cfg["training"].get("microbatch_size", effective_batch_size)),
        effective_batch_size,
    )
    replay_resume_steps = int(
        cfg.get("training", {}).get("replay_generator_resume_step", 0)
    )
    if replay_resume_steps:
        if not isinstance(replay, OfflineChunkReplay):
            raise ValueError(
                "exact replay-generator resume currently requires OfflineChunkReplay"
            )
        replay_advance = advance_replay_generator_for_exact_resume(
            generator,
            completed_steps=replay_resume_steps,
            train_replay_size=len(replay),
            train_batch_size=effective_batch_size,
            is_main=is_main,
            validation_replay_size=len(validation_replay),
            validation_batch_size=validation_batch_size,
            evaluation_interval=evaluation_interval,
            validation_consumes_generator=(
                not bool(cfg.get("evaluation", {}).get("full_validation", False))
                and fixed_validation_sample is None
            ),
        )
        if is_main:
            print(
                "[critic-resume] replay_generator_advanced=1 "
                f"completed_steps={replay_resume_steps} "
                f"train_draws={replay_advance['train_draws']} "
                f"validation_draws={replay_advance['validation_draws']}",
                flush=True,
    )
    if is_main:
        rankq_cfg = cfg["critic"].get("rankq", {})
        resolved_rankq = same_state_rankq_settings(
            cfg["critic"], optimizer_step=state.step
        )
        nested_rankq_active = bool(resolved_rankq["enabled"])
        print(
            "[critic] optimization "
            f"max_grad_norm={float(cfg['critic'].get('max_grad_norm', 10.0)):.12g} "
            f"world_size={world_size} global_batch={global_batch_size} "
            f"local_batch={effective_batch_size} microbatch={microbatch_size} "
            f"rankq_enabled={int(nested_rankq_active)} "
            f"rankq_configured_enabled={int(bool(resolved_rankq['configured_enabled']))} "
            f"lambda_rank_effective={float(resolved_rankq['lambda_rank_effective']):.12g} "
            f"lambda_rank_max={float(resolved_rankq['lambda_rank_max']):.12g} "
            f"lambda_rank_schedule_enabled={int(bool(resolved_rankq['lambda_rank_schedule_enabled']))} "
            f"rankq_warmup_start_step={resolved_rankq['rankq_warmup_start_step']} "
            f"rankq_warmup_end_step={resolved_rankq['rankq_warmup_end_step']} "
            f"rankq_logged_mild_weight={float(rankq_cfg.get('logged_mild_weight', 1.0)) if isinstance(rankq_cfg, dict) else 1.0:.12g} "
            f"rankq_mild_strong_weight={float(rankq_cfg.get('mild_strong_weight', 1.0)) if isinstance(rankq_cfg, dict) else 1.0:.12g} "
            f"rankq_strong_random_weight={float(rankq_cfg.get('strong_random_weight', 1.0)) if isinstance(rankq_cfg, dict) else 1.0:.12g} "
            f"v_q_aggregation={cfg['critic'].get('v_q_aggregation', 'min')} "
            f"v_tau_mode={cfg['critic'].get('v_tau_mode', 'legacy')} "
            f"v_tau_min={float(cfg['critic'].get('v_tau_min', cfg.get('divl', {}).get('tau_min', 0.0))):.12g} "
            f"v_tau_max={float(cfg['critic'].get('v_tau_max', cfg.get('divl', {}).get('tau_max', 1.0))):.12g}",
            flush=True,
        )
    best_snapshot = None
    checkpoint_interval = int(cfg["training"].get("checkpoint_interval", 0))
    latest_checkpoint = cfg["training"].get("latest_checkpoint_path")
    best_checkpoint = cfg["training"].get("best_checkpoint_path")
    milestone_steps = {int(value) for value in cfg["training"].get("milestone_steps", [])}
    milestone_dir = cfg["training"].get("milestone_checkpoint_dir")
    selection_cfg = cfg.get("evaluation", {}).get("checkpoint_selection", {})
    selection_enabled = bool(selection_cfg.get("enabled", False))
    best_selection_score = float("-inf")
    for step in range(critic_steps):
        stage_advanced = apply_scheduled_critic_stage(state, cfg)
        if stage_advanced and early_stopper is not None:
            early_stopper = ValidationEarlyStopper(
                mode=early_stopper.mode,
                patience=early_stopper.patience,
                min_delta=early_stopper.min_delta,
            )
            best_snapshot = None
        sample = replay.sample(effective_batch_size, generator=generator)
        metrics = (
            accumulated_critic_update(
                state,
                sample,
                cfg,
                microbatch_size=microbatch_size,
            )
            if microbatch_size < sample.batch_size
            else critic_update(state, sample, cfg)
        )
        metrics = _distributed_mean_metrics(
            metrics,
            next(state.critic.parameters()).device,
        )
        sample_metrics = distributed_sample_metrics(
            sample,
            cfg,
            next(state.critic.parameters()).device,
        )
        if step == 0 and str(cfg.get("critic", {}).get("stage", "head_td")) == "head_td":
            assert_head_only_trainability_and_gradients(state, require_gradients=True)
            if is_main:
                print(
                    "[critic-parameters] head_only_gradient_check="
                    "pretrained_none critic_specific_nonzero",
                    flush=True,
                )
        if stage_advanced:
            metrics["critic_stage_advanced"] = 1.0
        evaluation_due = (
            state.step == 1
            or state.step % evaluation_interval == 0
            or step == critic_steps - 1
        )
        if evaluation_due and is_main:
            validation_sample = (
                validation_batch
                if bool(cfg.get("evaluation", {}).get("full_validation", False))
                else (
                    fixed_validation_sample
                    if fixed_validation_sample is not None
                    else validation_replay.sample(
                        validation_batch_size,
                        generator=generator,
                    )
                )
            )
            metrics.update(validation_metrics_for_training(state, validation_sample, cfg))
            if not cfg.get("critic", {}).get("stage_schedule") and maybe_advance_critic_stage(
                state,
                metrics,
                cfg,
            ):
                metrics["critic_stage_advanced"] = 1.0
                if early_stopper is not None:
                    early_stopper = ValidationEarlyStopper(
                        mode=early_stopper.mode,
                        patience=early_stopper.patience,
                        min_delta=early_stopper.min_delta,
                    )
                    best_snapshot = None
            elif (
                early_stopper is not None
                and (early_start_stage is None or state.critic_stage == str(early_start_stage))
            ):
                if early_metric not in metrics:
                    raise KeyError(f"early-stopping metric is missing: {early_metric}")
                if early_stopper.update(float(metrics[early_metric])):
                    best_snapshot = TrainableSnapshot.capture(state.critic)
                metrics["early_stopping_best"] = early_stopper.best
                metrics["early_stopping_stale_evaluations"] = float(
                    early_stopper.stale_evaluations
                )
            if selection_enabled:
                selection_score, selection_eligible = critic_selection_score(metrics, cfg)
                metrics["validation_checkpoint_selection_score"] = selection_score
                metrics["validation_checkpoint_selection_eligible"] = float(selection_eligible)
                min_delta = float(selection_cfg.get("min_delta", 0.0))
                if (
                    selection_eligible
                    and selection_score > best_selection_score + min_delta
                    and best_checkpoint
                ):
                    best_selection_score = selection_score
                    save_checkpoint(state, cfg, ROOT / best_checkpoint)
                    metrics["validation_best_checkpoint_saved"] = 1.0
                    print(
                        f"[critic] best checkpoint step={state.step} "
                        f"score={selection_score:.6f} path={best_checkpoint}",
                        flush=True,
                    )
        if is_main:
            metrics.update(sample_metrics)
            metrics["sample_unique_episodes"] = float(torch.unique(sample.episode_ids).numel())
            metrics["optimizer_step"] = float(state.step)
            metrics["distributed_world_size"] = float(world_size)
            metrics["global_batch_size"] = float(global_batch_size)
            metrics["local_batch_size"] = float(effective_batch_size)
            add_run_metadata(metrics, config=cfg, step=step)
            assert writer is not None
            writer.write(metrics)
            print(
                f"[critic] local_step={step} optimizer_step={state.step} "
                f"lambda_rank_effective={float(metrics.get('lambda_rank_effective', 0.0)):.8g} "
                f"loss={metrics['critic_loss']:.4f}"
            )
        if is_main and checkpoint_interval > 0 and state.step % checkpoint_interval == 0:
            if latest_checkpoint:
                save_checkpoint(state, cfg, ROOT / latest_checkpoint)
                print(
                    f"[critic] latest checkpoint step={state.step} path={latest_checkpoint}",
                    flush=True,
                )
        if is_main and state.step in milestone_steps and milestone_dir:
            milestone_path = ROOT / milestone_dir / f"critic_step_{state.step:05d}.pt"
            save_checkpoint(state, cfg, milestone_path)
            print(f"[critic] milestone checkpoint saved: {milestone_path}", flush=True)
        if is_main and early_stopper is not None and early_stopper.should_stop:
            if best_snapshot is not None:
                best_snapshot.restore(state.critic)
                soft_update(state.target_critic, state.critic, 1.0)
            print(
                f"[critic] early stop stage={state.critic_stage} "
                f"metric={early_metric} best={early_stopper.best:.6f}"
            )
            break
        if distributed:
            dist.barrier()
    if is_main and (
        best_snapshot is not None
        and bool(early_cfg.get("restore_best", True))
        and (early_start_stage is None or state.critic_stage == str(early_start_stage))
    ):
        best_snapshot.restore(state.critic)
        soft_update(state.target_critic, state.critic, 1.0)
        print(
            f"[critic] restored best stage={state.critic_stage} "
            f"metric={early_metric} best={early_stopper.best:.6f}"
        )
    if is_main:
        save_checkpoint(state, cfg, ROOT / cfg["training"].get("checkpoint_path", "outputs/ogpo/udivl.pt"))
        print("[critic] checkpoint saved")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
