#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ogpo.metrics import add_run_metadata, create_metrics_writer
from ogpo.origin_cache import load_or_build_origin_feature_cache
from ogpo.replay import (
    CompositeChunkBatch,
    OfflineChunkReplay,
    load_replay,
    prepare_replay_from_config,
    split_success_buffers,
)
from ogpo.zarr_replay import concat_chunk_batches
from ogpo.trainer import (
    actor_guard_reason,
    actor_start_gate,
    build_train_state,
    critic_update,
    full_actor_update,
    freeze_critic_for_actor,
    load_critic_checkpoint,
    load_checkpoint,
    save_checkpoint,
    sync_old_policy,
)
from train_udivl_critic import _deep_update, load_config


class TaskCycleActorReplay:
    """Cycle tasks deterministically, sampling transitions uniformly within a task."""

    def __init__(self, batch, task_names: list[str]):
        self.batch = batch
        self.task_names = tuple(str(task) for task in task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("actor task cycle requires unique non-empty task names")
        self.indices = {
            task: torch.tensor(
                [index for index, value in enumerate(batch.task_ids) if str(value) == task],
                dtype=torch.long,
            )
            for task in self.task_names
        }
        missing = [task for task, indices in self.indices.items() if indices.numel() == 0]
        if missing:
            raise ValueError(f"actor task cycle has empty task pools: {missing}")

    def sample_task(self, task: str, batch_size: int, *, generator: torch.Generator):
        pool = self.indices[str(task)]
        offsets = torch.randint(pool.numel(), (int(batch_size),), generator=generator)
        return self.batch.index_select(pool.index_select(0, offsets))


class IndexedActorReplay:
    """Sample a subset of a file-backed replay without materializing its images."""

    def __init__(self, batch, indices: torch.Tensor):
        self.batch = batch
        self.indices = indices.to(dtype=torch.long, device="cpu").flatten()
        if self.indices.numel() == 0:
            raise ValueError("actor replay subset is empty")

    def __len__(self) -> int:
        return self.indices.numel()

    def sample(self, batch_size: int, *, generator: torch.Generator):
        offsets = torch.randint(len(self), (int(batch_size),), generator=generator)
        return self.batch.index_select(self.indices.index_select(0, offsets))


def load_actor_training_batch(root: Path, data_config: dict):
    """Load prepared shards with images file-backed; retain old raw-replay path."""
    configured_paths = data_config.get("dataset_paths")
    if configured_paths is None:
        configured_paths = [data_config["dataset_path"]]
    if not isinstance(configured_paths, list) or not configured_paths:
        raise ValueError("data.dataset_paths must be a non-empty list")
    batches = [
        load_replay(root / path, mmap=len(configured_paths) > 1)
        for path in configured_paths
    ]
    if len(batches) > 1 and bool(data_config.get("dataset_preprocessed", False)):
        return CompositeChunkBatch(batches)
    batch = batches[0] if len(batches) == 1 else concat_chunk_batches(batches)
    return prepare_replay_from_config(batch, data_config)


def smoke_output_root(config: dict) -> Path:
    """Derive an isolated smoke directory from the resolved run outputs."""
    training_cfg = config.get("training", {})
    periodic_dir = training_cfg.get("periodic_checkpoint_dir")
    if periodic_dir:
        run_name = Path(periodic_dir).name
    else:
        checkpoint_path = training_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(
                "actor smoke mode requires training.periodic_checkpoint_dir "
                "or training.checkpoint_path"
            )
        run_name = Path(checkpoint_path).stem.removesuffix("_final")
    if not run_name:
        raise ValueError("could not derive actor smoke run name from training outputs")
    return Path("outputs/ogpo/smoke") / run_name


def periodic_actor_checkpoint_path(
    training_config: dict,
    completed_step: int,
    *,
    root: Path = ROOT,
) -> Path | None:
    """Resolve a periodic checkpoint exactly at configured completed steps."""
    completed_step = int(completed_step)
    if completed_step <= 0:
        raise ValueError("completed actor step must be positive")
    interval = int(training_config.get("checkpoint_interval", 0))
    if interval < 0:
        raise ValueError("training.checkpoint_interval must be non-negative")
    if interval == 0 or completed_step % interval:
        return None
    checkpoint_path = root / training_config.get(
        "checkpoint_path", "outputs/ogpo/full.pt"
    )
    if not bool(training_config.get("keep_periodic_checkpoints", False)):
        return checkpoint_path
    periodic_dir = root / training_config.get(
        "periodic_checkpoint_dir",
        str(checkpoint_path.parent / f"{checkpoint_path.stem}_milestones"),
    )
    return periodic_dir / f"step_{completed_step:04d}" / checkpoint_path.name


def save_periodic_actor_checkpoint_if_due(
    state,
    config: dict,
    completed_step: int,
    *,
    root: Path = ROOT,
) -> Path | None:
    """Persist the full resumable actor transaction at an interval boundary."""
    completed_step = int(completed_step)
    if int(state.actor_step) != completed_step:
        raise ValueError(
            "refusing periodic checkpoint with mismatched actor_step: "
            f"state={state.actor_step} completed={completed_step}"
        )
    path = periodic_actor_checkpoint_path(
        config.get("training", {}), completed_step, root=root
    )
    if path is None:
        return None
    save_checkpoint(state, config, path)
    return path


def advance_actor_replay_generator_for_exact_resume(
    generator: torch.Generator,
    *,
    completed_steps: int,
    replay_size: int,
    batch_size: int,
    success_replay_size: int | None = None,
    success_batch_size: int = 0,
) -> dict[str, int]:
    """Replay deterministic sampler draws consumed by completed actor steps."""
    completed_steps = int(completed_steps)
    replay_size = int(replay_size)
    batch_size = int(batch_size)
    if completed_steps < 0 or replay_size <= 0 or batch_size <= 0:
        raise ValueError("invalid actor replay-resume geometry")
    if success_replay_size is not None and (
        int(success_replay_size) <= 0 or int(success_batch_size) <= 0
    ):
        raise ValueError("invalid success replay-resume geometry")
    for _ in range(completed_steps):
        # Each accepted actor iteration samples the policy batch and FM batch.
        torch.randint(replay_size, (batch_size,), generator=generator)
        torch.randint(replay_size, (batch_size,), generator=generator)
        if success_replay_size is not None:
            torch.randint(
                int(success_replay_size),
                (int(success_batch_size),),
                generator=generator,
            )
    return {
        "main_draws": completed_steps * batch_size,
        "fm_draws": completed_steps * batch_size,
        "success_draws": completed_steps * int(success_batch_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/full_ogpo.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--actor-steps", type=int)
    parser.add_argument("--critic-checkpoint")
    parser.add_argument("--base-checkpoint-dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    for overlay in args.overlay:
        cfg = _deep_update(cfg, load_config(ROOT / overlay))
    if args.actor_steps is not None:
        if args.actor_steps <= 0:
            raise ValueError("--actor-steps must be positive")
        cfg.setdefault("training", {})["actor_steps"] = args.actor_steps
    if args.critic_checkpoint:
        cfg.setdefault("training", {})["critic_checkpoint"] = args.critic_checkpoint
    if args.base_checkpoint_dir:
        cfg.setdefault("flow", {})["checkpoint_dir"] = args.base_checkpoint_dir
    if args.smoke:
        smoke_root = smoke_output_root(cfg)
        cfg["training"].update(
            {
                "checkpoint_path": str(smoke_root / "actor_smoke.pt"),
                "periodic_checkpoint_dir": str(smoke_root / "periodic"),
                "checkpoint_interval": 0,
                "metrics_path": str(smoke_root / "metrics.jsonl"),
                "tensorboard_dir": str(smoke_root / "tensorboard"),
                "config_snapshot_path": str(smoke_root / "resolved_config.yaml"),
            }
        )
    snapshot_path = cfg.get("training", {}).get("config_snapshot_path")
    if snapshot_path:
        resolved_path = ROOT / snapshot_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
    batch = load_actor_training_batch(ROOT, cfg["data"])
    # Apply the same explicitly configured offline transforms used by the
    # critic trainer (gamma rebase and, for RoboTwin, n-step folding) before
    # constructing replay samplers.  Historically this script loaded the raw
    # replay directly, silently ignoring ``data.apply_n_step_on_load``.
    replay = OfflineChunkReplay(batch)
    actor_sampling_cfg = cfg.get("training", {}).get("actor_sampling", {})
    actor_sampling_mode = str(actor_sampling_cfg.get("mode", "uniform")).lower()
    task_cycle_replay = None
    if actor_sampling_mode == "task_balanced_cycle":
        task_cycle_replay = TaskCycleActorReplay(
            batch, [str(value) for value in actor_sampling_cfg.get("task_names", [])]
        )
        print(
            f"[full] actor task-balanced cycle tasks={list(task_cycle_replay.task_names)}",
            flush=True,
        )
    elif actor_sampling_mode != "uniform":
        raise ValueError(f"unsupported training.actor_sampling.mode={actor_sampling_mode!r}")
    validation_path = cfg["data"].get("validation_path")
    validation_batch = load_replay(ROOT / validation_path) if validation_path else batch
    if validation_path:
        validation_batch = prepare_replay_from_config(
            validation_batch, cfg.get("data", {})
        )
    lambda_success = float(cfg.get("regularization", {}).get("lambda_success", 0.0))
    success_bc_enabled = lambda_success > 0.0
    success_indices = (
        torch.nonzero(batch.successes.bool(), as_tuple=False).flatten()
        if success_bc_enabled else torch.empty(0, dtype=torch.long)
    )
    success_replay = (
        IndexedActorReplay(batch, success_indices)
        if success_indices.numel() else None
    )
    print(
        "[full] regularization "
        f"lambda_success={lambda_success:.12g} "
        f"success_bc_enabled={int(success_bc_enabled)}",
        flush=True,
    )
    state_init_batch = (
        batch.index_select(torch.arange(min(8, batch.batch_size)))
        if isinstance(batch, CompositeChunkBatch) else batch
    )
    state = build_train_state(cfg, state_init_batch, device=cfg["training"].get("device", "cpu"))
    print(
        "[full] flow_sde "
        f"mode={state.policy.sde_mode} "
        f"constant_noise_std={float(getattr(state.policy, 'constant_noise_std', 0.0)):.6f} "
        f"learn_sde_std={bool(getattr(state.policy, 'learn_sde_std', True))} "
        f"log_std_trainable={bool(getattr(state.policy, 'log_std', torch.empty(0)).requires_grad)}",
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
        if actor_sampling_mode == "task_balanced_cycle":
            task_cycle_replay = TaskCycleActorReplay(
                batch, [str(value) for value in actor_sampling_cfg.get("task_names", [])]
            )
        success_batch = (
            split_success_buffers(batch).get("success") if success_bc_enabled else None
        )
        success_replay = OfflineChunkReplay(success_batch) if success_batch is not None else None
    resume = args.resume or cfg["training"].get("resume_checkpoint")
    critic_checkpoint = cfg["training"].get("critic_checkpoint")
    critic_steps_per_actor_step = int(cfg["critic"].get("steps_per_actor_step", 4))
    critic_frozen_during_actor = bool(
        cfg.get("training", {}).get("critic_frozen_during_actor", False)
    )
    if critic_frozen_during_actor and critic_steps_per_actor_step != 0:
        raise ValueError(
            "pure-offline actor improvement requires critic.steps_per_actor_step=0"
        )
    if critic_frozen_during_actor and not (
        resume or cfg.get("training", {}).get("critic_checkpoint")
    ):
        raise ValueError(
            "pure-offline actor improvement requires training.critic_checkpoint "
            "(or an actor resume checkpoint)"
        )
    if resume:
        load_checkpoint(ROOT / resume, state)
        print(f"[full] resumed checkpoint: {resume}")
    elif critic_checkpoint:
        load_critic_checkpoint(
            ROOT / critic_checkpoint,
            state,
            load_optimizer=critic_steps_per_actor_step > 0,
        )
        print(f"[full] loaded critic checkpoint: {critic_checkpoint}")
    if critic_frozen_during_actor:
        freeze_critic_for_actor(state)
        print("[full] critic frozen for offline actor improvement")
    if critic_steps_per_actor_step == 0:
        state.target_critic.to("cpu")
        if state.target_divl is not None:
            state.target_divl.to("cpu")
        print(
            "[full] actor-only critic mode: critic optimizer state is unused; "
            "target networks moved to CPU"
        )
    writer = create_metrics_writer(
        ROOT / cfg["training"].get("metrics_path", "outputs/ogpo/full_metrics.jsonl"),
        ROOT / cfg["training"]["tensorboard_dir"] if cfg["training"].get("tensorboard_dir") else None,
    )
    checkpoint_path = ROOT / cfg["training"].get(
        "checkpoint_path", "outputs/ogpo/full.pt"
    )
    generator = torch.Generator().manual_seed(13)
    replay_resume_step = int(
        cfg.get("training", {}).get("replay_generator_resume_step", 0)
    )
    if replay_resume_step:
        if not resume:
            raise ValueError(
                "training.replay_generator_resume_step requires an actor resume checkpoint"
            )
        if replay_resume_step != int(state.actor_step):
            raise ValueError(
                "actor replay resume step mismatch: "
                f"configured={replay_resume_step} checkpoint={state.actor_step}"
            )
        if not bool(cfg.get("critic", {}).get("force_actor", False)):
            raise ValueError(
                "exact replay-generator advancement currently requires critic.force_actor=true"
            )
        success_count = (
            max(
                1,
                round(
                    int(cfg["training"].get("batch_size", 16))
                    * float(cfg["data"].get("success_sampling_ratio", 0.5))
                ),
            )
            if success_replay is not None
            else 0
        )
        replay_draws = advance_actor_replay_generator_for_exact_resume(
            generator,
            completed_steps=replay_resume_step,
            replay_size=len(replay),
            batch_size=int(cfg["training"].get("batch_size", 16)),
            success_replay_size=(
                None if success_replay is None else len(success_replay)
            ),
            success_batch_size=success_count,
        )
        print(
            "[full] actor replay generator advanced for exact resume: "
            f"completed_steps={replay_resume_step} "
            f"main_draws={replay_draws['main_draws']} "
            f"fm_draws={replay_draws['fm_draws']} "
            f"success_draws={replay_draws['success_draws']}",
            flush=True,
        )
    consecutive_kl_rejections = 0
    actor_start_step = max(
        int(cfg["training"].get("actor_start_step", 0)), state.actor_step
    )
    last_periodic_path: Path | None = None
    last_periodic_step: int | None = None
    for local_step in range(int(cfg["training"].get("actor_steps", 2))):
        step = actor_start_step + local_step
        sampled_actor_task = None
        if task_cycle_replay is not None:
            sampled_actor_task = task_cycle_replay.task_names[
                local_step % len(task_cycle_replay.task_names)
            ]
            sample = task_cycle_replay.sample_task(
                sampled_actor_task,
                int(cfg["training"].get("batch_size", 16)),
                generator=generator,
            )
        else:
            sample = replay.sample(int(cfg["training"].get("batch_size", 16)), generator=generator)
        if not critic_frozen_during_actor:
            for _ in range(critic_steps_per_actor_step):
                critic_update(state, sample, cfg)
        gate_reason, gate_metrics = actor_start_gate(
            state, validation_batch, cfg, outer_step=step
        )
        if gate_reason:
            metrics: dict[str, float | str] = {
                **gate_metrics,
                "actor_skipped": 1.0,
                "stop_reason": gate_reason,
            }
            add_run_metadata(metrics, config=cfg, step=step)
            if sampled_actor_task is not None:
                metrics["sampled_actor_task"] = sampled_actor_task
            writer.write(metrics)
            print(f"[full] step={step} actor skipped: {gate_reason}")
            continue
        fm_sample = (
            task_cycle_replay.sample_task(
                sampled_actor_task,
                int(cfg["training"].get("batch_size", 16)),
                generator=generator,
            )
            if task_cycle_replay is not None and sampled_actor_task is not None
            else replay.sample(int(cfg["training"].get("batch_size", 16)), generator=generator)
        )
        success_sample = None
        if success_replay is not None:
            success_count = max(
                1,
                round(
                    int(cfg["training"].get("batch_size", 16))
                    * float(cfg["data"].get("success_sampling_ratio", 0.5))
                ),
            )
            success_sample = success_replay.sample(
                min(success_count, len(success_replay)),
                generator=generator,
            )
        metrics = full_actor_update(state, sample, cfg, fm_batch=fm_sample, success_batch=success_sample)
        metrics.update(gate_metrics)
        if bool(metrics.get("actor_update_rejected", 0.0)):
            consecutive_kl_rejections += 1
        else:
            consecutive_kl_rejections = 0
        metrics["consecutive_kl_rejections"] = float(consecutive_kl_rejections)
        accepted = bool(metrics.get("actor_update_accepted", 0.0))
        if str(cfg.get("flow", {}).get("adapter")) == "pi05_jax":
            # Preserve the pre-existing JAX lifecycle; this task changes only
            # the PyTorch transaction path.
            if accepted:
                state.accepted_actor_updates += 1
            state.actor_step = step + 1
            sync_period = int(cfg["actor"].get("old_policy_sync_period", 1))
            if sync_period > 0 and (step + 1) % sync_period == 0:
                sync_old_policy(
                    state, ema=float(cfg["actor"].get("old_policy_ema", 0.0))
                )
        add_run_metadata(metrics, config=cfg, step=step)
        if sampled_actor_task is not None:
            metrics["sampled_actor_task"] = sampled_actor_task
        stop_reason = actor_guard_reason(metrics, cfg)
        metrics["stop_reason"] = stop_reason or ""
        writer.write(metrics)
        print(f"[full] step={step} actor_loss={metrics['actor_loss']:.4f}")
        completed_step = step + 1
        periodic_path = save_periodic_actor_checkpoint_if_due(
            state, cfg, completed_step
        )
        if periodic_path is not None:
            last_periodic_path = periodic_path
            last_periodic_step = completed_step
            print(
                "[full] periodic checkpoint saved: "
                f"completed_step={completed_step} log_step={step} path={periodic_path}",
                flush=True,
            )
        if stop_reason:
            print(f"[full] stopping actor extraction: {stop_reason}")
            break
    if (
        last_periodic_path is not None
        and last_periodic_step == int(state.actor_step)
    ):
        # The complete resumable state for this exact actor step was just
        # serialized by the production periodic-save path. Avoid immediately
        # materializing the same ~24 GiB payload again: the duplicate CPU copy
        # can exceed host memory on four-role PI0.5 runs.
        print(
            "[full] terminal checkpoint already saved by periodic path: "
            f"actor_step={state.actor_step} path={last_periodic_path}",
            flush=True,
        )
    else:
        if bool(cfg.get("training", {}).get("save_final_checkpoint", True)):
            save_checkpoint(state, cfg, checkpoint_path)
            print(f"[full] checkpoint saved: path={checkpoint_path}", flush=True)
        else:
            print("[full] final checkpoint intentionally skipped", flush=True)


if __name__ == "__main__":
    main()
