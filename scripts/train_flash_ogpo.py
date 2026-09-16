#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import jax
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ogpo.metrics import add_run_metadata, create_metrics_writer
from ogpo.replay import (
    BalancedCriticReplay,
    OfflineChunkReplay,
    load_replay,
    split_success_buffers,
)
from ogpo.trainer import (
    accumulated_critic_update,
    actor_guard_reason,
    actor_start_gate,
    build_train_state,
    critic_update,
    finalize_actor_update_transaction,
    flash_actor_update,
    full_actor_update,
    freeze_critic_for_actor,
    load_critic_checkpoint,
    load_checkpoint,
    save_checkpoint,
    sync_old_policy,
)
from train_udivl_critic import _deep_update, load_config


def _optimizer_state_to(
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> None:
    for parameter_state in optimizer.state.values():
        for name, value in tuple(parameter_state.items()):
            if torch.is_tensor(value):
                parameter_state[name] = value.to(device)


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
        group["initial_lr"] = float(learning_rate)


def _mean_metric_dict(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    numeric_keys = {
        key
        for item in metrics
        for key, value in item.items()
        if isinstance(value, (int, float))
    }
    return {
        f"critic_refresh_{key}": sum(float(item[key]) for item in metrics if key in item)
        / sum(1 for item in metrics if key in item)
        for key in numeric_keys
    }


def _configure_jax_compilation_cache() -> None:
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")),
    )
    jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        int(os.environ.get("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")),
    )
    print(f"[flash] JAX persistent compilation cache: {cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ogpo/flash_ogpo.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--overlay", action="append", default=[])
    args = parser.parse_args()
    _configure_jax_compilation_cache()
    cfg = load_config(ROOT / args.config)
    for overlay in args.overlay:
        cfg = _deep_update(cfg, load_config(ROOT / overlay))
    batch = load_replay(ROOT / cfg["data"]["dataset_path"])
    replay = OfflineChunkReplay(batch)
    critic_cfg = cfg.get("critic", {})
    critic_gate_enabled = bool(critic_cfg.get("gate_enabled", True))
    validation_gate_batch = None
    if critic_gate_enabled:
        validation_path = cfg["data"].get("validation_path")
        validation_batch = load_replay(ROOT / validation_path) if validation_path else batch
        evaluation_cfg = cfg.get("evaluation", {})
        configured_gate_size = int(
            evaluation_cfg.get(
                "actor_gate_batch_size",
                evaluation_cfg.get(
                    "validation_batch_size", cfg["training"].get("batch_size", 16)
                ),
            )
        )
        gate_batch_size = (
            validation_batch.batch_size
            if configured_gate_size <= 0
            else min(validation_batch.batch_size, configured_gate_size)
        )
        gate_sampling_cfg = evaluation_cfg.get("actor_gate_sampling", {})
        if bool(gate_sampling_cfg.get("balanced", False)):
            gate_replay = BalancedCriticReplay(
                validation_batch,
                uniform_fraction=float(gate_sampling_cfg.get("uniform_fraction", 0.5)),
                success_fraction=float(gate_sampling_cfg.get("success_fraction", 0.25)),
                terminal_success_fraction=float(
                    gate_sampling_cfg.get("terminal_success_fraction", 0.125)
                ),
                failure_fraction=float(gate_sampling_cfg.get("failure_fraction", 0.125)),
            )
        else:
            gate_replay = OfflineChunkReplay(validation_batch)
        validation_gate_batch = gate_replay.sample(
            gate_batch_size, generator=torch.Generator().manual_seed(23)
        )
    success_batch = split_success_buffers(batch).get("success")
    success_replay = OfflineChunkReplay(success_batch) if success_batch is not None else None
    legacy_critic_steps = int(critic_cfg.get("steps_per_actor_step", 0))
    critic_refresh_period = int(
        critic_cfg.get("update_period_actor_steps", 1 if legacy_critic_steps > 0 else 0)
    )
    critic_steps_per_refresh = int(
        critic_cfg.get("steps_per_refresh", legacy_critic_steps)
    )
    critic_refresh_enabled = critic_refresh_period > 0 and critic_steps_per_refresh > 0
    critic_frozen_during_actor = bool(
        cfg.get("training", {}).get("critic_frozen_during_actor", False)
    )
    if critic_frozen_during_actor and critic_refresh_enabled:
        raise ValueError(
            "pure-offline actor improvement requires critic refresh/update to be disabled"
        )
    if critic_frozen_during_actor and not (
        args.resume or cfg.get("training", {}).get("critic_checkpoint")
    ):
        raise ValueError(
            "pure-offline actor improvement requires training.critic_checkpoint "
            "(or an actor resume checkpoint)"
        )
    if (critic_refresh_period < 0 or critic_steps_per_refresh < 0):
        raise ValueError("critic refresh period and steps must be non-negative")
    critic_refresh_batch_size = int(
        critic_cfg.get("refresh_batch_size", cfg["training"].get("batch_size", 16))
    )
    critic_refresh_microbatch_size = int(
        critic_cfg.get("refresh_microbatch_size", critic_refresh_batch_size)
    )
    if critic_refresh_enabled and critic_refresh_batch_size <= 0:
        raise ValueError("critic.refresh_batch_size must be positive")
    if critic_refresh_enabled and critic_refresh_microbatch_size <= 0:
        raise ValueError("critic.refresh_microbatch_size must be positive")
    refresh_sampling_cfg = critic_cfg.get("refresh_sampling", {})
    if bool(refresh_sampling_cfg.get("balanced", True)):
        critic_replay = BalancedCriticReplay(
            batch,
            uniform_fraction=float(refresh_sampling_cfg.get("uniform_fraction", 0.5)),
            success_fraction=float(refresh_sampling_cfg.get("success_fraction", 0.25)),
            terminal_success_fraction=float(
                refresh_sampling_cfg.get("terminal_success_fraction", 0.125)
            ),
            failure_fraction=float(refresh_sampling_cfg.get("failure_fraction", 0.125)),
        )
    else:
        critic_replay = OfflineChunkReplay(batch)
    state = build_train_state(cfg, batch, device=cfg["training"].get("device", "cpu"))
    print(
        "[actor] flow_sde "
        f"mode={state.policy.sde_mode} "
        f"constant_noise_std={float(getattr(state.policy, 'constant_noise_std', 0.0)):.6f} "
        f"learn_sde_std={bool(getattr(state.policy, 'learn_sde_std', True))} "
        f"log_std_trainable={bool(getattr(state.policy, 'log_std', torch.empty(0)).requires_grad)}",
        flush=True,
    )
    resume = args.resume or cfg["training"].get("resume_checkpoint")
    critic_checkpoint = cfg["training"].get("critic_checkpoint")
    override_critic_on_resume = bool(
        cfg["training"].get("override_critic_on_resume", False)
    )
    if override_critic_on_resume and not resume:
        raise ValueError("override_critic_on_resume requires a resume checkpoint")
    if override_critic_on_resume and not critic_checkpoint:
        raise ValueError("override_critic_on_resume requires a critic checkpoint")
    reset_actor_optimizer = False
    if resume:
        reset_actor_optimizer = bool(
            cfg["training"].get("reset_actor_optimizer_on_resume", False)
        )
        load_checkpoint(
            ROOT / resume,
            state,
            restore_actor_optimizer=not reset_actor_optimizer,
        )
        if reset_actor_optimizer and bool(
            cfg["training"].get("sync_old_policy_on_resume", True)
        ):
            sync_old_policy(state, ema=0.0)
        print(f"[flash] resumed checkpoint: {resume}")
        if reset_actor_optimizer:
            print(
                "[flash] reset actor optimizer on resume and synchronized "
                "old policy to the restored actor"
            )
        if override_critic_on_resume:
            load_critic_checkpoint(
                ROOT / critic_checkpoint,
                state,
                load_optimizer=critic_refresh_enabled,
            )
            print(
                "[flash] replaced resumed critic checkpoint: "
                f"{critic_checkpoint}"
            )
    elif critic_checkpoint:
        load_critic_checkpoint(
            ROOT / critic_checkpoint,
            state,
            load_optimizer=critic_refresh_enabled,
        )
        print(f"[flash] loaded critic checkpoint: {critic_checkpoint}")
    if critic_frozen_during_actor:
        freeze_critic_for_actor(state)
        print("[flash] critic frozen for offline actor improvement")
    if critic_refresh_enabled:
        refresh_learning_rate = float(
            critic_cfg.get("refresh_learning_rate", critic_cfg.get("learning_rate", 3e-4))
        )
        _set_optimizer_learning_rate(state.critic_optimizer, refresh_learning_rate)
        print(
            "[flash] periodic critic refresh: "
            f"period={critic_refresh_period} actor updates, "
            f"steps={critic_steps_per_refresh}, batch={critic_refresh_batch_size}, "
            f"microbatch={critic_refresh_microbatch_size}, "
            f"lr={refresh_learning_rate:g}"
        )
    if not critic_refresh_enabled:
        state.target_critic.to("cpu")
        if state.target_divl is not None:
            state.target_divl.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "[flash] actor-only critic mode: critic optimizer state is unused; "
            "target networks moved to CPU"
        )
    elif bool(critic_cfg.get("offload_refresh_state_between_updates", True)):
        state.target_critic.to("cpu")
        if state.target_divl is not None:
            state.target_divl.to("cpu")
        _optimizer_state_to(state.critic_optimizer, "cpu")
        torch.cuda.empty_cache()
        print("[flash] critic target and optimizer state offloaded between refreshes")
    writer = create_metrics_writer(
        ROOT / cfg["training"].get("metrics_path", "outputs/ogpo/flash_metrics.jsonl"),
        ROOT / cfg["training"]["tensorboard_dir"] if cfg["training"].get("tensorboard_dir") else None,
    )
    checkpoint_path = ROOT / cfg["training"].get("checkpoint_path", "outputs/ogpo/flash.pt")
    checkpoint_interval = int(cfg["training"].get("checkpoint_interval", 0))
    keep_periodic_checkpoints = bool(
        cfg["training"].get("keep_periodic_checkpoints", False)
    )
    periodic_checkpoint_dir = ROOT / cfg["training"].get(
        "periodic_checkpoint_dir",
        str(checkpoint_path.parent / f"{checkpoint_path.stem}_milestones"),
    )
    generator = torch.Generator().manual_seed(17)
    critic_generator = torch.Generator().manual_seed(29)
    consecutive_kl_rejections = 0
    flash_enabled = bool(cfg.get("actor", {}).get("flash_enabled", True))
    actor_update_mode = "flash" if flash_enabled else "full_chain"
    print(f"[actor] update_mode={actor_update_mode}")
    actor_start_step = max(
        int(cfg["training"].get("actor_start_step", 0)), state.actor_step
    )
    accepted_actor_updates = max(
        int(cfg["training"].get("accepted_actor_updates", 0)),
        state.accepted_actor_updates,
    )
    state.accepted_actor_updates = accepted_actor_updates
    if critic_gate_enabled:
        assert validation_gate_batch is not None
        gate_reason, gate_metrics = actor_start_gate(
            state,
            validation_gate_batch,
            cfg,
            outer_step=actor_start_step,
        )
        gate_metrics["critic_gate_enabled"] = 1.0
    else:
        gate_reason = None
        gate_metrics = {
            "critic_training_step": float(state.step),
            "critic_gate_enabled": 0.0,
        }
    gate_checked_at_actor_update = accepted_actor_updates
    if critic_gate_enabled:
        print(
            "[flash-gate] initial "
            f"passed={int(not bool(gate_reason))} reason={gate_reason or 'ok'} "
            f"ranking={gate_metrics.get('pairwise_ranking_accuracy', float('nan')):.4f} "
            f"rank_corr={gate_metrics.get('q_rank_correlation', float('nan')):.4f} "
            f"q_gap={gate_metrics.get('q_exploitation_gap', float('nan')):.4f} "
            f"coverage={gate_metrics.get('interval_coverage', float('nan')):.4f}"
        )
    else:
        print("[flash-gate] disabled; critic diagnostics do not gate actor training")
    for local_step in range(int(cfg["training"].get("actor_steps", 2))):
        outer_step_started = time.perf_counter()
        step = actor_start_step + local_step
        sample = replay.sample(int(cfg["training"].get("batch_size", 16)), generator=generator)
        if gate_reason:
            metrics: dict[str, float | str] = {
                **gate_metrics,
                "actor_skipped": 1.0,
                "stop_reason": gate_reason,
            }
            add_run_metadata(metrics, config=cfg, step=step)
            writer.write(metrics)
            print(
                f"[flash-gate] step={step} passed=0 reason={gate_reason} "
                f"ranking={metrics.get('pairwise_ranking_accuracy', float('nan')):.4f} "
                f"rank_corr={metrics.get('q_rank_correlation', float('nan')):.4f} "
                f"q_gap={metrics.get('q_exploitation_gap', float('nan')):.4f} "
                f"coverage={metrics.get('interval_coverage', float('nan')):.4f}"
            )
            print("[flash] critic failed its cached validation gate; stopping")
            break
        fm_sample = replay.sample(int(cfg["training"].get("batch_size", 16)), generator=generator)
        success_sample = None
        if success_replay is not None:
            success_count = max(
                1,
                round(
                    int(cfg["training"].get("batch_size", 16))
                    * float(cfg["data"].get("success_sampling_ratio", 0.5))
                ),
            )
            success_count = min(
                success_count,
                int(
                    cfg.get("regularization", {}).get(
                        "success_batch_size",
                        success_count,
                    )
                ),
            )
            success_sample = success_replay.sample(
                min(success_count, len(success_replay)),
                generator=generator,
            )
        actor_update_started = time.perf_counter()
        if flash_enabled:
            metrics = flash_actor_update(
                state,
                sample,
                cfg,
                fm_batch=fm_sample,
                success_batch=success_sample,
                actor_step=step,
            )
        else:
            metrics = full_actor_update(
                state,
                sample,
                cfg,
                fm_batch=fm_sample,
                success_batch=success_sample,
            )
        metrics["actor_update_seconds"] = time.perf_counter() - actor_update_started
        metrics["actor_update_mode"] = actor_update_mode
        metrics["actor_optimizer_reset_on_resume"] = float(
            reset_actor_optimizer
        )
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
        elif flash_enabled:
            metrics.update(
                finalize_actor_update_transaction(
                    state,
                    accepted=accepted,
                    config=cfg,
                )
            )
        accepted_actor_updates = state.accepted_actor_updates
        critic_refresh_due = (
            critic_refresh_enabled
            and bool(metrics.get("actor_update_accepted", 0.0))
            and accepted_actor_updates % critic_refresh_period == 0
        )
        metrics["critic_refresh_applied"] = float(critic_refresh_due)
        metrics["accepted_actor_updates"] = float(accepted_actor_updates)
        if critic_refresh_due:
            refresh_started = time.perf_counter()
            jax.effects_barrier()
            if bool(critic_cfg.get("clear_jax_cache_before_refresh", True)):
                jax.clear_caches()
            gc.collect()
            critic_device = next(state.critic.parameters()).device
            state.target_critic.to(critic_device)
            if state.target_divl is not None:
                state.target_divl.to(critic_device)
            _optimizer_state_to(state.critic_optimizer, critic_device)
            refresh_metrics = []
            for _ in range(critic_steps_per_refresh):
                critic_batch = critic_replay.sample(
                    critic_refresh_batch_size,
                    generator=critic_generator,
                )
                refresh_metrics.append(
                    accumulated_critic_update(
                        state,
                        critic_batch,
                        cfg,
                        microbatch_size=critic_refresh_microbatch_size,
                    )
                )
            state.critic_optimizer.zero_grad(set_to_none=True)
            metrics.update(_mean_metric_dict(refresh_metrics))
            metrics["critic_refresh_seconds"] = time.perf_counter() - refresh_started
            if bool(critic_cfg.get("offload_refresh_state_between_updates", True)):
                state.target_critic.to("cpu")
                if state.target_divl is not None:
                    state.target_divl.to("cpu")
                _optimizer_state_to(state.critic_optimizer, "cpu")
                gc.collect()
                torch.cuda.empty_cache()
            print(
                "[critic-refresh] "
                f"actor_updates={accepted_actor_updates} "
                f"critic_step={state.step} "
                f"loss={metrics.get('critic_refresh_critic_loss', float('nan')):.5f} "
                f"q={metrics.get('critic_refresh_q_mean', float('nan')):.5f} "
                f"target={metrics.get('critic_refresh_target_mean', float('nan')):.5f}"
            )
        if critic_gate_enabled:
            gate_eval_period = int(critic_cfg.get("gate_eval_period_actor_steps", 100))
            if gate_eval_period <= 0:
                raise ValueError("critic.gate_eval_period_actor_steps must be positive")
            if accepted_actor_updates - gate_checked_at_actor_update >= gate_eval_period:
                assert validation_gate_batch is not None
                gate_reason, gate_metrics = actor_start_gate(
                    state,
                    validation_gate_batch,
                    cfg,
                    outer_step=step + 1,
                )
                gate_metrics["critic_gate_enabled"] = 1.0
                gate_checked_at_actor_update = accepted_actor_updates
                print(
                    "[flash-gate] refresh "
                    f"actor_updates={accepted_actor_updates} "
                    f"passed={int(not bool(gate_reason))} reason={gate_reason or 'ok'} "
                    f"ranking={gate_metrics.get('pairwise_ranking_accuracy', float('nan')):.4f} "
                    f"rank_corr={gate_metrics.get('q_rank_correlation', float('nan')):.4f} "
                    f"q_gap={gate_metrics.get('q_exploitation_gap', float('nan')):.4f} "
                    f"coverage={gate_metrics.get('interval_coverage', float('nan')):.4f}"
                )
        metrics.update(gate_metrics)
        metrics["critic_gate_age_actor_steps"] = float(
            accepted_actor_updates - gate_checked_at_actor_update
            if critic_gate_enabled
            else 0
        )
        add_run_metadata(metrics, config=cfg, step=step)
        stop_reason = actor_guard_reason(metrics, cfg)
        metrics["stop_reason"] = stop_reason or ""
        metrics["outer_step_seconds"] = time.perf_counter() - outer_step_started
        metrics["periodic_checkpoint_due"] = float(
            checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0
        )
        writer.write(metrics)
        print(f"[flash] step={step} actor_loss={metrics['actor_loss']:.4f}")
        print(
            f"[flash-monitor] step={step} "
            f"gate_rank={metrics.get('pairwise_ranking_accuracy', float('nan')):.4f} "
            f"gate_corr={metrics.get('q_rank_correlation', float('nan')):.4f} "
            f"gate_gap={metrics.get('q_exploitation_gap', float('nan')):.4f} "
            f"post_kl={metrics.get('post_update_reference_kl', float('nan')):.5f} "
            f"kl_util={metrics.get('policy_reference_kl_utilization', float('nan')):.3f} "
            f"accepted={int(metrics.get('actor_update_accepted', 0.0))} "
            f"kl_reject_streak={int(metrics.get('consecutive_kl_rejections', 0.0))} "
            f"grad={metrics.get('actor_grad_norm', float('nan')):.3f} "
            f"grad_scale={metrics.get('actor_grad_clip_scale', float('nan')):.4f} "
            f"ratio={metrics.get('importance_ratio_mean', float('nan')):.5f} "
            f"ratio_std={metrics.get('importance_ratio_std', float('nan')):.5f} "
            f"ppo_clip={metrics.get('ppo_clip_fraction', float('nan')):.3f} "
            f"sign_agree={metrics.get('sign_agreement_ratio', float('nan')):.3f} "
            f"state_skip={metrics.get('state_skip_fraction', float('nan')):.3f}"
        )
        if bool(cfg.get("actor", {}).get("jax_gc_between_steps", False)):
            # Full PI0.5 NNX merges create large temporary pytrees. Ensure all
            # asynchronous work has completed, then collect wrapper cycles
            # before constructing the next step's value_and_grad closure.
            jax.effects_barrier()
            clear_each_step = bool(
                cfg.get("actor", {}).get("jax_clear_caches_between_steps", False)
            )
            clear_after_success = bool(
                cfg.get("actor", {}).get(
                    "jax_clear_caches_after_success_update",
                    False,
                )
            ) and bool(metrics.get("success_update_applied", 0.0))
            clear_period = int(
                cfg.get("actor", {}).get("jax_clear_caches_period", 0)
            )
            clear_periodic = clear_period > 0 and (step + 1) % clear_period == 0
            gc.collect()
            memory_threshold = int(
                cfg.get("actor", {}).get(
                    "jax_clear_caches_bytes_in_use_threshold",
                    0,
                )
            )
            memory_stats_by_device = [device.memory_stats() for device in jax.devices()]
            max_bytes_in_use = max(
                (
                    int(stats.get("bytes_in_use", 0))
                    for stats in memory_stats_by_device
                    if stats
                ),
                default=0,
            )
            clear_for_memory = (
                memory_threshold > 0 and max_bytes_in_use >= memory_threshold
            )
            clear_after_rejection = bool(
                metrics.get("jax_rejection_cleanup_applied", 0.0)
            )
            should_clear_cache = (
                clear_each_step
                or clear_after_success
                or clear_periodic
                or clear_for_memory
                or clear_after_rejection
            )
            if should_clear_cache:
                jax.clear_caches()
            gc.collect()
            memory_stats = jax.devices()[0].memory_stats()
            if memory_stats:
                print(
                    "[flash] jax_memory "
                    f"bytes_in_use={memory_stats.get('bytes_in_use', -1)} "
                    f"peak_bytes_in_use={memory_stats.get('peak_bytes_in_use', -1)} "
                    "cache_cleared="
                    f"{int(should_clear_cache)} "
                    f"rejection_cleanup={int(clear_after_rejection)} "
                    f"periodic_cleanup={int(clear_periodic)} "
                    f"memory_cleanup={int(clear_for_memory)} "
                    f"max_device_bytes_in_use={max_bytes_in_use} "
                    f"memory_cleanup_threshold={memory_threshold}"
                )
        if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            completed_step = step + 1
            periodic_path = checkpoint_path
            if keep_periodic_checkpoints:
                periodic_path = (
                    periodic_checkpoint_dir
                    / f"step_{completed_step:04d}"
                    / checkpoint_path.name
                )
            save_checkpoint(state, cfg, periodic_path)
            print(
                f"[flash] periodic checkpoint saved: completed_step={completed_step} "
                f"log_step={step} path={periodic_path}"
            )
        if stop_reason:
            print(f"[flash] stopping actor extraction: {stop_reason}")
            break
    save_checkpoint(state, cfg, checkpoint_path)
    print("[flash] checkpoint saved")


if __name__ == "__main__":
    main()
