from dataclasses import replace

import pytest
import torch

from ogpo import trainer
from ogpo.critic import assert_no_gradients
from ogpo.flash_ogpo import flash_ppo_loss
from ogpo.full_ogpo import full_chain_ais_ppo_loss, full_chain_ppo_loss
from ogpo.replay import make_synthetic_replay
from ogpo.trainer import (
    actor_delay_active,
    actor_start_gate,
    actor_guard_reason,
    awr_actor_update,
    build_train_state,
    critic_update,
    flash_actor_update,
    full_actor_update,
    sync_old_policy,
)
from ogpo.uncertainty import (
    actor_clip_for_uncertainty,
    kl_uncertainty_scale,
    state_adaptive_kl_penalty,
    state_entropy_weight,
)


def test_ppo_losses_are_finite():
    old_lp = torch.zeros(5, 3)
    new_lp = torch.zeros(5, 3)
    adv = torch.ones(5)
    full = full_chain_ppo_loss(new_lp, old_lp, adv, clip_eps=0.2)
    assert torch.isfinite(full.loss)
    flash = flash_ppo_loss(torch.zeros(5), torch.zeros(5), torch.ones(5), clip_eps=0.2)
    assert torch.isfinite(flash.loss)


def test_flash_timestep_sampling_is_seeded_and_step_varying():
    config = {"selected_timestep_distribution": "uniform"}
    first = trainer._select_flash_steps(
        config,
        batch_size=32,
        num_steps=8,
        device=torch.device("cpu"),
        seed=101,
    )
    repeated = trainer._select_flash_steps(
        config,
        batch_size=32,
        num_steps=8,
        device=torch.device("cpu"),
        seed=101,
    )
    next_step = trainer._select_flash_steps(
        config,
        batch_size=32,
        num_steps=8,
        device=torch.device("cpu"),
        seed=102,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_step)


def test_full_ppo_clip_can_vary_by_replay_state():
    old_lp = torch.zeros(2, 1)
    new_lp = torch.log(torch.tensor([[1.5], [1.5]]))
    result = full_chain_ppo_loss(
        new_lp,
        old_lp,
        torch.ones(2),
        clip_eps=torch.tensor([0.1, 0.5]),
    )

    assert torch.allclose(result.loss, torch.tensor(-1.3))
    assert result.clip_fraction == 0.5


def test_full_chain_ais_uses_one_joint_ratio_and_one_clip_per_chain():
    transition_ratios = torch.tensor([[1.1, 1.2], [0.9, 1.0]])
    old_log_probs = torch.zeros_like(transition_ratios)
    new_log_probs = transition_ratios.log()

    result = full_chain_ais_ppo_loss(
        new_log_probs,
        old_log_probs,
        torch.ones(2),
        clip_eps=0.1,
    )

    assert torch.allclose(torch.tensor(result.ratio_mean), torch.tensor((1.32 + 0.9) / 2))
    assert torch.allclose(result.loss, torch.tensor(-1.0))
    assert result.clip_fraction == 0.5


def test_reference_kl_penalty_is_weighted_per_state_group():
    penalty, raw_kl, beta = state_adaptive_kl_penalty(
        torch.ones(4),
        torch.tensor([0.0, 1.0]),
        group_size=2,
        beta_base=0.1,
        uncertainty_scale=2.0,
    )

    assert torch.allclose(penalty, torch.tensor(0.2))
    assert torch.allclose(raw_kl, torch.tensor(1.0))
    assert torch.allclose(beta, torch.tensor(0.2))


def test_ustate_actor_couplings_are_disabled_by_default():
    ustate = torch.tensor([0.0, 0.5, 1.0])
    actor_cfg = {"ppo_clip_min": 0.05, "ppo_clip_max": 0.2}
    regularization_cfg = {"kl_uncertainty_scale": 2.0}

    clip = actor_clip_for_uncertainty(ustate, actor_cfg, {})
    scale = kl_uncertainty_scale(regularization_cfg, {})

    assert torch.equal(clip, torch.full_like(ustate, 0.2))
    assert scale == 0.0


def test_ustate_actor_couplings_can_be_enabled_independently():
    ustate = torch.tensor([0.0, 0.5, 1.0])
    actor_cfg = {"ppo_clip_min": 0.05, "ppo_clip_max": 0.2}
    regularization_cfg = {"kl_uncertainty_scale": 2.0}

    adaptive_clip = actor_clip_for_uncertainty(
        ustate,
        actor_cfg,
        {"adapt_ppo_clip": True, "adapt_kl_beta": False},
    )
    fixed_scale = kl_uncertainty_scale(
        regularization_cfg,
        {"adapt_ppo_clip": True, "adapt_kl_beta": False},
    )
    fixed_clip = actor_clip_for_uncertainty(
        ustate,
        actor_cfg,
        {"adapt_ppo_clip": False, "adapt_kl_beta": True},
    )
    adaptive_scale = kl_uncertainty_scale(
        regularization_cfg,
        {"adapt_ppo_clip": False, "adapt_kl_beta": True},
    )

    assert torch.allclose(adaptive_clip, torch.tensor([0.2, 0.125, 0.05]))
    assert fixed_scale == 0.0
    assert torch.equal(fixed_clip, torch.full_like(ustate, 0.2))
    assert adaptive_scale == 2.0


def test_ustate_still_controls_wstate_when_actor_couplings_are_disabled():
    ustate = torch.tensor([0.0, 0.5, 1.0])

    wstate = state_entropy_weight(ustate, eta_entropy=0.5)

    assert torch.allclose(wstate, torch.exp(-0.5 * ustate))
    assert wstate[0] > wstate[1] > wstate[2]


def test_actor_backward_does_not_create_critic_or_reference_gradients():
    batch = make_synthetic_replay(num_samples=10, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"group_size": 2, "hidden_dim": 32},
        "flow": {"num_steps": 3, "selected_timestep": 1},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)
    metrics = full_actor_update(state, batch, cfg)
    assert metrics["ustate_adapt_ppo_clip"] == 0.0
    assert metrics["ustate_adapt_kl_beta"] == 0.0
    assert metrics["reference_kl_beta"] == pytest.approx(0.01)
    assert_no_gradients(state.critic, "critic")
    assert_no_gradients(state.reference_policy, "reference_policy")


def test_scalar_single_q_awr_updates_actor_with_weighted_flow_matching():
    batch = make_synthetic_replay(num_samples=10, generated_horizon=4, executed_horizon=2, action_dim=2)
    batch = replace(batch, mc_returns=batch.chunk_returns.clone())
    cfg = {
        "critic": {"ensemble_size": 1, "hidden_dim": 32, "num_layers": 1},
        "divl": {"enabled": False, "num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"algorithm": "awr", "hidden_dim": 32, "awr_temperature": 0.5, "awr_max_weight": 10.0},
        "flow": {"num_steps": 3},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)

    metrics = awr_actor_update(state, batch, cfg)

    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert metrics["awr_weight_max"] <= 10.0
    assert_no_gradients(state.critic, "critic")
    assert_no_gradients(state.reference_policy, "reference_policy")
    flash_metrics = flash_actor_update(state, batch, cfg)
    assert flash_metrics["ustate_adapt_ppo_clip"] == 0.0
    assert flash_metrics["ustate_adapt_kl_beta"] == 0.0
    assert flash_metrics["reference_kl_beta"] == pytest.approx(0.01)
    assert_no_gradients(state.critic, "critic")
    assert_no_gradients(state.reference_policy, "reference_policy")


def test_actor_update_reports_success_and_smoothness_losses_when_enabled():
    batch = make_synthetic_replay(num_samples=12, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"group_size": 2, "hidden_dim": 32},
        "flow": {"num_steps": 3, "selected_timestep": 1},
        "regularization": {"lambda_fm": 0.01, "lambda_success": 0.05, "lambda_smooth": 0.1},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)

    metrics = flash_actor_update(state, batch, cfg)

    assert metrics["success_buffer_loss"] >= 0.0
    assert metrics["action_smoothness"] >= 0.0


def test_actor_guard_reports_first_stop_reason():
    cfg = {
        "actor": {"max_policy_reference_kl": 0.5, "max_critic_disagreement": 2.0, "max_support_distance": 3.0}
    }

    reason = actor_guard_reason(
        {"reference_kl": 0.6, "candidate_ensemble_disagreement": 10.0, "support_distance_mean": 10.0},
        cfg,
    )

    assert reason == "policy_reference_kl_exceeded"


def test_actor_guard_defers_reference_kl_to_atomic_rejection_when_enabled():
    cfg = {
        "actor": {
            "max_policy_reference_kl": 0.5,
            "reject_update_on_kl": True,
        }
    }

    reason = actor_guard_reason({"reference_kl": 0.6}, cfg)

    assert reason is None


def test_actor_guard_stops_after_repeated_atomic_kl_rejections():
    cfg = {"actor": {"reject_update_on_kl": True, "max_consecutive_kl_rejections": 3}}

    reason = actor_guard_reason({"consecutive_kl_rejections": 3.0}, cfg)

    assert reason == "repeated_policy_reference_kl_rejections"


def test_actor_delay_active_uses_configured_step_boundary():
    cfg = {"actor": {"actor_delay": 2}}

    assert actor_delay_active(0, cfg)
    assert actor_delay_active(1, cfg)
    assert not actor_delay_active(2, cfg)


def test_actor_start_gate_checks_warmup_calibration_and_force_override():
    batch = make_synthetic_replay(num_samples=16, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {
            "ensemble_size": 2,
            "hidden_dim": 32,
            "num_layers": 1,
            "warmup_steps": 2,
            "min_ranking_accuracy": 1.1,
        },
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"hidden_dim": 32},
        "flow": {"num_steps": 3},
    }
    state = build_train_state(cfg, batch)

    reason, _ = actor_start_gate(state, batch, cfg, outer_step=0)
    assert reason == "critic_warmup"

    critic_update(state, batch, cfg)
    critic_update(state, batch, cfg)
    reason, metrics = actor_start_gate(state, batch, cfg, outer_step=1)
    assert reason == "critic_ranking_accuracy_below_min"
    assert "pairwise_ranking_accuracy" in metrics

    cfg["critic"]["force_actor"] = True
    reason, _ = actor_start_gate(state, batch, cfg, outer_step=1)
    assert reason is None


def test_flash_rectification_modes_are_configurable():
    batch = make_synthetic_replay(num_samples=12, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"group_size": 2, "hidden_dim": 32},
        "flow": {
            "num_steps": 3,
            "selected_timestep_distribution": "fixed",
            "selected_timestep": 1,
            "temporal_rectification_mode": "none",
        },
        "regularization": {"lambda_fm": 0.01},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)

    none_metrics = flash_actor_update(state, batch, cfg)

    assert none_metrics["rectification_weight"] == 1.0

    cfg["flow"]["temporal_rectification_mode"] = "empirical_ema"
    empirical_metrics = flash_actor_update(state, batch, cfg)

    assert empirical_metrics["selected_step_count_1"] == batch.batch_size
    assert empirical_metrics["rectifier_count_1"] >= batch.batch_size
    assert torch.isfinite(torch.tensor(empirical_metrics["flash_raw_loss_step_1"]))
    assert torch.isfinite(torch.tensor(empirical_metrics["flash_raw_grad_norm_step_1"]))
    assert torch.isfinite(torch.tensor(empirical_metrics["flash_rectified_grad_norm_step_1"]))


def test_old_policy_sync_supports_ema():
    batch = make_synthetic_replay(num_samples=4, generated_horizon=3, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {"hidden_dim": 16},
        "flow": {"num_steps": 2},
    }
    state = build_train_state(cfg, batch)
    old_before = next(state.old_policy.parameters()).detach().clone()
    with torch.no_grad():
        for param in state.policy.parameters():
            param.add_(1.0)
    policy_now = next(state.policy.parameters()).detach().clone()

    sync_old_policy(state, ema=0.5)

    old_after = next(state.old_policy.parameters()).detach()
    assert torch.allclose(old_after, 0.5 * old_before + 0.5 * policy_now)


def test_actor_epochs_reuse_one_flash_rollout_and_take_multiple_steps(monkeypatch):
    batch = make_synthetic_replay(num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"group_size": 2, "hidden_dim": 32, "actor_epochs_per_rollout": 2},
        "flow": {"num_steps": 3, "selected_timestep": 1},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)
    rollout_calls = 0
    optimizer_steps = 0
    original_rollout = state.old_policy.rollout
    original_step = state.actor_optimizer.step

    def counted_rollout(*args, **kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        return original_rollout(*args, **kwargs)

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(state.old_policy, "rollout", counted_rollout)
    monkeypatch.setattr(state.actor_optimizer, "step", counted_step)

    metrics = flash_actor_update(state, batch, cfg)

    assert rollout_calls == 1
    assert optimizer_steps == 2
    assert metrics["actor_epochs"] == 2.0
