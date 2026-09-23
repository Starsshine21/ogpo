from __future__ import annotations

import math

import pytest
import torch

from ogpo import trainer
from ogpo.chi2_regularization import (
    chi2_ppo_upper_bound,
    chi2_pessimistic_advantage,
    selected_transition_chi2_ratio,
)
from ogpo.replay import make_synthetic_replay
from ogpo.trainer import (
    build_train_state,
    critic_update,
    flash_actor_update,
    full_actor_update,
    load_checkpoint,
    save_checkpoint,
    slow_policy_update_due,
    sync_slow_policy,
)


def _config(*, variant: str = "chi2") -> dict:
    return {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {
            "group_size": 2,
            "candidate_group_size": 2,
            "hidden_dim": 32,
            "ogpo_variant": variant,
            "actor_epochs_per_rollout": 1,
            "max_grad_norm": 1.0,
            "chi2": {
                "ratio_scope": "selected_transition",
                "require_single_epoch": True,
                "normalize_action_dim": True,
                "logprob_microbatch_size": 3,
                "log_ratio_clip": 3.0,
                "ratio_max": 4.0,
                "beta_base": 0.25,
                "q_std_target": 1.0,
                "ensemble_alpha": 5.0,
                "r_max": 10.0,
                "normalize_group": False,
                "slow_policy_tau": 0.5,
                "slow_policy_update_period": 1,
                "apply_project_safety_gates": False,
            },
        },
        "flow": {
            "num_steps": 3,
            "selected_timestep_distribution": "fixed",
            "selected_timestep": 1,
            "stochastic_variance": 0.04,
        },
        "regularization": {"beta_kl": 0.0, "lambda_fm": 0.0, "lambda_success": 0.0},
    }


def test_selected_transition_ratio_normalizes_action_dimension_and_detaches():
    current = torch.tensor([math.log(4.0), math.log(0.25)], requires_grad=True)
    slow = torch.zeros_like(current)
    ratio, stats = selected_transition_chi2_ratio(
        current,
        slow,
        event_dim=2,
        normalize_action_dim=True,
        log_ratio_clip=10.0,
        ratio_max=4.0,
    )

    assert torch.allclose(ratio, torch.tensor([2.0, 0.5]))
    assert stats.logprob_normalizer == 2.0
    assert not ratio.requires_grad


def test_chipo_blends_q_mean_to_q_min_and_group_centers_advantage():
    q_values = torch.tensor(
        [
            [[2.0, 4.0]],
            [[0.0, 2.0]],
        ]
    )  # [ensemble=2, batch=1, candidate=2]
    ratio = torch.tensor([[1.0, 2.0]])
    advantage, stats = chi2_pessimistic_advantage(
        q_values,
        ratio,
        beta_base=1.0,
        q_std_target=1.0,
        ensemble_alpha=10.0,
        normalize_group=False,
        advantage_clip=None,
    )

    assert torch.allclose(advantage.mean(dim=-1), torch.zeros(1), atol=1e-6)
    assert stats.pessimism_weight_mean > 0.5
    assert stats.beta == pytest.approx(1.0)
    upper = chi2_ppo_upper_bound(torch.tensor([0.2]), beta=2.0, r_max=1.0)
    assert torch.allclose(upper, torch.tensor([1.2]))


def test_ca_default_does_not_allocate_slow_policy():
    batch = make_synthetic_replay(
        num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2
    )
    state = build_train_state(_config(variant="ca"), batch)
    assert state.slow_policy is None


@pytest.mark.parametrize('variant', ['chi2', 'ca_chi2'])
def test_flash_chipo_uses_selected_transition_and_reports_metrics(variant):
    batch = make_synthetic_replay(
        num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2
    )
    cfg = _config(variant=variant)
    state = build_train_state(cfg, batch)
    assert state.slow_policy is not None
    critic_update(state, batch, cfg)

    metrics = flash_actor_update(state, batch, cfg)

    assert metrics["ogpo_variant"] == variant
    assert metrics["advantage_mode"] == ('chi_po' if variant == 'chi2' else 'conservative')
    assert metrics["chi2_enabled"] == 1.0
    assert metrics["chi2_selected_logprob_normalizer"] == 8.0
    assert metrics["chi2_ratio_max"] <= 4.0
    assert metrics["actor_update_accepted"] == 1.0
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert all(parameter.grad is None for parameter in state.slow_policy.parameters())


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four policy-role GPUs')
def test_flash_ca_chipo_four_device_roles():
    batch = make_synthetic_replay(num_samples=4, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = _config(variant='ca_chi2')
    cfg['actor'].update(compute_step_grad_diagnostics=False, gradient_microbatch_size=2)
    state = build_train_state(cfg, batch, device='cuda:0')
    state.old_policy.to('cuda:1')
    state.slow_policy.to('cuda:2')
    state.reference_policy.to('cuda:3')
    metrics = flash_actor_update(state, batch, cfg)
    assert metrics['chi2_enabled'] == 1
    assert metrics['actor_update_accepted'] == 1
    assert math.isfinite(metrics['post_update_reference_kl'])
    assert all(p.grad is None for role in (state.old_policy, state.slow_policy, state.reference_policy) for p in role.parameters())


def test_chipo_full_actor_path_uses_joint_ratio():
    batch = make_synthetic_replay(
        num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2
    )
    cfg = _config()
    cfg["actor"]["chi2"]["ratio_scope"] = "full_chain"
    cfg["actor"]["full_ratio_mode"] = "ais_joint"
    cfg["actor"]["normalize_logprob_by_action_dim"] = True
    cfg["actor"]["normalize_logprob_by_denoising_steps"] = True
    metrics = full_actor_update(build_train_state(cfg, batch), batch, cfg)
    assert metrics["full_chain_joint_ratio"] == 1.0
    assert metrics["full_chain_one_clip"] == 1.0
    assert metrics["chi2_enabled"] == 1.0


def test_post_update_kl_cpu_rollback_restores_actor(monkeypatch):
    batch = make_synthetic_replay(
        num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2
    )
    cfg = _config()
    cfg["actor"].update(
        reject_update_on_kl=True,
        max_policy_reference_kl=0.1,
        post_update_kl_action="rollback_cpu",
    )
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)
    before = {name: value.detach().clone() for name, value in state.policy.state_dict().items()}
    monkeypatch.setattr(trainer, "_selected_transition_reference_kl", lambda *args, **kwargs: 1.0)

    metrics = flash_actor_update(state, batch, cfg)

    assert metrics["actor_update_rejected"] == 1.0
    assert metrics["actor_update_accepted"] == 0.0
    assert metrics["post_update_kl_exceeded"] == 1.0
    for name, value in state.policy.state_dict().items():
        assert torch.allclose(value, before[name])


def test_chipo_slow_policy_ema_and_checkpoint_round_trip(tmp_path):
    batch = make_synthetic_replay(
        num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2
    )
    cfg = _config()
    state = build_train_state(cfg, batch)
    assert state.slow_policy is not None
    before = {
        name: value.detach().clone()
        for name, value in state.slow_policy.state_dict().items()
    }
    with torch.no_grad():
        for parameter in state.policy.parameters():
            parameter.add_(1.0)
    policy_now = {
        name: value.detach().clone() for name, value in state.policy.state_dict().items()
    }

    # A rejected/rolled-back actor transaction does not call sync_slow_policy,
    # so the independent slow snapshot remains unchanged.
    for name, value in state.slow_policy.state_dict().items():
        assert torch.allclose(value, before[name])

    sync_slow_policy(state, ema=0.5)
    for name, value in state.slow_policy.state_dict().items():
        assert torch.allclose(value, 0.5 * before[name] + 0.5 * policy_now[name])

    checkpoint = tmp_path / "chi2.pt"
    state.actor_step = 17
    state.accepted_actor_updates = 13
    save_checkpoint(state, cfg, checkpoint)
    restored = build_train_state(cfg, batch)
    payload = load_checkpoint(checkpoint, restored)
    assert payload["chi2_slow_policy_restored"] is True
    assert restored.actor_step == 17
    assert restored.accepted_actor_updates == 13
    assert restored.slow_policy is not None
    assert restored.old_policy is not restored.slow_policy
    assert restored.policy is not restored.old_policy
    for name, value in state.slow_policy.state_dict().items():
        assert torch.allclose(value, restored.slow_policy.state_dict()[name])


def test_official_tau_slow_and_accepted_update_period_semantics():
    batch = make_synthetic_replay(
        num_samples=4, generated_horizon=3, executed_horizon=2, action_dim=2
    )
    cfg = _config()
    cfg["actor"]["chi2"]["slow_policy_tau"] = 0.0005
    state = build_train_state(cfg, batch)
    assert state.slow_policy is not None
    with torch.no_grad():
        for parameter in state.policy.parameters():
            parameter.fill_(1.0)
        for parameter in state.slow_policy.parameters():
            parameter.zero_()
    sync_slow_policy(state, ema=0.9995)
    first = next(state.slow_policy.parameters()).detach().clone()
    assert torch.allclose(first, torch.full_like(first, 0.0005), atol=1e-7)
    sync_slow_policy(state, ema=0.9995)
    second = next(state.slow_policy.parameters()).detach().clone()
    assert torch.allclose(second, torch.full_like(second, 0.00099975), atol=1e-7)
    assert slow_policy_update_due(
        accepted_actor_updates=1, update_period=1, update_accepted=True
    )
    assert not slow_policy_update_due(
        accepted_actor_updates=1, update_period=1, update_accepted=False
    )
    assert not slow_policy_update_due(
        accepted_actor_updates=1, update_period=2, update_accepted=True
    )
    assert slow_policy_update_due(
        accepted_actor_updates=2, update_period=2, update_accepted=True
    )
