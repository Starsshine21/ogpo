import torch
from dataclasses import replace
import pytest

from ogpo.critic import ScalarQEnsemble, clone_target
from ogpo.ensemble import bootstrap_mask
from ogpo.evaluator import (
    fit_conformal_calibration,
    offline_calibration_metrics,
    validation_metrics_for_training,
)
from ogpo.replay import make_synthetic_replay
from ogpo.temporal_rectification import EmpiricalGradientRectifier, analytic_rectification
from ogpo.trainer import build_train_state, conservative_advantages_for_candidates, critic_update
from ogpo.uncertainty import conformal_scale, support_weight
from ogpo.flow_sde import GaussianFlowPolicy


def test_target_network_has_no_gradients_enabled():
    policy = GaussianFlowPolicy(condition_dim=3, action_dim=2)
    target = clone_target(policy)
    assert all(not param.requires_grad for param in target.parameters())


def test_bootstrap_mask_can_create_member_differences():
    mask = bootstrap_mask(ensemble_size=5, batch_size=32, probability=0.5)
    assert mask.shape == (5, 32)
    assert mask.float().std() > 0.0


def test_randomized_prior_is_independent_and_frozen_per_q_member():
    critic = ScalarQEnsemble(
        ensemble_size=2,
        obs_dim=3,
        generated_horizon=2,
        action_dim=2,
        hidden_dim=8,
        num_layers=1,
        randomized_prior_scale=1.0,
    )
    critic.members[1].net.load_state_dict(critic.members[0].net.state_dict())
    observations = torch.randn(5, 3)
    actions = torch.randn(5, 2, 2)
    masks = torch.ones(5, 2, dtype=torch.bool)

    values = critic(observations, actions, masks)
    values.sum().backward()

    assert not torch.allclose(values[0], values[1])
    assert all(not parameter.requires_grad for member in critic.members for parameter in member.prior.parameters())
    assert all(parameter.grad is None for member in critic.members for parameter in member.prior.parameters())


def test_unsupported_pi_feature_sharing_is_not_silently_ignored():
    batch = make_synthetic_replay(num_samples=4)
    cfg = {
        "critic": {"shared_frozen_encoder": True},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
    }

    with pytest.raises(ValueError, match="feature_source"):
        build_train_state(cfg, batch)


def test_conformal_scale_and_support_weight_are_finite():
    q_mean = torch.tensor([0.0, 1.0, 2.0, 3.0])
    q_std = torch.ones(4) * 0.5
    returns = torch.tensor([0.0, 1.5, 1.0, 4.0])
    scale = conformal_scale(q_mean, q_std, returns, coverage_delta=0.25, min_samples=2)
    assert scale >= 1.0
    weights = support_weight(q_std, torch.tensor([0.0, 1.0, 20.0, 2.0]), lambda_epi=0.1, lambda_support=0.1, support_threshold=10.0)
    assert torch.isfinite(weights).all()
    assert weights[2] == 0.0


def test_temporal_rectification_is_finite():
    w = analytic_rectification(
        torch.tensor([[0.25], [0.5], [0.75]]),
        stochastic_variance=0.01,
        sde_mode="ogpo_corrected",
        clip_min=0.25,
        clip_max=4.0,
    )
    assert torch.isfinite(w).all()
    assert w.mean() == pytest.approx(1.0)
    assert torch.all(w[1:] > w[:-1])
    constant = analytic_rectification(
        torch.tensor([[0.25], [0.5], [0.75]]),
        stochastic_variance=0.01,
        sde_mode="gaussian_adapter",
        clip_min=0.25,
        clip_max=4.0,
    )
    assert torch.equal(constant, torch.ones_like(constant))
    rectifier = EmpiricalGradientRectifier(num_steps=4)
    rectifier.update(2, 3.0)
    weight = rectifier.weight(2)
    assert torch.isfinite(weight)


def test_reference_value_baseline_is_reported_when_enabled():
    batch = make_synthetic_replay(num_samples=10, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {
            "ensemble_size": 2,
            "hidden_dim": 32,
            "num_layers": 1,
            "reference_value_samples": 2,
            "lambda_divl_target": 0.5,
        },
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"hidden_dim": 32},
        "flow": {"num_steps": 3},
    }
    state = build_train_state(cfg, batch)

    metrics = critic_update(state, batch, cfg)

    assert "reference_value_mean" in metrics
    assert metrics["lambda_divl_target"] == 0.5


def test_target_update_period_is_honored():
    batch = make_synthetic_replay(num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {
            "ensemble_size": 2,
            "hidden_dim": 32,
            "num_layers": 1,
            "target_tau": 1.0,
            "target_update_period": 2,
        },
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"hidden_dim": 32},
        "flow": {"num_steps": 3},
    }
    state = build_train_state(cfg, batch)
    before = [parameter.detach().clone() for parameter in state.target_critic.parameters()]

    first_metrics = critic_update(state, batch, cfg)
    after_first = [parameter.detach().clone() for parameter in state.target_critic.parameters()]
    second_metrics = critic_update(state, batch, cfg)
    after_second = [parameter.detach().clone() for parameter in state.target_critic.parameters()]

    assert all(torch.equal(left, right) for left, right in zip(before, after_first, strict=True))
    assert any(not torch.equal(left, right) for left, right in zip(after_first, after_second, strict=True))
    assert first_metrics["target_updated"] == 0.0
    assert second_metrics["target_updated"] == 1.0


def test_no_divl_critic_uses_fixed_monte_carlo_returns():
    batch = make_synthetic_replay(num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2)
    mc_returns = torch.linspace(-1.0, 1.0, batch.batch_size)
    batch = replace(batch, mc_returns=mc_returns)
    cfg = {
        "critic": {"ensemble_size": 1, "hidden_dim": 16, "num_layers": 1},
        "divl": {"enabled": False, "num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {"hidden_dim": 16, "advantage_mode": "scalar_q"},
        "flow": {"num_steps": 2},
    }
    state = build_train_state(cfg, batch)

    metrics = critic_update(state, batch, cfg)

    assert metrics["divl_enabled"] == 0.0
    assert metrics["divl_loss"] == 0.0
    assert abs(metrics["target_mean"] - float(mc_returns.mean())) < 1e-6


def test_no_divl_scalar_q_advantage_uses_replay_action_baseline():
    batch = make_synthetic_replay(num_samples=6, generated_horizon=4, executed_horizon=2, action_dim=2)
    batch = replace(batch, mc_returns=batch.chunk_returns.clone())
    cfg = {
        "critic": {"ensemble_size": 1, "hidden_dim": 16, "num_layers": 1},
        "divl": {"enabled": False, "num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {"hidden_dim": 16, "advantage_mode": "scalar_q"},
        "flow": {"num_steps": 2},
    }
    state = build_train_state(cfg, batch)
    candidates = batch.action_chunks.reshape(batch.batch_size, 1, -1)

    advantages, metrics = conservative_advantages_for_candidates(
        state, batch.observations, candidates, batch, cfg
    )

    assert torch.count_nonzero(advantages) == 0
    assert metrics["state_entropy"] == 0.0


def test_support_gate_zeroes_far_candidate_advantages():
    batch = make_synthetic_replay(num_samples=6, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"hidden_dim": 32, "advantage_clip": 5.0},
        "flow": {"num_steps": 3},
        "uncertainty": {
            "use_support_weight": True,
            "lambda_support": 0.0,
            "lambda_epi": 0.0,
            "support_threshold": 0.01,
        },
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)
    far = batch.action_chunks.reshape(batch.batch_size, 1, -1) + 100.0

    advantages, metrics = conservative_advantages_for_candidates(state, batch.observations, far, batch, cfg)

    assert torch.count_nonzero(advantages) == 0
    assert metrics["support_weight_mean"] == 0.0


def test_entropy_skip_gate_zeroes_state_advantages():
    batch = make_synthetic_replay(num_samples=6, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"hidden_dim": 32, "advantage_clip": 5.0},
        "flow": {"num_steps": 3},
        "uncertainty": {"entropy_skip_threshold": 0.0, "consensus_skip_threshold": 0.0},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)
    candidates = batch.action_chunks.reshape(batch.batch_size, 1, -1)

    advantages, metrics = conservative_advantages_for_candidates(state, batch.observations, candidates, batch, cfg)

    assert torch.count_nonzero(advantages) == 0
    assert metrics["state_skip_fraction"] == 1.0


def test_advantage_ablation_modes_are_distinct_and_finite():
    batch = make_synthetic_replay(num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 3, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"group_size": 2, "hidden_dim": 32},
        "flow": {"num_steps": 3},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)
    candidates = torch.randn(batch.batch_size, 2, batch.generated_horizon * batch.action_dim)
    outputs = []
    for mode in ("sign_consensus", "lcb", "group_normalization", "scalar_q"):
        cfg["actor"]["advantage_mode"] = mode
        advantage, metrics = conservative_advantages_for_candidates(
            state, batch.observations, candidates, batch, cfg
        )
        assert torch.isfinite(advantage).all()
        assert metrics["advantage_mode"] == mode
        outputs.append(advantage)
    assert not torch.allclose(outputs[0], outputs[2])


def test_calibration_reports_required_metrics_and_updates_conformal_scale():
    batch = make_synthetic_replay(num_samples=24, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 3, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"hidden_dim": 32},
        "flow": {"num_steps": 3},
        "uncertainty": {"conformal_delta": 0.2, "min_calibration_samples": 8},
    }
    state = build_train_state(cfg, batch)
    critic_update(state, batch, cfg)

    scale = fit_conformal_calibration(state, batch, cfg)
    metrics = offline_calibration_metrics(
        state.critic,
        batch,
        divl=state.divl,
        conformal_scale=scale,
    )

    assert state.conformal_scale == scale
    for key in (
        "q_rank_correlation",
        "pairwise_ranking_accuracy",
        "disagreement_error_correlation",
        "interval_coverage",
        "expected_calibration_error",
        "q_exploitation_gap",
        "categorical_entropy",
    ):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))


def test_calibration_prefers_mc_returns_and_reports_ensemble_sign_disagreement():
    class FixedCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, observations, action_chunks, execution_masks):
            del action_chunks, execution_masks
            low = torch.zeros(observations.shape[0], device=observations.device) + self.anchor
            high = torch.full_like(low, 2.0) + self.anchor
            return torch.stack([low, high])

    batch = make_synthetic_replay(
        num_samples=6,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    batch = replace(
        batch,
        chunk_returns=torch.zeros_like(batch.chunk_returns),
        mc_returns=torch.ones_like(batch.chunk_returns),
    )

    metrics = offline_calibration_metrics(FixedCritic(), batch)

    assert metrics["calibration_target_is_mc"] == 1.0
    assert metrics["q_exploitation_gap"] == pytest.approx(0.0)
    assert metrics["ensemble_sign_disagreement"] == 1.0


def test_training_validation_metrics_are_namespaced_and_can_fit_conformal_scale():
    batch = make_synthetic_replay(
        num_samples=24,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    cfg = {
        "critic": {"ensemble_size": 3, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"hidden_dim": 32},
        "flow": {"num_steps": 3},
        "uncertainty": {
            "use_conformal": True,
            "conformal_delta": 0.2,
            "min_calibration_samples": 8,
        },
    }
    state = build_train_state(cfg, batch)

    metrics = validation_metrics_for_training(state, batch, cfg)

    assert "validation_q_rmse" in metrics
    assert "validation_pairwise_ranking_accuracy" in metrics
    assert "validation_ensemble_sign_disagreement" in metrics
    assert "validation_interval_coverage" in metrics
    assert "validation_q_exploitation_gap" in metrics
    assert metrics["validation_conformal_scale"] == state.conformal_scale
