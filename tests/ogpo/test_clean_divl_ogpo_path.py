from dataclasses import replace
from pathlib import Path
import sys

import torch
import torch.nn as nn
import pytest
from ogpo import trainer

from ogpo.chi2_regularization import (
    apply_chipo_to_ca_advantage,
    compute_chipo_beta,
    full_chain_chipo_ratio,
    full_chain_chipo_regularization_loss,
    sign_safe_advantage_intersection,
)
from ogpo.conservative_advantage import (
    _safe_max,
    group_relative_conservative_advantage,
)
from ogpo.distributional_value import categorical_quantile
from ogpo.full_ogpo import full_chain_ais_ppo_loss
from ogpo.full_ogpo import full_chain_chipo_ppo_loss
from ogpo.multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic
from ogpo.replay import make_n_step_replay, make_synthetic_replay
from ogpo.trainer import (
    build_train_state,
    critic_update,
    full_actor_update,
    load_critic_checkpoint,
    save_checkpoint,
    freeze_critic_for_actor,
)
from ogpo.evaluator import offline_calibration_metrics
from ogpo.divl import divl_quantile_values
from ogpo.divl import divl_clipped_double_q_projection_targets
from ogpo.flow_sde import GaussianFlowPolicy


# The project scripts are intentionally executable modules rather than a
# package.  Add their directory explicitly so config-loader tests work when
# pytest is launched from the repository root (the training entrypoint does
# the same path setup at runtime).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_clean_main_validation_rejects_flash_or_nonjoint_ratio():
    batch = make_synthetic_replay(num_samples=4)
    config = {
        "training": {"clean_pytorch_main": True},
        "critic": {"architecture": "mlp"},
        "actor": {
            "flash_enabled": True,
            "full_ratio_mode": "per_transition",
            "advantage_mode": "conservative",
            "ogpo_variant": "ca_chi2",
            "normalize_logprob_by_action_dim": False,
            "normalize_logprob_by_denoising_steps": False,
        },
        "flow": {"adapter": "gaussian_openpi", "sde_mode": "gaussian_adapter"},
        "offline": {"critic_update_during_actor": True},
    }
    with pytest.raises(ValueError, match="clean PyTorch main-path"):
        build_train_state(config, batch)


def test_full_chain_identity_ratio_is_one_and_unclipped():
    old = torch.randn(4, 5)
    result = full_chain_ais_ppo_loss(old, old, torch.ones(4), clip_eps=0.01)
    assert torch.allclose(result.loss, torch.tensor(-1.0))
    assert result.ratio_mean == 1.0
    assert result.clip_fraction == 0.0
    assert result.log_ratio_mean == 0.0


def test_corrected_sde_rollout_revaluation_has_identity_ratio():
    policy = GaussianFlowPolicy(
        condition_dim=3,
        action_dim=4,
        hidden_dim=8,
        num_steps=4,
        stochastic_variance=0.04,
        sde_mode="ogpo_corrected",
    )
    condition = torch.randn(2, 3)
    rollout = policy.rollout(condition, group_size=1)
    revaluated = torch.stack(
        [
            policy.log_prob(
                rollout.next_states[:, step],
                rollout.states[:, step],
                condition,
                rollout.timesteps[:, step],
            )
            for step in range(policy.num_steps)
        ],
        dim=1,
    )
    result = full_chain_ais_ppo_loss(
        revaluated,
        rollout.log_probs,
        torch.ones(2),
        clip_eps=0.01,
    )
    assert torch.allclose(result.ratio, torch.ones(2), atol=1e-5)
    assert result.clip_fraction == 0.0


def test_constant_corrected_sde_schedule_and_sampler_logprob_parity():
    policy = GaussianFlowPolicy(
        condition_dim=3,
        action_dim=4,
        hidden_dim=8,
        num_steps=4,
        sde_mode="ogpo_constant_corrected",
        constant_noise_std=0.005,
        learn_sde_std=False,
    )
    condition = torch.randn(2, 3)
    rollout = policy.rollout(condition)
    stds = torch.stack(
        [
            policy.transition_std(rollout.states[:, step], rollout.timesteps[:, step])
            for step in range(policy.num_steps)
        ],
        dim=1,
    )
    assert torch.allclose(stds[:, :-1], torch.full_like(stds[:, :-1], 0.005), atol=1e-6)
    assert torch.equal(stds[:, -1], torch.zeros_like(stds[:, -1]))
    generator = torch.Generator().manual_seed(123)
    before_rng = generator.get_state().clone()
    final_sample, final_log_prob = policy.sample_transition(
        rollout.states[:, -1],
        condition,
        rollout.timesteps[:, -1],
        generator=generator,
    )
    final_mean = policy.transition_mean(
        rollout.states[:, -1], condition, rollout.timesteps[:, -1]
    )
    assert torch.equal(generator.get_state(), before_rng)
    assert torch.allclose(final_sample, final_mean)
    assert torch.equal(final_log_prob, torch.zeros(2))
    reevaluated = torch.stack(
        [
            policy.log_prob(
                rollout.next_states[:, step],
                rollout.states[:, step],
                condition,
                rollout.timesteps[:, step],
            )
            for step in range(policy.num_steps)
        ],
        dim=1,
    )
    assert torch.allclose(reevaluated, rollout.log_probs, atol=1e-6)
    assert torch.equal(rollout.log_probs[:, -1], torch.zeros(2))
    identity = full_chain_ais_ppo_loss(
        reevaluated,
        rollout.log_probs,
        torch.ones(2),
        clip_eps=0.01,
        logprob_normalizer=policy.action_dim * policy.num_steps,
    )
    assert torch.allclose(identity.ratio, torch.ones(2), atol=1e-6)
    assert identity.clip_fraction == 0.0
    assert identity.log_ratio_mean == pytest.approx(0.0)
    assert torch.equal(
        policy.kl_to(
            policy,
            rollout.states[:, -1],
            condition,
            rollout.timesteps[:, -1],
        ),
        torch.zeros(2),
    )
    assert all(name != "log_std" for name, _ in policy.named_parameters())


def test_native_transition_excludes_constant_sde_correction():
    policy = GaussianFlowPolicy(
        condition_dim=3,
        action_dim=4,
        hidden_dim=8,
        num_steps=4,
        sde_mode="ogpo_constant_corrected",
        constant_noise_std=0.005,
        learn_sde_std=False,
    )
    condition = torch.randn(2, 3)
    x_t = torch.randn(2, 4)
    timestep = torch.full((2, 1), 0.5)
    native = policy.native_transition_mean(x_t, condition, timestep)
    raw = policy.flow_spec.euler_step(
        x_t, policy.predict_velocity(x_t, condition, timestep)
    )
    corrected = policy.transition_mean(x_t, condition, timestep)
    raw_velocity = policy.predict_velocity(x_t, condition, timestep)
    expected_velocity = raw_velocity + 0.5 * (0.005**2) * (
        ((1.0 - timestep) * raw_velocity + x_t) / timestep
    )
    expected_corrected = policy.flow_spec.euler_step(x_t, expected_velocity)
    assert torch.allclose(native, raw)
    assert torch.allclose(corrected, expected_corrected)
    assert (corrected - native).abs().max() > 0.0

    final_timestep = torch.full((2, 1), 1.0 / policy.num_steps)
    assert torch.allclose(
        policy.transition_mean(x_t, condition, final_timestep),
        policy.native_transition_mean(x_t, condition, final_timestep),
    )


def test_constant_sde_uses_explicit_local_to_ogpo_time_mapping():
    policy = GaussianFlowPolicy(
        condition_dim=3,
        action_dim=4,
        hidden_dim=8,
        num_steps=4,
        sde_mode="ogpo_constant_corrected",
        constant_noise_std=0.005,
        learn_sde_std=False,
    )
    local = torch.tensor([[1.0], [0.25]])
    assert torch.allclose(policy.ogpo_time(local), torch.tensor([[0.0], [0.75]]))


def test_full_chain_ratio_sums_log_ratios_before_normalization():
    old = torch.zeros(2, 3)
    new = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]])
    result = full_chain_ais_ppo_loss(
        new,
        old,
        torch.ones(2),
        clip_eps=10.0,
        logprob_normalizer=2.0,
    )
    expected = torch.exp(torch.tensor([(0.3) / 2.0, (0.7) / 2.0]))
    assert torch.allclose(result.ratio, expected)


def test_current_equals_slow_has_identity_chipo_ratio():
    logs = torch.randn(3, 4)
    ratio, stats = full_chain_chipo_ratio(
        logs,
        logs.clone(),
        action_dim=6,
        normalize_logprob_by_action_dim=True,
        normalize_logprob_by_denoising_steps=True,
    )
    assert torch.allclose(ratio, torch.ones(3))
    assert stats.log_ratio_mean == pytest.approx(0.0)


def test_chipo_ppo_uses_official_asymmetric_upper_bound():
    result = full_chain_chipo_ppo_loss(
        torch.tensor([[0.2, 0.0], [0.0, 0.0]]),
        torch.zeros(2, 2),
        torch.tensor([1.0, -1.0]),
        clip_eps=0.01,
        beta=2.0,
        r_max=0.001,
    )
    # min(1.01, 1 + .001/2) = 1.0005 exercises the ChiPO-specific bound.
    assert result.clipped_ratio is not None
    assert result.clipped_ratio[0] == pytest.approx(1.0005)
    assert result.clip_fraction == pytest.approx(0.5)


def test_memberwise_conservative_advantage_uses_group_mean_and_sign_consensus():
    q = torch.tensor(
        [
            [[2.0, 4.0, 3.0, 1.0]],
            [[1.0, 3.0, 2.0, 0.0]],
            [[3.0, 5.0, 4.0, 2.0]],
        ]
    )
    advantage, member, stats = group_relative_conservative_advantage(q)
    # Candidate 1 is unanimously positive; candidate 3 is unanimously
    # negative.  The conservative values are min-positive/max-negative.
    assert torch.allclose(advantage, torch.tensor([[-0.5, 1.5, 0.5, -1.5]]))
    assert member.shape == q.shape
    assert stats.positive_consensus_ratio == 0.5
    assert stats.negative_consensus_ratio == 0.5


def test_ca_synthetic_positive_negative_and_disagreement_cases():
    positive = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    negative = torch.tensor([[[-1.0]], [[-2.0]], [[-3.0]]])
    mixed = torch.tensor([[[1.0]], [[-1.0]], [[2.0]]])
    for values, expected in ((positive, 1.0), (negative, -1.0), (mixed, 0.0)):
        result, _, _ = group_relative_conservative_advantage(values)
        # A singleton group is centered to zero; use the direct consensus
        # helper for the hand-constructed sign cases instead.
        from ogpo.conservative_advantage import sign_consensus_advantage

        direct, _ = sign_consensus_advantage(values, torch.zeros(3, 1))
        assert direct.item() == expected


def test_safe_max_matches_upstream_sign_semantics():
    values = torch.tensor([[1.0, -1.0, 1.0], [2.0, -2.0, -1.0], [3.0, -3.0, 2.0]])
    assert torch.equal(_safe_max(values, dim=0), torch.tensor([1.0, -1.0, 0.0]))


def test_double_q_core_exposes_two_heads_and_memberwise_min():
    core = MultiHeadUdivlCore(
        state_dim=4,
        action_dim=2,
        max_horizon=3,
        action_hidden_dim=4,
        head_hidden_dim=8,
        num_attention_heads=2,
        num_value_atoms=7,
        num_pairs=3,
        q_representation="scalar",
        q_heads_per_member=2,
    )
    readout = torch.randn(5, 4)
    actions = torch.randn(5, 3, 2)
    mask = torch.ones(5, 3, dtype=torch.bool)
    pairs = core.q_pair_from_readout(readout, actions, mask)
    clipped = core.clipped_q_from_readout(readout, actions, mask)
    assert pairs.shape == (3, 2, 5)
    assert torch.equal(clipped, pairs.min(dim=1).values)


def test_divl_v_target_projects_memberwise_clipped_double_q():
    pairs = torch.tensor([[[0.2, 0.8], [0.6, 0.1]], [[0.4, 0.3], [0.5, 0.9]]])
    support = torch.linspace(-0.1, 1.1, 7)
    projected = divl_clipped_double_q_projection_targets(pairs, support)
    assert projected.shape == (2, 2, 7)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(2, 2))


def test_double_q_divl_update_uses_memberwise_targets():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 8)

        def forward(self, batch, *, next_observation=False):
            values = batch.next_observations if next_observation else batch.observations
            return self.projection(values)

    batch = make_synthetic_replay(
        num_samples=8,
        obs_dim=3,
        proprio_dim=3,
        generated_horizon=3,
        executed_horizon=2,
        action_dim=2,
    )

    def factory(_, __):
        return MultiHeadUdivlCritic(
            Encoder(),
            MultiHeadUdivlCore(
                state_dim=8,
                action_dim=2,
                max_horizon=3,
                action_hidden_dim=8,
                head_hidden_dim=16,
                num_attention_heads=2,
                num_value_atoms=11,
                num_pairs=3,
                q_representation="scalar",
                q_heads_per_member=2,
            ),
        )

    config = {
        "critic": {
            "architecture": "gemma_siglip_multihead",
            "ensemble_size": 3,
            "double_q_divl": True,
            "q_heads_per_member": 2,
            "q_representation": "scalar",
            "hidden_dim": 16,
            "num_layers": 1,
            "bootstrap_probability": 1.0,
        },
        "divl": {
            "num_atoms": 11,
            "v_min": -0.1,
            "v_max": 1.1,
            "adaptive_tau": True,
            "tau_base": 0.6,
            "entropy_coefficient": 0.3,
            "interpolate_quantile": False,
        },
        "actor": {"hidden_dim": 16},
        "flow": {"num_steps": 3},
    }
    state = build_train_state(config, batch, multimodal_critic_factory=factory)
    metrics = critic_update(state, batch, config)
    assert metrics["bootstrap_target_is_memberwise"] == 1.0
    assert metrics["bootstrap_target_is_min"] == 1.0
    assert metrics["adaptive_tau_min"] <= metrics["adaptive_tau_mean"] <= metrics["adaptive_tau_max"]
    for member in range(3):
        assert f"q_mean_member_{member}_q1" in metrics
        assert f"q_mean_member_{member}_q2" in metrics
        assert f"v_tau_member_{member}" in metrics


def test_evaluator_reports_clipped_double_q_heads():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 8)

        def forward(self, batch, *, next_observation=False):
            values = batch.next_observations if next_observation else batch.observations
            return self.projection(values)

    batch = make_synthetic_replay(
        num_samples=4,
        obs_dim=3,
        proprio_dim=3,
        generated_horizon=3,
        executed_horizon=2,
        action_dim=2,
    )
    critic = MultiHeadUdivlCritic(
        Encoder(),
        MultiHeadUdivlCore(
            state_dim=8,
            action_dim=2,
            max_horizon=3,
            action_hidden_dim=8,
            head_hidden_dim=16,
            num_attention_heads=2,
            num_value_atoms=11,
            num_pairs=3,
            q_representation="scalar",
            q_heads_per_member=2,
        ),
    )
    metrics = offline_calibration_metrics(
        critic,
        batch,
        config={"critic": {"ensemble_size": 3, "q_heads_per_member": 2}},
    )
    assert "critic/q_member_0_q1_mean" in metrics
    assert "critic/q_member_2_clipped_mean" in metrics


def test_clean_double_q_checkpoint_round_trip(tmp_path):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 8)

        def forward(self, batch, *, next_observation=False):
            values = batch.next_observations if next_observation else batch.observations
            return self.projection(values)

    batch = make_synthetic_replay(
        num_samples=4,
        obs_dim=3,
        proprio_dim=3,
        generated_horizon=3,
        executed_horizon=2,
        action_dim=2,
    )

    def factory(_, __):
        return MultiHeadUdivlCritic(
            Encoder(),
            MultiHeadUdivlCore(
                state_dim=8,
                action_dim=2,
                max_horizon=3,
                action_hidden_dim=8,
                head_hidden_dim=16,
                num_attention_heads=2,
                num_value_atoms=11,
                num_pairs=3,
                q_representation="scalar",
                q_heads_per_member=2,
            ),
        )

    config = {
        "critic": {
            "architecture": "gemma_siglip_multihead",
            "ensemble_size": 3,
            "double_q_divl": True,
            "q_heads_per_member": 2,
            "q_representation": "scalar",
            "hidden_dim": 16,
            "num_layers": 1,
        },
        "divl": {"num_atoms": 11, "v_min": -0.1, "v_max": 1.1},
        "actor": {"hidden_dim": 16},
        "flow": {"num_steps": 2},
        "training": {"checkpoint_path": str(tmp_path / "clean.pt")},
    }
    first = build_train_state(config, batch, multimodal_critic_factory=factory)
    critic_update(first, batch, config)
    path = tmp_path / "clean.pt"
    save_checkpoint(first, config, path)
    second = build_train_state(config, batch, multimodal_critic_factory=factory)
    load_critic_checkpoint(path, second, load_optimizer=False)
    left = first.critic.q_pair_from_features(
        first.critic.encode_state(batch), batch.action_chunks, batch.execution_masks
    )
    right = second.critic.q_pair_from_features(
        second.critic.encode_state(batch), batch.action_chunks, batch.execution_masks
    )
    assert torch.allclose(left, right)


def test_adaptive_tau_decreases_with_normalized_entropy_without_interpolation():
    support = torch.linspace(-0.1, 1.1, 5)
    low = torch.zeros(1, 1, 5)
    low[..., 2] = 1.0
    high = torch.full((1, 1, 5), 0.2)
    middle = torch.tensor([[[0.7, 0.1, 0.1, 0.05, 0.05]]])
    low_stats = divl_quantile_values(
        low,
        support,
        alpha_min=0.0,
        alpha_max=1.0,
        adaptive_tau_mode=True,
        tau_base=0.6,
        entropy_coefficient=0.3,
        interpolate_quantile=False,
    )
    high_stats = divl_quantile_values(
        high,
        support,
        alpha_min=0.0,
        alpha_max=1.0,
        adaptive_tau_mode=True,
        tau_base=0.6,
        entropy_coefficient=0.3,
        interpolate_quantile=False,
    )
    middle_stats = divl_quantile_values(
        middle,
        support,
        alpha_min=0.0,
        alpha_max=1.0,
        adaptive_tau_mode=True,
        tau_base=0.6,
        entropy_coefficient=0.3,
        tau_min=0.3,
        tau_max=0.6,
        interpolate_quantile=False,
    )
    assert high_stats.tau.item() < middle_stats.tau.item() < low_stats.tau.item()
    assert low_stats.tau.item() == pytest.approx(0.6)
    assert high_stats.tau.item() == pytest.approx(0.6 - 0.3)


def test_full_chain_chipo_ratio_and_ca_composition():
    current = torch.tensor([[0.2, 0.1], [0.0, 0.0]])
    slow = torch.zeros_like(current)
    ratio, stats = full_chain_chipo_ratio(
        current,
        slow,
        action_dim=2,
        normalize_logprob_by_action_dim=True,
        normalize_logprob_by_denoising_steps=True,
    )
    assert torch.all(ratio >= 1.0)
    assert stats.logprob_normalizer == 4.0
    ca = torch.zeros(1, 2)
    q = torch.tensor([[[1.0, 2.0]], [[0.8, 2.2]], [[1.2, 1.8]]])
    combined, _ = apply_chipo_to_ca_advantage(
        ca,
        q,
        torch.tensor([[1.2, 0.8]]),
        beta_base=1.0,
        q_std_target=1.0,
        ensemble_alpha=5.0,
        r_max=10.0,
    )
    assert torch.allclose(combined.mean(dim=-1), torch.zeros(1), atol=1e-6)
    # CA+ChiPO is a sign-safe intersection: a zero CA signal cannot be
    # resurrected by policy-drift pessimism.
    assert torch.equal(combined, ca)


def test_official_chipo_q_target_and_beta_parity_reference_formula():
    # Reference written directly in the upstream formula's order, then
    # compared with the PyTorch helper (local layout is [M,B,G]).
    q = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 5.0]],
            [[0.0, 4.0], [1.0, 4.0]],
            [[2.0, 2.0], [3.0, 6.0]],
        ]
    )
    ratio = torch.tensor([[1.2, 0.8], [1.1, 1.4]])
    beta, q_std = compute_chipo_beta(q, beta_base=0.1, q_std_target=1.0)
    expected_std = q.std(dim=0, unbiased=False).mean()
    expected_beta = 0.1 * expected_std
    q_mean = q.mean(dim=0)
    q_min = q.min(dim=0).values
    weight = torch.sigmoid(5.0 * (ratio - 1.0))
    q_target = (1.0 - weight) * q_mean + weight * q_min
    penalized = q_target - expected_beta * ratio
    expected_adv = penalized - penalized.mean(dim=-1, keepdim=True)
    actual_adv, stats = trainer.chi2_pessimistic_advantage(
        q,
        ratio,
        beta_base=0.1,
        q_std_target=1.0,
        ensemble_alpha=5.0,
        normalize_group=False,
        advantage_clip=None,
    )
    assert torch.allclose(q_std, expected_std)
    assert torch.allclose(beta, expected_beta)
    assert torch.allclose(actual_adv, expected_adv)
    assert stats.q_target_mean == pytest.approx(float(q_target.mean()))
    assert stats.q_penalized_mean == pytest.approx(float(penalized.mean()))


def test_ca_chipo_sign_safe_intersection_vectorized():
    ca = torch.tensor([[0.0, 1.0, -2.0, 1.0], [2.0, -1.0, 0.0, -3.0]])
    chi = torch.tensor([[4.0, 3.0, -1.0, -2.0], [1.0, -4.0, 5.0, -2.0]])
    final, stats = sign_safe_advantage_intersection(ca, chi)
    expected = torch.tensor([[0.0, 1.0, -1.0, 0.0], [1.0, -1.0, 0.0, -2.0]])
    assert torch.equal(final, expected)
    assert stats["gate_positive_ratio"] == pytest.approx(2 / 8)
    assert stats["gate_negative_ratio"] == pytest.approx(3 / 8)
    assert stats["gate_sign_conflict_ratio"] == pytest.approx(1 / 8)


def test_chipo_regularizer_has_live_gradient_when_ca_advantage_is_zero():
    current = torch.full((3, 4), 0.2, requires_grad=True)
    slow = torch.zeros_like(current)
    loss, metrics = full_chain_chipo_regularization_loss(
        current,
        slow,
        action_dim=2,
        beta=0.1,
        normalize_logprob_by_action_dim=True,
        normalize_logprob_by_denoising_steps=True,
    )
    loss.backward()
    assert metrics["chi2_regularization_loss"] > 0.0
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert current.grad.abs().sum() > 0.0


def test_n_step_success_uses_traversed_indices_when_replay_is_interleaved():
    batch = make_synthetic_replay(
        num_samples=4,
        generated_horizon=2,
        executed_horizon=1,
        action_dim=2,
    )
    # Episode 0 transitions are at indices 0 and 2; index 1 belongs to a
    # successful episode and must not leak into start=0's n-step success.
    batch = replace(
        batch,
        episode_ids=torch.tensor([0, 1, 0, 1]),
        timesteps=torch.tensor([0, 0, 1, 1]),
        successes=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        dones=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        executed_lengths=torch.ones(4, dtype=torch.long),
    )
    result = make_n_step_replay(batch, n_step=2)
    assert result.successes[0].item() == 0.0
    assert result.behavior_metadata[0]["n_step_visited_indices"] == [0, 2]


def test_full_chain_success_bc_survives_gradient_microbatching():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=3,
        executed_horizon=2,
        action_dim=2,
    )
    config = {
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {
            "group_size": 2,
            "hidden_dim": 16,
            "ogpo_variant": "chi2",
            "full_ratio_mode": "ais_joint",
            "gradient_microbatch_size": 1,
            "normalize_logprob_by_action_dim": True,
            "normalize_logprob_by_denoising_steps": True,
            "chi2": {
                "ratio_scope": "full_chain",
                "slow_policy_tau": 0.5,
                "beta_base": 0.1,
                "q_std_target": 1.0,
                "ensemble_alpha": 5.0,
                "r_max": 10.0,
            },
        },
        "flow": {
            "num_steps": 3,
            "sde_mode": "ogpo_constant_corrected",
            "constant_noise_std": 0.005,
            "learn_sde_std": False,
        },
        "regularization": {
            "beta_kl": 0.0,
            "lambda_fm": 0.0,
            "lambda_success": 0.1,
        },
    }
    state = build_train_state(config, batch)
    critic_update(state, batch, config)
    metrics = full_actor_update(state, batch, config)
    assert metrics["success_bc_weighted_loss"] >= 0.0
    assert metrics["full_chain_joint_ratio"] == 1.0
    assert metrics["full_chain_one_clip"] == 1.0
    assert metrics["chi2_loss"] == 0.0
    assert metrics["sde_std_mean"] == pytest.approx(0.005, abs=1e-6)
    assert metrics["num_stochastic_flow_steps"] == 2.0
    assert metrics["num_logprob_flow_steps"] == 3.0


def test_full_chain_kl_rollback_is_atomic(monkeypatch):
    batch = make_synthetic_replay(
        num_samples=6,
        generated_horizon=3,
        executed_horizon=2,
        action_dim=2,
    )
    config = {
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {
            "group_size": 2,
            "hidden_dim": 16,
            "ogpo_variant": "ca",
            "full_ratio_mode": "ais_joint",
            "reject_update_on_kl": True,
            "post_update_kl_action": "rollback_cpu",
            "max_policy_reference_kl": 0.01,
            "gradient_microbatch_size": 1,
        },
        "flow": {"num_steps": 3, "stochastic_variance": 0.04},
        "regularization": {"beta_kl": 0.0, "lambda_fm": 0.0, "lambda_success": 0.0},
    }
    state = build_train_state(config, batch)
    critic_update(state, batch, config)
    before = {name: value.detach().clone() for name, value in state.policy.state_dict().items()}
    monkeypatch.setattr(trainer, "_full_chain_policy_kl", lambda *args, **kwargs: 1.0)
    metrics = full_actor_update(state, batch, config)
    assert metrics["actor_update_rejected"] == 1.0
    assert metrics["actor_update_accepted"] == 0.0
    for name, value in state.policy.state_dict().items():
        assert torch.allclose(value, before[name])


def test_offline_actor_update_requires_explicit_frozen_critic():
    batch = make_synthetic_replay(num_samples=6, generated_horizon=3, executed_horizon=2, action_dim=2)
    config = {
        "training": {"critic_frozen_during_actor": True},
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {"group_size": 2, "hidden_dim": 16},
        "flow": {"num_steps": 2},
    }
    state = build_train_state(config, batch)
    with pytest.raises(RuntimeError, match="critic_frozen_during_actor"):
        full_actor_update(state, batch, config)
    freeze_critic_for_actor(state)
