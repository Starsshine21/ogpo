from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch
import torch.nn as nn

from ogpo import trainer
from ogpo.categorical_q import decode_categorical_q
from ogpo.multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic
from ogpo.rankq import (
    compute_rankq_loss,
    compute_same_state_rankq_loss,
    ddp_global_valid_mean_scale,
    make_rankq_actions,
    make_same_state_rankq_actions,
    rank_pair_loss,
    same_state_rankq_settings,
)
from ogpo.replay import make_synthetic_replay
from ogpo.trainer import (
    _multimodal_double_q_divl_update,
    accumulated_critic_update,
    build_train_state,
    critic_update,
)


class ReplayEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, batch, *, next_observation: bool = False):
        values = batch.next_observations if next_observation else batch.observations
        return self.projection(values)


def _factory(batch, config):
    critic_cfg = config["critic"]
    return MultiHeadUdivlCritic(
        ReplayEncoder(batch.obs_dim),
        MultiHeadUdivlCore(
            state_dim=8,
            action_dim=batch.action_dim,
            max_horizon=batch.generated_horizon,
            action_hidden_dim=8,
            head_hidden_dim=16,
            num_attention_heads=2,
            num_value_atoms=int(config["divl"]["num_atoms"]),
            num_pairs=int(critic_cfg.get("ensemble_size", 3)),
            q_heads_per_member=int(critic_cfg.get("q_heads_per_member", 1)),
            q_representation=str(critic_cfg.get("q_representation", "scalar")),
            q_num_bins=201,
            q_vmin=-0.1,
            q_vmax=1.1,
        ),
    )


def _config(*, enable_rankq: bool, q_representation: str = "categorical") -> dict:
    return {
        "critic": {
            "architecture": "gemma_siglip_multihead",
            "ensemble_size": 3,
            "learning_rate": 3e-4,
            "q_loss": "mse",
            "q_representation": q_representation,
            "q_num_bins": 201,
            "q_vmin": -0.1,
            "q_vmax": 1.1,
            "q_hl_gauss_sigma_bins": 0.75,
            "rank_consensus_enabled": False,
            "enable_rankq": enable_rankq,
            "lambda_rank": 1.0,
            "rankq_noise_sigma": 0.15,
        },
        "divl": {
            "num_atoms": 201,
            "v_min": -0.1,
            "v_max": 1.1,
            "alpha_min": 0.5,
            "alpha_max": 0.6,
            "loss_weight": 1.0,
        },
        "actor": {"hidden_dim": 16},
        "flow": {"num_steps": 3},
        "training": {"seed": 7},
    }


def _q_values_from_logits(logits: dict[str, torch.Tensor], support: torch.Tensor):
    return {name: decode_categorical_q(value, support) for name, value in logits.items()}


def test_rank_pair_loss_has_vanilla_softplus_behavior():
    equal = rank_pair_loss(torch.tensor([0.0]), torch.tensor([0.0]))
    ordered = rank_pair_loss(torch.tensor([10.0]), torch.tensor([-10.0]))
    reversed_pair = rank_pair_loss(torch.tensor([-1.0]), torch.tensor([1.0]))

    assert equal.item() == pytest.approx(math.log(2.0))
    assert ordered.item() < 1e-8
    assert reversed_pair.item() > equal.item()


def test_rankq_categorical_expectation_shape_and_gradients_reach_all_q_heads():
    support = torch.linspace(-0.1, 1.1, 201)
    logits = {
        name: torch.randn(3, 4, 201, requires_grad=True)
        for name in ("positive", "noisy", "very_noisy", "random", "permuted")
    }
    values = _q_values_from_logits(logits, support)
    v_logits = torch.randn(3, 4, 201, requires_grad=True)

    output = compute_rankq_loss(values, torch.tensor([True, True, False, False]))
    output.loss.backward()

    assert values["positive"].shape == (3, 4)
    assert output.loss_per_head.shape == (3,)
    assert all(logits["positive"].grad[head].abs().sum().item() > 0.0 for head in range(3))
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in logits.values()
    )
    assert v_logits.grad is None


def test_rankq_module_gradient_reaches_q_heads_but_not_value_heads():
    batch = make_synthetic_replay(
        num_samples=6,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config(enable_rankq=True)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    features = state.critic.encode_state(batch)
    action_pool = state.critic.core.action_pool
    actions = make_rankq_actions(
        batch.action_chunks,
        batch.execution_masks,
        action_mean=action_pool.action_mean,
        action_std=action_pool.action_std,
        action_min=action_pool.action_min,
        action_max=action_pool.action_max,
        noise_sigma=0.15,
        generator=torch.Generator().manual_seed(17),
    )
    q_values = {
        name: state.critic.q_from_features(features, action, batch.execution_masks)
        for name, action in (
            ("positive", actions.positive),
            ("noisy", actions.noisy),
            ("very_noisy", actions.very_noisy),
            ("random", actions.random),
            ("permuted", actions.permuted),
        )
    }

    state.critic.zero_grad(set_to_none=True)
    compute_rankq_loss(q_values, batch.successes).loss.backward()

    for q_head in state.critic.core.q_heads:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
            for parameter in q_head.parameters()
        )
    assert all(
        parameter.grad is None
        for value_head in state.critic.core.value_heads
        for parameter in value_head.parameters()
    )


@pytest.mark.parametrize("success_value", [True, False])
def test_rankq_empty_success_or_failure_category_is_finite(success_value: bool):
    q_values = {
        name: torch.randn(3, 5, requires_grad=True)
        for name in ("positive", "noisy", "very_noisy", "random", "permuted")
    }
    successes = torch.full((5,), success_value, dtype=torch.bool)

    output = compute_rankq_loss(q_values, successes)
    output.loss.backward()

    assert torch.isfinite(output.loss)
    if success_value:
        assert output.failure_loss.item() == 0.0
    else:
        assert output.success_loss.item() == 0.0


def test_rankq_actions_respect_mask_bounds_and_shared_noise_direction():
    actions = torch.zeros(3, 4, 2)
    actions[1] = 0.95
    actions[2] = -0.95
    mask = torch.tensor([[True, True, False, False]] * 3)
    generated = make_rankq_actions(
        actions,
        mask,
        action_mean=torch.zeros(2),
        action_std=torch.ones(2),
        action_min=torch.full((2,), -1.0),
        action_max=torch.full((2,), 1.0),
        noise_sigma=0.15,
        generator=torch.Generator().manual_seed(13),
    )

    noisy_delta = generated.noisy - actions
    very_noisy_delta = generated.very_noisy - actions
    assert torch.allclose(very_noisy_delta[:, :2], 2.0 * noisy_delta[:, :2], atol=1e-6)
    for candidate in (
        generated.noisy,
        generated.very_noisy,
        generated.random,
        generated.permuted,
    ):
        assert torch.equal(candidate[:, 2:], actions[:, 2:])
        assert bool((candidate >= -1.0).all() and (candidate <= 1.0).all())


def test_rankq_permutation_uses_other_transition_when_batch_has_multiple_rows():
    actions = torch.arange(4, dtype=torch.float32).view(4, 1, 1).expand(-1, 3, -1).clone()
    mask = torch.ones(4, 3, dtype=torch.bool)
    generated = make_rankq_actions(
        actions,
        mask,
        action_mean=torch.zeros(1),
        action_std=torch.ones(1),
        action_min=torch.zeros(1),
        action_max=torch.full((1,), 3.0),
        noise_sigma=0.15,
        generator=torch.Generator().manual_seed(3),
    )

    assert bool((generated.permutation != torch.arange(4)).all())
    assert torch.equal(generated.permuted, actions.index_select(0, generated.permutation))


def test_rankq_random_actions_use_paper_normalized_uniform_range():
    actions = torch.full((32, 2, 1), 10.0)
    generated = make_rankq_actions(
        actions,
        torch.ones(32, 2, dtype=torch.bool),
        action_mean=torch.tensor([10.0]),
        action_std=torch.tensor([2.0]),
        action_min=torch.tensor([0.0]),
        action_max=torch.tensor([20.0]),
        noise_sigma=0.15,
        generator=torch.Generator().manual_seed(23),
    )

    random_normalized = (generated.random - 10.0) / 2.0
    assert bool((random_normalized >= -1.0).all() and (random_normalized <= 1.0).all())
    assert random_normalized.min().item() < -0.5
    assert random_normalized.max().item() > 0.5


def test_rankq_reduction_averages_heads_and_adds_success_and_failure_categories():
    positive = torch.tensor([[2.0, 0.5], [1.5, 0.4], [1.0, 0.3]])
    q_values = {
        "positive": positive,
        "noisy": positive - 0.1,
        "very_noisy": positive - 0.2,
        "random": positive - 0.3,
        "permuted": positive - 0.4,
    }
    output = compute_rankq_loss(q_values, torch.tensor([True, False]))

    expected_success = (
        3.0 * torch.nn.functional.softplus(torch.tensor(-0.1))
        + torch.nn.functional.softplus(torch.tensor(-0.2))
        + torch.nn.functional.softplus(torch.tensor(-0.3))
        + torch.nn.functional.softplus(torch.tensor(-0.4))
    )
    expected_failure = torch.nn.functional.softplus(torch.tensor(-0.3))

    assert output.loss.item() == pytest.approx(output.loss_per_head.mean().item())
    assert output.success_loss.item() == pytest.approx(expected_success.item())
    assert output.failure_loss.item() == pytest.approx(expected_failure.item())
    assert torch.allclose(
        output.loss_per_head,
        output.success_loss_per_head + output.failure_loss_per_head,
    )
    assert output.metrics["rankq/loss_q1"] == pytest.approx(output.loss_per_head[0].item())
    assert output.metrics["rankq/loss_q2"] == pytest.approx(output.loss_per_head[1].item())
    assert output.metrics["rankq/loss_q3"] == pytest.approx(output.loss_per_head[2].item())


def test_disabled_rankq_keeps_original_td10_loss_composition():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config(enable_rankq=False)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    metrics = critic_update(state, batch, config)

    assert metrics["rankq/enabled"] == 0.0
    assert metrics["rankq/loss"] == 0.0
    assert metrics["critic/rank_loss"] == 0.0
    assert metrics["critic_loss"] == pytest.approx(
        metrics["q_loss"] + metrics["divl_loss"]
    )


def test_enabled_rankq_is_an_additive_auxiliary_only():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config(enable_rankq=True)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    metrics = critic_update(state, batch, config)

    assert metrics["rankq/enabled"] == 1.0
    assert metrics["rankq/loss"] > 0.0
    assert metrics["critic/rank_loss"] == 0.0
    assert metrics["critic_loss"] == pytest.approx(
        metrics["q_loss"] + metrics["divl_loss"] + metrics["rankq/loss"]
    )
    assert metrics["q_representation_is_categorical"] == 1.0
    assert "categorical_q/near_upper_support_fraction" in metrics


def test_scalar_q_uses_the_same_vanilla_rankq_auxiliary():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config(enable_rankq=True, q_representation="scalar")
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    metrics = critic_update(state, batch, config)

    assert state.critic.core.q_representation == "scalar"
    assert metrics["q_loss_is_mse"] == 1.0
    assert metrics["q_representation_is_categorical"] == 0.0
    assert metrics["rankq/enabled"] == 1.0
    assert metrics["rankq/loss"] > 0.0
    assert metrics["critic_loss"] == pytest.approx(
        metrics["q_loss"] + metrics["divl_loss"] + metrics["rankq/loss"]
    )
    assert "categorical_q/near_upper_support_fraction" not in metrics


def test_rankq_gradient_accumulation_uses_effective_batch_category_reductions():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config(enable_rankq=True)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    metrics = accumulated_critic_update(
        state,
        batch,
        config,
        microbatch_size=4,
    )

    success_count = float(batch.successes.bool().sum().item())
    assert metrics["rankq/success_count"] == success_count
    assert metrics["rankq/failure_count"] == batch.batch_size - success_count
    assert metrics["rankq/loss"] == pytest.approx(
        metrics["rankq/success_loss"] + metrics["rankq/failure_loss"]
    )
    assert metrics["critic_loss"] == pytest.approx(
        metrics["q_loss"] + metrics["divl_loss"] + metrics["rankq/loss"]
    )


def _nested_rankq_config(*, enabled: bool, lambda_rank: float) -> dict:
    config = _config(enable_rankq=False, q_representation="scalar")
    config["critic"].update(
        {
            "ensemble_size": 5,
            "double_q_divl": True,
            "q_heads_per_member": 2,
            "bootstrap_probability": 1.0,
            "max_grad_norm": 1000.0,
            "rankq": {
                "enabled": enabled,
                "lambda_rank": lambda_rank,
                "mild_sigma": 0.02,
                "strong_sigma": 0.05,
                "use_success_only": True,
                "use_random_negative": True,
            },
        }
    )
    return config


def _gradient_vector(module: nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (
                torch.zeros_like(parameter).reshape(-1)
                if parameter.grad is None
                else parameter.grad.detach().reshape(-1).clone()
            )
            for parameter in module.parameters()
        ]
    )


def _identical_rankq_state(config: dict, reference_batch, *, seed: int = 1701):
    torch.manual_seed(seed)
    return build_train_state(
        config,
        reference_batch,
        multimodal_critic_factory=_factory,
    )


def test_nested_rankq_lambda_zero_total_loss_is_original_divl_loss():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _nested_rankq_config(enabled=True, lambda_rank=0.0)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    metrics = critic_update(state, batch, config)
    assert metrics["rankq_enabled"] == 0.0
    assert metrics["rankq_weighted_loss"] == 0.0
    assert metrics["total_critic_loss"] == metrics["divl_objective_loss"]
    assert metrics["divl_objective_loss"] == pytest.approx(
        metrics["q_loss"] + metrics["divl_loss"]
    )


def test_nested_rankq_lambda_zero_update_is_bitwise_identical_to_disabled():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    disabled = _nested_rankq_config(enabled=False, lambda_rank=0.02)
    zero = _nested_rankq_config(enabled=True, lambda_rank=0.0)
    torch.manual_seed(101)
    disabled_state = build_train_state(
        disabled, batch, multimodal_critic_factory=_factory
    )
    torch.manual_seed(101)
    zero_state = build_train_state(zero, batch, multimodal_critic_factory=_factory)
    torch.manual_seed(202)
    disabled_metrics = critic_update(disabled_state, batch, disabled)
    torch.manual_seed(202)
    zero_metrics = critic_update(zero_state, batch, zero)
    assert disabled_metrics["critic_loss"] == zero_metrics["critic_loss"]
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            disabled_state.critic.parameters(),
            zero_state.critic.parameters(),
            strict=True,
        )
    )
    for left, right in zip(
        disabled_state.critic.parameters(),
        zero_state.critic.parameters(),
        strict=True,
    ):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad)


def test_nested_rankq_only_q_heads_receive_auxiliary_gradient_and_all_ten_are_active():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _nested_rankq_config(enabled=True, lambda_rank=0.02)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    features = state.critic.encode_state(batch)
    pool = state.critic.core.action_pool
    actions = make_same_state_rankq_actions(
        batch.action_chunks,
        batch.execution_masks,
        action_mean=pool.action_mean,
        action_std=pool.action_std,
        action_min=pool.action_min,
        action_max=pool.action_max,
        mild_sigma=0.02,
        strong_sigma=0.05,
        use_random_negative=True,
        generator=torch.Generator().manual_seed(303),
    )
    q_values = {
        name: state.critic.raw_q_ensemble_from_features(
            features, action, batch.execution_masks
        )
        for name, action in (
            ("logged", actions.logged),
            ("mild", actions.mild),
            ("strong", actions.strong),
            ("random", actions.random),
        )
    }
    state.critic.zero_grad(set_to_none=True)
    compute_same_state_rankq_loss(q_values, batch.successes).loss.backward()
    assert len(state.critic.core.q_heads) == 10
    assert all(
        any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
            for parameter in head.parameters()
        )
        for head in state.critic.core.q_heads
    )
    assert all(
        parameter.grad is None
        for head in state.critic.core.value_heads
        for parameter in head.parameters()
    )


def test_nested_rankq_ordered_chain_has_much_lower_loss_than_reverse_chain():
    successes = torch.ones(4, dtype=torch.bool)
    ordered = {
        "logged": torch.full((10, 4), 6.0),
        "mild": torch.full((10, 4), 2.0),
        "strong": torch.full((10, 4), -2.0),
        "random": torch.full((10, 4), -6.0),
    }
    reverse = {name: -value for name, value in ordered.items()}
    good = compute_same_state_rankq_loss(ordered, successes)
    bad = compute_same_state_rankq_loss(reverse, successes)
    assert good.loss.item() < 0.1
    assert bad.loss.item() > good.loss.item() + 10.0


def _relation_test_values(*, seed: int = 3301):
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(10, 4, generator=generator, requires_grad=True)
        for name in ("logged", "mild", "strong", "random")
    }


def _loss_gradients(loss: torch.Tensor, values: dict[str, torch.Tensor]):
    gradients = torch.autograd.grad(
        loss,
        tuple(values.values()),
        retain_graph=True,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(values.values(), gradients, strict=True)
    )


def test_same_state_rankq_unit_relation_weights_preserve_loss_and_gradient():
    values = _relation_test_values()
    successes = torch.ones(4, dtype=torch.bool)
    original = compute_same_state_rankq_loss(values, successes)
    original_gradients = _loss_gradients(original.loss, values)
    unit = compute_same_state_rankq_loss(
        values,
        successes,
        logged_mild_weight=1.0,
        mild_strong_weight=1.0,
        strong_random_weight=1.0,
    )
    unit_gradients = _loss_gradients(unit.loss, values)
    assert torch.equal(original.loss, unit.loss)
    assert all(
        torch.equal(original_gradient, unit_gradient)
        for original_gradient, unit_gradient in zip(
            original_gradients, unit_gradients, strict=True
        )
    )


def test_same_state_rankq_weighted_loss_is_exact_relation_sum():
    values = _relation_test_values(seed=3302)
    output = compute_same_state_rankq_loss(
        values,
        torch.ones(4, dtype=torch.bool),
        logged_mild_weight=50.0,
        mild_strong_weight=33.0,
        strong_random_weight=0.25,
    )
    expected = (
        50.0 * output.relation_losses["logged_vs_mild"]
        + 33.0 * output.relation_losses["mild_vs_strong"]
        + 0.25 * output.relation_losses["strong_vs_random"]
    )
    assert torch.equal(output.loss, expected)


@pytest.mark.parametrize(
    ("relation", "weights", "weight"),
    [
        ("logged_vs_mild", (50.0, 0.0, 0.0), 50.0),
        ("mild_vs_strong", (0.0, 33.0, 0.0), 33.0),
        ("strong_vs_random", (0.0, 0.0, 0.25), 0.25),
    ],
)
def test_same_state_rankq_each_relation_gradient_scales_by_its_weight(
    relation, weights, weight
):
    values = _relation_test_values(seed=3303)
    raw = compute_same_state_rankq_loss(values, torch.ones(4, dtype=torch.bool))
    raw_gradients = _loss_gradients(raw.relation_losses[relation], values)
    weighted = compute_same_state_rankq_loss(
        values,
        torch.ones(4, dtype=torch.bool),
        logged_mild_weight=weights[0],
        mild_strong_weight=weights[1],
        strong_random_weight=weights[2],
    )
    weighted_gradients = _loss_gradients(weighted.loss, values)
    assert all(
        torch.allclose(
            weighted_gradient,
            weight * raw_gradient,
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        for weighted_gradient, raw_gradient in zip(
            weighted_gradients, raw_gradients, strict=True
        )
    )


def test_nested_rankq_actions_respect_real_bounds_and_execution_mask():
    actions = torch.tensor([[[0.95], [0.0], [0.0]], [[-0.95], [0.0], [0.0]]])
    mask = torch.tensor([[True, False, False], [True, False, False]])
    generated = make_same_state_rankq_actions(
        actions,
        mask,
        action_mean=torch.zeros(1),
        action_std=torch.ones(1),
        action_min=torch.full((1,), -1.0),
        action_max=torch.full((1,), 1.0),
        mild_sigma=0.02,
        strong_sigma=0.05,
        use_random_negative=True,
        generator=torch.Generator().manual_seed(404),
    )
    for candidate in (generated.mild, generated.strong, generated.random):
        assert candidate is not None
        assert bool((candidate >= -1.0).all() and (candidate <= 1.0).all())
        assert torch.equal(candidate[:, 1:], actions[:, 1:])
    mild_delta = generated.mild[:, :1] - actions[:, :1]
    strong_delta = generated.strong[:, :1] - actions[:, :1]
    assert torch.allclose(strong_delta, mild_delta * (0.05 / 0.02), atol=1e-6)


def test_rankq_ddp_global_success_weighting_matches_global_sample_mean():
    # Rank 0 has one success with mean loss 2; rank 1 has three with mean 4.
    # DDP averages the two pre-scaled local gradients/losses.
    rank0 = 2.0 * ddp_global_valid_mean_scale(1, 4, 2)
    rank1 = 4.0 * ddp_global_valid_mean_scale(3, 4, 2)
    assert (rank0 + rank1) / 2.0 == pytest.approx((2.0 + 3.0 * 4.0) / 4.0)


def test_direct_rankq_ddp_gradient_matches_explicit_four_rank_global_weighted_mean(
    monkeypatch,
):
    counts = [1, 2, 5, 7]
    global_count = sum(counts)
    reference = make_synthetic_replay(
        num_samples=global_count,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    reference = replace(reference, successes=torch.ones(global_count))
    active = _nested_rankq_config(enabled=True, lambda_rank=0.02)
    disabled = _nested_rankq_config(enabled=False, lambda_rank=0.02)

    monkeypatch.setattr(trainer.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer.dist, "get_world_size", lambda: 4)

    def reduce_global_count(value, op=None):
        assert value.numel() == 1
        value.fill_(global_count)

    monkeypatch.setattr(trainer.dist, "all_reduce", reduce_global_count)
    # Capture each rank's pre-DDP gradient. The explicit average below applies
    # the same final SUM/world-size operation as production.
    monkeypatch.setattr(trainer, "_all_reduce_gradients", lambda parameters: None)

    direct_rankq_parts = []
    explicit_global = None
    offset = 0
    for rank, count in enumerate(counts):
        indices = torch.arange(offset, offset + count)
        offset += count
        local_batch = reference.index_select(indices)
        action_state = _identical_rankq_state(active, reference)
        pool = action_state.critic.core.action_pool
        actions = make_same_state_rankq_actions(
            local_batch.action_chunks,
            local_batch.execution_masks,
            action_mean=pool.action_mean,
            action_std=pool.action_std,
            action_min=pool.action_min,
            action_max=pool.action_max,
            mild_sigma=0.02,
            strong_sigma=0.05,
            use_random_negative=True,
            generator=torch.Generator().manual_seed(5000 + rank),
        )
        monkeypatch.setattr(
            trainer,
            "make_same_state_rankq_actions",
            lambda *args, fixed=actions, **kwargs: fixed,
        )

        direct_active = _identical_rankq_state(active, reference)
        direct_disabled = _identical_rankq_state(disabled, reference)
        metrics = critic_update(direct_active, local_batch, active)
        critic_update(direct_disabled, local_batch, disabled)
        assert metrics["rankq_success_count"] == float(global_count)
        direct_rankq_parts.append(
            _gradient_vector(direct_active.critic)
            - _gradient_vector(direct_disabled.critic)
        )

        raw_active = _identical_rankq_state(active, reference)
        raw_disabled = _identical_rankq_state(disabled, reference)
        _multimodal_double_q_divl_update(
            raw_active,
            local_batch,
            active,
            optimizer_step=False,
            same_state_rankq_actions=actions,
            same_state_rankq_success_scale=1.0,
        )
        _multimodal_double_q_divl_update(
            raw_disabled,
            local_batch,
            disabled,
            optimizer_step=False,
        )
        raw_rankq = (
            _gradient_vector(raw_active.critic)
            - _gradient_vector(raw_disabled.critic)
        )
        weighted = raw_rankq * (count / global_count)
        explicit_global = weighted if explicit_global is None else explicit_global + weighted

    assert explicit_global is not None
    ddp_direct = torch.stack(direct_rankq_parts).mean(dim=0)
    assert torch.allclose(ddp_direct, explicit_global, rtol=2e-5, atol=2e-7)


def test_direct_and_accumulated_rankq_gradients_match_on_equivalent_batch():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    batch = replace(
        batch,
        successes=torch.tensor([True, False, True, True, False, True, False, True]),
    )
    active = _nested_rankq_config(enabled=True, lambda_rank=0.02)
    disabled = _nested_rankq_config(enabled=False, lambda_rank=0.02)

    direct_active = _identical_rankq_state(active, batch, seed=1901)
    direct_disabled = _identical_rankq_state(disabled, batch, seed=1901)
    torch.manual_seed(2901)
    critic_update(direct_active, batch, active)
    critic_update(direct_disabled, batch, disabled)
    direct_rankq = (
        _gradient_vector(direct_active.critic)
        - _gradient_vector(direct_disabled.critic)
    )

    accumulated_active = _identical_rankq_state(active, batch, seed=1901)
    accumulated_disabled = _identical_rankq_state(disabled, batch, seed=1901)
    torch.manual_seed(2901)
    accumulated_critic_update(
        accumulated_active, batch, active, microbatch_size=2
    )
    accumulated_critic_update(
        accumulated_disabled, batch, disabled, microbatch_size=2
    )
    accumulated_rankq = (
        _gradient_vector(accumulated_active.critic)
        - _gradient_vector(accumulated_disabled.critic)
    )

    assert torch.allclose(
        direct_rankq,
        accumulated_rankq,
        rtol=2e-5,
        atol=2e-7,
    )


def test_direct_rankq_global_zero_success_is_finite_zero():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    batch = replace(batch, successes=torch.zeros(8))
    config = _nested_rankq_config(enabled=True, lambda_rank=0.02)
    state = _identical_rankq_state(config, batch)
    metrics = critic_update(state, batch, config)
    assert metrics["rankq_success_count"] == 0.0
    assert metrics["rankq_loss"] == 0.0
    assert metrics["rankq_weighted_loss"] == 0.0
    assert math.isfinite(metrics["critic_loss"])


def test_nested_rankq_lambda_zero_does_not_generate_perturbations(monkeypatch):
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _nested_rankq_config(enabled=True, lambda_rank=0.0)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    monkeypatch.setattr(
        "ogpo.trainer.make_same_state_rankq_actions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled RankQ generated perturbations")
        ),
    )
    monkeypatch.setattr(
        state.critic,
        "raw_q_ensemble_from_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled RankQ ran an auxiliary raw-Q forward")
        ),
    )
    metrics = accumulated_critic_update(state, batch, config, microbatch_size=2)
    assert metrics["rankq_enabled"] == 0.0
    assert metrics["rankq_weighted_loss"] == 0.0


def test_nested_rankq_config_is_fully_disabled_by_either_gate():
    disabled = same_state_rankq_settings(
        {"rankq": {"enabled": False, "lambda_rank": 0.02}}
    )
    zero = same_state_rankq_settings(
        {"rankq": {"enabled": True, "lambda_rank": 0.0}}
    )
    assert disabled["enabled"] is False
    assert zero["enabled"] is False


@pytest.mark.parametrize(
    ("lambda_rank_max", "midpoint"),
    [(0.02, 0.01), (0.05, 0.025)],
)
def test_nested_rankq_absolute_step_warmup_schedule(lambda_rank_max, midpoint):
    config = {
        "rankq": {
            "enabled": True,
            "lambda_rank_max": lambda_rank_max,
            "rankq_warmup_start_step": 500,
            "rankq_warmup_end_step": 1500,
        }
    }
    expected = {
        0: 0.0,
        499: 0.0,
        500: 0.0,
        1000: midpoint,
        1500: lambda_rank_max,
        8000: lambda_rank_max,
    }
    for step, value in expected.items():
        settings = same_state_rankq_settings(config, optimizer_step=step)
        assert settings["lambda_rank_effective"] == pytest.approx(value)
        assert settings["enabled"] is (value > 0.0)


def test_nested_rankq_warmup_uses_restored_absolute_step_and_legacy_is_constant():
    scheduled = {
        "rankq": {
            "enabled": True,
            "lambda_rank_max": 0.02,
            "rankq_warmup_start_step": 500,
            "rankq_warmup_end_step": 1500,
        }
    }
    # A resumed state only supplies its restored optimizer step; no local-run
    # counter participates in resolving the coefficient.
    assert same_state_rankq_settings(scheduled, optimizer_step=1200)[
        "lambda_rank_effective"
    ] == pytest.approx(0.014)
    legacy = same_state_rankq_settings(
        {"rankq": {"enabled": True, "lambda_rank": 0.02}}, optimizer_step=1200
    )
    assert legacy["lambda_rank_effective"] == pytest.approx(0.02)
    assert legacy["lambda_rank_schedule_enabled"] is False


def test_nested_rankq_relation_weights_default_to_one_and_resolve_independently():
    defaults = same_state_rankq_settings(
        {"rankq": {"enabled": True, "lambda_rank": 0.02}}
    )
    weighted = same_state_rankq_settings(
        {
            "rankq": {
                "enabled": True,
                "lambda_rank": 0.02,
                "logged_mild_weight": 50.0,
                "mild_strong_weight": 33.0,
                "strong_random_weight": 0.25,
            }
        }
    )
    assert [
        defaults["logged_mild_weight"],
        defaults["mild_strong_weight"],
        defaults["strong_random_weight"],
    ] == [1.0, 1.0, 1.0]
    assert [
        weighted["logged_mild_weight"],
        weighted["mild_strong_weight"],
        weighted["strong_random_weight"],
    ] == [50.0, 33.0, 0.25]


def test_nested_rankq_integrates_as_small_additive_double_q_divl_loss():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _nested_rankq_config(enabled=True, lambda_rank=0.02)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    metrics = accumulated_critic_update(state, batch, config, microbatch_size=2)
    assert metrics["rankq_enabled"] == 1.0
    assert metrics["rankq_lambda"] == 0.02
    assert metrics["rankq_loss"] > 0.0
    assert metrics["rankq_weighted_loss"] == pytest.approx(
        0.02 * metrics["rankq_loss"]
    )
    assert metrics["rankq_total_raw_relation_loss"] == pytest.approx(
        metrics["rankq_mild_loss"]
        + metrics["rankq_strong_loss"]
        + metrics["rankq_random_loss"]
    )
    assert metrics["rankq_total_relation_weighted_loss"] == pytest.approx(
        metrics["rankq_loss"]
    )
    assert metrics["rankq_total_lambda_weighted_loss"] == pytest.approx(
        metrics["rankq_weighted_loss"]
    )
    assert metrics["critic_grad_norm_preclip"] == metrics["critic_grad_norm"]
    assert metrics["critic_grad_norm_postclip"] <= 1000.0 + 1.0e-4
    assert 0.0 < metrics["critic_grad_clip_scale"] <= 1.0
    assert metrics["total_critic_loss"] == pytest.approx(
        metrics["divl_objective_loss"] + metrics["rankq_weighted_loss"]
    )
    for key in (
        "rankq_mild_loss",
        "rankq_strong_loss",
        "rankq_random_loss",
        "rankq_logged_vs_mild_acc",
        "rankq_mild_vs_strong_acc",
        "rankq_strong_vs_random_acc",
        "rankq_raw10_unanimous_logged_vs_mild",
        "rankq_raw10_unanimous_mild_vs_strong",
    ):
        assert key in metrics


def test_nested_rankq_component_probe_never_updates_state():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _nested_rankq_config(enabled=True, lambda_rank=0.02)
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    before = [parameter.detach().clone() for parameter in state.critic.parameters()]
    metrics = _multimodal_double_q_divl_update(
        state,
        batch,
        config,
        diagnostic_component_gradients=True,
    )
    assert state.step == 0
    assert metrics["divl_grad_norm"] > 0.0
    assert metrics["rankq_grad_norm"] > 0.0
    assert metrics["rankq_weighted_grad_norm"] > 0.0
    assert math.isfinite(metrics["divl_rankq_gradient_cosine"])
    assert -1.0 <= metrics["divl_rankq_gradient_cosine"] <= 1.0
    assert metrics["divl_backbone_grad_norm"] >= 0.0
    assert metrics["divl_q_head_grad_norm"] > 0.0
    assert metrics["rankq_backbone_grad_norm"] >= 0.0
    assert metrics["rankq_q_head_grad_norm"] > 0.0
    assert metrics["rankq_value_head_grad_norm"] == 0.0
    assert metrics["rankq_q_head_gradient_count"] == 10.0
    for relation in (
        "logged_vs_mild",
        "mild_vs_strong",
        "strong_vs_random",
    ):
        assert metrics[f"rankq_{relation}_grad_norm"] > 0.0
        assert metrics[f"rankq_{relation}_backbone_grad_norm"] >= 0.0
        assert metrics[f"rankq_{relation}_q_head_grad_norm"] > 0.0
        assert -1.0 <= metrics[f"rankq_{relation}_divl_gradient_cosine"] <= 1.0
        assert metrics[f"rankq_{relation}_weighted_individual_grad_norm"] > 0.0
        assert (
            -1.0
            <= metrics[f"rankq_{relation}_weighted_rankq_gradient_cosine"]
            <= 1.0
        )
        assert 0.0 <= metrics[f"rankq_{relation}_softplus_slope_mean"] <= 1.0
    for pair in (
        "logged_vs_mild__mild_vs_strong",
        "logged_vs_mild__strong_vs_random",
        "mild_vs_strong__strong_vs_random",
    ):
        assert -1.0 <= metrics[f"rankq_relation_gradient_cosine_{pair}"] <= 1.0
    assert all(
        torch.equal(saved, current)
        for saved, current in zip(before, state.critic.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in state.critic.parameters())
