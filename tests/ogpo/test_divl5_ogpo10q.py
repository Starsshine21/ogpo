import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ogpo import trainer
from ogpo.chi2_regularization import (
    chi2_pessimistic_advantage,
    compute_chipo_beta,
)
from ogpo.conservative_advantage import safe_max
from ogpo.divl import (
    aggregate_double_q_for_v,
    divl_clipped_double_q_projection_targets,
    divl_double_q_projection_targets,
)
from ogpo.multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic
from ogpo.replay import make_synthetic_replay
from ogpo.trainer import build_train_state, critic_update


class _Encoder(nn.Module):
    def __init__(self, state_dim: int = 4):
        super().__init__()
        self.projection = nn.Linear(3, state_dim)

    def forward(self, batch, *, next_observation=False):
        source = batch.next_observations if next_observation else batch.observations
        return self.projection(source)


def _critic(num_pairs: int = 5) -> MultiHeadUdivlCritic:
    core = MultiHeadUdivlCore(
        state_dim=4,
        action_dim=2,
        max_horizon=2,
        action_hidden_dim=4,
        head_hidden_dim=8,
        num_attention_heads=2,
        num_value_atoms=201,
        num_pairs=num_pairs,
        q_representation="scalar",
        q_heads_per_member=2,
    )
    return MultiHeadUdivlCritic(_Encoder(), core)


def _set_raw_q_constants(critic: MultiHeadUdivlCritic) -> None:
    with torch.no_grad():
        for index, head in enumerate(critic.core.q_heads, start=1):
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias.fill_(float(index))


def test_five_pair_critic_has_one_encoder_ten_independent_q_and_five_v_heads():
    critic = _critic()
    assert critic.ensemble_size == 5
    assert critic.num_raw_q_heads == 10
    assert len(critic.core.q_heads) == 10
    assert len(critic.core.value_heads) == 5
    assert len({id(head) for head in critic.core.q_heads}) == 10
    assert len({id(head) for head in critic.core.value_heads}) == 5
    assert sum(1 for _ in critic.children()) == 2  # one encoder plus one shared head core


def test_raw_q_interface_shape_and_member_major_order_are_stable():
    critic = _critic()
    _set_raw_q_constants(critic)
    batch = make_synthetic_replay(
        num_samples=3, obs_dim=3, generated_horizon=2, executed_horizon=2, action_dim=2
    )
    features = critic.encode_state(batch)
    raw = critic.raw_q_ensemble_from_features(
        features, batch.action_chunks, batch.execution_masks
    )
    pairs = critic.q_pair_from_features(features, batch.action_chunks, batch.execution_masks)
    assert pairs.shape == (5, 2, 3)
    assert raw.shape == (10, 3)
    assert torch.equal(raw[:, 0], torch.arange(1.0, 11.0))
    assert torch.equal(raw, pairs.reshape(10, 3))


def test_divl_uses_pair_min_while_ogpo_sees_both_raw_q_heads():
    critic = _critic()
    _set_raw_q_constants(critic)
    batch = make_synthetic_replay(
        num_samples=2, obs_dim=3, generated_horizon=2, executed_horizon=2, action_dim=2
    )
    features = critic.encode_state(batch)
    pairs = critic.q_pair_from_features(features, batch.action_chunks, batch.execution_masks)
    raw = critic.raw_q_ensemble_from_features(
        features, batch.action_chunks, batch.execution_masks
    )
    projected = divl_clipped_double_q_projection_targets(
        pairs, torch.arange(0.0, 12.0)
    )
    assert torch.equal(pairs[0, :, 0], torch.tensor([1.0, 2.0]))
    assert projected[0, 0].argmax().item() == 1  # DIVL target uses min(1, 2)
    assert torch.equal(raw[:2, 0], torch.tensor([1.0, 2.0]))  # OGPO keeps both


def test_configurable_v_q_aggregation_mean_and_min_are_memberwise():
    pairs = torch.tensor(
        [
            [[0.1, 0.9], [0.5, 0.3]],
            [[-0.1, 0.7], [0.3, 1.1]],
        ]
    )
    assert torch.equal(aggregate_double_q_for_v(pairs, "min"), pairs.min(dim=1).values)
    assert torch.equal(aggregate_double_q_for_v(pairs, "mean"), pairs.mean(dim=1))
    support = torch.linspace(-0.1, 1.1, 13)
    legacy = divl_clipped_double_q_projection_targets(pairs, support)
    explicit_min = divl_double_q_projection_targets(pairs, support, aggregation="min")
    projected_mean = divl_double_q_projection_targets(pairs, support, aggregation="mean")
    assert torch.equal(legacy, explicit_min)
    assert not torch.equal(projected_mean, explicit_min)


def _controlled_v_config() -> dict:
    return {
        "critic": {
            "architecture": "gemma_siglip_multihead",
            "ensemble_size": 5,
            "expected_num_pairs": 5,
            "expected_num_raw_q_heads": 10,
            "double_q_divl": True,
            "q_heads_per_member": 2,
            "q_representation": "scalar",
            "bootstrap_probability": 1.0,
            "max_grad_norm": 1000.0,
        },
        "divl": {
            "num_atoms": 201,
            "v_min": -0.1,
            "v_max": 1.1,
            "adaptive_tau": True,
            "use_adaptive_quantile": True,
            "tau_base": 0.6,
            "entropy_coefficient": 0.3,
            "tau_min": 0.3,
            "tau_max": 0.6,
            "interpolate_quantile": False,
        },
        "actor": {"hidden_dim": 8},
        "flow": {"num_steps": 2},
        "training": {"seed": 17},
    }


def test_explicit_min_and_original_tau_bounds_are_exact_update_parity():
    batch = make_synthetic_replay(
        num_samples=6,
        obs_dim=3,
        generated_horizon=2,
        executed_horizon=2,
        action_dim=2,
    )
    original = _controlled_v_config()
    fallback = copy.deepcopy(original)
    fallback["critic"].update(
        {
            "v_q_aggregation": "min",
            "v_tau_mode": "adaptive",
            "v_tau_min": 0.3,
            "v_tau_max": 0.6,
        }
    )
    torch.manual_seed(101)
    original_state = build_train_state(
        original, batch, multimodal_critic_factory=lambda *_args: _critic()
    )
    torch.manual_seed(101)
    fallback_state = build_train_state(
        fallback, batch, multimodal_critic_factory=lambda *_args: _critic()
    )
    torch.manual_seed(202)
    original_metrics = critic_update(original_state, batch, original)
    torch.manual_seed(202)
    fallback_metrics = critic_update(fallback_state, batch, fallback)
    for key in (
        "critic_loss",
        "q_loss",
        "divl_loss",
        "target_mean",
        "v_divl_mean",
        "adaptive_tau_mean",
    ):
        assert original_metrics[key] == fallback_metrics[key]
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            original_state.critic.parameters(),
            fallback_state.critic.parameters(),
            strict=True,
        )
    )


def test_pair_mean_high_tau_mode_is_reported_and_bounded():
    batch = make_synthetic_replay(
        num_samples=6,
        obs_dim=3,
        generated_horizon=2,
        executed_horizon=2,
        action_dim=2,
    )
    config = _controlled_v_config()
    config["critic"].update(
        {
            "v_q_aggregation": "mean",
            "v_tau_mode": "adaptive",
            "v_tau_min": 0.5,
            "v_tau_max": 0.7,
        }
    )
    state = build_train_state(
        config, batch, multimodal_critic_factory=lambda *_args: _critic()
    )
    metrics = critic_update(state, batch, config)
    assert metrics["v_q_aggregation_is_mean"] == 1.0
    assert metrics["v_q_aggregation_is_min"] == 0.0
    assert metrics["v_tau_config_min"] == 0.5
    assert metrics["v_tau_config_max"] == 0.7
    assert 0.5 <= metrics["adaptive_tau_min"]
    assert metrics["adaptive_tau_max"] <= 0.7


def test_ten_q_ca_requires_unanimous_sign_not_majority_or_pair_min():
    advantages = torch.empty(10, 3)
    advantages[:, 0] = torch.arange(1.0, 11.0)
    advantages[:, 1] = -torch.arange(1.0, 11.0)
    advantages[:, 2] = 1.0
    advantages[-1, 2] = -1.0
    result = safe_max(advantages, dim=0)
    assert torch.equal(result, torch.tensor([1.0, -1.0, 0.0]))


def test_chipo_statistics_use_all_ten_raw_q_heads():
    q_values = torch.arange(40.0).reshape(10, 1, 4)
    ratio = torch.tensor([[0.8, 1.0, 1.2, 1.4]])
    advantage, stats = chi2_pessimistic_advantage(
        q_values,
        ratio,
        beta_base=0.1,
        q_std_target=1.0,
        ensemble_alpha=5.0,
        normalize_group=False,
        advantage_clip=None,
    )
    expected_beta, expected_std = compute_chipo_beta(
        q_values, beta_base=0.1, q_std_target=1.0
    )
    assert advantage.shape == (1, 4)
    assert stats.q_mean == pytest.approx(float(q_values.mean(dim=0).mean()))
    assert stats.q_min == pytest.approx(float(q_values.min(dim=0).values.mean()))
    assert stats.q_ensemble_std == pytest.approx(float(expected_std))
    assert stats.beta == pytest.approx(float(expected_beta))


def test_clean_actor_raw_q_path_does_not_read_value_heads(monkeypatch):
    critic = _critic()
    batch = make_synthetic_replay(
        num_samples=2, obs_dim=3, generated_horizon=2, executed_horizon=2, action_dim=2
    )
    monkeypatch.setattr(
        critic,
        "value_logits_from_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("actor read V")),
    )
    state = SimpleNamespace(
        critic=critic,
        divl=None,
        support=torch.linspace(-0.1, 1.1, 201),
        running_mad=SimpleNamespace(value=1.0),
        conformal_scale=1.0,
        actor_step=0,
    )
    candidates = batch.action_chunks.reshape(2, 1, -1).repeat(1, 4, 1)
    config = {
        "critic": {"use_execution_mask": True},
        "divl": {"enabled": True},
        "actor": {
            "ogpo_variant": "ca",
            "advantage_mode": "conservative",
            "q_ensemble_source": "raw_heads",
            "q_ensemble_size": 10,
            "q_correlation_log_period": 100,
        },
        "uncertainty": {},
    }
    advantage, diagnostics = trainer.conservative_advantages_for_candidates(
        state, batch.observations, candidates, batch, config
    )
    assert advantage.shape == (2, 4)
    assert diagnostics["q_ensemble_size"] == 10.0
    assert diagnostics["final_adv_abs_mean"] == pytest.approx(
        float(advantage.abs().mean().item())
    )


def test_three_pair_checkpoint_fails_loudly_for_five_pair_actor(tmp_path):
    source = _critic(num_pairs=3)
    destination = _critic(num_pairs=5)
    payload = {
        "critic_format": "gemma_siglip_multihead",
        "critic_metadata": {
            "num_pairs": 3,
            "num_raw_q_heads": 6,
            "q_heads_per_member": 2,
        },
        "multimodal_critic": source.state_dict(),
    }
    checkpoint = tmp_path / "legacy_3pair.pt"
    torch.save(payload, checkpoint)
    state = SimpleNamespace(critic=destination)
    with pytest.raises(ValueError, match="5-pair / 10-Q.*3 pairs / 6 raw Q heads"):
        trainer.load_critic_checkpoint(checkpoint, state, load_optimizer=False)


def test_clean_configs_resolve_to_five_pairs_and_ten_raw_q():
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from train_udivl_critic import load_config

    critic_cfg = load_config(
        root / "configs/ogpo/robotwin_pick_dual_bottles_pytorch_divl_5pair_g995.yaml"
    )
    actor_cfg = load_config(
        root
        / "configs/ogpo/robotwin_pick_dual_bottles_pytorch_divl5_ogpo10q_ca_chipo_fullchain.yaml"
    )
    assert critic_cfg["critic"]["ensemble_size"] == 5
    assert critic_cfg["critic"]["q_heads_per_member"] == 2
    assert critic_cfg["data"]["gamma"] == pytest.approx(0.995)
    assert critic_cfg["data"]["n_step"] == 2
    assert critic_cfg["critic"]["bootstrap_probability"] == pytest.approx(1.0)
    assert actor_cfg["critic"]["ensemble_size"] == 5
    assert actor_cfg["actor"]["q_ensemble_source"] == "raw_heads"
    assert actor_cfg["actor"]["q_ensemble_size"] == 10
    assert actor_cfg["actor"]["group_size"] == 4
    assert actor_cfg["flow"]["adapter"] == "pi05_pytorch"


def test_five_pair_divl_update_reports_per_pair_and_raw_ensemble_metrics():
    batch = make_synthetic_replay(
        num_samples=6, obs_dim=3, generated_horizon=2, executed_horizon=2, action_dim=2
    )
    config = {
        "critic": {
            "architecture": "gemma_siglip_multihead",
            "ensemble_size": 5,
            "expected_num_pairs": 5,
            "expected_num_raw_q_heads": 10,
            "double_q_divl": True,
            "q_heads_per_member": 2,
            "q_representation": "scalar",
            "bootstrap_probability": 1.0,
            "q_correlation_log_period": 1,
        },
        "divl": {
            "num_atoms": 201,
            "v_min": -0.1,
            "v_max": 1.1,
            "adaptive_tau": True,
            "use_adaptive_quantile": True,
            "tau_base": 0.6,
            "entropy_coefficient": 0.3,
            "tau_min": 0.3,
            "tau_max": 0.6,
            "interpolate_quantile": False,
        },
        "actor": {"hidden_dim": 8},
        "flow": {"num_steps": 2},
    }
    state = build_train_state(
        config,
        batch,
        multimodal_critic_factory=lambda *_args: _critic(),
    )
    metrics = critic_update(state, batch, config)
    assert metrics["raw_10q_mean"] == metrics["raw_q_mean"]
    assert metrics["raw_10q_std"] == metrics["raw_q_std"]
    assert "raw_q_ensemble_std" in metrics
    assert "raw_q_pair_correlation" in metrics
    assert "mean_pair_disagreement" in metrics
    for member in range(5):
        for key in (
            f"q_loss_member_{member}",
            f"q_mean_member_{member}_q1",
            f"q_std_member_{member}_q1",
            f"q_mean_member_{member}_q2",
            f"q_std_member_{member}_q2",
            f"q_clipped_member_{member}_mean",
            f"q_clipped_member_{member}_std",
            f"v_expected_member_{member}",
            f"v_expected_member_{member}_std",
            f"v_entropy_member_{member}",
            f"v_tau_member_{member}",
            f"v_loss_member_{member}",
        ):
            assert key in metrics


def test_click_bell_100ep_configs_resolve_to_8k_critic_then_4k_actor():
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from train_udivl_critic import load_config

    critic_cfg = load_config(
        root
        / "configs/ogpo/robotwin_click_bell_100ep_pytorch_divl_5pair_g995_8k.yaml"
    )
    actor_cfg = load_config(
        root
        / "configs/ogpo/robotwin_click_bell_100ep_pytorch_divl5_ogpo10q_ca_chipo_4k.yaml"
    )
    assert critic_cfg["training"]["critic_steps"] == 8000
    assert critic_cfg["critic"]["ensemble_size"] == 5
    assert critic_cfg["data"]["dataset_path"].endswith("click_bell_dense_100ep_g999_train.pt")
    assert critic_cfg["data"]["gamma"] == pytest.approx(0.995)
    assert critic_cfg["data"]["n_step"] == 2
    assert actor_cfg["training"]["actor_steps"] == 4000
    assert actor_cfg["actor"]["q_ensemble_source"] == "raw_heads"
    assert actor_cfg["actor"]["q_ensemble_size"] == 10
    assert actor_cfg["actor"]["group_size"] == 4
    assert actor_cfg["evaluation"]["tasks"] == ["click_bell"]
