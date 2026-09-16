from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from ogpo.gemma_siglip_backbone import LoRALinear
from ogpo.multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic
from ogpo.replay import make_synthetic_replay
from ogpo.evaluator import offline_calibration_metrics
from ogpo.trainer import (
    accumulated_critic_update,
    apply_scheduled_critic_stage,
    build_train_state,
    conservative_advantages_for_candidates,
    critic_update,
    load_checkpoint,
    maybe_advance_critic_stage,
    save_checkpoint,
)


class ReplayStateEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, batch, *, next_observation: bool = False):
        values = batch.next_observations if next_observation else batch.observations
        return self.projection(values)


def _factory(batch, config):
    atoms = int(config["divl"]["num_atoms"])
    return MultiHeadUdivlCritic(
        ReplayStateEncoder(batch.obs_dim, 8),
        MultiHeadUdivlCore(
            state_dim=8,
            action_dim=batch.action_dim,
            max_horizon=batch.generated_horizon,
            action_hidden_dim=8,
            head_hidden_dim=16,
            num_attention_heads=2,
            num_value_atoms=atoms,
            num_pairs=3,
        ),
    )


class _LoraReplayEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, 8)
        self.vision_model = nn.Linear(3, 8)
        self.gemma_model = nn.Module()
        layer = nn.Module()
        layer.q_proj = nn.Linear(8, 8)
        self.gemma_model.layers = nn.ModuleList([layer])
        self.visual_projection = nn.Linear(8, 8)
        self.proprio_projection = nn.Linear(4, 8)
        self.readout_token = nn.Parameter(torch.zeros(1, 1, 8))

    def forward(self, batch, *, next_observation=False):
        values = batch.next_observations if next_observation else batch.observations
        return self.projection(values)


def _lora_factory(batch, config):
    return MultiHeadUdivlCritic(
        _LoraReplayEncoder(batch.obs_dim),
        MultiHeadUdivlCore(
            state_dim=8,
            action_dim=batch.action_dim,
            max_horizon=batch.generated_horizon,
            action_hidden_dim=8,
            head_hidden_dim=16,
            num_attention_heads=2,
            num_value_atoms=int(config["divl"]["num_atoms"]),
            num_pairs=3,
        ),
    )


def _config():
    return {
        "critic": {
            "architecture": "gemma_siglip_multihead",
            "ensemble_size": 3,
            "learning_rate": 3e-4,
        },
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {"hidden_dim": 16},
        "flow": {"num_steps": 3},
    }


def test_multimodal_builder_and_critic_update_use_paired_heads():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    state = build_train_state(_config(), batch, multimodal_critic_factory=_factory)

    assert isinstance(state.critic, MultiHeadUdivlCritic)
    assert state.critic.ensemble_size == 3
    assert state.divl is None
    assert all(not parameter.requires_grad for parameter in state.target_critic.parameters())
    metrics = critic_update(state, batch, _config())
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))


def test_multimodal_builder_uses_real_config_factory_by_default(monkeypatch):
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    calls = []

    def fake_real_factory(replay_batch, config):
        calls.append((replay_batch, config))
        return _factory(replay_batch, config)

    monkeypatch.setattr("ogpo.trainer.build_gemma_siglip_critic", fake_real_factory)

    state = build_train_state(_config(), batch)

    assert isinstance(state.critic, MultiHeadUdivlCritic)
    assert calls == [(batch, _config())]


def test_multimodal_head_mc_stage_uses_mc_targets_and_is_checkpointed(tmp_path):
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    batch = replace(batch, mc_returns=batch.chunk_returns + 0.5)
    config = _config()
    config["critic"].update({"stage": "head_mc", "lambda_mc": 0.0})
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    metrics = critic_update(state, batch, config)
    checkpoint = tmp_path / "stage.pt"
    save_checkpoint(state, config, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)

    assert metrics["critic_stage_head_mc"] == 1.0
    assert metrics["lambda_mc"] == 1.0
    assert state.critic_stage == "head_mc"
    assert state.critic_stage_step == 1
    assert payload["critic_stage"] == "head_mc"
    assert payload["critic_stage_step"] == 1


def test_multimodal_gradient_accumulation_is_one_effective_step():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    batch = replace(batch, mc_returns=batch.chunk_returns + 0.5)
    config = _config()
    config["critic"]["stage"] = "head_mc"
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    metrics = accumulated_critic_update(
        state,
        batch,
        config,
        microbatch_size=2,
    )

    assert state.step == 1
    assert state.critic_stage_step == 1
    assert metrics["effective_batch_size"] == 8.0
    assert metrics["microbatch_size"] == 2.0
    assert metrics["gradient_accumulation_steps"] == 4.0
    assert metrics["critic_grad_norm"] > 0.0


def test_scheduled_warmup_rebuilds_trainability_at_boundaries():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    config["critic"].update(
        {
            "stage": "head_mc",
            "stage_schedule": [
                {"stage": "head_mc", "steps": 2},
                {"stage": "head_td", "steps": 3},
                {"stage": "full_td"},
            ],
        }
    )
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    state.step = 2
    assert apply_scheduled_critic_stage(state, config)
    assert state.critic_stage == "head_td"
    state.step = 5
    assert apply_scheduled_critic_stage(state, config)
    assert state.critic_stage == "full_td"
    assert all(parameter.requires_grad for parameter in state.critic.parameters())


def test_multimodal_reference_action_samples_reduce_td_target_variance_path():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    config["critic"].update(
        {
            "reference_value_samples": 3,
            "lambda_divl_target": 0.5,
            "bootstrap_target": "subsample_min",
        }
    )
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)

    metrics = critic_update(state, batch, config)

    assert "reference_value_mean" in metrics
    assert metrics["reference_value_samples"] == 3.0
    assert torch.isfinite(torch.tensor(metrics["target_mean"]))


def test_critic_stage_advances_only_when_validation_gates_pass():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    config["critic"].update(
        {
            "stage": "head_mc",
            "stage_gates": {
                "min_stage_steps": 1,
                "min_pairwise_ranking_accuracy": 0.6,
                "min_interval_coverage": 0.8,
                "max_categorical_saturation": 0.2,
                "max_abs_exploitation_gap": 0.3,
            },
        }
    )
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    state.critic_stage_step = 1

    assert not maybe_advance_critic_stage(state, {"critic_loss": 0.0}, config)
    advanced = maybe_advance_critic_stage(
        state,
        {
            "validation_pairwise_ranking_accuracy": 0.7,
            "validation_interval_coverage": 0.9,
            "validation_categorical_saturation": 0.1,
            "validation_q_exploitation_gap": -0.2,
        },
        config,
    )

    assert advanced
    assert state.critic_stage == "head_td"
    assert state.critic_stage_step == 0


def test_multimodal_checkpoint_roundtrip_restores_online_and_target(tmp_path):
    torch.manual_seed(5)
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    critic_update(state, batch, config)
    with torch.no_grad():
        expected = state.critic(batch, batch.action_chunks, batch.execution_masks).clone()
    checkpoint = tmp_path / "multimodal.pt"
    save_checkpoint(state, config, checkpoint)

    with torch.no_grad():
        next(state.critic.parameters()).add_(100.0)
    load_checkpoint(checkpoint, state)

    with torch.no_grad():
        actual = state.critic(batch, batch.action_chunks, batch.execution_masks)
    assert torch.equal(expected, actual)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["critic_format"] == "gemma_siglip_multihead"
    assert "multimodal_critic" in payload
    assert "target_multimodal_critic" in payload
    assert torch.equal(
        payload["critic_metadata"]["action_mean"],
        state.critic.core.action_pool.action_mean.cpu(),
    )
    assert torch.equal(
        payload["critic_metadata"]["action_std"],
        state.critic.core.action_pool.action_std.cpu(),
    )


def test_multimodal_checkpoint_rebuilds_optimizer_when_restoring_lora_stage(tmp_path):
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    source_config = _config()
    source_config["critic"].update(
        {
            "stage": "gemma_lora_td",
            "gemma_lora": {
                "final_n_layers": 1,
                "rank": 2,
                "alpha": 4.0,
                "learning_rate": 1e-5,
                "target_modules": ["q_proj"],
            },
        }
    )
    source = build_train_state(source_config, batch, multimodal_critic_factory=_lora_factory)
    checkpoint = tmp_path / "lora_stage.pt"
    save_checkpoint(source, source_config, checkpoint)
    target_config = {**source_config, "critic": {**source_config["critic"], "stage": "head_mc"}}
    target = build_train_state(target_config, batch, multimodal_critic_factory=_lora_factory)

    load_checkpoint(checkpoint, target)

    assert target.critic_stage == "gemma_lora_td"
    assert len(target.critic_optimizer.param_groups) == 2
    assert any(
        module.lora_a.requires_grad
        for module in target.critic.state_encoder.gemma_model.modules()
        if isinstance(module, LoRALinear)
    )


def test_multimodal_critic_scores_actor_candidates_and_validation_batch():
    batch = make_synthetic_replay(
        num_samples=8,
        generated_horizon=4,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    candidates = torch.randn(batch.batch_size, 3, batch.generated_horizon * batch.action_dim)

    advantages, diagnostics = conservative_advantages_for_candidates(
        state,
        batch.observations,
        candidates,
        batch,
        config,
    )
    metrics = offline_calibration_metrics(state.critic, batch)

    assert advantages.shape == (batch.batch_size, 3)
    assert "state_entropy" in diagnostics
    assert "q_rmse" in metrics
