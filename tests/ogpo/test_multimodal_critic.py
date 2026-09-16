from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ogpo.multimodal_critic import MultiHeadUdivlCritic, MultiHeadUdivlCore


class CountingStateEncoder(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.projection = nn.Linear(3, state_dim)
        self.calls = 0

    def forward(self, batch, *, next_observation: bool = False) -> torch.Tensor:
        self.calls += 1
        values = batch.next_observations if next_observation else batch.observations
        return self.projection(values)


class TinyBatch:
    def __init__(self):
        self.observations = torch.randn(2, 3)
        self.next_observations = torch.randn(2, 3)


def _core() -> MultiHeadUdivlCore:
    return MultiHeadUdivlCore(
        state_dim=8,
        action_dim=2,
        max_horizon=4,
        action_hidden_dim=8,
        head_hidden_dim=16,
        num_attention_heads=2,
        num_value_atoms=5,
        num_pairs=3,
    )


def test_shared_state_encoding_serves_all_q_and_value_heads():
    encoder = CountingStateEncoder(state_dim=8)
    critic = MultiHeadUdivlCritic(encoder, _core())
    batch = TinyBatch()
    actions = torch.randn(2, 4, 2)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    features = critic.encode_state(batch)
    q_values = critic.q_from_features(features, actions, mask)
    value_logits = critic.value_logits_from_features(features)

    assert encoder.calls == 1
    assert q_values.shape == (3, 2)
    assert value_logits.shape == (3, 2, 5)


def test_masked_action_suffix_cannot_change_q_values():
    torch.manual_seed(0)
    critic = MultiHeadUdivlCritic(CountingStateEncoder(8), _core())
    batch = TinyBatch()
    actions = torch.randn(2, 4, 2)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    changed = actions.clone()
    changed[~mask] = 10_000.0
    features = critic.encode_state(batch)

    original_q = critic.q_from_features(features, actions, mask)
    changed_q = critic.q_from_features(features, changed, mask)

    assert torch.allclose(original_q, changed_q, atol=1e-6)


def test_q_value_heads_are_independent_and_receive_gradients():
    torch.manual_seed(1)
    core = _core()
    features = torch.randn(2, 8, requires_grad=True)
    actions = torch.randn(2, 4, 2)
    mask = torch.ones(2, 4, dtype=torch.bool)

    q_values = core.q_from_readout(features, actions, mask)
    value_logits = core.value_logits_from_readout(features)
    (q_values.mean() + value_logits.mean()).backward()

    assert not torch.equal(core.q_heads[0][0].weight, core.q_heads[1][0].weight)
    assert not torch.equal(core.value_heads[0][0].weight, core.value_heads[1][0].weight)
    assert core.action_pool.action_projection.weight.grad is not None
    for head in [*core.q_heads, *core.value_heads]:
        assert all(parameter.grad is not None for parameter in head.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_action_pool_rejects_a_sample_without_executed_actions():
    core = _core()
    actions = torch.randn(2, 4, 2)
    mask = torch.tensor([[True, False, False, False], [False, False, False, False]])

    with pytest.raises(ValueError, match="at least one executed action"):
        core.q_from_readout(torch.randn(2, 8), actions, mask)


def test_value_heads_accept_bfloat16_backbone_readout_with_float32_heads():
    core = _core()
    readout = torch.randn(2, 8, dtype=torch.bfloat16)

    logits = core.value_logits_from_readout(readout)

    assert logits.dtype == torch.float32
    assert logits.shape == (3, 2, 5)
