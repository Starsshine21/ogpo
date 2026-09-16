import torch
import torch.nn as nn

from ogpo.critic_targets import aggregate_value_heads
from ogpo.gemma_siglip_backbone import (
    LoRALinear,
    configure_critic_stage,
    install_final_gemma_lora,
)
from ogpo.multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic


def test_subsample_min_uses_two_distinct_heads_per_sample():
    values = torch.tensor(
        [[1.0, 8.0, 3.0, 7.0], [2.0, 6.0, 9.0, 4.0], [5.0, 1.0, 4.0, 2.0]]
    )
    generator = torch.Generator().manual_seed(17)

    aggregated, indices = aggregate_value_heads(values, "subsample_min", generator=generator)

    assert indices.shape == (2, values.shape[1])
    assert torch.all(indices[0] != indices[1])
    expected = torch.minimum(
        values.gather(0, indices[0].unsqueeze(0)).squeeze(0),
        values.gather(0, indices[1].unsqueeze(0)).squeeze(0),
    )
    assert torch.equal(aggregated, expected)


def test_mean_and_global_min_aggregation_are_exact():
    values = torch.tensor([[1.0, 4.0], [3.0, 2.0], [2.0, 8.0]])

    mean, mean_indices = aggregate_value_heads(values, "mean")
    minimum, min_indices = aggregate_value_heads(values, "min")

    assert torch.equal(mean, torch.tensor([2.0, 14.0 / 3.0]))
    assert torch.equal(minimum, torch.tensor([1.0, 2.0]))
    assert mean_indices is None
    assert min_indices is None


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.other = nn.Linear(8, 8)


class _StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(3, 8)
        self.gemma_model = nn.Module()
        self.gemma_model.layers = nn.ModuleList([_Layer() for _ in range(5)])
        self.visual_projection = nn.Linear(8, 8)
        self.proprio_projection = nn.Linear(5, 8)
        self.readout_token = nn.Parameter(torch.zeros(1, 1, 8))


def _critic():
    return MultiHeadUdivlCritic(
        _StateEncoder(),
        MultiHeadUdivlCore(
            state_dim=8,
            action_dim=2,
            max_horizon=4,
            action_hidden_dim=8,
            head_hidden_dim=8,
            num_attention_heads=2,
            num_value_atoms=11,
            num_pairs=3,
        ),
    )


def test_head_stages_train_fusion_and_heads_but_freeze_pretrained_backbone():
    critic = _critic()

    configure_critic_stage(critic, "head_mc")

    assert all(not parameter.requires_grad for parameter in critic.state_encoder.vision_model.parameters())
    assert all(not parameter.requires_grad for parameter in critic.state_encoder.gemma_model.parameters())
    assert all(parameter.requires_grad for parameter in critic.state_encoder.visual_projection.parameters())
    assert all(parameter.requires_grad for parameter in critic.state_encoder.proprio_projection.parameters())
    assert critic.state_encoder.readout_token.requires_grad
    assert all(parameter.requires_grad for parameter in critic.core.parameters())


def test_lora_stage_only_adapts_requested_final_gemma_layers():
    critic = _critic()
    replaced = install_final_gemma_lora(
        critic.state_encoder,
        final_n_layers=2,
        rank=2,
        alpha=4.0,
        target_suffixes=("q_proj",),
    )

    configure_critic_stage(critic, "gemma_lora_td")

    assert replaced == (3, 4)
    for index, layer in enumerate(critic.state_encoder.gemma_model.layers):
        if index >= 3:
            assert isinstance(layer.q_proj, LoRALinear)
            assert layer.q_proj.lora_a.requires_grad
            assert layer.q_proj.lora_b.requires_grad
            assert not layer.q_proj.base.weight.requires_grad
        else:
            assert isinstance(layer.q_proj, nn.Linear)
            assert not layer.q_proj.weight.requires_grad
        assert not layer.other.weight.requires_grad


def test_full_td_stage_trains_one_shared_vlm_and_all_heads():
    critic = _critic()

    configure_critic_stage(critic, "full_td")

    assert all(parameter.requires_grad for parameter in critic.state_encoder.parameters())
    assert all(parameter.requires_grad for parameter in critic.core.parameters())
