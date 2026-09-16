from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .chunk_transition import flatten_masked_action


def _mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class ScalarQMember(nn.Module):
    """One independent scalar Q(s, A_prefix) member."""

    def __init__(
        self,
        obs_dim: int,
        generated_horizon: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        randomized_prior_scale: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.generated_horizon = generated_horizon
        self.action_dim = action_dim
        input_dim = obs_dim + generated_horizon * action_dim
        self.net = _mlp(input_dim, hidden_dim, num_layers, 1)
        self.randomized_prior_scale = float(randomized_prior_scale)
        self.prior = _mlp(input_dim, hidden_dim, num_layers, 1)
        self.prior.requires_grad_(False)

    def forward(
        self,
        observations: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        assert observations.ndim == 2
        assert action_chunks.shape[:2] == execution_masks.shape
        action_flat = flatten_masked_action(action_chunks, execution_masks)
        x = torch.cat([observations, action_flat], dim=-1)
        value = self.net(x)
        if self.randomized_prior_scale != 0.0:
            with torch.no_grad():
                prior_value = self.prior(x)
            value = value + self.randomized_prior_scale * prior_value
        return value.squeeze(-1)


class ScalarQEnsemble(nn.Module):
    """Independent action-chunk Q ensemble."""

    def __init__(
        self,
        ensemble_size: int,
        obs_dim: int,
        generated_horizon: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        randomized_prior_scale: float = 0.0,
    ):
        super().__init__()
        self.members = nn.ModuleList(
            [
                ScalarQMember(
                    obs_dim,
                    generated_horizon,
                    action_dim,
                    hidden_dim,
                    num_layers,
                    randomized_prior_scale,
                )
                for _ in range(ensemble_size)
            ]
        )

    @property
    def ensemble_size(self) -> int:
        return len(self.members)

    def forward(
        self,
        observations: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [member(observations, action_chunks, execution_masks) for member in self.members],
            dim=0,
        )


def clone_target(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    for param in target.parameters():
        param.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
        target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def assert_no_gradients(module: nn.Module, name: str) -> None:
    for param_name, param in module.named_parameters():
        if param.grad is not None:
            raise AssertionError(f"{name}.{param_name} unexpectedly has a gradient")
