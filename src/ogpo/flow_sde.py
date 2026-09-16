from __future__ import annotations

import torch
import torch.nn as nn

from .openpi_flow_spec import FlowRollout, OpenPIStochasticFlowPolicy


class GaussianFlowPolicy(OpenPIStochasticFlowPolicy):
    """Torch stochastic flow-transition adapter.

    This compact trainable fallback follows the OpenPI PI0/PI0.5 flow
    convention: t=1 is Gaussian noise, t=0 is the action endpoint, and each
    Euler transition steps backward with dt=-1/K. The PI0.5 adapter exposes
    the same transition interface while delegating `predict_velocity` to the
    OpenPI action expert.
    """

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 8,
        stochastic_variance: float = 0.04,
        sde_mode: str = "gaussian_adapter",
        constant_noise_std: float = 0.005,
        learn_sde_std: bool = True,
        randn_clip_value: float = 3.0,
    ):
        super().__init__(
            action_dim=action_dim,
            num_steps=num_steps,
            stochastic_variance=stochastic_variance,
            sde_mode=sde_mode,
            constant_noise_std=constant_noise_std,
            learn_sde_std=learn_sde_std,
            randn_clip_value=randn_clip_value,
        )
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.Linear(condition_dim + action_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _features(self, x_t: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=x_t.dtype, device=x_t.device)
        while t.ndim < x_t.ndim:
            t = t.unsqueeze(-1)
        if t.shape[-1] != 1:
            t = t[..., :1]
        return torch.cat([x_t, condition, t], dim=-1)

    def predict_velocity(self, x_t: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.net(self._features(x_t, condition, timestep))

    def condition_from_batch(self, batch, *, next_observation: bool = False) -> torch.Tensor:
        return batch.next_observations if next_observation else batch.observations
