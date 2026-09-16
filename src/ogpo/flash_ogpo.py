from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FlashRollout:
    x_t: torch.Tensor
    x_prev: torch.Tensor
    timestep: torch.Tensor
    old_log_prob: torch.Tensor
    endpoint: torch.Tensor
    selected_step: int | torch.Tensor


@dataclass(frozen=True)
class FlashStats:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    clip_fraction: float


def flash_ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: torch.Tensor | float,
    upper_ratio_bound: torch.Tensor | float | None = None,
    rectification_weight: torch.Tensor | float = 1.0,
    log_ratio_clip: float = 20.0,
) -> FlashStats:
    """Selected-transition Flash-OGPO objective."""
    assert new_log_prob.shape == old_log_prob.shape == advantages.shape
    log_ratio = (new_log_prob - old_log_prob.detach()).clamp(-log_ratio_clip, log_ratio_clip)
    ratio = torch.exp(log_ratio)
    eps = torch.as_tensor(clip_eps, dtype=ratio.dtype, device=ratio.device)
    while eps.ndim < ratio.ndim:
        eps = eps.unsqueeze(-1)
    eps = eps.expand_as(ratio)
    if upper_ratio_bound is None:
        upper = 1.0 + eps
    else:
        upper = torch.as_tensor(
            upper_ratio_bound, dtype=ratio.dtype, device=ratio.device
        )
        while upper.ndim < ratio.ndim:
            upper = upper.unsqueeze(-1)
        upper = upper.expand_as(ratio)
        if torch.any(upper < 1.0 - eps):
            raise ValueError("PPO upper ratio bound must not be below the lower bound")
    clipped_ratio = torch.maximum(torch.minimum(ratio, upper), 1.0 - eps)
    adv = advantages.detach()
    objective = torch.minimum(ratio * adv, clipped_ratio * adv)
    weight = torch.as_tensor(rectification_weight, dtype=ratio.dtype, device=ratio.device)
    objective = objective * weight
    loss = -objective.mean()
    return FlashStats(
        loss=loss,
        per_sample_loss=-objective,
        ratio_mean=float(ratio.detach().mean().item()),
        ratio_std=float(ratio.detach().std(unbiased=False).item()),
        ratio_min=float(ratio.detach().min().item()),
        ratio_max=float(ratio.detach().max().item()),
        clip_fraction=float((ratio.ne(clipped_ratio)).float().mean().item()),
    )


@torch.no_grad()
def sample_flash_rollout(
    old_policy,
    condition: Any,
    *,
    group_size: int,
    selected_step: int | torch.Tensor,
    generator: torch.Generator | None = None,
) -> FlashRollout:
    rollout = old_policy.rollout(
        condition,
        group_size=group_size,
        selected_timestep=selected_step,
        generator=generator,
        deterministic_except_selected=True,
    )
    if isinstance(selected_step, torch.Tensor):
        batch = old_policy.condition_batch_size(condition)
        selected = selected_step.to(device=rollout.endpoint.device, dtype=torch.long)
        if selected.shape != (batch,):
            raise ValueError("selected_step tensor must have shape [batch]")
        selected_g = selected.repeat_interleave(group_size)
        state_index = selected_g.view(-1, 1, 1).expand(-1, 1, rollout.states.shape[-1])
        scalar_index = selected_g.view(-1, 1)
        time_index = selected_g.view(-1, 1, 1).expand(-1, 1, rollout.timesteps.shape[-1])
        return FlashRollout(
            x_t=rollout.states.gather(1, state_index).squeeze(1),
            x_prev=rollout.next_states.gather(1, state_index).squeeze(1),
            timestep=rollout.timesteps.gather(1, time_index).squeeze(1),
            old_log_prob=rollout.log_probs.gather(1, scalar_index).squeeze(1),
            endpoint=rollout.endpoint,
            selected_step=selected,
        )
    return FlashRollout(
        x_t=rollout.states[:, selected_step],
        x_prev=rollout.next_states[:, selected_step],
        timestep=rollout.timesteps[:, selected_step],
        old_log_prob=rollout.log_probs[:, selected_step],
        endpoint=rollout.endpoint,
        selected_step=selected_step,
    )
