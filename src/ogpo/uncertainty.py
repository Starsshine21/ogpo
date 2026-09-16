from __future__ import annotations

from collections.abc import Mapping

import torch


def conformal_scale(
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    returns: torch.Tensor,
    *,
    coverage_delta: float,
    min_samples: int = 16,
    eps: float = 1e-6,
) -> float:
    if q_mean.numel() < min_samples:
        return 1.0
    ratios = (q_mean - returns).abs() / q_std.clamp_min(eps)
    q = torch.quantile(ratios.detach().float(), 1.0 - float(coverage_delta))
    return float(q.clamp_min(1.0).item())


def state_entropy_weight(entropy_norm: torch.Tensor, eta_entropy: float) -> torch.Tensor:
    return torch.exp(-float(eta_entropy) * entropy_norm.clamp(0.0, 1.0))


def adaptive_clip(entropy_norm: torch.Tensor, eps_min: float, eps_max: float) -> torch.Tensor:
    return eps_min + (1.0 - entropy_norm.clamp(0.0, 1.0)) * (eps_max - eps_min)


def actor_clip_for_uncertainty(
    entropy_norm: torch.Tensor,
    actor_config: Mapping[str, object],
    uncertainty_config: Mapping[str, object],
) -> torch.Tensor:
    eps_max = float(actor_config.get("ppo_clip_max", 0.2))
    if not bool(uncertainty_config.get("adapt_ppo_clip", False)):
        return torch.full_like(entropy_norm, eps_max)
    return adaptive_clip(
        entropy_norm,
        float(actor_config.get("ppo_clip_min", 0.05)),
        eps_max,
    )


def kl_uncertainty_scale(
    regularization_config: Mapping[str, object],
    uncertainty_config: Mapping[str, object],
) -> float:
    if not bool(uncertainty_config.get("adapt_kl_beta", False)):
        return 0.0
    return float(regularization_config.get("kl_uncertainty_scale", 1.0))


def adaptive_kl_beta(entropy_norm: torch.Tensor, beta_base: float, uncertainty_scale: float) -> torch.Tensor:
    return float(beta_base) * (1.0 + float(uncertainty_scale) * entropy_norm.clamp(0.0, 1.0))


def state_adaptive_kl_penalty(
    transition_kl: torch.Tensor,
    entropy_norm: torch.Tensor,
    *,
    group_size: int,
    beta_base: float,
    uncertainty_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weight candidate transition KL by its replay state's DIVL entropy."""
    if transition_kl.ndim != 1:
        raise ValueError("transition KL must have shape [batch * group_size]")
    if transition_kl.numel() != entropy_norm.numel() * int(group_size):
        raise ValueError("transition KL does not match the state groups")
    per_state_kl = transition_kl.reshape(entropy_norm.numel(), int(group_size)).mean(dim=1)
    beta = adaptive_kl_beta(entropy_norm, beta_base, uncertainty_scale)
    return (beta * per_state_kl).mean(), per_state_kl.mean(), beta.mean()


def support_weight(
    sigma_epi: torch.Tensor,
    support_distance: torch.Tensor,
    *,
    lambda_epi: float,
    lambda_support: float,
    support_threshold: float | None = None,
) -> torch.Tensor:
    weight = torch.exp(-float(lambda_epi) * sigma_epi - float(lambda_support) * support_distance)
    if support_threshold is not None:
        weight = torch.where(support_distance > float(support_threshold), torch.zeros_like(weight), weight)
    return weight
