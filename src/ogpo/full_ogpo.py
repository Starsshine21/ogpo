from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOStats:
    loss: torch.Tensor
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    clip_fraction: float
    ratio_p5: float = 0.0
    ratio_p50: float = 0.0
    ratio_p95: float = 0.0
    log_ratio_mean: float = 0.0
    log_ratio_std: float = 0.0
    ratio: torch.Tensor | None = None
    clipped_ratio: torch.Tensor | None = None
    log_ratio: torch.Tensor | None = None


def full_chain_ais_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: torch.Tensor | float,
    log_ratio_clip: float = 20.0,
    logprob_normalizer: float = 1.0,
    upper_ratio_bound: torch.Tensor | float | None = None,
) -> PPOStats:
    """OGPO AIS objective with one joint likelihood ratio per flow chain.

    ``logprob_normalizer`` scales the summed joint log-ratio, which is the
    mathematically equivalent placement used by the released OGPO
    ``normalize_denoising_horizon``/``normalize_act_space_dimension`` options.
    It is deliberately applied before ``exp`` and before the single clip.
    """
    assert new_log_probs.shape == old_log_probs.shape
    assert new_log_probs.ndim == 2
    assert advantages.shape == new_log_probs.shape[:1]
    if float(logprob_normalizer) <= 0.0:
        raise ValueError("logprob_normalizer must be positive")
    if float(log_ratio_clip) <= 0.0:
        raise ValueError("log_ratio_clip must be positive")
    log_ratio = (new_log_probs - old_log_probs.detach()).sum(dim=1)
    log_ratio = log_ratio / float(logprob_normalizer)
    ratio = torch.exp(log_ratio.clamp(-log_ratio_clip, log_ratio_clip))
    eps = torch.as_tensor(clip_eps, dtype=ratio.dtype, device=ratio.device)
    if eps.ndim > 1 or (
        eps.ndim == 1 and eps.numel() not in {1, ratio.numel()}
    ):
        raise ValueError("chain clip must be scalar or have one value per trajectory")
    eps = eps.expand_as(ratio)
    clipped_ratio = torch.maximum(torch.minimum(ratio, 1.0 + eps), 1.0 - eps)
    if upper_ratio_bound is not None:
        upper = torch.as_tensor(upper_ratio_bound, dtype=ratio.dtype, device=ratio.device).expand_as(ratio)
        if torch.any(upper < 1.0 - eps):
            raise ValueError("upper_ratio_bound must not be below the PPO lower bound")
        clipped_ratio = torch.minimum(clipped_ratio, upper)
    advantage = advantages.detach()
    objective = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    quantiles = torch.quantile(ratio.detach(), ratio.new_tensor([0.05, 0.50, 0.95]))
    return PPOStats(
        loss=-objective.mean(),
        ratio_mean=float(ratio.detach().mean().item()),
        ratio_std=float(ratio.detach().std(unbiased=False).item()),
        ratio_min=float(ratio.detach().min().item()),
        ratio_max=float(ratio.detach().max().item()),
        clip_fraction=float(ratio.ne(clipped_ratio).float().mean().item()),
        ratio_p5=float(quantiles[0].item()),
        ratio_p50=float(quantiles[1].item()),
        ratio_p95=float(quantiles[2].item()),
        log_ratio_mean=float(log_ratio.detach().mean().item()),
        log_ratio_std=float(log_ratio.detach().std(unbiased=False).item()),
        ratio=ratio.detach(),
        clipped_ratio=clipped_ratio.detach(),
        log_ratio=log_ratio.detach(),
    )


def full_chain_joint_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: torch.Tensor | float,
    logprob_normalizer: float = 1.0,
    upper_ratio_bound: torch.Tensor | float | None = None,
    log_ratio_clip: float = 20.0,
) -> PPOStats:
    """Explicit alias for one joint/full-chain PPO ratio and one clip."""
    return full_chain_ais_ppo_loss(
        new_log_probs,
        old_log_probs,
        advantages,
        clip_eps=clip_eps,
        logprob_normalizer=logprob_normalizer,
        upper_ratio_bound=upper_ratio_bound,
        log_ratio_clip=log_ratio_clip,
    )


def full_chain_chipo_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: torch.Tensor | float,
    beta: torch.Tensor | float,
    r_max: float,
    logprob_normalizer: float = 1.0,
    log_ratio_clip: float = 20.0,
) -> PPOStats:
    """Official ChiPO asymmetric full-chain PPO surrogate.

    PyTorch port of OGPO_public ``compute_chi_po_ppo_loss`` (upstream commit
    ``0b3be413cde766a41257c6b19c0c2b06393a557f``): the lower bound remains
    ``1-eps`` while the upper bound is
    ``min(1+eps, 1+R_max/beta)``.  The joint chain ratio is formed once.
    """
    eps = torch.as_tensor(clip_eps, dtype=new_log_probs.dtype, device=new_log_probs.device)
    beta_t = torch.as_tensor(beta, dtype=eps.dtype, device=eps.device)
    if torch.any(beta_t < 0) or not float(r_max) > 0.0:
        raise ValueError("ChiPO beta must be non-negative and r_max positive")
    if torch.all(beta_t > 0):
        upper = torch.minimum(
            1.0 + eps,
            1.0 + float(r_max) / beta_t,
        )
    else:
        upper = 1.0 + eps
    return full_chain_ais_ppo_loss(
        new_log_probs,
        old_log_probs,
        advantages,
        clip_eps=eps,
        upper_ratio_bound=upper,
        logprob_normalizer=logprob_normalizer,
        log_ratio_clip=log_ratio_clip,
    )


def full_chain_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: torch.Tensor | float,
    timestep_weights: torch.Tensor | None = None,
    log_ratio_clip: float = 20.0,
) -> PPOStats:
    """Full-chain OGPO objective.

    new_log_probs/old_log_probs: [N, K]
    advantages: [N], broadcast over all stochastic transitions.
    """
    assert new_log_probs.shape == old_log_probs.shape
    assert advantages.shape == new_log_probs.shape[:1]
    log_ratio = (new_log_probs - old_log_probs.detach()).clamp(-log_ratio_clip, log_ratio_clip)
    ratio = torch.exp(log_ratio)
    adv = advantages.detach().unsqueeze(-1)
    unclipped = ratio * adv
    eps = torch.as_tensor(clip_eps, dtype=ratio.dtype, device=ratio.device)
    if eps.ndim == 1:
        eps = eps.unsqueeze(-1)
    eps = eps.expand_as(ratio)
    clipped_ratio = torch.maximum(torch.minimum(ratio, 1.0 + eps), 1.0 - eps)
    clipped = clipped_ratio * adv
    objective = torch.minimum(unclipped, clipped)
    if timestep_weights is not None:
        assert timestep_weights.shape == objective.shape
        objective = objective * timestep_weights.detach()
    loss = -objective.mean()
    clip_fraction = (ratio.ne(clipped_ratio)).float().mean()
    flat_ratio = ratio.detach().reshape(-1)
    flat_clipped = clipped_ratio.detach().reshape(-1)
    flat_log_ratio = log_ratio.detach().reshape(-1)
    quantiles = torch.quantile(flat_ratio, flat_ratio.new_tensor([0.05, 0.50, 0.95]))
    return PPOStats(
        loss=loss,
        ratio_mean=float(ratio.detach().mean().item()),
        ratio_std=float(ratio.detach().std(unbiased=False).item()),
        ratio_min=float(ratio.detach().min().item()),
        ratio_max=float(ratio.detach().max().item()),
        clip_fraction=float(clip_fraction.item()),
        ratio_p5=float(quantiles[0].item()),
        ratio_p50=float(quantiles[1].item()),
        ratio_p95=float(quantiles[2].item()),
        log_ratio_mean=float(flat_log_ratio.mean().item()),
        log_ratio_std=float(flat_log_ratio.std(unbiased=False).item()),
        ratio=flat_ratio,
        clipped_ratio=flat_clipped,
        log_ratio=flat_log_ratio,
    )
