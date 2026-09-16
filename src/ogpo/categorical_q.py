from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def hl_gauss_projection(
    target_values: torch.Tensor,
    support: torch.Tensor,
    *,
    sigma_bins: float,
) -> torch.Tensor:
    """Project scalar targets to Gaussian-smoothed categorical labels."""
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("support must be rank-1 with at least two bins")
    if sigma_bins <= 0.0:
        raise ValueError("sigma_bins must be positive")
    support_f = support.detach().float()
    target_f = target_values.detach().to(device=support.device, dtype=torch.float32)
    clipped = target_f.clamp(float(support_f[0]), float(support_f[-1]))
    bin_width = (support_f[-1] - support_f[0]) / (support_f.numel() - 1)
    sigma = bin_width * float(sigma_bins)
    log_weights = -0.5 * (
        (support_f.view(*([1] * clipped.ndim), -1) - clipped.unsqueeze(-1)) / sigma
    ).square()
    return torch.softmax(log_weights, dim=-1).detach()


def nearest_support_indices(
    target_values: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Map clipped scalar targets to their nearest categorical support bin."""
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("support must be rank-1 with at least two bins")
    support_f = support.detach().to(device=target_values.device, dtype=torch.float32)
    target_f = target_values.detach().to(dtype=torch.float32)
    clipped = target_f.clamp(min=support_f[0], max=support_f[-1])
    right = torch.searchsorted(support_f, clipped.contiguous()).clamp(
        min=1, max=support_f.numel() - 1
    )
    left = right - 1
    choose_right = (support_f[right] - clipped).abs() < (
        clipped - support_f[left]
    ).abs()
    return torch.where(choose_right, right, left).detach()


def one_hot_categorical_cross_entropy(
    q_logits: torch.Tensor,
    target_values: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy against the one-hot nearest-bin projection of scalars.

    Class-index cross-entropy is exactly equivalent to explicitly materializing
    a one-hot label, while avoiding an extra tensor with one entry per bin.
    Leading logits dimensions (for example ensemble and batch) are preserved.
    """
    if q_logits.ndim < 2 or q_logits.shape[-1] != support.numel():
        raise ValueError("Q logits and support have incompatible dimensions")
    indices = nearest_support_indices(target_values, support)
    try:
        expanded_indices = torch.broadcast_to(indices, q_logits.shape[:-1])
    except RuntimeError as error:
        raise ValueError(
            "target values must broadcast to the leading Q-logit dimensions"
        ) from error
    losses = F.cross_entropy(
        q_logits.float().reshape(-1, q_logits.shape[-1]),
        expanded_indices.reshape(-1),
        reduction="none",
    )
    return losses.view(q_logits.shape[:-1])


def decode_categorical_q(q_logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Decode categorical Q logits to expectations in FP32."""
    if q_logits.shape[-1] != support.numel():
        raise ValueError("Q logits and support have incompatible final dimensions")
    probabilities = torch.softmax(q_logits.float(), dim=-1)
    return (probabilities * support.to(q_logits.device, dtype=torch.float32)).sum(dim=-1)


def categorical_q_entropy(q_logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(q_logits.float(), dim=-1)
    log_probabilities = torch.log_softmax(q_logits.float(), dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1)


def ranking_action_negatives(
    action_chunks: torch.Tensor,
    execution_masks: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    noise_sigma: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create strong-noise and random negatives in normalized coordinates."""
    if action_chunks.ndim != 3 or execution_masks.shape != action_chunks.shape[:2]:
        raise ValueError("actions must be [B,H,D] and execution_masks must be [B,H]")
    if noise_sigma < 0.0:
        raise ValueError("noise_sigma must be non-negative")
    values = action_chunks.float()
    mean = action_mean.to(values.device, dtype=torch.float32).view(1, 1, -1)
    std = action_std.to(values.device, dtype=torch.float32).clamp_min(1e-6).view(1, 1, -1)
    lower = ((action_min.to(values.device, dtype=torch.float32) - mean.flatten()) / std.flatten()).view(
        1, 1, -1
    )
    upper = ((action_max.to(values.device, dtype=torch.float32) - mean.flatten()) / std.flatten()).view(
        1, 1, -1
    )
    normalized = (values - mean) / std
    executed = execution_masks.to(device=values.device, dtype=torch.bool).unsqueeze(-1)

    noise = torch.randn(
        normalized.shape,
        device=normalized.device,
        dtype=torch.float32,
        generator=generator,
    )
    strong_normalized = torch.where(
        executed,
        normalized + float(noise_sigma) * noise,
        normalized,
    )
    random_unit = torch.rand(
        normalized.shape,
        device=normalized.device,
        dtype=torch.float32,
        generator=generator,
    )
    random_normalized = torch.where(
        executed,
        lower + random_unit * (upper - lower),
        normalized,
    )
    strong = (strong_normalized * std + mean).to(action_chunks.dtype)
    random = (random_normalized * std + mean).to(action_chunks.dtype)
    return strong, random


def soft_worst_member_margin(member_margins: torch.Tensor, *, tau: float) -> torch.Tensor:
    """Stable normalized soft minimum over the ensemble/member dimension."""
    if member_margins.ndim < 1 or member_margins.shape[0] < 1:
        raise ValueError("member_margins must have a non-empty member dimension")
    if tau <= 0.0:
        raise ValueError("soft-min tau must be positive")
    margins = member_margins.float()
    return -float(tau) * (
        torch.logsumexp(-margins / float(tau), dim=0) - math.log(margins.shape[0])
    )


def consensus_ranking_loss(
    positive_q: torch.Tensor,
    negative_q: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    margin: float,
    softmin_tau: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Worst-member soft-margin ranking loss for one negative type."""
    if positive_q.shape != negative_q.shape or positive_q.ndim != 2:
        raise ValueError("positive_q and negative_q must both be [members,batch]")
    if valid_mask.shape != positive_q.shape[1:]:
        raise ValueError("valid_mask must have shape [batch]")
    if temperature <= 0.0:
        raise ValueError("ranking temperature must be positive")
    member_margins = positive_q.float() - negative_q.float()
    worst_margin = soft_worst_member_margin(member_margins, tau=softmin_tau)
    per_pair = float(temperature) * F.softplus(
        (float(margin) - worst_margin) / float(temperature)
    )
    valid = valid_mask.to(device=per_pair.device, dtype=torch.bool)
    if bool(valid.any()):
        loss = per_pair[valid].mean()
    else:
        loss = positive_q.sum().float() * 0.0
    return loss, member_margins, worst_margin
