from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_support(v_min: float, v_max: float, num_atoms: int, device: torch.device | str | None = None) -> torch.Tensor:
    if num_atoms < 2:
        raise ValueError("num_atoms must be at least 2")
    if not v_min < v_max:
        raise ValueError("v_min must be less than v_max")
    return torch.linspace(float(v_min), float(v_max), int(num_atoms), device=device)


def make_support_from_targets(
    targets: torch.Tensor,
    *,
    num_atoms: int,
    margin_fraction: float = 0.05,
) -> torch.Tensor:
    """Create fixed categorical atoms from a frozen replay return range."""
    finite = targets.detach().flatten()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        raise ValueError("cannot estimate DIVL support without finite return targets")
    if margin_fraction < 0.0:
        raise ValueError("margin_fraction must be non-negative")
    lower = finite.min()
    upper = finite.max()
    span = upper - lower
    if bool(span <= 0):
        span = torch.maximum(lower.abs(), lower.new_tensor(1.0))
    margin = torch.maximum(span * margin_fraction, span.new_tensor(1e-6))
    return torch.linspace(
        float((lower - margin).item()),
        float((upper + margin).item()),
        int(num_atoms),
        device=targets.device,
    )


def categorical_projection(target_values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project scalar targets to a fixed categorical support.

    Returns probabilities with shape target_values.shape + [num_atoms].
    """
    if support.ndim != 1:
        raise ValueError("support must be rank-1")
    target = target_values.to(dtype=support.dtype, device=support.device).clamp(
        float(support[0]), float(support[-1])
    )
    num_atoms = int(support.numel())
    delta = (support[-1] - support[0]) / (num_atoms - 1)
    b = (target - support[0]) / delta
    lower = torch.floor(b).long().clamp(0, num_atoms - 1)
    upper = torch.ceil(b).long().clamp(0, num_atoms - 1)
    upper_w = b - lower.to(b.dtype)
    lower_w = 1.0 - upper_w

    projected = torch.zeros(*target.shape, num_atoms, dtype=support.dtype, device=support.device)
    projected.scatter_add_(-1, lower.unsqueeze(-1), lower_w.unsqueeze(-1))
    projected.scatter_add_(-1, upper.unsqueeze(-1), upper_w.unsqueeze(-1))
    projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return projected


def categorical_entropy(probs: torch.Tensor, *, normalized: bool = True, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)
    if normalized:
        entropy = entropy / math.log(probs.shape[-1])
    return entropy.clamp(0.0, 1.0 if normalized else float("inf"))


def adaptive_alpha(
    entropy_norm: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    *,
    temperature: float = 1.0,
    mode: str = "linear",
) -> torch.Tensor:
    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError("alpha range must lie within [0, 1]")
    clarity = (1.0 - entropy_norm).clamp(0.0, 1.0)
    if mode == "linear":
        factor = clarity
        alpha = alpha_min + (alpha_max - alpha_min) * factor
    elif mode == "sigmoid":
        factor = torch.sigmoid((clarity - 0.5) / max(float(temperature), 1e-6))
        alpha = alpha_min + (alpha_max - alpha_min) * factor
    elif mode == "lwd_linear":
        alpha = (alpha_max - float(temperature) * entropy_norm).clamp(
            min=float(alpha_min),
            max=float(alpha_max),
        )
    else:
        raise ValueError(f"unknown alpha mode: {mode}")
    return alpha


def adaptive_tau(
    entropy_norm: torch.Tensor,
    *,
    tau_base: float = 0.6,
    entropy_coefficient: float = 0.3,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
) -> torch.Tensor:
    """LWD/DIVL entropy-adaptive quantile.

    The offline DIVL setting uses ``tau = tau_base - alpha * H`` where H is
    entropy normalized by ``log(num_atoms)``.  Lower quantiles are selected in
    high-entropy states, making the value bootstrap more conservative.  This
    helper is additive to the historical ``adaptive_alpha`` API so fixed-
    quantile experiments remain loadable.
    """
    if not 0.0 <= float(tau_min) <= float(tau_max) <= 1.0:
        raise ValueError("tau bounds must satisfy 0 <= tau_min <= tau_max <= 1")
    if not math.isfinite(float(tau_base)):
        raise ValueError("tau_base must be finite")
    if not math.isfinite(float(entropy_coefficient)) or float(entropy_coefficient) < 0.0:
        raise ValueError("entropy_coefficient must be finite and non-negative")
    tau = float(tau_base) - float(entropy_coefficient) * entropy_norm
    return tau.clamp(min=float(tau_min), max=float(tau_max))


def categorical_quantile(
    probs: torch.Tensor,
    support: torch.Tensor,
    alpha: torch.Tensor | float,
    *,
    interpolate: bool = True,
) -> torch.Tensor:
    """Extract categorical quantiles, optionally using LWD's atom selection."""
    if probs.ndim < 1:
        raise ValueError("probs must have at least one dimension")
    support = support.to(device=probs.device, dtype=probs.dtype)
    if probs.shape[-1] != support.numel():
        raise ValueError("the final probability dimension must match support")
    alpha_t = torch.as_tensor(alpha, dtype=probs.dtype, device=probs.device)
    target_shape = probs.shape[:-1]
    # Accept scalar, per-batch, or broadcastable quantile tensors.  The
    # previous implementation happened to work for [M, B] but could produce
    # a non-broadcastable [1, 1, 1] value tensor when callers supplied a
    # scalar tau for a multi-member distribution.
    if alpha_t.numel() == 1:
        alpha_t = alpha_t.expand(target_shape)
    else:
        try:
            alpha_t = torch.broadcast_to(alpha_t, target_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"alpha with shape {tuple(alpha_t.shape)} is not broadcastable "
                f"to probability batch shape {tuple(target_shape)}"
            ) from exc
    alpha_t = alpha_t.clamp(0.0, 1.0)

    cdf = probs.cumsum(dim=-1)
    idx = torch.searchsorted(cdf.contiguous(), alpha_t.unsqueeze(-1), right=False).squeeze(-1)
    idx = idx.clamp(0, support.numel() - 1)
    if not interpolate:
        return support[idx]
    low_idx = (idx - 1).clamp_min(0)
    high_idx = idx
    cdf_low_gathered = torch.gather(cdf, -1, low_idx.unsqueeze(-1)).squeeze(-1)
    cdf_low = torch.where(idx == 0, torch.zeros_like(cdf_low_gathered), cdf_low_gathered)
    cdf_high = torch.gather(cdf, -1, high_idx.unsqueeze(-1)).squeeze(-1)
    support_low = support[low_idx]
    support_high = support[high_idx]
    denom = (cdf_high - cdf_low).clamp_min(1e-8)
    interp = ((alpha_t - cdf_low) / denom).clamp(0.0, 1.0)
    return support_low + interp * (support_high - support_low)


class DistributionalValueHead(nn.Module):
    """State-conditioned categorical replay-value distribution Z(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int, num_atoms: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_atoms))
        self.net = nn.Sequential(*layers)

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        assert observations.ndim == 2
        return self.net(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(observations), dim=-1)


class DistributionalValueEnsemble(nn.Module):
    def __init__(self, ensemble_size: int, obs_dim: int, hidden_dim: int, num_layers: int, num_atoms: int):
        super().__init__()
        self.members = nn.ModuleList(
            [
                DistributionalValueHead(obs_dim, hidden_dim, num_layers, num_atoms)
                for _ in range(ensemble_size)
            ]
        )

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.stack([member.logits(observations) for member in self.members], dim=0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(observations), dim=-1)
