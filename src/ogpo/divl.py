from __future__ import annotations

from dataclasses import dataclass

import torch

from .distributional_value import (
    adaptive_alpha,
    adaptive_tau,
    categorical_entropy,
    categorical_projection,
    categorical_quantile,
)


@dataclass(frozen=True)
class DIVLStats:
    entropy: torch.Tensor
    alpha: torch.Tensor
    quantile_value: torch.Tensor
    tau: torch.Tensor | None = None


def divl_projection_targets(q_target_data: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project per-member replay-action Q targets into Z_m(s) targets."""
    assert q_target_data.ndim == 2  # [M, B]
    return categorical_projection(q_target_data, support)


def divl_clipped_double_q_projection_targets(
    q_target_pairs: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project each member's ``min(Q^-_1,Q^-_2)`` into its V target."""
    if q_target_pairs.ndim != 3 or q_target_pairs.shape[1] != 2:
        raise ValueError("q_target_pairs must have shape [member, 2, batch]")
    return divl_projection_targets(q_target_pairs.min(dim=1).values, support)


def aggregate_double_q_for_v(
    q_target_pairs: torch.Tensor,
    aggregation: str = "min",
) -> torch.Tensor:
    """Aggregate each Q pair only for the categorical-V supervision target."""
    if q_target_pairs.ndim != 3 or q_target_pairs.shape[1] != 2:
        raise ValueError("q_target_pairs must have shape [member, 2, batch]")
    mode = str(aggregation).lower()
    if mode == "min":
        return q_target_pairs.min(dim=1).values
    if mode == "mean":
        return q_target_pairs.mean(dim=1)
    raise ValueError("v_q_aggregation must be 'min' or 'mean'")


def divl_double_q_projection_targets(
    q_target_pairs: torch.Tensor,
    support: torch.Tensor,
    *,
    aggregation: str = "min",
) -> torch.Tensor:
    """Project configurable pair aggregation into each member's V target."""
    return divl_projection_targets(
        aggregate_double_q_for_v(q_target_pairs, aggregation),
        support,
    )


def divl_quantile_values(
    probs: torch.Tensor,
    support: torch.Tensor,
    *,
    alpha_min: float,
    alpha_max: float,
    entropy_temperature: float = 1.0,
    alpha_mode: str = "linear",
    use_adaptive_quantile: bool = True,
    interpolate_quantile: bool = True,
    # New LWD offline parameterization.  When ``adaptive_tau_mode`` is true,
    # these take precedence over the legacy alpha range below.
    adaptive_tau_mode: bool = False,
    tau_base: float = 0.6,
    entropy_coefficient: float = 0.3,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
) -> DIVLStats:
    """Compute adaptive replay-value quantile V_m(s) from Z_m(s)."""
    assert probs.ndim == 3  # [M, B, atoms]
    entropy = categorical_entropy(probs, normalized=True)
    if adaptive_tau_mode:
        tau = adaptive_tau(
            entropy,
            tau_base=float(tau_base),
            entropy_coefficient=float(entropy_coefficient),
            tau_min=float(tau_min),
            tau_max=float(tau_max),
        )
        # ``alpha`` is retained as a compatibility alias in diagnostics and
        # checkpoints; for the new mode it is exactly the selected tau.
        alpha = tau
    elif use_adaptive_quantile:
        alpha = adaptive_alpha(
            entropy,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            temperature=entropy_temperature,
            mode=alpha_mode,
        )
    else:
        alpha = torch.full_like(entropy, float(alpha_max))
    quantile = categorical_quantile(
        probs,
        support.to(probs.device),
        alpha,
        interpolate=interpolate_quantile,
    )
    return DIVLStats(entropy=entropy, alpha=alpha, quantile_value=quantile, tau=alpha)
