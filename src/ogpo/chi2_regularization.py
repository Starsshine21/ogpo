"""ChiPO helpers for the legacy Flash and clean full-chain PyTorch paths.

The selected-transition functions remain for historical Flash experiments.
The clean main path uses :func:`full_chain_chipo_ratio`, which forms one
joint trajectory ratio before normalization and applies the official-style
mean-to-min pessimism/ratio penalty without mutating the frozen critic. The
historical squared-ratio regularizer remains available only behind an explicit
legacy configuration flag.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Chi2RatioStats:
    log_ratio_mean: float
    log_ratio_std: float
    log_ratio_min: float
    log_ratio_max: float
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    ratio_clipped_fraction: float
    divergence_proxy: float
    logprob_normalizer: float


@dataclass(frozen=True)
class Chi2PessimismStats:
    beta: float
    q_ensemble_std: float
    pessimism_weight_mean: float
    q_target_mean: float
    q_target_min: float
    penalty_mean: float
    advantage_mean: float
    advantage_std: float
    pessimism_weight_min: float = 0.0
    pessimism_weight_max: float = 0.0
    q_mean: float = 0.0
    q_min: float = 0.0
    q_penalized_mean: float = 0.0
    chi_advantage_mean: float = 0.0
    chi_advantage_std: float = 0.0
    gate_positive_ratio: float = 0.0
    gate_negative_ratio: float = 0.0
    gate_zero_ratio: float = 0.0
    gate_sign_conflict_ratio: float = 0.0


def compute_chipo_beta(
    q_values: torch.Tensor,
    *,
    beta_base: float,
    q_std_target: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch port of OGPO_public ``compute_chi_po_beta``.

    Upstream source: ``ogpo/agents/modules/pg_helper.py`` at commit
    ``0b3be413cde766a41257c6b19c0c2b06393a557f``. The upstream layout is
    ``[G, M, B]``; this adapter accepts the repository's equivalent
    ``[M, B, G]`` layout and reduces over the member axis before averaging
    candidates/states.
    """
    if q_values.ndim != 3:
        raise ValueError("q_values must have shape [M, B, G]")
    if not math.isfinite(float(beta_base)) or float(beta_base) < 0.0:
        raise ValueError("beta_base must be finite and non-negative")
    if not math.isfinite(float(q_std_target)) or float(q_std_target) <= 0.0:
        raise ValueError("q_std_target must be finite and positive")
    q_std = q_values.std(dim=0, unbiased=False).mean()
    beta = q_values.new_tensor(float(beta_base)) * q_std / float(q_std_target)
    return beta, q_std


def sign_safe_advantage_intersection(
    ca_advantage: torch.Tensor,
    chi_advantage: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the repository-specific CA+ChiPO sign gate.

    Upstream OGPO has no joint CA+ChiPO branch.  This helper is our explicit
    composition: zero/disagreement is rejected, positive agreement keeps the
    smaller signal, and negative agreement keeps the larger (less negative)
    signal.
    """
    if ca_advantage.shape != chi_advantage.shape:
        raise ValueError("CA and ChiPO advantages must have matching shapes")
    ca = ca_advantage.detach()
    chi = chi_advantage.detach()
    positive = (ca > 0.0) & (chi > 0.0)
    negative = (ca < 0.0) & (chi < 0.0)
    final = torch.where(positive, torch.minimum(ca, chi), torch.zeros_like(ca))
    final = torch.where(negative, torch.maximum(ca, chi), final)
    stats = {
        "gate_positive_ratio": float(positive.float().mean().item()),
        "gate_negative_ratio": float(negative.float().mean().item()),
        "gate_zero_ratio": float((~(positive | negative)).float().mean().item()),
        "gate_sign_conflict_ratio": float(((ca * chi) < 0.0).float().mean().item()),
    }
    return final, stats


def full_chain_chipo_ratio(
    current_log_probs: torch.Tensor,
    slow_log_probs: torch.Tensor,
    *,
    action_dim: int,
    normalize_logprob_by_action_dim: bool = True,
    normalize_logprob_by_denoising_steps: bool = True,
    log_ratio_clip: float | None = None,
    ratio_max: float | None = None,
) -> tuple[torch.Tensor, Chi2RatioStats]:
    """Compute the official-style χ² drift ratio over a whole flow chain.

    The ratio/log-prob normalization mirrors OGPO_public's
    ``compute_flow_log_prob`` at commit
    ``0b3be413cde766a41257c6b19c0c2b06393a557f``; this function only adapts
    the stored PyTorch chain layout and returns a detached ratio for use in
    advantage construction.

    The likelihood ratio is formed from the *sum* of transition log
    likelihood differences.  Optional normalization is applied to that joint
    log-ratio scale (never to probabilities), preserving one ratio and one
    clipping decision per trajectory.
    """
    if current_log_probs.ndim != 2 or current_log_probs.shape != slow_log_probs.shape:
        raise ValueError("full-chain log probabilities must have shape [N, K]")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if log_ratio_clip is not None and (
        not math.isfinite(float(log_ratio_clip)) or float(log_ratio_clip) <= 0
    ):
        raise ValueError("log_ratio_clip must be finite and positive")
    if ratio_max is not None and (
        not math.isfinite(float(ratio_max)) or float(ratio_max) < 1.0
    ):
        raise ValueError("ratio_max must be finite and at least one")
    raw = (current_log_probs - slow_log_probs.detach()).sum(dim=1)
    normalizer = 1.0
    if normalize_logprob_by_action_dim:
        normalizer *= float(action_dim)
    if normalize_logprob_by_denoising_steps:
        # The upstream denominator counts the initial Gaussian prior plus
        # K-1 stochastic transitions.  The prior cancels from this difference,
        # but the resulting count is still K for a full chain.
        normalizer *= float(current_log_probs.shape[1])
    scaled = raw / normalizer
    effective_clip = None
    if log_ratio_clip is not None:
        effective_clip = float(log_ratio_clip)
    if ratio_max is not None:
        ratio_clip = math.log(float(ratio_max))
        effective_clip = ratio_clip if effective_clip is None else min(effective_clip, ratio_clip)
    clipped = scaled if effective_clip is None else scaled.clamp(-effective_clip, effective_clip)
    ratio = clipped.exp()
    if ratio_max is not None:
        ratio = ratio.clamp_max(float(ratio_max))
    ratio = ratio.detach()
    stats = Chi2RatioStats(
        log_ratio_mean=float(scaled.detach().mean().item()),
        log_ratio_std=float(scaled.detach().std(unbiased=False).item()),
        log_ratio_min=float(scaled.detach().min().item()),
        log_ratio_max=float(scaled.detach().max().item()),
        ratio_mean=float(ratio.mean().item()),
        ratio_std=float(ratio.std(unbiased=False).item()),
        ratio_min=float(ratio.min().item()),
        ratio_max=float(ratio.max().item()),
        ratio_clipped_fraction=float(scaled.ne(clipped).float().mean().item()),
        divergence_proxy=float((ratio - 1.0).square().mean().item()),
        logprob_normalizer=float(normalizer),
    )
    return ratio, stats


def full_chain_chipo_regularization_loss(
    current_log_probs: torch.Tensor,
    slow_log_probs: torch.Tensor,
    *,
    action_dim: int,
    beta: float,
    normalize_logprob_by_action_dim: bool = True,
    normalize_logprob_by_denoising_steps: bool = True,
    log_ratio_clip: float = 20.0,
    ratio_max: float = 20.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Legacy differentiable squared-ratio drift penalty.

    Upstream ChiPO does not add this extra loss; clean CA+ChiPO consumes the
    official ``beta * ratio`` term through its detached advantage instead.
    This helper is retained solely for an explicit legacy ablation.
    """
    if current_log_probs.ndim != 2 or current_log_probs.shape != slow_log_probs.shape:
        raise ValueError("full-chain log probabilities must have shape [N, K]")
    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("ChiPO regularization beta must be finite and non-negative")
    if not math.isfinite(float(log_ratio_clip)) or float(log_ratio_clip) <= 0.0:
        raise ValueError("log_ratio_clip must be finite and positive")
    if not math.isfinite(float(ratio_max)) or float(ratio_max) < 1.0:
        raise ValueError("ratio_max must be finite and at least one")
    normalizer = 1.0
    if normalize_logprob_by_action_dim:
        normalizer *= float(action_dim)
    if normalize_logprob_by_denoising_steps:
        normalizer *= float(current_log_probs.shape[1])
    scaled_log_ratio = (
        (current_log_probs - slow_log_probs.detach()).sum(dim=1) / normalizer
    )
    effective_clip = min(float(log_ratio_clip), math.log(float(ratio_max)))
    ratio = scaled_log_ratio.clamp(-effective_clip, effective_clip).exp()
    loss = 0.5 * float(beta) * (ratio - 1.0).square().mean()
    stats = {
        "chi2_regularization_loss": float(loss.detach().item()),
        "chi2_regularization_ratio_mean": float(ratio.detach().mean().item()),
        "chi2_regularization_ratio_std": float(ratio.detach().std(unbiased=False).item()),
        "chi2_regularization_beta": float(beta),
    }
    return loss, stats


def apply_chipo_to_ca_advantage(
    ca_advantage: torch.Tensor,
    q_values: torch.Tensor,
    chi2_ratio: torch.Tensor,
    *,
    beta_base: float,
    q_std_target: float,
    ensemble_alpha: float,
    r_max: float,
    normalize_group: bool = False,
) -> tuple[torch.Tensor, Chi2PessimismStats]:
    """Our CA+ChiPO composition using a sign-safe intersection gate.

    The upstream OGPO branch ends at ``A_chi`` (the pessimistic Q target and
    ``beta * ratio`` penalty).  It does not define a CA+ChiPO branch.  This
    repository-specific composition therefore computes upstream ``A_chi``
    first, then permits an update only when CA and ChiPO agree in sign:
    positive uses ``min(A_CA, A_chi)``, negative uses ``max(...)``, and every
    disagreement—including ``A_CA == 0``—is exactly zero.
    """
    if ca_advantage.ndim != 2 or q_values.ndim != 3:
        raise ValueError("CA advantage/q values must have shapes [B,G] and [M,B,G]")
    if chi2_ratio.shape != ca_advantage.shape:
        raise ValueError("full-chain χ² ratio must have shape [B,G]")
    if not math.isfinite(float(r_max)) or float(r_max) <= 0.0:
        raise ValueError("r_max must be finite and positive")
    chi_advantage, chi_stats = chi2_pessimistic_advantage(
        q_values,
        chi2_ratio,
        beta_base=beta_base,
        q_std_target=q_std_target,
        ensemble_alpha=ensemble_alpha,
        normalize_group=bool(normalize_group),
        advantage_clip=None,
    )
    combined, gate_stats = sign_safe_advantage_intersection(
        ca_advantage, chi_advantage
    )
    ca = ca_advantage.detach()
    chi = chi_advantage.detach()
    gate_positive = gate_stats["gate_positive_ratio"]
    gate_negative = gate_stats["gate_negative_ratio"]
    gate_zero = gate_stats["gate_zero_ratio"]
    gate_conflict = gate_stats["gate_sign_conflict_ratio"]
    stats = Chi2PessimismStats(
        beta=chi_stats.beta,
        q_ensemble_std=chi_stats.q_ensemble_std,
        pessimism_weight_mean=chi_stats.pessimism_weight_mean,
        pessimism_weight_max=chi_stats.pessimism_weight_max,
        pessimism_weight_min=chi_stats.pessimism_weight_min,
        q_target_mean=chi_stats.q_target_mean,
        q_target_min=chi_stats.q_target_min,
        penalty_mean=chi_stats.penalty_mean,
        advantage_mean=float(combined.detach().mean().item()),
        advantage_std=float(combined.detach().std(unbiased=False).item()),
        q_mean=chi_stats.q_mean,
        q_min=chi_stats.q_min,
        q_penalized_mean=chi_stats.q_penalized_mean,
        chi_advantage_mean=chi_stats.chi_advantage_mean,
        chi_advantage_std=chi_stats.chi_advantage_std,
        gate_positive_ratio=gate_positive,
        gate_negative_ratio=gate_negative,
        gate_zero_ratio=gate_zero,
        gate_sign_conflict_ratio=gate_conflict,
    )
    return combined.detach(), stats


def selected_transition_chi2_ratio(
    current_log_prob: torch.Tensor,
    slow_log_prob: torch.Tensor,
    *,
    event_dim: int,
    normalize_action_dim: bool,
    log_ratio_clip: float,
    ratio_max: float,
) -> tuple[torch.Tensor, Chi2RatioStats]:
    """Return detached current/slow ratio for one Flash transition per sample."""
    if current_log_prob.ndim != 1 or slow_log_prob.shape != current_log_prob.shape:
        raise ValueError("χ² selected-transition log probabilities must have matching shape [N]")
    if event_dim <= 0:
        raise ValueError("χ² selected transition event_dim must be positive")
    if not math.isfinite(float(log_ratio_clip)) or float(log_ratio_clip) <= 0.0:
        raise ValueError("actor.chi2.log_ratio_clip must be finite and positive")
    if not math.isfinite(float(ratio_max)) or float(ratio_max) < 1.0:
        raise ValueError("actor.chi2.ratio_max must be finite and at least one")

    normalizer = float(event_dim) if normalize_action_dim else 1.0
    raw_log_ratio = (current_log_prob - slow_log_prob) / normalizer
    effective_log_clip = min(float(log_ratio_clip), math.log(float(ratio_max)))
    clipped_log_ratio = raw_log_ratio.clamp(-effective_log_clip, effective_log_clip)
    ratio = clipped_log_ratio.exp().clamp_max(float(ratio_max)).detach()
    stats = Chi2RatioStats(
        log_ratio_mean=float(raw_log_ratio.detach().mean().item()),
        log_ratio_std=float(raw_log_ratio.detach().std(unbiased=False).item()),
        log_ratio_min=float(raw_log_ratio.detach().min().item()),
        log_ratio_max=float(raw_log_ratio.detach().max().item()),
        ratio_mean=float(ratio.mean().item()),
        ratio_std=float(ratio.std(unbiased=False).item()),
        ratio_min=float(ratio.min().item()),
        ratio_max=float(ratio.max().item()),
        ratio_clipped_fraction=float(raw_log_ratio.ne(clipped_log_ratio).float().mean().item()),
        divergence_proxy=float((ratio - 1.0).square().mean().item()),
        logprob_normalizer=normalizer,
    )
    return ratio, stats


def chi2_pessimistic_advantage(
    q_values: torch.Tensor,
    chi2_ratio: torch.Tensor,
    *,
    beta_base: float,
    q_std_target: float,
    ensemble_alpha: float,
    normalize_group: bool,
    advantage_clip: float | None,
) -> tuple[torch.Tensor, Chi2PessimismStats]:
    """Official ChiPO Q mean→min blending and group-relative advantage.

    PyTorch port of OGPO_public
    ``pg_helper.py::compute_chi_po_advantages`` at upstream commit
    ``0b3be413cde766a41257c6b19c0c2b06393a557f``.

    ``q_values`` has shape ``[ensemble, batch, candidate]`` and ``chi2_ratio``
    has shape ``[batch, candidate]``.  The ratio is intentionally detached.
    """
    if q_values.ndim != 3:
        raise ValueError("χ² Q values must have shape [ensemble, batch, candidate]")
    if chi2_ratio.shape != q_values.shape[1:]:
        raise ValueError("χ² ratio must have shape [batch, candidate]")
    if not math.isfinite(float(beta_base)) or float(beta_base) < 0.0:
        raise ValueError("actor.chi2.beta_base must be finite and non-negative")
    if not math.isfinite(float(q_std_target)) or float(q_std_target) <= 0.0:
        raise ValueError("actor.chi2.q_std_target must be finite and positive")
    if not math.isfinite(float(ensemble_alpha)) or float(ensemble_alpha) < 0.0:
        raise ValueError("actor.chi2.ensemble_alpha must be finite and non-negative")

    ratio = chi2_ratio.detach()
    q_mean = q_values.mean(dim=0)
    q_min = q_values.min(dim=0).values
    beta_tensor, q_ensemble_std = compute_chipo_beta(
        q_values,
        beta_base=beta_base,
        q_std_target=q_std_target,
    )
    beta = beta_tensor
    pessimism_weight = torch.sigmoid(float(ensemble_alpha) * (ratio - 1.0))
    q_target = (1.0 - pessimism_weight) * q_mean + pessimism_weight * q_min
    penalty = beta.detach() * ratio
    penalized_q = q_target - penalty
    advantage = penalized_q - penalized_q.mean(dim=-1, keepdim=True)
    if normalize_group:
        advantage = advantage / advantage.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-8)
    if advantage_clip is not None:
        if float(advantage_clip) <= 0.0:
            raise ValueError("actor.advantage_clip must be positive when χ² clipping is enabled")
        advantage = advantage.clamp(-float(advantage_clip), float(advantage_clip))
    stats = Chi2PessimismStats(
        beta=float(beta.detach().item()),
        q_ensemble_std=float(q_ensemble_std.detach().item()),
        pessimism_weight_mean=float(pessimism_weight.mean().item()),
        pessimism_weight_max=float(pessimism_weight.max().item()),
        pessimism_weight_min=float(pessimism_weight.min().item()),
        q_target_mean=float(q_target.mean().item()),
        q_target_min=float(q_target.min().item()),
        penalty_mean=float(penalty.mean().item()),
        advantage_mean=float(advantage.mean().item()),
        advantage_std=float(advantage.std(unbiased=False).item()),
        q_mean=float(q_mean.mean().item()),
        q_min=float(q_min.mean().item()),
        q_penalized_mean=float(penalized_q.mean().item()),
        chi_advantage_mean=float(advantage.mean().item()),
        chi_advantage_std=float(advantage.std(unbiased=False).item()),
    )
    return advantage.detach(), stats


def chi2_ppo_upper_bound(
    clip_eps: torch.Tensor,
    *,
    beta: float,
    r_max: float,
) -> torch.Tensor:
    """chiPO's asymmetric PPO upper bound for a detached beta coefficient."""
    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("χ² PPO beta must be finite and non-negative")
    if not math.isfinite(float(r_max)) or float(r_max) <= 0.0:
        raise ValueError("actor.chi2.r_max must be finite and positive")
    regular_upper = 1.0 + clip_eps
    if float(beta) == 0.0:
        return regular_upper
    return torch.minimum(
        regular_upper,
        torch.full_like(clip_eps, 1.0 + float(r_max) / float(beta)),
    )
