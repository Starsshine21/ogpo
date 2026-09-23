"""Pure metrics and selection logic for the 10-raw-Q OGPO critic boundary."""

from __future__ import annotations

from typing import Any

import torch

from .chi2_regularization import apply_chipo_to_ca_advantage
from .conservative_advantage import group_relative_conservative_advantage


RAW_Q_HEADS = 10
DIVL_PAIRS = 5


def _require_finite(*values: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("non-finite scalar/raw10 evaluator input")


def raw10_from_q_pairs(q_pairs: torch.Tensor) -> torch.Tensor:
    """Flatten ``[5,2,...]`` as Q11,Q12,...,Q51,Q52 without clipping."""
    if q_pairs.ndim < 3 or q_pairs.shape[:2] != (DIVL_PAIRS, 2):
        raise ValueError("raw10 evaluator requires q_pairs with leading shape [5,2]")
    raw = q_pairs.reshape(RAW_Q_HEADS, *q_pairs.shape[2:])
    if raw.shape[0] != RAW_Q_HEADS:
        raise AssertionError("raw-Q flattening did not produce ten heads")
    return raw


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    if values.numel() == 0:
        return ranks
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    offset = 0
    for count in counts.tolist():
        midpoint = offset + (count - 1) / 2.0
        ranks[order[offset : offset + count]] = midpoint
        offset += count
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel() or x.numel() < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = rx.square().mean().sqrt() * ry.square().mean().sqrt()
    if float(denominator) <= 1e-12:
        return 0.0
    return float(((rx * ry).mean() / denominator).clamp(-1.0, 1.0).item())


def pairwise_rank_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    pair_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> float:
    predictions = predictions.detach().float().flatten()
    targets = targets.detach().float().flatten()
    if predictions.shape != targets.shape or predictions.numel() < 2:
        raise ValueError("pairwise ranking inputs must have equal length >= 2")
    if pair_indices is None:
        pair_tensor = torch.triu_indices(
            predictions.numel(), predictions.numel(), offset=1
        )
        pair_i, pair_j = pair_tensor[0], pair_tensor[1]
    elif isinstance(pair_indices, torch.Tensor):
        if pair_indices.shape[0] != 2:
            raise ValueError("pair_indices tensor must have shape [2,P]")
        pair_i, pair_j = pair_indices[0], pair_indices[1]
    else:
        pair_i, pair_j = pair_indices
    target_delta = targets[pair_i] - targets[pair_j]
    valid = target_delta != 0
    if not bool(valid.any()):
        return 0.0
    prediction_delta = predictions[pair_i] - predictions[pair_j]
    return float(
        (torch.sign(prediction_delta[valid]) == torch.sign(target_delta[valid]))
        .float()
        .mean()
        .item()
    )


def off_diagonal_q_correlation_metrics(raw_q: torch.Tensor) -> dict[str, float]:
    """Pearson correlation range across non-constant raw-Q prediction vectors."""
    if raw_q.ndim != 2 or raw_q.shape[0] != RAW_Q_HEADS:
        raise ValueError("raw_q must have shape [10,N]")
    centered = raw_q - raw_q.mean(dim=1, keepdim=True)
    norms = centered.square().sum(dim=1).sqrt()
    valid = norms > 1e-12
    if int(valid.sum()) < 2:
        return {
            "mean_off_diagonal_q_correlation": 1.0,
            "min_off_diagonal_q_correlation": 1.0,
            "max_off_diagonal_q_correlation": 1.0,
            "q_correlation_valid_head_count": float(valid.sum().item()),
        }
    normalized = centered[valid] / norms[valid, None]
    correlation = (normalized @ normalized.t()).clamp(-1.0, 1.0)
    off_diagonal = ~torch.eye(
        correlation.shape[0], dtype=torch.bool, device=correlation.device
    )
    values = correlation[off_diagonal]
    return {
        "mean_off_diagonal_q_correlation": float(values.mean().item()),
        "min_off_diagonal_q_correlation": float(values.min().item()),
        "max_off_diagonal_q_correlation": float(values.max().item()),
        "q_correlation_valid_head_count": float(valid.sum().item()),
    }


def raw10_fidelity_metrics(raw_q: torch.Tensor, mc_return: torch.Tensor) -> dict[str, Any]:
    """Evaluate all ten heads; pair-min is intentionally absent."""
    if raw_q.ndim != 2 or raw_q.shape[0] != RAW_Q_HEADS:
        raise ValueError("raw_q must have shape [10,N]")
    _require_finite(raw_q, mc_return)
    target = mc_return.detach().float().flatten().cpu()
    raw = raw_q.detach().float().cpu()
    if raw.shape[1] != target.numel():
        raise ValueError("raw-Q batch and MC-return batch do not match")
    mean_q = raw.mean(dim=0)
    error = mean_q - target
    disagreement = raw.std(dim=0, unbiased=False)
    pair_indices = torch.triu_indices(target.numel(), target.numel(), offset=1)
    member_spearman = [spearman_correlation(head, target) for head in raw]
    member_pairwise = [
        pairwise_rank_accuracy(head, target, pair_indices=pair_indices) for head in raw
    ]
    member_rank = torch.tensor(member_spearman)
    metrics: dict[str, Any] = {
        "raw10_mc_spearman": spearman_correlation(mean_q, target),
        "raw10_pairwise_rank_acc": pairwise_rank_accuracy(
            mean_q, target, pair_indices=pair_indices
        ),
        "member_mc_spearman": member_spearman,
        "member_pairwise_rank_acc": member_pairwise,
        "raw10_member_rank_mean": float(member_rank.mean().item()),
        "raw10_member_rank_min": float(member_rank.min().item()),
        "raw10_member_rank_std": float(member_rank.std(unbiased=False).item()),
        "raw10_rmse": float(error.square().mean().sqrt().item()),
        "raw10_mae": float(error.abs().mean().item()),
        "raw10_abs_bias": float(error.mean().abs().item()),
        "raw10_bias": float(error.mean().item()),
        "raw10_ensemble_std_mean": float(disagreement.mean().item()),
        "raw10_ensemble_std_p50": float(torch.quantile(disagreement, 0.5).item()),
        "raw10_ensemble_std_p90": float(torch.quantile(disagreement, 0.9).item()),
        "error_disagreement_spearman": spearman_correlation(
            disagreement, error.abs()
        ),
    }
    metrics.update(off_diagonal_q_correlation_metrics(raw))
    for index, value in enumerate(member_spearman):
        metrics[f"member_mc_spearman_{index}"] = value
    for index, value in enumerate(member_pairwise):
        metrics[f"member_pairwise_rank_acc_{index}"] = value
    return metrics


def focused_return_metrics(
    raw_q: torch.Tensor,
    mc_return: torch.Tensor,
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
    dones: torch.Tensor,
    *,
    high_return_threshold: float = 0.8,
) -> dict[str, Any]:
    """Episode-wise success ranking and high/terminal-return calibration."""
    if raw_q.ndim != 2 or raw_q.shape[0] != RAW_Q_HEADS:
        raise ValueError("raw_q must have shape [10,N]")
    count = raw_q.shape[1]
    fields = (mc_return, episode_ids, successes, dones)
    if any(value.ndim != 1 or value.numel() != count for value in fields):
        raise ValueError("return-focus fields must all have shape [N]")
    mean_q = raw_q.detach().float().cpu().mean(dim=0)
    target = mc_return.detach().float().cpu()
    ids = episode_ids.detach().long().cpu()
    success = successes.detach().bool().cpu()
    done = dones.detach().bool().cpu()

    episode_values: list[dict[str, float | int]] = []
    for episode_id in sorted(int(value) for value in torch.unique(ids[success])):
        mask = (ids == episode_id) & success
        if int(mask.sum()) < 2:
            continue
        correlation = spearman_correlation(mean_q[mask], target[mask])
        episode_values.append(
            {
                "episode_id": episode_id,
                "transition_count": int(mask.sum().item()),
                "mc_spearman": correlation,
            }
        )
    correlations = torch.tensor(
        [float(row["mc_spearman"]) for row in episode_values],
        dtype=torch.float64,
    )
    if correlations.numel() == 0:
        episode_median = episode_p10 = 0.0
    else:
        episode_median = float(torch.quantile(correlations, 0.5).item())
        episode_p10 = float(torch.quantile(correlations, 0.1).item())

    high = target > float(high_return_threshold)
    if bool(high.any()):
        high_error = mean_q[high] - target[high]
        high_mae = float(high_error.abs().mean().item())
        high_bias = float(high_error.mean().item())
    else:
        high_mae = high_bias = 0.0
    terminal_success = done & success
    if bool(terminal_success.any()):
        terminal_error = (mean_q[terminal_success] - 1.0).abs()
        terminal_abs_mean = float(terminal_error.mean().item())
        terminal_abs_median = float(torch.quantile(terminal_error, 0.5).item())
    else:
        terminal_abs_mean = terminal_abs_median = 0.0
    return {
        "success_episode_mc_spearman_median": episode_median,
        "success_episode_mc_spearman_p10": episode_p10,
        "success_episode_mc_spearman_count": len(episode_values),
        "success_episode_mc_spearman": episode_values,
        "high_return_threshold": float(high_return_threshold),
        "high_return_count": int(high.sum().item()),
        "high_return_mae": high_mae,
        "high_return_bias": high_bias,
        "terminal_success_count": int(terminal_success.sum().item()),
        "terminal_success_abs_q_minus_1_mean": terminal_abs_mean,
        "terminal_success_abs_q_minus_1_median": terminal_abs_median,
    }


def scalar_td_fidelity_metrics(raw_q: torch.Tensor, member_targets: torch.Tensor) -> dict[str, Any]:
    """Member-local TD errors; raw10-mean compares mean Q with mean member target."""
    if raw_q.ndim != 2 or raw_q.shape[0] != RAW_Q_HEADS:
        raise ValueError("raw_q must have shape [10,N]")
    if member_targets.shape != (DIVL_PAIRS, raw_q.shape[1]):
        raise ValueError("member_targets must have shape [5,N]")
    _require_finite(raw_q, member_targets)
    raw, targets = raw_q.detach().float().cpu(), member_targets.detach().float().cpu()
    errors = raw - targets.repeat_interleave(2, dim=0)
    mean_error = raw.mean(dim=0) - targets.mean(dim=0)
    rmses = errors.square().mean(dim=1).sqrt()
    return {
        "td_rmse": float(mean_error.square().mean().sqrt()),
        "td_mae": float(mean_error.abs().mean()),
        "td_error_p90": float(torch.quantile(mean_error.abs(), 0.9)),
        "td_rawhead_rmse": float(errors.square().mean().sqrt()),
        "member_td_rmse": rmses.tolist(),
        "member_td_mae": errors.abs().mean(dim=1).tolist(),
        "member_td_error_p90": torch.quantile(errors.abs(), 0.9, dim=1).tolist(),
        "worst_head_td_rmse": float(rmses.max()),
        "worst_td_head_index": int(rmses.argmax()),
    }


def same_state_action_ranking_metrics(
    positive_q: torch.Tensor,
    negative_q: torch.Tensor,
) -> dict[str, Any]:
    """Measure model-imposed same-state ordering across all ten Q heads.

    These are synthetic-relation diagnostics, not action-return correctness.
    Legacy ``*_rank_*`` keys remain for old reports; new consumers should use
    the explicitly neutral ``*_ordering_rate`` names.
    """
    if positive_q.ndim != 2 or positive_q.shape[0] != RAW_Q_HEADS:
        raise ValueError("positive_q must have shape [10,B]")
    if negative_q.ndim != 3 or negative_q.shape[1:] != positive_q.shape:
        raise ValueError("negative_q must have shape [K,10,B]")
    _require_finite(positive_q, negative_q)
    margins = positive_q.unsqueeze(0) - negative_q
    conservative = margins.min(dim=1).values
    head_ordering = margins > 0
    per_head_ordering = head_ordering.float().mean(dim=(0, 2))
    ensemble_mean_margins = positive_q.mean(dim=0).unsqueeze(0) - negative_q.mean(dim=1)
    head_ordering_rate = float(head_ordering.float().mean().item())
    ensemble_mean_ordering_rate = float((ensemble_mean_margins > 0).float().mean().item())
    unanimous_rate = float((conservative > 0).float().mean().item())
    metrics: dict[str, Any] = {
        "head_ordering_rate": head_ordering_rate,
        "ensemble_mean_ordering_rate": ensemble_mean_ordering_rate,
        "unanimous_rate": unanimous_rate,
        "per_head_ordering_rate": [float(value) for value in per_head_ordering.tolist()],
        # Backward-compatible aliases for historical reports.
        "raw10_action_rank_mean": head_ordering_rate,
        "raw10_action_rank_unanimous": unanimous_rate,
        "member_action_rank_acc": [float(value) for value in per_head_ordering.tolist()],
        "raw10_member_action_rank_mean": float(per_head_ordering.mean().item()),
        "raw10_member_action_rank_min": float(per_head_ordering.min().item()),
        "raw10_member_action_rank_std": float(
            per_head_ordering.std(unbiased=False).item()
        ),
        "raw10_mean_action_margin": float(margins.mean().item()),
        "raw10_min_action_margin": float(conservative.mean().item()),
        "raw10_min_action_margin_global": float(conservative.min().item()),
        "raw10_action_margin_p10": float(torch.quantile(conservative.float(), 0.1).item()),
    }
    slopes = torch.sigmoid(-margins.float())
    for prefix, values in (("margin", margins), ("worst_head_margin", conservative)):
        for suffix, quantile in (("median", 0.5), ("p10", 0.1), ("p90", 0.9)):
            metrics[f"{prefix}_{suffix}"] = float(torch.quantile(values.float(), quantile))
    metrics.update({
        "softplus_slope_mean": float(slopes.mean()),
        "softplus_slope_median": float(torch.quantile(slopes, 0.5)),
        "softplus_slope_p90": float(torch.quantile(slopes, 0.9)),
        "softplus_slope_below_1e2": float((slopes < 1e-2).float().mean()),
    })
    for index, value in enumerate(per_head_ordering.tolist()):
        metrics[f"member_action_rank_acc_{index}"] = float(value)
    return metrics


def actor_signal_metrics(
    candidate_q: torch.Tensor,
    *,
    chi2_config: dict[str, Any],
    positive_margin: float = 0.0,
    negative_margin: float = 0.0,
) -> dict[str, float]:
    """Call the production CA and CA+ChiPO functions with the initialization ratio 1."""
    if candidate_q.ndim != 3 or candidate_q.shape[0] != RAW_Q_HEADS:
        raise ValueError("candidate_q must have shape [10,B,G]")
    _require_finite(candidate_q)
    ca, _, ca_stats = group_relative_conservative_advantage(
        candidate_q,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
    )
    ratio = torch.ones_like(ca)
    final, chipo_stats = apply_chipo_to_ca_advantage(
        ca,
        candidate_q,
        ratio,
        beta_base=float(chi2_config.get("beta_base", chi2_config.get("beta", 0.1))),
        q_std_target=float(chi2_config.get("q_std_target", 1.0)),
        ensemble_alpha=float(chi2_config.get("ensemble_alpha", 5.0)),
        r_max=float(chi2_config.get("r_max", 10.0)),
        normalize_group=bool(chi2_config.get("normalize_group", False)),
    )
    ca_nonzero = ca != 0
    final_nonzero = final != 0
    filtered = ca_nonzero & ~final_nonzero
    ca_abs_mean = float(ca.abs().mean().item())
    final_abs_mean = float(final.abs().mean().item())
    group_size = candidate_q.shape[-1]
    if group_size < 2:
        raise ValueError("actor signal requires at least two candidates")
    # No pre-existing worst-head candidate ranking definition: use mean-Q
    # winner versus the leave-one-out mean of other candidates, then raw10 min.
    best = candidate_q.mean(dim=0).argmax(dim=-1)
    best_q = candidate_q.gather(2, best[None, :, None].expand(RAW_Q_HEADS, -1, 1)).squeeze(-1)
    other_mean = (candidate_q.sum(dim=-1) - best_q) / (group_size - 1)
    worst_margin = (best_q - other_mean).min(dim=0).values
    disagreement = candidate_q.std(dim=0, unbiased=False)
    return {
        "actor_raw10_disagreement_mean": float(disagreement.mean()),
        "actor_raw10_disagreement_median": float(torch.quantile(disagreement, 0.5)),
        "actor_raw10_disagreement_p90": float(torch.quantile(disagreement, 0.9)),
        "actor_state_disagreement_mean": float(disagreement.mean(dim=-1).mean()),
        "actor_state_disagreement_p90": float(torch.quantile(disagreement.mean(dim=-1), 0.9)),
        "worst_head_ranking_accuracy": float((worst_margin > 0).float().mean()),
        "worst_head_margin_mean": float(worst_margin.mean()),
        "ca_mean": float(ca.mean()),
        "ca_abs_p90": float(torch.quantile(ca.abs(), 0.9)),
        "ca_positive_ratio": float((ca > 0).float().mean().item()),
        "ca_negative_ratio": float((ca < 0).float().mean().item()),
        "ca_zero_ratio": float((ca == 0).float().mean().item()),
        "ca_nonzero_ratio": float(ca_nonzero.float().mean().item()),
        "ca_abs_adv_mean": ca_abs_mean,
        "ca_abs_adv_std": float(ca.abs().std(unbiased=False).item()),
        "ca_adv_abs_mean": ca_abs_mean,
        "ca_adv_std": float(ca.std(unbiased=False).item()),
        "final_positive_ratio": float((final > 0).float().mean().item()),
        "final_negative_ratio": float((final < 0).float().mean().item()),
        "final_zero_ratio": float((final == 0).float().mean().item()),
        "final_nonzero_ratio": float(final_nonzero.float().mean().item()),
        "final_adv_abs_mean": final_abs_mean,
        "final_adv_std": float(final.std(unbiased=False).item()),
        "chipo_adv_scale_ratio": final_abs_mean / (ca_abs_mean + 1e-12),
        "chipo_filtered_fraction": float(filtered.float().mean().item()),
        "chipo_ratio_mean": 1.0,
        "chipo_beta": float(chipo_stats.beta),
        "chipo_gate_sign_conflict_ratio": float(
            chipo_stats.gate_sign_conflict_ratio
        ),
        "ca_sign_agreement_ratio": float(ca_stats.sign_agreement_ratio),
    }


def fixed_validation_indices(batch_size: int, count: int, seed: int) -> torch.Tensor:
    if batch_size <= 0 or count <= 0:
        raise ValueError("batch_size and count must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(batch_size, generator=generator)[: min(count, batch_size)]


def validate_candidate_cache(
    payload: dict[str, Any],
    *,
    expected_count: int | None = None,
    expected_seed: int | None = None,
) -> None:
    required = {
        "schema_version",
        "split",
        "seed",
        "group_size",
        "candidate_state_count",
        "validation_indices",
        "episode_ids",
        "timesteps",
        "batch",
        "candidate_actions",
        "base_actor_checkpoint",
        "base_actor_identity",
        "validation_replay_identity",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"candidate cache is missing fields: {sorted(missing)}")
    if payload["split"] != "validation":
        raise ValueError("candidate cache must be generated from validation split")
    if int(payload["schema_version"]) != 2:
        raise ValueError("raw10 v2 evaluator requires candidate cache schema_version=2")
    candidates = payload["candidate_actions"]
    if not isinstance(candidates, torch.Tensor) or candidates.ndim != 4:
        raise ValueError("candidate_actions must have shape [B,G,H,D]")
    if candidates.shape[1] != 4 or int(payload["group_size"]) != 4:
        raise ValueError("OGPO evaluator requires a fixed G=4 candidate cache")
    count = int(payload["candidate_state_count"])
    if count <= 0 or candidates.shape[0] != count:
        raise ValueError("candidate cache count does not match candidate_actions")
    if payload["validation_indices"].numel() != count:
        raise ValueError("candidate cache count does not match validation_indices")
    if payload["episode_ids"].numel() != count or payload["timesteps"].numel() != count:
        raise ValueError("candidate cache count does not match episode/timestep identity")
    if expected_count is not None and count != int(expected_count):
        raise ValueError(
            f"candidate cache count mismatch: expected {expected_count}, got {count}"
        )
    if expected_seed is not None and int(payload["seed"]) != int(expected_seed):
        raise ValueError(
            f"candidate cache seed mismatch: expected {expected_seed}, got {payload['seed']}"
        )


def validate_selection_split(split: str, heldout_replay: str | None = None) -> None:
    if split != "validation":
        raise ValueError("checkpoint selection sweep must use validation split")
    if heldout_replay is not None:
        raise ValueError("heldout replay must not participate in the 1k~8k selection sweep")


def _percentile(values: list[float], *, higher_is_better: bool = True) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    ranks = _average_ranks(tensor).double()
    if len(values) == 1:
        normalized = torch.ones_like(ranks)
    else:
        normalized = ranks / (len(values) - 1)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return [float(value) for value in normalized.tolist()]


def rank_checkpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply ordered gates and a non-duplicative percentile reference score."""
    if not rows:
        raise ValueError("no checkpoint rows to rank")
    mild_action = _percentile(
        [row["mild_raw10_action_rank_unanimous"] for row in rows]
    )
    all_action = _percentile([row["raw10_action_rank_unanimous"] for row in rows])
    mc_spearman = _percentile([row["raw10_mc_spearman"] for row in rows])
    mc_pairwise = _percentile([row["raw10_pairwise_rank_acc"] for row in rows])
    success_episode_median = _percentile(
        [row["success_episode_mc_spearman_median"] for row in rows]
    )
    success_episode_p10 = _percentile(
        [row["success_episode_mc_spearman_p10"] for row in rows]
    )
    worst_action_head = _percentile(
        [row["raw10_member_action_rank_min"] for row in rows]
    )
    uncertainty = _percentile([row["error_disagreement_spearman"] for row in rows])
    ca_signal_std = _percentile([row["ca_adv_std"] for row in rows])
    final_signal_std = _percentile([row["final_adv_std"] for row in rows])
    rmse = _percentile([row["raw10_rmse"] for row in rows], higher_is_better=False)
    bias = _percentile([row["raw10_abs_bias"] for row in rows], higher_is_better=False)
    high_return_mae = _percentile(
        [row["high_return_mae"] for row in rows], higher_is_better=False
    )
    high_return_bias = _percentile(
        [abs(row["high_return_bias"]) for row in rows], higher_is_better=False
    )
    terminal_error = _percentile(
        [row["terminal_success_abs_q_minus_1_mean"] for row in rows],
        higher_is_better=False,
    )
    ranked: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        local_action_score = (2.0 * mild_action[index] + all_action[index]) / 3.0
        mc_rank_score = (
            mc_spearman[index]
            + mc_pairwise[index]
            + success_episode_median[index]
            + success_episode_p10[index]
        ) / 4.0
        # CA nonzero is diagnostic-only: it is intentionally absent here.
        actor_signal_score = (ca_signal_std[index] + final_signal_std[index]) / 2.0
        calibration_score = (
            rmse[index]
            + bias[index]
            + high_return_mae[index]
            + high_return_bias[index]
            + terminal_error[index]
        ) / 5.0
        uncertainty_uninformative = row["error_disagreement_spearman"] <= 0.0
        collapse = (
            row["raw10_ensemble_std_mean"] < 1e-4
            or (
                row["mean_off_diagonal_q_correlation"] > 0.999
                and row["error_disagreement_spearman"] <= 0.05
            )
        )
        poor_global_ranking = (
            row["raw10_mc_spearman"] <= 0.0
            or row["raw10_pairwise_rank_acc"] < 0.55
            or row["raw10_member_rank_min"] <= 0.0
        )
        poor_local_action_ranking = (
            row["mild_raw10_action_rank_unanimous"] < 0.20
            or row["raw10_action_rank_unanimous"] < 0.25
        )
        bad_worst_action_head = row["raw10_member_action_rank_min"] < 0.50
        starved = row["final_zero_ratio"] > 0.95
        row["gate_q_correctness_pass"] = not poor_global_ranking
        row["gate_local_action_pass"] = not poor_local_action_ranking
        row["gate_worst_action_head_pass"] = not bad_worst_action_head
        row["gate_uncertainty_pass"] = not (
            collapse or uncertainty_uninformative
        )
        row["gate_actor_signal_pass"] = not starved
        row["actor_signal_starved"] = starved
        row["ensemble_collapse"] = collapse
        row["uncertainty_uninformative"] = uncertainty_uninformative
        row["score_local_action"] = local_action_score
        row["score_mc_rank"] = mc_rank_score
        row["score_worst_action_head"] = worst_action_head[index]
        row["score_uncertainty"] = uncertainty[index]
        row["score_actor_signal"] = actor_signal_score
        row["score_calibration"] = calibration_score
        row["ca_nonzero_diagnostic_only"] = True
        row["composite_score"] = (
            0.30 * local_action_score
            + 0.20 * mc_rank_score
            + 0.15 * worst_action_head[index]
            + 0.15 * uncertainty[index]
            + 0.10 * actor_signal_score
            + 0.10 * calibration_score
        )
        row["recommendation"] = (
            "POOR_LOCAL_ACTION_RANKING"
            if poor_local_action_ranking
            else "BAD_WORST_ACTION_HEAD"
            if bad_worst_action_head
            else "POOR_GLOBAL_RANKING"
            if poor_global_ranking
            else "ENSEMBLE_COLLAPSE"
            if collapse
            else "UNCERTAINTY_UNINFORMATIVE"
            if uncertainty_uninformative
            else "ACTOR_SIGNAL_STARVED"
            if starved
            else "ELIGIBLE"
        )
        ranked.append(row)
    eligible = [row for row in ranked if row["recommendation"] == "ELIGIBLE"]
    eligible.sort(key=lambda row: row["composite_score"], reverse=True)
    if eligible:
        eligible[0]["recommendation"] = "BEST"
    if len(eligible) > 1:
        eligible[1]["recommendation"] = "SECOND"
    ranked.sort(
        key=lambda row: (
            row["recommendation"] in {"BEST", "SECOND", "ELIGIBLE"},
            row["composite_score"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked
