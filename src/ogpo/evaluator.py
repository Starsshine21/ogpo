from __future__ import annotations

import torch

from .categorical_q import (
    categorical_q_entropy,
    decode_categorical_q,
    ranking_action_negatives,
)
from .ensemble import ensemble_mean_std
from .multimodal_critic import MultiHeadScalarQCritic, MultiHeadUdivlCritic
from .types import ChunkBatch
from .uncertainty import conformal_scale as compute_conformal_scale


def _critic_predictions(
    critic,
    batch: ChunkBatch,
    *,
    divl=None,
    inference_batch_size: int | None = None,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    device = next(critic.parameters()).device
    batch_size = batch.batch_size if inference_batch_size is None else max(1, int(inference_batch_size))
    q_chunks = []
    probability_chunks = []
    diagnostic_chunks: dict[str, list[torch.Tensor]] = {}
    critic_cfg = (config or {}).get("critic", {})
    rank_enabled = bool(critic_cfg.get("rank_consensus_enabled", False))
    rank_generator = torch.Generator(device=device).manual_seed(
        int((config or {}).get("training", {}).get("seed", 0)) + 7919
    )
    for start in range(0, batch.batch_size, batch_size):
        indices = torch.arange(start, min(start + batch_size, batch.batch_size))
        sample = batch.index_select(indices).to(device)
        if isinstance(critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
            features = critic.encode_state(sample)
            if (
                isinstance(critic, MultiHeadUdivlCritic)
                and critic.core.q_representation == "categorical"
            ):
                q_logits = critic.q_logits_from_features(
                    features,
                    sample.action_chunks,
                    sample.execution_masks,
                )
                categorical_values = decode_categorical_q(
                    q_logits, critic.core.q_support
                )
                if critic.core.q_heads_per_member == 2:
                    q_values = categorical_values.min(dim=1).values
                    diagnostic_chunks.setdefault("q_pairs", []).append(
                        categorical_values.detach().cpu()
                    )
                else:
                    q_values = categorical_values
                diagnostic_chunks.setdefault("q_entropy", []).append(
                    categorical_q_entropy(q_logits).detach().cpu()
                )
            elif isinstance(critic, MultiHeadUdivlCritic) and critic.core.q_heads_per_member == 2:
                q_pairs = critic.q_pair_from_features(
                    features,
                    sample.action_chunks,
                    sample.execution_masks,
                )
                q_values = q_pairs.min(dim=1).values
                diagnostic_chunks.setdefault("q_pairs", []).append(
                    q_pairs.detach().cpu()
                )
            else:
                q_values = critic.q_from_features(
                    features,
                    sample.action_chunks,
                    sample.execution_masks,
                )
            probabilities = (
                torch.softmax(critic.value_logits_from_features(features), dim=-1)
                if isinstance(critic, MultiHeadUdivlCritic)
                else None
            )
            if rank_enabled and isinstance(critic, MultiHeadUdivlCritic):
                action_pool = critic.core.action_pool
                strong, random = ranking_action_negatives(
                    sample.action_chunks,
                    sample.execution_masks,
                    action_mean=action_pool.action_mean,
                    action_std=action_pool.action_std,
                    action_min=action_pool.action_min,
                    action_max=action_pool.action_max,
                    noise_sigma=float(critic_cfg.get("rank_noise_sigma", 0.15)),
                    generator=rank_generator,
                )
                negatives = []
                if bool(critic_cfg.get("rank_use_strong_noise", True)):
                    negatives.append(strong)
                if bool(critic_cfg.get("rank_use_random_negative", True)):
                    negatives.append(random)
                if negatives:
                    negative_actions = torch.cat(negatives, dim=0)
                    negative_masks = sample.execution_masks.repeat(len(negatives), 1)
                    negative_features = type(features)(
                        readout=features.readout.repeat(len(negatives), 1)
                    )
                    negative_q = critic.q_from_features(
                        negative_features,
                        negative_actions,
                        negative_masks,
                    ).reshape(critic.ensemble_size, len(negatives), sample.batch_size)
                    margins = q_values.unsqueeze(1) - negative_q
                    valid = (
                        sample.successes.bool()
                        if bool(critic_cfg.get("rank_only_success", True))
                        else torch.ones(sample.batch_size, device=device, dtype=torch.bool)
                    )
                    diagnostic_chunks.setdefault("rank_margins", []).append(
                        margins[:, :, valid].detach().cpu()
                    )
        else:
            q_values = critic(sample.observations, sample.action_chunks, sample.execution_masks)
            probabilities = divl(sample.observations) if divl is not None else None
        q_chunks.append(q_values.detach().cpu())
        if probabilities is not None:
            probability_chunks.append(probabilities.detach().cpu())
    q_values = torch.cat(q_chunks, dim=1)
    probabilities = None
    if probability_chunks:
        probability_batch_dim = 1 if probability_chunks[0].ndim == 3 else 0
        probabilities = torch.cat(probability_chunks, dim=probability_batch_dim)
    diagnostics: dict[str, torch.Tensor] = {}
    if diagnostic_chunks.get("q_entropy"):
        diagnostics["q_entropy"] = torch.cat(diagnostic_chunks["q_entropy"], dim=1)
    if diagnostic_chunks.get("q_pairs"):
        diagnostics["q_pairs"] = torch.cat(diagnostic_chunks["q_pairs"], dim=2)
    if diagnostic_chunks.get("rank_margins"):
        diagnostics["rank_margins"] = torch.cat(
            diagnostic_chunks["rank_margins"], dim=2
        )
    return q_values, probabilities, diagnostics


def _correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return x.new_tensor(0.0)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.std(unbiased=False) * y.std(unbiased=False)
    if denominator <= 1e-8:
        return x.new_tensor(0.0)
    return ((x * y).mean() / denominator).clamp(-1.0, 1.0)


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


@torch.no_grad()
def offline_calibration_metrics(
    critic,
    batch: ChunkBatch,
    *,
    divl=None,
    conformal_scale: float = 1.0,
    inference_batch_size: int | None = None,
    config: dict | None = None,
) -> dict[str, float]:
    was_training = critic.training
    critic.eval()
    q_values, probs, diagnostics = _critic_predictions(
        critic,
        batch,
        divl=divl,
        inference_batch_size=inference_batch_size,
        config=config,
    )
    q_mean, q_std = ensemble_mean_std(q_values)
    target_is_mc = batch.mc_returns is not None
    target = (batch.mc_returns if target_is_mc else batch.chunk_returns).cpu()
    error = q_mean - target
    rmse = torch.sqrt(torch.mean(error.pow(2)))
    huber = torch.nn.functional.huber_loss(q_mean, target)
    error_abs = error.abs()
    rank_corr = _correlation(_ranks(q_mean), _ranks(target))
    disagreement_error_corr = _correlation(q_std, error_abs)
    calibrated_std = q_std * float(conformal_scale)
    interval_coverage = (error_abs <= calibrated_std).float().mean()
    member_error = q_values - target.unsqueeze(0)
    all_positive = (member_error > 0).all(dim=0)
    all_negative = (member_error < 0).all(dim=0)
    sign_disagreement = (~(all_positive | all_negative)).float().mean()
    pair_i, pair_j = torch.triu_indices(batch.batch_size, batch.batch_size, offset=1, device=q_mean.device)
    target_delta = target[pair_i] - target[pair_j]
    valid_pairs = target_delta != 0
    if valid_pairs.any():
        q_delta = q_mean[pair_i] - q_mean[pair_j]
        pairwise_accuracy = (torch.sign(q_delta[valid_pairs]) == torch.sign(target_delta[valid_pairs])).float().mean()
    else:
        pairwise_accuracy = q_mean.new_tensor(0.0)
    calibration_terms = []
    order = torch.argsort(calibrated_std)
    for indices in torch.tensor_split(order, min(10, max(1, batch.batch_size))):
        if indices.numel():
            calibration_terms.append((error_abs[indices].mean() - calibrated_std[indices].mean()).abs())
    ece = torch.stack(calibration_terms).mean() if calibration_terms else q_mean.new_tensor(0.0)
    metrics = {
        "q_rmse": float(rmse.item()),
        "q_huber": float(huber.item()),
        "q_rank_correlation": float(rank_corr.item()),
        "pairwise_ranking_accuracy": float(pairwise_accuracy.item()),
        "ensemble_disagreement": float(q_std.mean().item()),
        "ensemble_sign_disagreement": float(sign_disagreement.item()),
        "disagreement_error_correlation": float(disagreement_error_corr.item()),
        "interval_coverage": float(interval_coverage.item()),
        "expected_calibration_error": float(ece.item()),
        "q_exploitation_gap": float(error.mean().item()),
        "conformal_scale": float(conformal_scale),
        "q_mean": float(q_mean.mean().item()),
        "calibration_target_is_mc": float(target_is_mc),
    }
    if probs is not None:
        probs = probs.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(probs.shape[-1], device=probs.device, dtype=probs.dtype))
        metrics["categorical_entropy"] = float(entropy.mean().item())
        metrics["categorical_saturation"] = float(
            ((probs[..., 0] + probs[..., -1]) > 0.5).float().mean().item()
        )
    if "q_entropy" in diagnostics:
        metrics["critic/q_entropy_mean"] = float(diagnostics["q_entropy"].mean().item())
    if "q_pairs" in diagnostics:
        q_pairs = diagnostics["q_pairs"]
        for member in range(q_pairs.shape[0]):
            metrics[f"critic/q_member_{member}_q1_mean"] = float(q_pairs[member, 0].mean().item())
            metrics[f"critic/q_member_{member}_q2_mean"] = float(q_pairs[member, 1].mean().item())
            metrics[f"critic/q_member_{member}_clipped_mean"] = float(
                q_pairs[member].min(dim=0).values.mean().item()
            )
    if "rank_margins" in diagnostics:
        margins = diagnostics["rank_margins"]
        pair_count = margins.shape[1] * margins.shape[2]
        if pair_count:
            for member in range(min(3, margins.shape[0])):
                metrics[f"critic/rank_acc_q{member + 1}"] = float(
                    (margins[member] > 0.0).float().mean().item()
                )
            metrics["critic/rank_acc_mean"] = float(
                (margins.mean(dim=0) > 0.0).float().mean().item()
            )
            metrics["critic/rank_acc_unanimous"] = float(
                (margins > 0.0).all(dim=0).float().mean().item()
            )
            critic_cfg = (config or {}).get("critic", {})
            bin_width = (
                float(critic_cfg.get("q_vmax", 1.1))
                - float(critic_cfg.get("q_vmin", -0.1))
            ) / (int(critic_cfg.get("q_num_bins", 201)) - 1)
            margin = float(critic_cfg.get("rank_margin_bins", 2.0)) * bin_width
            metrics["critic/rank_margin_satisfied"] = float(
                (margins.min(dim=0).values > margin).float().mean().item()
            )
            metrics["critic/rank_pair_count"] = float(pair_count)
    critic.train(was_training)
    return metrics


@torch.no_grad()
def fit_conformal_calibration(
    state,
    batch: ChunkBatch,
    config: dict,
    *,
    inference_batch_size: int | None = None,
) -> float:
    was_training = state.critic.training
    state.critic.eval()
    q_values, _, _ = _critic_predictions(
        state.critic,
        batch,
        inference_batch_size=inference_batch_size,
        config=config,
    )
    q_mean, q_std = ensemble_mean_std(q_values)
    uncertainty_cfg = config.get("uncertainty", {})
    scale = compute_conformal_scale(
        q_mean,
        q_std,
        (batch.mc_returns if batch.mc_returns is not None else batch.chunk_returns).cpu(),
        coverage_delta=float(uncertainty_cfg.get("conformal_delta", uncertainty_cfg.get("coverage_delta", 0.1))),
        min_samples=int(uncertainty_cfg.get("min_calibration_samples", 16)),
    )
    state.conformal_scale = scale
    state.critic.train(was_training)
    return scale


def validation_metrics_for_training(state, batch: ChunkBatch, config: dict) -> dict[str, float]:
    """Compute fixed-replay critic diagnostics for periodic training logs."""
    uncertainty_cfg = config.get("uncertainty", {})
    inference_batch_size = config.get("evaluation", {}).get("inference_batch_size")
    if bool(uncertainty_cfg.get("use_conformal", False)):
        fit_conformal_calibration(
            state,
            batch,
            config,
            inference_batch_size=inference_batch_size,
        )
    metrics = offline_calibration_metrics(
        state.critic,
        batch,
        divl=state.divl if bool(config.get("divl", {}).get("enabled", True)) else None,
        conformal_scale=state.conformal_scale,
        inference_batch_size=inference_batch_size,
        config=config,
    )
    if config.get("training", {}).get("critic_sampling", {}).get("mode") in {"member_episode_bootstrap", "shared_episode_bootstrap"}:
        # Fixed validation samples, all ten raw heads (not pair-min).
        with torch.no_grad():
            was_training = state.critic.training
            state.critic.eval()
            _, _, diagnostics = _critic_predictions(state.critic, batch,
                inference_batch_size=inference_batch_size, config=config)
            raw = diagnostics["q_pairs"].flatten(0, 1).float()
            corr = torch.corrcoef(raw)
            off = ~torch.eye(raw.shape[0], dtype=torch.bool)
            metrics["raw10_q_head_correlation"] = float(corr[off].mean())
            metrics["raw10_ensemble_std"] = float(raw.std(dim=0, unbiased=False).mean())
            state.critic.train(was_training)
    validation = {f"validation_{key}": value for key, value in metrics.items()}
    validation.update(
        {
            key: value
            for key, value in metrics.items()
            if key.startswith("critic/rank_acc_")
            or key == "critic/rank_margin_satisfied"
        }
    )
    return validation
