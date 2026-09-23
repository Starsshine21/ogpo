from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RankQActions:
    positive: torch.Tensor
    noisy: torch.Tensor
    very_noisy: torch.Tensor
    random: torch.Tensor
    permuted: torch.Tensor
    permutation: torch.Tensor
    permuted_valid_mask: torch.Tensor | None = None
    same_episode_permuted_mask: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "RankQActions":
        return RankQActions(
            positive=self.positive.to(device),
            noisy=self.noisy.to(device),
            very_noisy=self.very_noisy.to(device),
            random=self.random.to(device),
            permuted=self.permuted.to(device),
            permutation=self.permutation.to(device),
            permuted_valid_mask=None if self.permuted_valid_mask is None else self.permuted_valid_mask.to(device),
            same_episode_permuted_mask=None if self.same_episode_permuted_mask is None else self.same_episode_permuted_mask.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> "RankQActions":
        action_indices = indices.to(self.positive.device)
        return RankQActions(
            positive=self.positive.index_select(0, action_indices),
            noisy=self.noisy.index_select(0, action_indices),
            very_noisy=self.very_noisy.index_select(0, action_indices),
            random=self.random.index_select(0, action_indices),
            permuted=self.permuted.index_select(0, action_indices),
            permutation=self.permutation.index_select(0, indices.to(self.permutation.device)),
            permuted_valid_mask=None if self.permuted_valid_mask is None else self.permuted_valid_mask.index_select(0, action_indices),
            same_episode_permuted_mask=None if self.same_episode_permuted_mask is None else self.same_episode_permuted_mask.index_select(0, action_indices),
        )


@dataclass(frozen=True)
class RankQLossOutput:
    loss: torch.Tensor
    loss_per_head: torch.Tensor
    success_loss: torch.Tensor
    failure_loss: torch.Tensor
    success_loss_per_head: torch.Tensor
    failure_loss_per_head: torch.Tensor
    metrics: dict[str, float]
    chain_loss: torch.Tensor | None = None
    relation_losses: dict[str, torch.Tensor] | None = None


def full_rankq_settings(critic_config: dict, *, optimizer_step: int | None = None) -> dict:
    """Explicit full semantics; old nested settings remain a legacy adapter."""
    nested = critic_config.get("rankq") or {}
    if not isinstance(nested, dict):
        raise TypeError("critic.rankq must be a mapping")
    full = nested.get("mode") == "full" or "noise_sigma" in nested
    if not full:
        return {"full": False, **same_state_rankq_settings(critic_config, optimizer_step=optimizer_step)}
    obsolete = {"mild_sigma", "strong_sigma", "use_success_only", "success_only",
                "logged_mild_weight", "mild_strong_weight", "strong_random_weight"}.intersection(nested)
    if obsolete:
        raise ValueError(f"full RankQ rejects legacy parameters {sorted(obsolete)}; use noise_sigma (strong=2*sigma)")
    # Reuse the optional absolute-step schedule, but not the legacy defaults.
    schedule_config = {"rankq": {**nested, "lambda_rank": nested.get("lambda_rank", 1.0)}}
    settings = same_state_rankq_settings(schedule_config, optimizer_step=optimizer_step)
    for key, default in (("noise_sigma", 0.04), ("alpha_success", 1.0), ("alpha_failure", 1.0)):
        value = float(nested.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"critic.rankq.{key} must be finite and non-negative")
        settings[key] = value
    if nested.get("permutation_mode", "same_task") != "same_task":
        raise ValueError("full RankQ requires permutation_mode=same_task")
    pair_loss = str(nested.get("pair_loss", "softplus"))
    if pair_loss not in {
        "softplus",
        "temperature_softplus",
        "temperature_softplus_guarded",
    }:
        raise ValueError(f"unsupported critic.rankq.pair_loss={pair_loss!r}")
    temperature = float(nested.get("temperature", 1.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("critic.rankq.temperature must be finite and positive")
    configured_max_gap = nested.get("max_gap")
    max_gap = None if configured_max_gap is None else float(configured_max_gap)
    if max_gap is not None and (not math.isfinite(max_gap) or max_gap <= 0.0):
        raise ValueError("critic.rankq.max_gap must be finite and positive")
    if pair_loss == "temperature_softplus_guarded" and max_gap is None:
        raise ValueError("guarded RankQ requires critic.rankq.max_gap")
    if pair_loss != "temperature_softplus_guarded" and max_gap is not None:
        raise ValueError("critic.rankq.max_gap is only valid for temperature_softplus_guarded")
    settings.update(
        full=True,
        use_permuted_negative=bool(nested.get("use_permuted_negative", True)),
        pair_loss=pair_loss,
        temperature=temperature,
        max_gap=max_gap,
    )
    for key in ("mild_sigma", "strong_sigma", "use_success_only", "logged_mild_weight", "mild_strong_weight", "strong_random_weight"):
        settings.pop(key, None)
    return settings


@dataclass(frozen=True)
class SameStateRankQActions:
    """Legal same-state actions for the nested ``critic.rankq`` auxiliary."""

    logged: torch.Tensor
    mild: torch.Tensor
    strong: torch.Tensor
    random: torch.Tensor | None

    def to(self, device: torch.device | str) -> "SameStateRankQActions":
        return SameStateRankQActions(
            logged=self.logged.to(device),
            mild=self.mild.to(device),
            strong=self.strong.to(device),
            random=None if self.random is None else self.random.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> "SameStateRankQActions":
        selected = indices.to(self.logged.device)
        return SameStateRankQActions(
            logged=self.logged.index_select(0, selected),
            mild=self.mild.index_select(0, selected),
            strong=self.strong.index_select(0, selected),
            random=(
                None
                if self.random is None
                else self.random.index_select(0, selected)
            ),
        )


@dataclass(frozen=True)
class SameStateRankQLossOutput:
    loss: torch.Tensor
    loss_per_head: torch.Tensor
    relation_losses: dict[str, torch.Tensor]
    metrics: dict[str, float]


_SAME_STATE_RELATION_WEIGHT_KEYS = {
    "logged_vs_mild": "logged_mild_weight",
    "mild_vs_strong": "mild_strong_weight",
    "strong_vs_random": "strong_random_weight",
}


def ddp_global_valid_mean_scale(
    local_valid_count: int,
    global_valid_count: int,
    world_size: int,
) -> float:
    """Pre-scale a local valid-sample mean before DDP averages gradients."""
    local = int(local_valid_count)
    global_count = int(global_valid_count)
    world = int(world_size)
    if local < 0 or global_count < 0 or world <= 0:
        raise ValueError("RankQ DDP counts must be non-negative and world_size positive")
    if local > global_count:
        raise ValueError("local RankQ count cannot exceed global count")
    if global_count == 0:
        return 0.0
    return float(world * local / global_count)


def same_state_rankq_settings(
    critic_config: dict,
    *,
    optimizer_step: int | None = None,
) -> dict[str, float | bool | int | None]:
    """Resolve the new nested RankQ config without changing legacy flat RankQ."""
    nested = critic_config.get("rankq", {})
    if nested is None or nested is False:
        nested = {}
    if not isinstance(nested, dict):
        raise TypeError("critic.rankq must be a mapping")
    enabled = bool(nested.get("enabled", False))
    lambda_rank_constant = float(nested.get("lambda_rank", 0.02))
    schedule_keys = {
        "lambda_rank_max",
        "rankq_warmup_start_step",
        "rankq_warmup_end_step",
    }
    configured_schedule_keys = schedule_keys.intersection(nested)
    if configured_schedule_keys and configured_schedule_keys != schedule_keys:
        missing = sorted(schedule_keys - configured_schedule_keys)
        raise ValueError(f"critic.rankq schedule is incomplete; missing {missing}")
    schedule_enabled = bool(configured_schedule_keys)
    if schedule_enabled:
        lambda_rank_max = float(nested["lambda_rank_max"])
        warmup_start = int(nested["rankq_warmup_start_step"])
        warmup_end = int(nested["rankq_warmup_end_step"])
        if warmup_start < 0 or warmup_end <= warmup_start:
            raise ValueError(
                "critic.rankq warmup requires 0 <= start_step < end_step"
            )
        if not math.isfinite(lambda_rank_max) or lambda_rank_max < 0.0:
            raise ValueError("critic.rankq.lambda_rank_max must be finite and non-negative")
        if optimizer_step is None:
            # Non-training consumers such as action-ranking evaluators need
            # the configured settings but do not apply the loss.
            lambda_rank = lambda_rank_max
        else:
            step = int(optimizer_step)
            if step <= warmup_start:
                lambda_rank = 0.0
            elif step >= warmup_end:
                lambda_rank = lambda_rank_max
            else:
                lambda_rank = lambda_rank_max * (
                    (step - warmup_start) / (warmup_end - warmup_start)
                )
    else:
        lambda_rank_max = lambda_rank_constant
        warmup_start = warmup_end = None
        lambda_rank = lambda_rank_constant
    mild_sigma = float(nested.get("mild_sigma", 0.02))
    strong_sigma = float(nested.get("strong_sigma", 0.05))
    use_success_only = bool(
        nested.get("use_success_only", nested.get("success_only", True))
    )
    use_random_negative = bool(nested.get("use_random_negative", True))
    relation_weights = {
        relation: float(nested.get(config_key, 1.0))
        for relation, config_key in _SAME_STATE_RELATION_WEIGHT_KEYS.items()
    }
    if not math.isfinite(lambda_rank_constant) or lambda_rank_constant < 0.0:
        raise ValueError("critic.rankq.lambda_rank must be non-negative")
    if mild_sigma < 0.0 or strong_sigma < 0.0:
        raise ValueError("critic.rankq sigmas must be non-negative")
    if strong_sigma < mild_sigma:
        raise ValueError("critic.rankq.strong_sigma must be >= mild_sigma")
    if any(
        not torch.isfinite(torch.tensor(weight)).item() or weight < 0.0
        for weight in relation_weights.values()
    ):
        raise ValueError("critic.rankq relation weights must be finite and non-negative")
    return {
        "enabled": enabled and lambda_rank > 0.0,
        "configured_enabled": enabled,
        "lambda_rank": lambda_rank,
        "lambda_rank_effective": lambda_rank,
        "lambda_rank_max": lambda_rank_max,
        "lambda_rank_schedule_enabled": schedule_enabled,
        "rankq_warmup_start_step": warmup_start,
        "rankq_warmup_end_step": warmup_end,
        "lambda_rank_schedule_step": optimizer_step,
        "mild_sigma": mild_sigma,
        "strong_sigma": strong_sigma,
        "use_success_only": use_success_only,
        "use_random_negative": use_random_negative,
        **{
            config_key: relation_weights[relation]
            for relation, config_key in _SAME_STATE_RELATION_WEIGHT_KEYS.items()
        },
    }


def make_same_state_rankq_actions(
    action_chunks: torch.Tensor,
    execution_masks: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    mild_sigma: float,
    strong_sigma: float,
    use_random_negative: bool,
    generator: torch.Generator | None = None,
) -> SameStateRankQActions:
    """Perturb normalized executed actions and clamp every result to legal bounds."""
    if action_chunks.ndim != 3 or execution_masks.shape != action_chunks.shape[:2]:
        raise ValueError("actions must be [batch,horizon,action_dim] with masks [batch,horizon]")
    if mild_sigma < 0.0 or strong_sigma < mild_sigma:
        raise ValueError("RankQ requires 0 <= mild_sigma <= strong_sigma")
    values = action_chunks.float()
    mean, std, lower, upper = _normalized_bounds(
        action_mean=action_mean,
        action_std=action_std,
        action_min=action_min,
        action_max=action_max,
        device=values.device,
    )
    normalized = (values - mean) / std
    executed = execution_masks.to(device=values.device, dtype=torch.bool).unsqueeze(-1)

    # Mild and strong must share one direction. Clamp epsilon using the
    # stronger perturbation so both actions stay legal while preserving
    # a_m=a+sigma_m*epsilon and a_s=a+sigma_s*epsilon exactly.
    epsilon = torch.randn(
        normalized.shape,
        device=normalized.device,
        dtype=torch.float32,
        generator=generator,
    )
    if strong_sigma > 0.0:
        epsilon = torch.maximum(
            torch.minimum(epsilon, (upper - normalized) / float(strong_sigma)),
            (lower - normalized) / float(strong_sigma),
        )
    else:
        epsilon = torch.zeros_like(epsilon)
    mild_normalized = torch.where(
        executed,
        normalized + float(mild_sigma) * epsilon,
        normalized,
    )
    strong_normalized = torch.where(
        executed,
        normalized + float(strong_sigma) * epsilon,
        normalized,
    )
    random_normalized = None
    if use_random_negative:
        random_unit = torch.rand(
            normalized.shape,
            device=normalized.device,
            dtype=torch.float32,
            generator=generator,
        )
        candidate = lower + random_unit * (upper - lower)
        random_normalized = torch.where(executed, candidate, normalized)

    def restore(candidate: torch.Tensor | None) -> torch.Tensor | None:
        if candidate is None:
            return None
        return (candidate * std + mean).to(action_chunks.dtype)

    return SameStateRankQActions(
        logged=action_chunks,
        mild=restore(mild_normalized),
        strong=restore(strong_normalized),
        random=restore(random_normalized),
    )


def compute_same_state_rankq_loss(
    q_values: dict[str, torch.Tensor],
    successes: torch.Tensor,
    *,
    use_success_only: bool = True,
    logged_mild_weight: float = 1.0,
    mild_strong_weight: float = 1.0,
    strong_random_weight: float = 1.0,
) -> SameStateRankQLossOutput:
    """Apply the exact raw-head chain logged > mild > strong > random."""
    required = {"logged", "mild", "strong"}
    missing = required - set(q_values)
    if missing:
        raise KeyError(f"missing same-state RankQ values: {sorted(missing)}")
    logged = q_values["logged"]
    if logged.ndim != 2:
        raise ValueError("same-state RankQ values must have shape [raw_heads,batch]")
    if any(value.shape != logged.shape for value in q_values.values()):
        raise ValueError("all same-state RankQ values must have identical shape")
    if successes.shape != logged.shape[1:]:
        raise ValueError("successes must have shape [batch]")
    valid = (
        successes.to(device=logged.device, dtype=torch.bool)
        if use_success_only
        else torch.ones_like(successes, device=logged.device, dtype=torch.bool)
    )
    relations: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "logged_vs_mild": (logged, q_values["mild"]),
        "mild_vs_strong": (q_values["mild"], q_values["strong"]),
    }
    if "random" in q_values:
        relations["strong_vs_random"] = (q_values["strong"], q_values["random"])
    relation_weights = {
        "logged_vs_mild": float(logged_mild_weight),
        "mild_vs_strong": float(mild_strong_weight),
        "strong_vs_random": float(strong_random_weight),
    }
    if any(
        not torch.isfinite(torch.tensor(weight)).item() or weight < 0.0
        for weight in relation_weights.values()
    ):
        raise ValueError("same-state RankQ relation weights must be finite and non-negative")

    relation_losses: dict[str, torch.Tensor] = {}
    per_head_parts = []
    metrics: dict[str, float] = {
        "rankq_success_count": float(valid.sum().item()),
    }
    for name, (preferred, inferior) in relations.items():
        pair = rank_pair_loss(preferred, inferior)
        if bool(valid.any()):
            selected = pair[:, valid]
            per_head = selected.mean(dim=1)
            relation_loss = selected.mean()
            margin = (preferred - inferior)[:, valid].detach()
            # d softplus(Q_inferior-Q_preferred) / d(Q_inferior-Q_preferred).
            # Near-zero values expose an already-saturated ordering relation.
            softplus_slope = torch.sigmoid(-margin)
            accuracy = float((margin > 0.0).float().mean().item())
            unanimous = float((margin > 0.0).all(dim=0).float().mean().item())
            slope_mean = float(softplus_slope.mean().item())
            slope_median = float(torch.quantile(softplus_slope.float(), 0.5).item())
            slope_p90 = float(torch.quantile(softplus_slope.float(), 0.9).item())
            slope_below_1e2 = float((softplus_slope < 1.0e-2).float().mean().item())
        else:
            per_head = pair.sum(dim=1) * 0.0
            relation_loss = pair.sum() * 0.0
            accuracy = 0.0
            unanimous = 0.0
            slope_mean = 0.0
            slope_median = 0.0
            slope_p90 = 0.0
            slope_below_1e2 = 0.0
        relation_losses[name] = relation_loss
        weight = relation_weights[name]
        # Preserve the original graph exactly for the default configuration.
        per_head_parts.append(per_head if weight == 1.0 else weight * per_head)
        metrics[f"rankq_{name}_acc"] = accuracy
        metrics[f"rankq_raw10_unanimous_{name}"] = unanimous
        metrics[f"rankq_{name}_softplus_slope_mean"] = slope_mean
        metrics[f"rankq_{name}_softplus_slope_median"] = slope_median
        metrics[f"rankq_{name}_softplus_slope_p90"] = slope_p90
        metrics[f"rankq_{name}_softplus_slope_below_1e2"] = slope_below_1e2
        metrics[f"rankq_{name}_weight"] = weight
        metrics[f"rankq_{name}_weighted_loss"] = float(
            (relation_loss.detach() * weight).item()
        )
    loss_per_head = torch.stack(per_head_parts, dim=0).sum(dim=0)
    if all(relation_weights[name] == 1.0 for name in relations):
        # Exact legacy reduction/graph for the default 1/1/1 behavior.
        loss = loss_per_head.mean()
    else:
        weighted_losses = {
            name: relation_weights[name] * relation_losses[name]
            for name in relations
        }
        loss = weighted_losses["logged_vs_mild"] + weighted_losses[
            "mild_vs_strong"
        ]
        if "strong_vs_random" in weighted_losses:
            loss = loss + weighted_losses["strong_vs_random"]
    if "strong_vs_random" not in relations:
        metrics.update(
            {
                "rankq_strong_vs_random_acc": 0.0,
                "rankq_raw10_unanimous_strong_vs_random": 0.0,
                "rankq_strong_vs_random_softplus_slope_mean": 0.0,
                "rankq_strong_vs_random_softplus_slope_median": 0.0,
                "rankq_strong_vs_random_softplus_slope_p90": 0.0,
                "rankq_strong_vs_random_softplus_slope_below_1e2": 0.0,
                "rankq_strong_vs_random_weight": relation_weights[
                    "strong_vs_random"
                ],
                "rankq_strong_vs_random_weighted_loss": 0.0,
            }
        )
    metrics.update(
        {
            "rankq_loss": float(loss.detach().item()),
            "rankq_mild_loss": float(
                relation_losses["logged_vs_mild"].detach().item()
            ),
            "rankq_strong_loss": float(
                relation_losses["mild_vs_strong"].detach().item()
            ),
            "rankq_random_loss": float(
                relation_losses.get("strong_vs_random", loss * 0.0)
                .detach()
                .item()
            ),
            "rankq_total_raw_relation_loss": float(
                sum(
                    relation_loss.detach()
                    for relation_loss in relation_losses.values()
                ).item()
            ),
            "rankq_total_relation_weighted_loss": float(loss.detach().item()),
        }
    )
    return SameStateRankQLossOutput(
        loss=loss,
        loss_per_head=loss_per_head,
        relation_losses=relation_losses,
        metrics=metrics,
    )


def disabled_same_state_rankq_metrics(
    *,
    lambda_rank: float = 0.0,
    logged_mild_weight: float = 1.0,
    mild_strong_weight: float = 1.0,
    strong_random_weight: float = 1.0,
) -> dict[str, float]:
    metrics = {
        "rankq_enabled": 0.0,
        "rankq_lambda": float(lambda_rank),
        "rankq_loss": 0.0,
        "rankq_mild_loss": 0.0,
        "rankq_strong_loss": 0.0,
        "rankq_random_loss": 0.0,
        "rankq_total_raw_relation_loss": 0.0,
        "rankq_total_relation_weighted_loss": 0.0,
        "rankq_total_lambda_weighted_loss": 0.0,
        "rankq_success_count": 0.0,
        "rankq_weighted_loss": 0.0,
    }
    weights = {
        "logged_vs_mild": float(logged_mild_weight),
        "mild_vs_strong": float(mild_strong_weight),
        "strong_vs_random": float(strong_random_weight),
    }
    for relation in ("logged_vs_mild", "mild_vs_strong", "strong_vs_random"):
        metrics[f"rankq_{relation}_acc"] = 0.0
        metrics[f"rankq_raw10_unanimous_{relation}"] = 0.0
        metrics[f"rankq_{relation}_softplus_slope_mean"] = 0.0
        metrics[f"rankq_{relation}_softplus_slope_median"] = 0.0
        metrics[f"rankq_{relation}_softplus_slope_p90"] = 0.0
        metrics[f"rankq_{relation}_softplus_slope_below_1e2"] = 0.0
        metrics[f"rankq_{relation}_weight"] = weights[relation]
        metrics[f"rankq_{relation}_weighted_loss"] = 0.0
    return metrics


def _normalized_bounds(
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = action_mean.to(device=device, dtype=torch.float32).view(1, 1, -1)
    std = action_std.to(device=device, dtype=torch.float32).clamp_min(1e-6).view(1, 1, -1)
    lower = (
        (action_min.to(device=device, dtype=torch.float32).view(1, 1, -1) - mean) / std
    )
    upper = (
        (action_max.to(device=device, dtype=torch.float32).view(1, 1, -1) - mean) / std
    )
    return mean, std, lower, upper


def make_rankq_actions(
    action_chunks: torch.Tensor,
    execution_masks: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    noise_sigma: float,
    generator: torch.Generator | None = None,
    task_ids: list[str] | None = None,
    episode_ids: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
) -> RankQActions:
    """Construct vanilla RankQ actions in the critic's normalized action space."""
    if action_chunks.ndim != 3 or execution_masks.shape != action_chunks.shape[:2]:
        raise ValueError("actions must be [batch, horizon, action_dim] with masks [batch, horizon]")
    if not math.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError("rankq_noise_sigma must be non-negative")

    values = action_chunks.float()
    mean, std, lower, upper = _normalized_bounds(
        action_mean=action_mean,
        action_std=action_std,
        action_min=action_min,
        action_max=action_max,
        device=values.device,
    )
    normalized = (values - mean) / std
    executed = execution_masks.to(device=values.device, dtype=torch.bool).unsqueeze(-1)

    # Use the same Gaussian direction for noisy and very-noisy actions.
    epsilon = torch.randn(
        normalized.shape,
        device=normalized.device,
        dtype=torch.float32,
        generator=generator,
    )
    delta = float(noise_sigma) * epsilon
    # Clip the shared one-step delta so that both a+delta and a+2*delta stay
    # legal without breaking the exact two-times perturbation relation.
    delta = torch.maximum(
        torch.minimum(delta, (upper - normalized) / 2.0),
        (lower - normalized) / 2.0,
    )
    noisy_normalized = normalized + delta
    very_noisy_normalized = normalized + 2.0 * delta
    random_unit = torch.rand(
        normalized.shape,
        device=normalized.device,
        dtype=torch.float32,
        generator=generator,
    )
    random_normalized = lower + random_unit * (upper - lower)

    noisy_normalized = torch.where(executed, noisy_normalized, normalized)
    very_noisy_normalized = torch.where(executed, very_noisy_normalized, normalized)
    random_normalized = torch.where(executed, random_normalized, normalized)

    batch_size = action_chunks.shape[0]
    permutation = torch.arange(batch_size, device=values.device)
    if batch_size > 1:
        permutation = torch.roll(permutation, shifts=1)
    valid = torch.full((batch_size,), batch_size > 1, device=values.device, dtype=torch.bool)
    same_episode = torch.zeros_like(valid)
    if task_ids is not None:
        if len(task_ids) != batch_size or episode_ids is None or episode_ids.shape != (batch_size,):
            raise ValueError("same-task permutation requires task_ids and episode_ids for the full batch")
        ids = episode_ids.cpu().tolist()
        times = None if timesteps is None else timesteps.cpu().tolist()
        valid.zero_()
        for i, task in enumerate(task_ids):
            candidates = [j for j, donor_task in enumerate(task_ids)
                          if donor_task == task and j != i
                          and (times is None or ids[j] != ids[i] or times[j] != times[i])]
            preferred = [j for j in candidates if ids[j] != ids[i]]
            donors = preferred or candidates
            if donors:
                offset = int(torch.randint(len(donors), (1,), device=values.device, generator=generator).item())
                permutation[i] = donors[offset]
                valid[i] = True
                same_episode[i] = ids[donors[offset]] == ids[i]
            else:
                permutation[i] = i
    donor_normalized = normalized.index_select(0, permutation)
    donor_executed = executed.index_select(0, permutation)
    permuted_normalized = torch.where(
        executed & donor_executed,
        donor_normalized,
        normalized,
    )

    def restore(candidate: torch.Tensor) -> torch.Tensor:
        restored = (candidate * std + mean).to(action_chunks.dtype)
        return torch.where(executed, restored, action_chunks)

    return RankQActions(
        positive=action_chunks,
        noisy=restore(noisy_normalized),
        very_noisy=restore(very_noisy_normalized),
        random=restore(random_normalized),
        permuted=restore(permuted_normalized),
        permutation=permutation,
        permuted_valid_mask=valid,
        same_episode_permuted_mask=same_episode,
    )


def rank_pair_loss(
    q_positive: torch.Tensor,
    q_negative: torch.Tensor,
    *,
    pair_loss: str = "softplus",
    temperature: float = 1.0,
    max_gap: float | None = None,
) -> torch.Tensor:
    """Canonical RankQ pair loss; both Q arguments always receive gradients.

    ``temperature_softplus_guarded`` uses tau=0.1 to make gaps on the
    critic's natural 0--1 value scale saturate around 0.1 rather than 1.0.
    Its intentionally loose max_gap=1.0 cutoff is a safety fuse: at that gap
    the natural slope is already sigmoid(-10)~=4.5e-5, so the temperature,
    rather than the guard, should normally control training.
    """
    if q_positive.shape != q_negative.shape:
        raise ValueError("positive and negative Q tensors must have identical shapes")
    if pair_loss == "softplus":
        return F.softplus(q_negative - q_positive)
    if pair_loss not in {"temperature_softplus", "temperature_softplus_guarded"}:
        raise ValueError(f"unsupported RankQ pair_loss={pair_loss!r}")
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("RankQ temperature must be finite and positive")
    gap = q_positive - q_negative
    base = tau * F.softplus(-gap / tau)
    if pair_loss == "temperature_softplus":
        if max_gap is not None:
            raise ValueError("max_gap is not used by temperature_softplus")
        return base
    if max_gap is None or not math.isfinite(float(max_gap)) or float(max_gap) <= 0.0:
        raise ValueError("guarded RankQ requires a finite positive max_gap")
    cutoff = tau * F.softplus(base.new_tensor(-float(max_gap) / tau))
    # Continuous at max_gap. PyTorch ReLU's derivative at exactly zero is 0,
    # so gap >= max_gap receives exactly no RankQ gradient.
    return F.relu(base - cutoff)


def _masked_head_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-head dimensions while preserving the leading Q-head axis."""
    if values.ndim < 2 or mask.shape != values.shape[-1:]:
        raise ValueError("RankQ values must end in the batch dimension selected by mask")
    if bool(mask.any()):
        selected = values[..., mask]
        return selected.reshape(values.shape[0], -1).mean(dim=1)
    return values.reshape(values.shape[0], -1).sum(dim=1) * 0.0


def _masked_success_pair_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum RankQ relations per success sample, then average over samples."""
    if values.ndim != 3 or mask.shape != values.shape[-1:]:
        raise ValueError("success RankQ values must have shape [heads, pairs, batch]")
    if bool(mask.any()):
        return values[..., mask].sum(dim=1).mean(dim=1)
    return values.reshape(values.shape[0], -1).sum(dim=1) * 0.0


def _masked_diagnostic(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values.detach()[..., mask].mean().item())


def compute_rankq_loss(
    q_values: dict[str, torch.Tensor],
    successes: torch.Tensor,
    *,
    permuted_valid_mask: torch.Tensor | None = None,
    alpha_success: float = 1.0,
    alpha_failure: float = 1.0,
    use_random_negative: bool = True,
    use_permuted_negative: bool = True,
    mean_scales: dict[str, float] | None = None,
    pair_loss: str = "softplus",
    temperature: float = 1.0,
    max_gap: float | None = None,
) -> RankQLossOutput:
    """Canonical full RankQ on raw scalar Q (or legacy decoded expectations).

    ``success_loss`` retains the legacy success+chain contract; separate
    success/chain metrics expose the four-plus-two relation decomposition.
    Each relation is averaged over its own valid samples, then raw heads.
    """
    required = {"positive", "noisy", "very_noisy", "random", "permuted"}
    missing = required - set(q_values)
    if missing:
        raise KeyError(f"missing RankQ values: {sorted(missing)}")
    positive = q_values["positive"]
    if positive.ndim != 2:
        raise ValueError("RankQ Q values must have shape [heads, batch]")
    if any(value.shape != positive.shape for value in q_values.values()):
        raise ValueError("all RankQ Q tensors must have shape [heads, batch]")
    if successes.shape != positive.shape[1:]:
        raise ValueError("successes must have shape [batch]")

    success_mask = successes.to(device=positive.device, dtype=torch.bool)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in (alpha_success, alpha_failure)):
        raise ValueError("RankQ outcome weights must be finite and non-negative")
    failure_mask = ~success_mask
    relations = {
        "pos_noisy": (positive, q_values["noisy"]),
        "pos_very_noisy": (positive, q_values["very_noisy"]),
        "pos_random": (positive, q_values["random"]),
        "pos_permuted": (positive, q_values["permuted"]),
        "noisy_very_noisy": (q_values["noisy"], q_values["very_noisy"]),
        "very_noisy_random": (q_values["very_noisy"], q_values["random"]),
    }
    def relation_enabled(name: str) -> bool:
        return (
            (use_random_negative or name not in {"pos_random", "very_noisy_random", "failure_random"})
            and (use_permuted_negative or name != "pos_permuted")
        )
    if not all(bool(torch.isfinite(value).all()) for value in q_values.values()):
        raise FloatingPointError("non-finite raw Q in RankQ")
    valid_perm = torch.ones_like(success_mask) if permuted_valid_mask is None else permuted_valid_mask.to(positive.device).bool()
    if valid_perm.shape != success_mask.shape:
        raise ValueError("permuted_valid_mask must have shape [batch]")
    scales = mean_scales or {}
    relation_losses = {}
    relation_heads = {}
    for name, (preferred, inferior) in relations.items():
        mask = success_mask & valid_perm if name == "pos_permuted" else success_mask
        enabled = relation_enabled(name)
        per_head = _masked_head_mean(
            rank_pair_loss(
                preferred,
                inferior,
                pair_loss=pair_loss,
                temperature=temperature,
                max_gap=max_gap,
            ),
            mask,
        ) * float(scales.get("permuted" if name == "pos_permuted" else "success", 1.0))
        relation_heads[name] = per_head if enabled else per_head * 0.0
        relation_losses[name] = relation_heads[name].mean()
    chain_per_head = relation_heads["noisy_very_noisy"] + relation_heads["very_noisy_random"]
    success_only_per_head = sum(relation_heads[name] for name in ("pos_noisy", "pos_very_noisy", "pos_random", "pos_permuted"))
    success_loss_per_head = success_only_per_head + chain_per_head
    failure_pair_loss = rank_pair_loss(
        positive,
        q_values["random"],
        pair_loss=pair_loss,
        temperature=temperature,
        max_gap=max_gap,
    )
    failure_loss_per_head = _masked_head_mean(failure_pair_loss, failure_mask) * float(scales.get("failure", 1.0))
    if not use_random_negative:
        failure_loss_per_head = failure_loss_per_head * 0.0
    loss_per_head = alpha_success * success_loss_per_head + alpha_failure * failure_loss_per_head
    success_loss = success_loss_per_head.mean()
    failure_loss = failure_loss_per_head.mean()
    loss = loss_per_head.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite full RankQ loss")

    metrics: dict[str, float] = {
        "rankq/loss": float(loss.detach().item()),
        "rankq/success_loss": float(success_loss.detach().item()),
        "rankq/failure_loss": float(failure_loss.detach().item()),
        "rankq/success_fraction": float(success_mask.float().mean().item()),
        "rankq/failure_fraction": float(failure_mask.float().mean().item()),
        "rankq/success_count": float(success_mask.sum().item()),
        "rankq/failure_count": float(failure_mask.sum().item()),
    }
    for head, head_loss in enumerate(loss_per_head):
        metrics[f"rankq/loss_q{head + 1}"] = float(head_loss.detach().item())
        metrics[f"rankq/success_loss_q{head + 1}"] = float(
            success_loss_per_head[head].detach().item()
        )
        metrics[f"rankq/failure_loss_q{head + 1}"] = float(
            failure_loss_per_head[head].detach().item()
        )
    for name, (q_positive, q_negative) in relations.items():
        margin = q_positive - q_negative
        relation_mask = success_mask & valid_perm if name == "pos_permuted" else success_mask
        metrics[f"rankq/margin_{name}"] = _masked_diagnostic(margin, relation_mask)
        metrics[f"rankq/acc_{name}"] = _masked_diagnostic(
            (margin > 0.0).to(margin.dtype),
            relation_mask,
        )
    failure_margin = positive - q_values["random"]
    metrics["rankq/margin_failure_random"] = _masked_diagnostic(
        failure_margin,
        failure_mask,
    )
    metrics["rankq/acc_failure_random"] = _masked_diagnostic(
        (failure_margin > 0.0).to(failure_margin.dtype),
        failure_mask,
    )
    canonical = {"pos_noisy": "pos_vs_noisy", "pos_very_noisy": "pos_vs_very_noisy",
                 "pos_random": "pos_vs_random", "pos_permuted": "pos_vs_permuted",
                 "noisy_very_noisy": "noisy_vs_very_noisy", "very_noisy_random": "very_noisy_vs_random"}
    all_relations = {**relations, "failure_random": (positive, q_values["random"])}
    relation_losses["failure_random"] = failure_loss_per_head.mean()
    natural_temperature = float(temperature) if pair_loss != "softplus" else 1.0
    guard_enabled = pair_loss == "temperature_softplus_guarded"
    all_margins: list[torch.Tensor] = []
    all_triggers: list[torch.Tensor] = []
    all_natural_slopes: list[torch.Tensor] = []
    all_any_head: list[torch.Tensor] = []
    all_all_heads: list[torch.Tensor] = []
    for name, (preferred, inferior) in all_relations.items():
        mask = failure_mask if name == "failure_random" else success_mask
        if name == "pos_permuted":
            mask = mask & valid_perm
        label = canonical.get(name, "failure_pos_vs_random")
        enabled = relation_enabled(name)
        margin = (preferred - inferior).detach()[:, mask].float()
        if not enabled:
            margin = margin[:, :0]
        slope = torch.sigmoid(-margin / natural_temperature)
        triggered = (
            margin >= float(max_gap)
            if guard_enabled and max_gap is not None
            else torch.zeros_like(margin, dtype=torch.bool)
        )
        metrics[f"rankq_{name}_weight"] = alpha_failure if name == "failure_random" else alpha_success
        metrics[f"rankq_{label}_loss"] = float(relation_losses[name].detach())
        metrics[f"rankq_{label}_margin"] = float(margin.mean()) if margin.numel() else 0.0
        metrics[f"rankq_{label}_acc"] = float((margin > 0).float().mean()) if margin.numel() else 0.0
        metrics[f"rankq_raw10_unanimous_{label}"] = float((margin > 0).all(0).float().mean()) if margin.numel() else 0.0
        for suffix, value in (("mean", slope.mean() if slope.numel() else 0),
                              ("median", torch.quantile(slope, 0.5) if slope.numel() else 0),
                              ("p90", torch.quantile(slope, 0.9) if slope.numel() else 0),
                              ("below_1e2", (slope < 1e-2).float().mean() if slope.numel() else 0)):
            metrics[f"rankq_{label}_softplus_slope_{suffix}"] = float(value)
        for suffix, value in (
            ("mean", margin.mean() if margin.numel() else 0),
            ("median", torch.quantile(margin, 0.5) if margin.numel() else 0),
            ("p10", torch.quantile(margin, 0.1) if margin.numel() else 0),
            ("p90", torch.quantile(margin, 0.9) if margin.numel() else 0),
        ):
            metrics[f"rankq/gap/{label}/{suffix}"] = float(value)
        for threshold, threshold_label in ((0.1, "0p1"), (0.2, "0p2"), (0.5, "0p5"), (1.0, "1p0")):
            metrics[f"rankq/gap/{label}/ge_{threshold_label}_fraction"] = (
                float((margin >= threshold).float().mean()) if margin.numel() else 0.0
            )
        relation_valid_samples = margin.shape[1] if margin.ndim == 2 else 0
        guard_label = "failure_pos_random" if name == "failure_random" else name
        metrics[f"rankq/guard/{guard_label}_fraction"] = (
            float(triggered.float().mean()) if triggered.numel() else 0.0
        )
        metrics[f"rankq/guard/{guard_label}_any_head_fraction"] = (
            float(triggered.any(dim=0).float().mean()) if relation_valid_samples else 0.0
        )
        metrics[f"rankq/guard/{guard_label}_all_heads_fraction"] = (
            float(triggered.all(dim=0).float().mean()) if relation_valid_samples else 0.0
        )
        if margin.numel():
            all_margins.append(margin.reshape(-1))
            all_triggers.append(triggered.reshape(-1))
            all_natural_slopes.append(slope.reshape(-1))
            all_any_head.append(triggered.any(dim=0))
            all_all_heads.append(triggered.all(dim=0))
    if all_margins:
        overall_margin = torch.cat(all_margins)
        overall_trigger = torch.cat(all_triggers)
        overall_slope = torch.cat(all_natural_slopes)
        overall_any = torch.cat(all_any_head)
        overall_all = torch.cat(all_all_heads)
    else:
        overall_margin = positive.detach().new_empty(0, dtype=torch.float32)
        overall_trigger = torch.empty(0, dtype=torch.bool, device=positive.device)
        overall_slope = positive.detach().new_empty(0, dtype=torch.float32)
        overall_any = torch.empty(0, dtype=torch.bool, device=positive.device)
        overall_all = torch.empty(0, dtype=torch.bool, device=positive.device)
    trigger_count = int(overall_trigger.sum().item())
    valid_count = int(overall_trigger.numel())
    metrics.update({
        "rankq/guard/overall_fraction": float(overall_trigger.float().mean()) if valid_count else 0.0,
        "rankq/guard/overall_count": float(trigger_count),
        "rankq/guard/overall_valid_count": float(valid_count),
        "rankq/guard/overall_any_head_fraction": float(overall_any.float().mean()) if overall_any.numel() else 0.0,
        "rankq/guard/overall_all_heads_fraction": float(overall_all.float().mean()) if overall_all.numel() else 0.0,
        "rankq/natural_slope_mean": float(overall_slope.mean()) if overall_slope.numel() else 0.0,
        "rankq/natural_slope_median": float(torch.quantile(overall_slope, 0.5)) if overall_slope.numel() else 0.0,
        "rankq/natural_slope_p90": float(torch.quantile(overall_slope, 0.9)) if overall_slope.numel() else 0.0,
        "rankq/natural_slope_below_1e2_fraction": float((overall_slope < 1e-2).float().mean()) if overall_slope.numel() else 0.0,
        "rankq/guard_trigger_but_natural_slope_gt_1e2_fraction": (
            float((overall_slope[overall_trigger] > 1e-2).float().mean()) if trigger_count else 0.0
        ),
    })
    for threshold, threshold_label in ((0.1, "0p1"), (0.2, "0p2"), (0.5, "0p5"), (1.0, "1p0")):
        metrics[f"rankq/gap/overall_ge_{threshold_label}_fraction"] = (
            float((overall_margin >= threshold).float().mean()) if overall_margin.numel() else 0.0
        )
    metrics.update(rankq_success_loss=float(success_only_per_head.mean().detach()),
                   rankq_chain_loss=float(chain_per_head.mean().detach()),
                   rankq_failure_loss=float(failure_loss.detach()), rankq_raw_loss=float(loss.detach()),
                   rankq_loss=float(loss.detach()), rankq_success_count=float(success_mask.sum()),
                   rankq_failure_count=float(failure_mask.sum()),
                   rankq_permuted_valid_fraction=float(valid_perm.float().mean()),
                   alpha_success=float(alpha_success), alpha_failure=float(alpha_failure))
    return RankQLossOutput(
        loss=loss,
        loss_per_head=loss_per_head,
        success_loss=success_loss,
        failure_loss=failure_loss,
        success_loss_per_head=success_loss_per_head,
        failure_loss_per_head=failure_loss_per_head,
        metrics=metrics,
        chain_loss=chain_per_head.mean(),
        relation_losses=relation_losses,
    )


def disabled_rankq_metrics(*, ensemble_size: int) -> dict[str, float]:
    metrics = {
        "rankq/enabled": 0.0,
        "rankq/loss": 0.0,
        "rankq/success_loss": 0.0,
        "rankq/failure_loss": 0.0,
        "rankq/success_fraction": 0.0,
        "rankq/failure_fraction": 0.0,
        "rankq/success_count": 0.0,
        "rankq/failure_count": 0.0,
    }
    for head in range(ensemble_size):
        metrics[f"rankq/loss_q{head + 1}"] = 0.0
        metrics[f"rankq/success_loss_q{head + 1}"] = 0.0
        metrics[f"rankq/failure_loss_q{head + 1}"] = 0.0
    for name in (
        "pos_noisy",
        "pos_very_noisy",
        "pos_random",
        "pos_permuted",
        "noisy_very_noisy",
        "very_noisy_random",
    ):
        metrics[f"rankq/margin_{name}"] = 0.0
        metrics[f"rankq/acc_{name}"] = 0.0
    metrics["rankq/margin_failure_random"] = 0.0
    metrics["rankq/acc_failure_random"] = 0.0
    return metrics
