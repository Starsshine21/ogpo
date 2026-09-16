from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import partial
import copy
import gc
import math
import warnings
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .conservative_advantage import (
    RunningMAD,
    group_normalized_advantage,
    group_relative_conservative_advantage,
    lcb_advantage,
    scheduled_lambda_abs,
    sign_consensus_advantage,
)
from .chi2_regularization import (
    Chi2PessimismStats,
    Chi2RatioStats,
    apply_chipo_to_ca_advantage,
    chi2_ppo_upper_bound,
    chi2_pessimistic_advantage,
    full_chain_chipo_ratio,
    full_chain_chipo_regularization_loss,
    selected_transition_chi2_ratio,
)
from .categorical_q import (
    categorical_q_entropy,
    consensus_ranking_loss,
    decode_categorical_q,
    hl_gauss_projection,
    one_hot_categorical_cross_entropy,
    ranking_action_negatives,
)
from .critic import ScalarQEnsemble, assert_no_gradients, clone_target, soft_update
from .critic_targets import aggregate_value_heads
from .distributional_value import DistributionalValueEnsemble, make_support, make_support_from_targets
from .divl import (
    aggregate_double_q_for_v,
    divl_clipped_double_q_projection_targets,
    divl_double_q_projection_targets,
    divl_projection_targets,
    divl_quantile_values,
)
from .ensemble import bootstrap_mask, ensemble_mean_std
from .flash_ogpo import flash_ppo_loss, sample_flash_rollout
from .flow_logprob import gaussian_kl_diag, gaussian_log_prob
from .flow_sde import GaussianFlowPolicy
from .full_ogpo import (
    full_chain_ais_ppo_loss,
    full_chain_chipo_ppo_loss,
    full_chain_ppo_loss,
)
from .gemma_siglip_backbone import (
    build_gemma_siglip_critic,
    build_gemma_siglip_scalar_q_critic,
    configure_critic_stage,
    install_final_gemma_lora,
)
from .losses import (
    action_smoothness_loss,
    flow_matching_anchor_loss,
    success_buffer_loss,
    weighted_flow_matching_loss,
)
from .metrics import grad_norm
from .multimodal_critic import MultiHeadScalarQCritic, MultiHeadUdivlCritic
from .openpi_flow_spec import OpenPIStochasticFlowPolicy
from .pi05_jax_adapter import PI05JaxFlowPolicy, ema_actor_state, load_pi05_jax_flow_policy
from .pi05_jax_flow_core import (
    OpenPIJaxFlowSpec,
    flash_ppo_loss as jax_flash_ppo_loss,
    flow_matching_loss as jax_flow_matching_loss,
    full_chain_ais_ppo_loss as jax_full_chain_ais_ppo_loss,
    full_chain_ppo_loss as jax_full_chain_ppo_loss,
    gaussian_kl_diag as jax_gaussian_kl_diag,
    gaussian_log_prob as jax_gaussian_log_prob,
    rollout as jax_rollout,
    sample_flash_rollout as sample_jax_flash_rollout,
    state_adaptive_kl_penalty as jax_state_adaptive_kl_penalty,
    transition_kl as jax_transition_kl,
    transition_log_prob as jax_transition_log_prob,
    transition_log_std as jax_transition_log_std,
    transition_mean as jax_transition_mean,
)
from .rankq import (
    RankQActions,
    SameStateRankQActions,
    compute_same_state_rankq_loss,
    compute_rankq_loss,
    ddp_global_valid_mean_scale,
    disabled_same_state_rankq_metrics,
    disabled_rankq_metrics,
    make_same_state_rankq_actions,
    make_rankq_actions,
    same_state_rankq_settings,
)
from .pi05_pytorch_adapter import PI05FlowCondition, PI05PytorchFlowPolicy, load_pi05_pytorch_flow_policy
from .temporal_rectification import EmpiricalGradientRectifier, analytic_rectification
from .types import ChunkBatch
from .uncertainty import (
    actor_clip_for_uncertainty,
    kl_uncertainty_scale,
    state_adaptive_kl_penalty,
    state_entropy_weight,
    support_weight,
)
from .value_critic_protocol import StateFeatures


@dataclass
class OGPOTrainState:
    critic: Any
    target_critic: Any
    divl: DistributionalValueEnsemble | None
    target_divl: DistributionalValueEnsemble | None
    policy: OpenPIStochasticFlowPolicy
    old_policy: OpenPIStochasticFlowPolicy
    reference_policy: OpenPIStochasticFlowPolicy
    # Only OGPO+χ² owns this independent EMA snapshot. It never aliases the
    # current actor, PPO-old snapshot, or immutable base/SFT reference.
    slow_policy: OpenPIStochasticFlowPolicy | None
    critic_optimizer: torch.optim.Optimizer
    actor_optimizer: torch.optim.Optimizer
    support: torch.Tensor
    running_mad: RunningMAD
    rectifier: EmpiricalGradientRectifier
    conformal_scale: float = 1.0
    step: int = 0
    actor_step: int = 0
    accepted_actor_updates: int = 0
    critic_stage: str = "head_td"
    critic_stage_step: int = 0
    target_generator: torch.Generator | None = None


def _using_jax_actor(policy: OpenPIStochasticFlowPolicy) -> bool:
    return isinstance(policy, PI05JaxFlowPolicy)


def ogpo_variant(config: dict[str, Any]) -> str:
    """Return the OGPO actor variant.

    Existing configs omit this field and therefore preserve the historical
    U-DIVL sign-consensus OGPO+CA path exactly.
    """
    actor_cfg = config.get("actor", {})
    if "use_ca" in actor_cfg or "use_chipo" in actor_cfg:
        use_ca = bool(actor_cfg.get("use_ca", False))
        use_chipo = bool(actor_cfg.get("use_chipo", False))
        if use_ca and use_chipo:
            return "ca_chi2"
        if use_chipo:
            return "chi2"
        return "ca"
    raw = str(actor_cfg.get("ogpo_variant", "ca")).strip().lower()
    aliases = {
        "ca": "ca",
        "ogpo_ca": "ca",
        "ogpo+ca": "ca",
        "chi2": "chi2",
        "chi-square": "chi2",
        "chi_square": "chi2",
        "ogpo_chi2": "chi2",
        "ogpo+chi2": "chi2",
        "ca_chi2": "ca_chi2",
        "ca+chi2": "ca_chi2",
        "ca_chipo": "ca_chi2",
        "chipo_ca": "ca_chi2",
        "chi_po_ca": "ca_chi2",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            "actor.ogpo_variant must be one of ca, chi2, or ca_chi2, "
            f"got {raw!r}"
        ) from exc


def _chi2_config(config: dict[str, Any]) -> dict[str, Any]:
    actor_cfg = config.get("actor", {})
    value = actor_cfg.get("chi2")
    if value is None:
        value = actor_cfg.get("chi_po", {})
    if not isinstance(value, dict):
        raise ValueError("actor.chi2 must be a mapping")
    return value


def _divl_adaptive_tau_enabled(divl_cfg: dict[str, Any]) -> bool:
    """Resolve the LWD tau switch while retaining legacy DIVL configs.

    Older experiments used ``use_adaptive_quantile`` for the alpha-range
    heuristic and therefore cannot infer the new formula from that flag
    alone.  A config that explicitly supplies the LWD parameters (or either
    tau switch alias) is unambiguously requesting
    ``tau_base - entropy_coefficient * H``.
    """
    if "adaptive_tau" in divl_cfg:
        return bool(divl_cfg["adaptive_tau"])
    if "adaptive_tau_mode" in divl_cfg:
        return bool(divl_cfg["adaptive_tau_mode"])
    return "tau_base" in divl_cfg and "entropy_coefficient" in divl_cfg


def _double_q_v_tau_kwargs(
    critic_cfg: dict[str, Any],
    divl_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Resolve optional V-only tau controls while preserving legacy values."""
    mode = critic_cfg.get("v_tau_mode")
    if mode is None:
        return {
            "adaptive_tau_mode": _divl_adaptive_tau_enabled(divl_cfg),
            "tau_base": float(divl_cfg.get("tau_base", 0.6)),
            "entropy_coefficient": float(
                divl_cfg.get("entropy_coefficient", 0.3)
            ),
            "tau_min": float(divl_cfg.get("tau_min", 0.0)),
            "tau_max": float(divl_cfg.get("tau_max", 1.0)),
        }
    if str(mode).lower() != "adaptive":
        raise ValueError("critic.v_tau_mode currently supports only 'adaptive'")
    tau_min = float(critic_cfg.get("v_tau_min", 0.3))
    tau_max = float(critic_cfg.get("v_tau_max", 0.6))
    if not 0.0 <= tau_min <= tau_max <= 1.0:
        raise ValueError("critic V tau bounds must satisfy 0 <= min <= max <= 1")
    return {
        "adaptive_tau_mode": True,
        "tau_base": tau_max,
        "entropy_coefficient": tau_max - tau_min,
        "tau_min": tau_min,
        "tau_max": tau_max,
    }


def _ppo_chain_clip(actor_cfg: dict[str, Any]) -> float:
    """Read the chain clip with the concise ``ppo_clip`` alias supported."""
    return float(actor_cfg.get("ppo_clip_chain", actor_cfg.get("ppo_clip", 0.01)))


def chi2_slow_policy_ema(config: dict[str, Any]) -> float:
    """Convert official ``tau_slow`` (new-weight) to local EMA convention."""
    chi2_cfg = _chi2_config(config)
    if "slow_policy_tau" in chi2_cfg or "tau_slow" in chi2_cfg:
        tau = float(chi2_cfg.get("slow_policy_tau", chi2_cfg.get("tau_slow")))
        if not 0.0 < tau <= 1.0:
            raise ValueError("actor.chi2.slow_policy_tau must be in (0, 1]")
        return 1.0 - tau
    return float(chi2_cfg.get("slow_policy_ema", 0.9995))


def slow_policy_update_due(
    *,
    accepted_actor_updates: int,
    update_period: int,
    update_accepted: bool,
) -> bool:
    """Return whether an accepted actor transaction should advance pi_slow."""
    if int(update_period) <= 0:
        raise ValueError("slow policy update period must be positive")
    return bool(update_accepted) and int(accepted_actor_updates) > 0 and (
        int(accepted_actor_updates) % int(update_period) == 0
    )


def _validate_chi2_config(config: dict[str, Any]) -> None:
    if ogpo_variant(config) not in {"chi2", "ca_chi2"} and not bool(
        _chi2_config(config).get("enabled", False)
    ):
        return
    chi2_cfg = _chi2_config(config)
    beta_base = float(chi2_cfg.get("beta_base", chi2_cfg.get("beta", 0.1)))
    regularization_beta = float(chi2_cfg.get("regularization_beta", beta_base))
    slow_policy_ema = chi2_slow_policy_ema(config)
    slow_policy_update_period = int(chi2_cfg.get("slow_policy_update_period", 1))
    log_ratio_clip = float(chi2_cfg.get("log_ratio_clip", 20.0))
    ratio_max = float(chi2_cfg.get("ratio_max", 20.0))
    q_std_target = float(chi2_cfg.get("q_std_target", 1.0))
    ensemble_alpha = float(chi2_cfg.get("ensemble_alpha", 5.0))
    r_max = float(chi2_cfg.get("r_max", 10.0))
    if not math.isfinite(beta_base) or beta_base < 0.0:
        raise ValueError("actor.chi2.beta_base must be finite and non-negative")
    if not math.isfinite(regularization_beta) or regularization_beta < 0.0:
        raise ValueError("actor.chi2.regularization_beta must be finite and non-negative")
    if not 0.0 <= slow_policy_ema < 1.0:
        raise ValueError("actor.chi2.slow_policy_ema must be in [0, 1)")
    if slow_policy_update_period <= 0:
        raise ValueError("actor.chi2.slow_policy_update_period must be positive")
    if not math.isfinite(log_ratio_clip) or log_ratio_clip <= 0.0:
        raise ValueError("actor.chi2.log_ratio_clip must be finite and positive")
    if not math.isfinite(ratio_max) or ratio_max < 1.0:
        raise ValueError("actor.chi2.ratio_max must be finite and at least one")
    if not math.isfinite(q_std_target) or q_std_target <= 0.0:
        raise ValueError("actor.chi2.q_std_target must be finite and positive")
    if not math.isfinite(ensemble_alpha) or ensemble_alpha < 0.0:
        raise ValueError("actor.chi2.ensemble_alpha must be finite and non-negative")
    if not math.isfinite(r_max) or r_max <= 0.0:
        raise ValueError("actor.chi2.r_max must be finite and positive")
    scope = str(chi2_cfg.get("ratio_scope", "selected_transition"))
    if scope not in {"selected_transition", "full_chain"}:
        raise ValueError(
            "actor.chi2.ratio_scope must be selected_transition or full_chain"
        )
    if scope == "selected_transition" and bool(chi2_cfg.get("require_single_epoch", True)) and int(
        config.get("actor", {}).get("actor_epochs_per_rollout", 1)
    ) != 1:
        raise ValueError(
            "Flash-chiPO requires actor.actor_epochs_per_rollout=1; "
            "set actor.chi2.require_single_epoch=false only for an explicit ablation"
        )


def _effective_advantage_mode(config: dict[str, Any]) -> str:
    variant = ogpo_variant(config)
    if variant == "chi2":
        return "chi_po"
    if variant == "ca_chi2":
        return "conservative"
    return str(config.get("actor", {}).get("advantage_mode", "sign_consensus"))


def _single_epoch_on_policy_log_prob(
    new_log_prob: jax.Array,
    old_log_prob: jax.Array,
) -> jax.Array:
    """Use the exact on-policy value while retaining the current-policy gradient."""
    return old_log_prob + new_log_prob - jax.lax.stop_gradient(new_log_prob)


def _anchor_current_to_old_value(current: jax.Array, old: jax.Array) -> jax.Array:
    """Evaluate a one-step surrogate at the old value with the current Jacobian."""
    return old + current - jax.lax.stop_gradient(current)


def _conditionally_anchor_current_to_old_value(
    current: jax.Array,
    old: jax.Array,
    anchor_strength: jax.Array | float,
) -> jax.Array:
    """Anchor the value to old while preserving the current-policy Jacobian."""
    strength = jnp.asarray(anchor_strength, dtype=current.dtype)
    return current + strength * jax.lax.stop_gradient(old - current)


def _numpy_gaussian_kl_diag(
    mean_p: np.ndarray,
    log_std_p: np.ndarray,
    mean_q: np.ndarray,
    log_std_q: np.ndarray,
    *,
    event_dim: int | None = None,
) -> np.ndarray:
    """Host-side diagonal Gaussian KL used for the post-update trust-region gate."""
    mean_p = np.asarray(mean_p, dtype=np.float32)
    log_std_p = np.asarray(log_std_p, dtype=np.float32)
    mean_q = np.asarray(mean_q, dtype=np.float32)
    log_std_q = np.asarray(log_std_q, dtype=np.float32)
    if event_dim is not None:
        if event_dim <= 0 or event_dim > mean_p.shape[-1]:
            raise ValueError(
                f"event_dim must be in [1, {mean_p.shape[-1]}], got {event_dim}"
            )
        mean_p = mean_p[..., :event_dim]
        log_std_p = log_std_p[..., :event_dim]
        mean_q = mean_q[..., :event_dim]
        log_std_q = log_std_q[..., :event_dim]
    var_p = np.exp(2.0 * log_std_p)
    var_q = np.exp(2.0 * log_std_q)
    kl = log_std_q - log_std_p + (var_p + np.square(mean_p - mean_q)) / (2.0 * var_q) - 0.5
    return kl.reshape(kl.shape[0], -1).sum(axis=-1)


@partial(
    nnx.jit,
    static_argnames=("num_steps", "sde_mode", "group_size"),
)
def _sample_frozen_jax_flash_rollout(
    actor,
    observation,
    selected_step,
    rng,
    *,
    num_steps: int,
    sde_mode: str,
    group_size: int,
):
    rollout = sample_jax_flash_rollout(
        actor=actor,
        flow_spec=OpenPIJaxFlowSpec(num_steps),
        observation=observation,
        group_size=group_size,
        selected_step=selected_step,
        rng=rng,
        sde_mode=sde_mode,
    )
    return rollout.x_t, rollout.x_prev, rollout.timestep, rollout.endpoint


def _make_critic_optimizer(
    critic,
    divl,
    critic_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    named_parameters = list(critic.named_parameters())
    if divl is not None:
        named_parameters.extend((f"divl.{name}", parameter) for name, parameter in divl.named_parameters())
    trainable = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("critic stage has no trainable parameters")
    base_lr = float(critic_cfg.get("learning_rate", 3e-4))
    backbone_lr = float(critic_cfg.get("backbone", {}).get("learning_rate", base_lr))
    lora_parameters = [parameter for name, parameter in trainable if ".lora_" in name]
    backbone_parameters = [
        parameter
        for name, parameter in trainable
        if ".lora_" not in name
        and (
            name.startswith("state_encoder.gemma_model.")
            or name.startswith("state_encoder.vision_model.")
        )
    ]
    excluded = {id(parameter) for parameter in (*lora_parameters, *backbone_parameters)}
    regular_parameters = [parameter for _, parameter in trainable if id(parameter) not in excluded]
    groups = []
    if regular_parameters:
        groups.append({"params": regular_parameters, "lr": base_lr})
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_lr})
    if lora_parameters:
        lora_cfg = critic_cfg.get("gemma_lora", {})
        groups.append({"params": lora_parameters, "lr": float(lora_cfg.get("learning_rate", base_lr * 0.1))})
    for group in groups:
        group["initial_lr"] = group["lr"]
    optimizer_name = str(critic_cfg.get("optimizer", "adamw")).lower()
    optimizer_kwargs = {
        "lr": base_lr,
        "weight_decay": float(critic_cfg.get("weight_decay", 1e-4)),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(groups, **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, **optimizer_kwargs)
    raise ValueError(f"unsupported critic.optimizer={optimizer_name!r}")


def _apply_critic_lr_schedule(
    state: OGPOTrainState,
    config: dict[str, Any],
) -> float:
    critic_cfg = config.get("critic", {})
    schedule = str(critic_cfg.get("lr_schedule", "constant")).lower()
    if schedule == "constant":
        scale = 1.0
    elif schedule == "cosine":
        total_steps = int(
            critic_cfg.get(
                "lr_schedule_steps",
                config.get("training", {}).get("critic_steps", 1),
            )
        )
        if total_steps <= 0:
            raise ValueError("critic cosine lr_schedule requires positive total steps")
        progress = min(float(state.step) / max(total_steps - 1, 1), 1.0)
        min_ratio = float(critic_cfg.get("min_lr_ratio", 0.0))
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError("critic.min_lr_ratio must be in [0, 1]")
        scale = min_ratio + 0.5 * (1.0 - min_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
    else:
        raise ValueError(f"unsupported critic.lr_schedule={schedule!r}")
    for group in state.critic_optimizer.param_groups:
        initial_lr = float(group.get("initial_lr", group["lr"]))
        group["initial_lr"] = initial_lr
        group["lr"] = initial_lr * scale
    return scale


def build_train_state(
    config: dict[str, Any],
    batch: ChunkBatch,
    *,
    device: str | torch.device = "cpu",
    multimodal_critic_factory: Callable[[ChunkBatch, dict[str, Any]], Any] | None = None,
) -> OGPOTrainState:
    method_name = str(config.get("method", {}).get("name", "ogpo-divl"))
    critic_cfg = config.get("critic", {})
    divl_cfg = config.get("divl", {})
    actor_cfg = config.get("actor", {})
    flow_cfg = config.get("flow", {})
    _validate_chi2_config(config)
    selected_ogpo_variant = ogpo_variant(config)
    if bool(config.get("training", {}).get("clean_pytorch_main", False)):
        required = {
            "flow.adapter": flow_cfg.get("adapter") == "pi05_pytorch",
            "actor.flash_enabled": not bool(actor_cfg.get("flash_enabled", False)),
            "actor.temporal_rectification": not bool(
                actor_cfg.get("temporal_rectification", False)
            ),
            "actor.full_ratio_mode": str(actor_cfg.get("full_ratio_mode")) == "ais_joint",
            "actor.full_chain": bool(actor_cfg.get("full_chain", True)),
            "actor.residual_enabled": not bool(actor_cfg.get("residual_enabled", True)),
            "inference.dynamics": str(
                config.get("inference", {}).get("dynamics", "native_ode")
            ) == "native_ode",
            "flow.sde_mode": str(flow_cfg.get("sde_mode")) == "ogpo_constant_corrected",
            "flow.constant_noise_std": math.isclose(
                float(flow_cfg.get("constant_noise_std", 0.0)), 0.005, rel_tol=0.0, abs_tol=1e-12
            ),
            "flow.learn_sde_std": not bool(flow_cfg.get("learn_sde_std", True)),
            "flow.use_constant_noise": bool(flow_cfg.get("use_constant_noise", True)),
            "flow.error_correct_sde_to_ode": bool(
                flow_cfg.get("error_correct_sde_to_ode", True)
            ),
            "flow.temporal_rectification_mode": str(
                flow_cfg.get("temporal_rectification_mode", "none")
            ) == "none",
            "flow.temporal_rectification": not bool(
                flow_cfg.get("temporal_rectification", False)
            ),
            "actor.normalize_logprob_by_action_dim": bool(
                actor_cfg.get("normalize_logprob_by_action_dim", False)
            ),
            "actor.normalize_logprob_by_denoising_steps": bool(
                actor_cfg.get("normalize_logprob_by_denoising_steps", False)
            ),
            "actor.advantage_mode": (
                (
                    selected_ogpo_variant in {"ca", "ca_chi2"}
                    and str(actor_cfg.get("advantage_mode")) == "conservative"
                )
                or (
                    selected_ogpo_variant == "chi2"
                    and str(actor_cfg.get("advantage_mode")) == "chi_po"
                )
            ),
            "actor.ogpo_variant": selected_ogpo_variant in {"ca", "chi2", "ca_chi2"},
            "actor.chi2.ratio_scope": (
                selected_ogpo_variant == "ca"
                or str(actor_cfg.get("chi2", {}).get("ratio_scope")) == "full_chain"
            ),
            "actor.legacy_extra_chi2_loss": not bool(
                actor_cfg.get(
                    "legacy_extra_chi2_loss",
                    actor_cfg.get("chi2", {}).get("legacy_extra_chi2_loss", False),
                )
            ),
            "offline.critic_update_during_actor": not bool(
                config.get("offline", {}).get("critic_update_during_actor", False)
            ),
        }
        invalid = [name for name, valid in required.items() if not valid]
        if invalid:
            raise ValueError(
                "clean PyTorch main-path invariants violated: " + ", ".join(invalid)
            )
    architecture = str(critic_cfg.get("architecture", "mlp"))
    if method_name == "ogpo-origin":
        required = {
            "critic.architecture": architecture == "gemma_siglip_scalar_q",
            "divl.enabled": not bool(divl_cfg.get("enabled", True)),
            "actor.advantage_mode": str(actor_cfg.get("advantage_mode")) == "group_mean",
            "flow.sde_mode": str(flow_cfg.get("sde_mode")) == "ogpo_corrected",
        }
        invalid = [name for name, valid in required.items() if not valid]
        if invalid:
            raise ValueError(
                "ogpo-origin configuration violates original-path invariants: "
                + ", ".join(invalid)
            )
    if architecture not in {"mlp", "gemma_siglip_multihead", "gemma_siglip_scalar_q"}:
        raise ValueError(f"unsupported critic.architecture={architecture!r}")
    if architecture == "mlp":
        feature_source = str(critic_cfg.get("feature_source", "replay_state"))
        if feature_source != "replay_state":
            raise ValueError("critic.feature_source currently supports only 'replay_state'")
        if bool(critic_cfg.get("shared_frozen_encoder", False)):
            raise ValueError(
                "critic.shared_frozen_encoder requires an unavailable PI0.5 feature_source; "
                "use feature_source: replay_state and shared_frozen_encoder: false"
            )
        if int(critic_cfg.get("train_last_n_backbone_layers", 0)) != 0:
            raise ValueError("replay-state critic has no backbone layers to unfreeze")
        if not bool(critic_cfg.get("detach_policy_features", True)):
            raise ValueError("replay-state critic requires critic.detach_policy_features=true")
    flow_adapter = str(flow_cfg.get("adapter", "gaussian_openpi"))
    if flow_adapter not in {"gaussian_openpi", "gaussian", "pi05_pytorch", "pi05_jax"}:
        raise ValueError(
            f"unsupported flow.adapter={flow_adapter!r}; expected gaussian_openpi, pi05_pytorch, or pi05_jax"
        )
    if selected_ogpo_variant in {"chi2", "ca_chi2"} and flow_adapter == "pi05_jax":
        raise ValueError(
            "full-chain ChiPO variants currently support the production "
            "PyTorch/Gaussian flow adapters only; the JAX multi-device path "
            "does not yet persist an independent Flash-chiPO slow-policy state"
        )

    ensemble_size = int(critic_cfg.get("ensemble_size", 3))
    hidden_dim = int(critic_cfg.get("hidden_dim", 128))
    num_layers = int(critic_cfg.get("num_layers", 2))
    if bool(critic_cfg.get("double_q_divl", False)):
        rankq_value = critic_cfg.get("rankq", False)
        nested_rankq_mapping = isinstance(rankq_value, dict)
        q_representation = str(critic_cfg.get("q_representation", "scalar"))
        required = {
            "critic.q_representation": q_representation in {"scalar", "categorical"},
            "critic.categorical_q_loss": (
                q_representation != "categorical"
                or str(critic_cfg.get("categorical_q_loss", "")).lower()
                == "one_hot_ce"
            ),
            "critic.q_heads_per_member": int(critic_cfg.get("q_heads_per_member", 1)) == 2,
            "critic.architecture": architecture == "gemma_siglip_multihead",
            "critic.enable_rankq": not bool(critic_cfg.get("enable_rankq", False)),
            # Historical configs used ``rankq: false`` as a boolean guard.
            # The new same-state auxiliary is an explicit nested mapping and
            # is supported only by this raw-double-Q path.
            "critic.rankq": nested_rankq_mapping or not bool(rankq_value),
            "critic.rank_consensus_enabled": not bool(critic_cfg.get("rank_consensus_enabled", False)),
        }
        invalid = [name for name, valid in required.items() if not valid]
        if invalid:
            raise ValueError(
                "double_q_divl main-path invariants violated: " + ", ".join(invalid)
            )
    if architecture == "gemma_siglip_multihead":
        critic_factory = multimodal_critic_factory or build_gemma_siglip_critic
        critic = critic_factory(batch, config)
        if torch.device(device).type == "cuda":
            parameter_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in critic.parameters()
            )
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
            print(
                "[ogpo] critic_cuda_load "
                f"parameter_gib={parameter_bytes / 2**30:.3f} "
                f"free_gib={free_bytes / 2**30:.3f} "
                f"total_gib={total_bytes / 2**30:.3f}",
                flush=True,
            )
        critic = critic.to(device)
        if critic.ensemble_size != ensemble_size:
            raise ValueError(
                "multimodal critic factory pair count does not match "
                f"critic.ensemble_size: expected {ensemble_size}, got {critic.ensemble_size}"
            )
        expected_pairs = critic_cfg.get("expected_num_pairs")
        expected_raw_q = critic_cfg.get("expected_num_raw_q_heads")
        if expected_pairs is not None and critic.core.num_pairs != int(expected_pairs):
            raise ValueError(
                f"Expected a {int(expected_pairs)}-pair critic, got {critic.core.num_pairs} pairs"
            )
        if expected_raw_q is not None and critic.core.num_q_heads != int(expected_raw_q):
            raise ValueError(
                f"Expected {int(expected_raw_q)} raw Q heads, got {critic.core.num_q_heads}"
            )
        actor_expected_raw_q = actor_cfg.get("q_ensemble_size")
        if actor_expected_raw_q is not None and critic.core.num_q_heads != int(actor_expected_raw_q):
            raise ValueError(
                "OGPO actor Q ensemble mismatch: expected "
                f"{int(actor_expected_raw_q)} raw Q heads, got {critic.core.num_q_heads}"
            )
        critic_stage = str(critic_cfg.get("stage", "head_td"))
        lora_cfg = critic_cfg.get("gemma_lora", {})
        if int(lora_cfg.get("final_n_layers", 0)) > 0 and hasattr(critic.state_encoder, "gemma_model"):
            install_final_gemma_lora(
                critic.state_encoder,
                final_n_layers=int(lora_cfg["final_n_layers"]),
                rank=int(lora_cfg.get("rank", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                target_suffixes=tuple(lora_cfg.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
            )
        if "member_initialization_seed" in critic_cfg:
            seed = int(critic_cfg["member_initialization_seed"])
            devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                for member in range(critic.core.num_pairs):
                    torch.manual_seed(seed + 1009 * member)
                    modules = list(critic.core.q_heads[2 * member:2 * member + 2]) + [critic.core.value_heads[member]]
                    for head in modules:
                        for layer in head.modules():
                            if isinstance(layer, torch.nn.Linear):
                                layer.reset_parameters()
        configure_critic_stage(critic, critic_stage)
        target_critic = clone_target(critic).to(device)
        divl = None
        target_divl = None
    elif architecture == "gemma_siglip_scalar_q":
        if bool(divl_cfg.get("enabled", True)):
            raise ValueError("gemma_siglip_scalar_q requires divl.enabled=false")
        critic_factory = multimodal_critic_factory or build_gemma_siglip_scalar_q_critic
        critic = critic_factory(batch, config)
        if not isinstance(critic, MultiHeadScalarQCritic):
            raise TypeError("gemma_siglip_scalar_q factory must return MultiHeadScalarQCritic")
        if critic.ensemble_size != ensemble_size:
            raise ValueError(
                "gemma_siglip_scalar_q factory ensemble size does not match critic.ensemble_size"
            )
        critic = critic.to(device)
        critic_stage = str(critic_cfg.get("stage", "head_td"))
        target_critic = clone_target(critic).to(device)
        divl = None
        target_divl = None
    else:
        critic_stage = "legacy"
        critic = ScalarQEnsemble(
            ensemble_size,
            batch.obs_dim,
            batch.generated_horizon,
            batch.action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            randomized_prior_scale=float(critic_cfg.get("randomized_prior_scale", 0.0)),
        ).to(device)
        target_critic = clone_target(critic).to(device)
        divl = DistributionalValueEnsemble(
            ensemble_size,
            batch.obs_dim,
            hidden_dim,
            num_layers,
            int(divl_cfg.get("num_atoms", 51)),
        ).to(device)
        target_divl = clone_target(divl).to(device)
    num_atoms = int(divl_cfg.get("num_atoms", 51))
    if bool(divl_cfg.get("auto_support", False)):
        support_targets = batch.mc_returns if batch.mc_returns is not None else batch.chunk_returns
        support = make_support_from_targets(
            support_targets.to(device),
            num_atoms=num_atoms,
            margin_fraction=float(divl_cfg.get("support_margin_fraction", 0.05)),
        )
    else:
        support = make_support(
            float(divl_cfg.get("v_min", -5.0)),
            float(divl_cfg.get("v_max", 5.0)),
            num_atoms,
            device=device,
        )
    load_frozen_pytorch_role: Callable[[str], PI05PytorchFlowPolicy] | None = None
    if flow_adapter in {"gaussian_openpi", "gaussian"}:
        action_flat_dim = batch.generated_horizon * batch.action_dim
        policy = GaussianFlowPolicy(
            condition_dim=batch.obs_dim,
            action_dim=action_flat_dim,
            hidden_dim=int(actor_cfg.get("hidden_dim", 128)),
            num_steps=int(flow_cfg.get("num_steps", 8)),
            stochastic_variance=float(flow_cfg.get("stochastic_variance", 0.04)),
            sde_mode=str(flow_cfg.get("sde_mode", "gaussian_adapter")),
            constant_noise_std=float(flow_cfg.get("constant_noise_std", 0.005)),
            learn_sde_std=bool(flow_cfg.get("learn_sde_std", True)),
            randn_clip_value=float(flow_cfg.get("randn_clip_value", 3.0)),
        ).to(device)
        old_policy = clone_target(policy).to(device)
        reference_policy = clone_target(policy).to(device)
    else:
        if flow_adapter == "pi05_jax":
            _load_flow_policy = load_pi05_jax_flow_policy
        else:
            _load_flow_policy = load_pi05_pytorch_flow_policy
        flow_policy_kwargs = dict(
            checkpoint_dir=str(flow_cfg["checkpoint_dir"]),
            train_config_name=str(flow_cfg["train_config"]),
            image_mapping=dict(
                flow_cfg.get("image_mapping_override", flow_cfg.get("image_mapping", {}))
            ),
            image_container_key=flow_cfg.get("image_container_key"),
            transpose_images_to_chw=bool(flow_cfg.get("transpose_images_to_chw", False)),
            environment_action_dim=batch.action_dim,
            num_steps=int(flow_cfg.get("num_steps", 10)),
            stochastic_variance=float(flow_cfg.get("stochastic_variance", 0.04)),
            sde_mode=str(flow_cfg.get("sde_mode", "gaussian_adapter")),
            residual_hidden_dim=int(actor_cfg.get("hidden_dim", 128)),
            device=device,
        )
        if flow_adapter == "pi05_jax":
            flow_policy_kwargs["flow_action_dim"] = flow_cfg.get("latent_action_dim")
            flow_policy_kwargs["inference_dynamics"] = flow_cfg.get(
                "inference_dynamics",
                "training_sde_drift",
            )
        elif flow_adapter == "pi05_pytorch":
            role_devices = flow_cfg.get("pytorch_role_devices", {})
            if role_devices:
                required_roles = {"current", "old", "slow", "reference"}
                missing_roles = required_roles.difference(role_devices)
                if missing_roles:
                    raise ValueError(
                        "flow.pytorch_role_devices is missing roles: "
                        f"{sorted(missing_roles)}"
                    )
                resolved_role_devices = {
                    role: torch.device(str(role_devices[role]))
                    for role in required_roles
                }
                if any(value.type != "cuda" for value in resolved_role_devices.values()):
                    raise ValueError("PyTorch policy-role parallelism requires CUDA devices")
                device_indices = {
                    value.index if value.index is not None else 0
                    for value in resolved_role_devices.values()
                }
                if len(device_indices) != len(required_roles):
                    raise ValueError("current/old/slow/reference must use distinct CUDA devices")
                if max(device_indices) >= torch.cuda.device_count():
                    raise ValueError(
                        "flow.pytorch_role_devices requests unavailable CUDA device; "
                        f"visible count={torch.cuda.device_count()}"
                    )
                flow_policy_kwargs["device"] = resolved_role_devices["current"]
            flow_policy_kwargs["backend_train_mode"] = str(
                flow_cfg.get("backend_train_mode", "none")
            )
            flow_policy_kwargs["constant_noise_std"] = float(
                flow_cfg.get("constant_noise_std", 0.005)
            )
            flow_policy_kwargs["learn_sde_std"] = bool(
                flow_cfg.get("learn_sde_std", True)
            )
            flow_policy_kwargs["randn_clip_value"] = float(
                flow_cfg.get("randn_clip_value", 3.0)
            )
            flow_policy_kwargs["residual_enabled"] = bool(
                actor_cfg.get("residual_enabled", True)
            )
        policy = _load_flow_policy(**flow_policy_kwargs)
        if bool(config.get("training", {}).get("clean_pytorch_main", False)):
            if isinstance(policy, PI05PytorchFlowPolicy):
                if bool(getattr(policy, "residual_enabled", True)):
                    raise ValueError(
                        "clean PyTorch main path requires residual_enabled=false"
                    )
                if bool(getattr(policy, "learn_sde_std", True)):
                    raise ValueError(
                        "clean PyTorch main path requires learn_sde_std=false"
                    )
        if policy.model_horizon != batch.generated_horizon:
            raise ValueError(
                f"PI0.5 action horizon {policy.model_horizon} does not match replay horizon {batch.generated_horizon}"
            )
        if isinstance(policy, PI05PytorchFlowPolicy) and role_devices:
            def _load_frozen_role(role: str) -> PI05PytorchFlowPolicy:
                frozen_kwargs = dict(flow_policy_kwargs)
                frozen_kwargs["device"] = resolved_role_devices[role]
                frozen_kwargs["backend_train_mode"] = "none"
                frozen_policy = _load_flow_policy(**frozen_kwargs)
                assert isinstance(frozen_policy, PI05PytorchFlowPolicy)
                frozen_policy._backend_trainable_names = policy._backend_trainable_names
                frozen_policy.requires_grad_(False)
                return frozen_policy

            load_frozen_pytorch_role = _load_frozen_role
            # Place each frozen policy as a real module on its dedicated role
            # device, avoiding three full functional snapshots on current's GPU.
            old_policy = load_frozen_pytorch_role("old")
            reference_policy = load_frozen_pytorch_role("reference")
            print(
                "[ogpo] pytorch_policy_roles "
                + " ".join(
                    f"{role}={resolved_role_devices[role]}"
                    for role in ("current", "old", "slow", "reference")
                ),
                flush=True,
            )
        else:
            # Legacy single-device PyTorch uses functional snapshots so the
            # backend module itself remains shared without aliasing state.
            old_policy = policy.clone_adapter()
            reference_policy = policy.clone_adapter()
    if selected_ogpo_variant in {"chi2", "ca_chi2"}:
        if isinstance(policy, PI05PytorchFlowPolicy):
            slow_policy = (
                load_frozen_pytorch_role("slow")
                if load_frozen_pytorch_role is not None
                else policy.clone_adapter()
            )
        elif isinstance(policy, PI05JaxFlowPolicy):  # guarded above; keeps type narrowing explicit.
            raise AssertionError("JAX OGPO+χ² must be rejected before slow-policy construction")
        else:
            slow_policy = clone_target(policy).to(device)
    else:
        slow_policy = None
    actor_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if isinstance(policy, PI05JaxFlowPolicy):
        policy.init_actor_optimizer(
            learning_rate=float(actor_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(actor_cfg.get("weight_decay", 0.0)),
            optimizer=str(actor_cfg.get("optimizer", "adafactor")),
            max_grad_norm=float(actor_cfg.get("max_grad_norm", 1.0)),
            preserve_state_for_rollback=bool(actor_cfg.get("reject_update_on_kl", False)),
        )
    actor_optimizer_name = str(actor_cfg.get("optimizer", "adamw")).lower()
    if actor_parameters and actor_optimizer_name == "adafactor":
        actor_optimizer = torch.optim.Adafactor(
            actor_parameters,
            lr=float(actor_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(actor_cfg.get("weight_decay", 0.0)),
        )
    elif actor_parameters and actor_optimizer_name == "adamw":
        actor_optimizer = torch.optim.AdamW(
            actor_parameters,
            lr=float(actor_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(actor_cfg.get("weight_decay", 0.0)),
        )
    elif actor_parameters:
        raise ValueError(f"unsupported PyTorch actor optimizer: {actor_optimizer_name!r}")
    else:
        actor_optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-4)
    return OGPOTrainState(
        critic=critic,
        target_critic=target_critic,
        divl=divl,
        target_divl=target_divl,
        policy=policy,
        old_policy=old_policy,
        reference_policy=reference_policy,
        slow_policy=slow_policy,
        critic_optimizer=_make_critic_optimizer(critic, divl, critic_cfg),
        actor_optimizer=actor_optimizer,
        support=support,
        running_mad=RunningMAD(),
        rectifier=EmpiricalGradientRectifier(
            int(flow_cfg.get("num_steps", 8)),
            momentum=float(flow_cfg.get("rectification_momentum", 0.95)),
            clip_min=float(flow_cfg.get("rectification_clip_min", 0.25)),
            clip_max=float(flow_cfg.get("rectification_clip_max", 4.0)),
        ),
        conformal_scale=float(config.get("uncertainty", {}).get("conformal_scale", 1.0)),
        critic_stage=critic_stage,
        target_generator=torch.Generator(device=torch.device(device).type).manual_seed(
            int(config.get("training", {}).get("seed", 0)) + 1701
        ),
    )


def maybe_advance_critic_stage(
    state: OGPOTrainState,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> bool:
    """Advance staged critic training only after fixed-validation gates pass."""
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        return False
    gates = config.get("critic", {}).get("stage_gates", {})
    required = {
        "validation_pairwise_ranking_accuracy": ("min_pairwise_ranking_accuracy", lambda x, y: x >= y),
        "validation_interval_coverage": ("min_interval_coverage", lambda x, y: x >= y),
        "validation_categorical_saturation": ("max_categorical_saturation", lambda x, y: x <= y),
        "validation_q_exploitation_gap": ("max_abs_exploitation_gap", lambda x, y: abs(x) <= y),
    }
    if state.critic_stage_step < int(gates.get("min_stage_steps", 0)):
        return False
    for metric_name, (gate_name, predicate) in required.items():
        if gate_name not in gates:
            continue
        if metric_name not in metrics or not predicate(float(metrics[metric_name]), float(gates[gate_name])):
            return False
    if not gates or not any(gate_name in gates for gate_name, _ in required.values()):
        return False
    if state.critic_stage == "head_mc":
        next_stage = "head_td"
    elif state.critic_stage == "head_td" and int(
        config.get("critic", {}).get("gemma_lora", {}).get("final_n_layers", 0)
    ) > 0:
        next_stage = "gemma_lora_td"
    else:
        return False
    configure_critic_stage(state.critic, next_stage)
    state.critic_optimizer = _make_critic_optimizer(state.critic, state.divl, config.get("critic", {}))
    state.critic_stage = next_stage
    state.critic_stage_step = 0
    return True


def apply_scheduled_critic_stage(
    state: OGPOTrainState,
    config: dict[str, Any],
) -> bool:
    """Apply a deterministic warmup schedule keyed by optimizer steps."""
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        return False
    schedule = config.get("critic", {}).get("stage_schedule", [])
    if not schedule:
        return False
    elapsed = 0
    desired_stage = str(schedule[-1]["stage"])
    for index, entry in enumerate(schedule):
        desired_stage = str(entry["stage"])
        steps = entry.get("steps")
        if steps is None:
            if index != len(schedule) - 1:
                raise ValueError("only the final critic.stage_schedule entry may omit steps")
            break
        steps = int(steps)
        if steps <= 0:
            raise ValueError("critic.stage_schedule steps must be positive")
        elapsed += steps
        if state.step < elapsed:
            break
    if desired_stage == state.critic_stage:
        return False
    configure_critic_stage(state.critic, desired_stage)
    state.critic_optimizer = _make_critic_optimizer(
        state.critic,
        state.divl,
        config.get("critic", {}),
    )
    state.critic_stage = desired_stage
    state.critic_stage_step = 0
    return True


def _policy_condition(
    policy: OpenPIStochasticFlowPolicy,
    batch: ChunkBatch,
    *,
    next_observation: bool = False,
):
    return policy.condition_from_batch(batch, next_observation=next_observation)


def _jax_actor_step(policy: PI05JaxFlowPolicy, loss_fn: Callable[[Any], jax.Array]) -> float:
    if policy.actor_tx is None or policy.actor_opt_state is None:
        raise RuntimeError("JAX PI0.5 actor optimizer is not initialized")

    def _loss_from_state(actor_state):
        actor = nnx.merge(policy.actor_graphdef, actor_state)
        return loss_fn(actor)

    loss_value, grads = jax.value_and_grad(_loss_from_state)(policy.actor_state)
    policy.apply_actor_gradients(grads)
    return float(loss_value)


def _torch_condition_to_jax(condition):
    if isinstance(condition, PI05FlowCondition):
        return policy_observation_to_jax(condition.observation)
    return condition


def policy_observation_to_jax(observation):
    from .pi05_jax_adapter import _observation_to_jax  # noqa: PLC0415

    return _observation_to_jax(observation)


def _jax_flow_matching_inputs(
    policy: PI05JaxFlowPolicy,
    batch: ChunkBatch,
    *,
    seed: int,
) -> dict[str, Any]:
    device = next(policy.parameters()).device
    source = batch.to(device)
    condition = _policy_condition(policy, source)
    observation = policy_observation_to_jax(condition.observation)
    endpoint = policy.action_chunks_to_flow(source).reshape(source.batch_size, -1)
    actions = jnp.asarray(endpoint.detach().cpu().numpy())
    noise_key, time_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    noise = jax.random.normal(noise_key, actions.shape, dtype=actions.dtype)
    time = jax.random.beta(time_key, 1.5, 1.0, (source.batch_size,)) * 0.999 + 0.001
    return {
        "observation": observation,
        "action_endpoint": actions,
        "noise": noise,
        "timestep": time,
    }


def _prepare_jax_regularization(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None,
    success_batch: ChunkBatch | None,
    enable_success: bool = True,
    seed_step: int | None = None,
) -> dict[str, Any]:
    assert isinstance(state.policy, PI05JaxFlowPolicy)
    regularization_cfg = config.get("regularization", {})
    lambda_smooth = float(regularization_cfg.get("lambda_smooth", 0.0))
    if lambda_smooth != 0.0:
        raise ValueError(
            "JAX PI0.5 full finetuning does not yet support differentiable raw-action smoothness; "
            "set regularization.lambda_smooth=0"
        )
    result: dict[str, Any] = {
        "lambda_fm": float(regularization_cfg.get("lambda_fm", 0.1)),
        "lambda_success": float(regularization_cfg.get("lambda_success", 0.0)),
        "fm": None,
        "success": None,
    }
    stochastic_step = int(state.step if seed_step is None else seed_step)
    if result["lambda_fm"] != 0.0:
        result["fm"] = _jax_flow_matching_inputs(
            state.policy,
            fm_batch if fm_batch is not None else batch,
            seed=stochastic_step + 4101,
        )
    if enable_success and result["lambda_success"] != 0.0:
        success_source = success_batch if success_batch is not None else _success_subset(batch)
        if success_source is not None:
            result["success"] = _jax_flow_matching_inputs(
                state.policy,
                success_source,
                seed=stochastic_step + 5101,
            )
    return result


def _jax_regularization_loss(actor: Any, inputs: dict[str, Any]) -> tuple[jax.Array, jax.Array, jax.Array]:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    fm_loss = zero
    if inputs["fm"] is not None:
        fm_loss = jax_flow_matching_loss(actor=actor, **inputs["fm"])
    success_loss = zero
    if inputs["success"] is not None:
        success_loss = jax_flow_matching_loss(actor=actor, **inputs["success"])
    total = float(inputs["lambda_fm"]) * fm_loss + float(inputs["lambda_success"]) * success_loss
    return total, fm_loss, success_loss


def _accumulate_jax_grads_on_host(existing: Any, grads: Any, *, weight: float) -> Any:
    host_grads = jax.device_get(grads)

    def _scaled_copy(value):
        array = np.asarray(value)
        scale = np.asarray(weight, dtype=array.dtype)
        return np.array(array * scale, copy=True)

    if existing is None:
        return jax.tree_util.tree_map(_scaled_copy, host_grads)

    def _add(accumulator, value):
        array = np.asarray(value, dtype=accumulator.dtype)
        scale = np.asarray(weight, dtype=accumulator.dtype)
        np.add(accumulator, array * scale, out=accumulator)
        return accumulator

    return jax.tree_util.tree_map(_add, existing, host_grads)


def _jax_tree_l2_norm(tree: Any) -> float:
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(jax.device_get(tree)):
        flat = np.asarray(leaf).reshape(-1)
        # Some scanned PI0.5 parameter leaves exceed 4 GiB. Squaring a whole
        # leaf materializes another equally large host array, so accumulate
        # the norm in bounded chunks instead.
        for start in range(0, flat.size, 16 * 1024 * 1024):
            chunk = flat[start : start + 16 * 1024 * 1024]
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            total += float(np.dot(chunk, chunk))
    return total**0.5


def _slice_jax_batch(tree: Any, start: int, stop: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: value[start:stop] if value is not None else None,
        tree,
    )


def _shard_jax_batch(tree: Any, num_devices: int) -> Any:
    """Reshape a leading candidate batch into [device, local_batch, ...]."""
    num_devices = int(num_devices)
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")

    def _shard(value):
        if value is None:
            return None
        if value.shape[0] % num_devices:
            raise ValueError(
                f"candidate batch {value.shape[0]} is not divisible by {num_devices} devices"
            )
        return value.reshape(
            num_devices,
            value.shape[0] // num_devices,
            *value.shape[1:],
        )

    return jax.tree_util.tree_map(_shard, tree)


def _first_pmap_replica(tree: Any) -> Any:
    """Keep one copy of a pmean-replicated pytree on the first local device."""
    return jax.tree_util.tree_map(
        lambda value: _pmap_replica_shard(value, 0).reshape(value.shape[1:]),
        tree,
    )


def _pmap_replica_shard(value: jax.Array, replica: int) -> jax.Array:
    """Return one pmap shard without dispatching a cross-device slice op."""
    replica = int(replica)
    if len(value.addressable_shards) == 1:
        local = value.addressable_shards[0].data
        if local.shape[0] == value.shape[0]:
            # Unit tests and single-device fallback arrays are not sharded.
            # This slice stays on the array's only committed device.
            return local[replica : replica + 1]
    for shard in value.addressable_shards:
        leading_index = shard.index[0]
        start = (
            int(leading_index)
            if isinstance(leading_index, int)
            else int(leading_index.start or 0)
        )
        if start == replica:
            return shard.data
    raise ValueError(
        f"pmap output has no addressable replica {replica}; "
        f"available shards={[shard.index for shard in value.addressable_shards]}"
    )


def _jax_tree_to_device(tree: Any, device: jax.Device) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(
            value,
            device,
            donate=False,
            may_alias=True,
        ),
        tree,
    )


def _jax_tree_copy_to_device(tree: Any, device: jax.Device) -> Any:
    """Copy a pytree so its source buffers can be released or donated."""
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(
            value,
            device,
            donate=False,
            may_alias=False,
        ),
        tree,
    )


def _mean_pmap_gradients_on_host(tree: Any) -> Any:
    """Average data-parallel gradients without a multi-GiB NCCL workspace."""
    def _mean_leaf(value):
        replicas = int(value.shape[0])
        first = np.asarray(jax.device_get(value[0]))
        # Preserve the gradient dtype. PI0.5's largest scanned bf16 leaf is
        # about 255 MiB; promoting every accumulated leaf to fp32 doubles the
        # persistent host gradient tree and can exceed the process VM limit.
        # This matches the established single-GPU host accumulation semantics.
        accumulator = np.array(first, copy=True)
        for replica in range(1, replicas):
            current = np.asarray(jax.device_get(value[replica]), dtype=accumulator.dtype)
            np.add(accumulator, current, out=accumulator)
        accumulator /= float(replicas)
        return accumulator

    return jax.tree_util.tree_map(_mean_leaf, tree)


@partial(jax.jit, donate_argnums=(0,), static_argnames=("other_weight",))
def _jax_tree_add_scaled(tree: Any, other: Any, *, other_weight: float) -> Any:
    return jax.tree_util.tree_map(
        lambda left, right: left + float(other_weight) * right,
        tree,
        other,
    )


@partial(jax.jit, donate_argnums=(0,), static_argnames=("scale",))
def _jax_tree_scale(tree: Any, *, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: value * float(scale), tree)


def _mean_pmap_gradients_on_device(tree: Any, device: jax.Device) -> Any:
    """Move local gradient shards to one device and average without NCCL."""
    replicas = int(jax.tree_util.tree_leaves(tree)[0].shape[0])
    accumulator = jax.tree_util.tree_map(
        lambda value: jax.device_put(
            jnp.zeros(value.shape[1:], dtype=value.dtype),
            device,
        ),
        tree,
    )
    for replica in range(replicas):
        current = jax.tree_util.tree_map(
            lambda value: jax.device_put(
                _pmap_replica_shard(value, replica),
                device,
                donate=False,
                may_alias=False,
            ).reshape(value.shape[1:]),
            tree,
        )
        accumulator = _jax_tree_add_scaled(
            accumulator,
            current,
            other_weight=1.0,
        )
        jax.effects_barrier()
        del current
        gc.collect()
    return _jax_tree_scale(accumulator, scale=1.0 / replicas)


@jax.jit
def _jax_tree_l2_norm_device(tree: Any) -> jax.Array:
    squared_norms = [
        jnp.sum(jnp.square(value.astype(jnp.float32)))
        for value in jax.tree_util.tree_leaves(tree)
    ]
    return jnp.sqrt(jnp.sum(jnp.stack(squared_norms)))


def _stack_candidate_rollouts(parts: list[jax.Array]) -> jax.Array:
    """Stack candidate-major [G, B, ...] parts into state-major [B * G, ...]."""
    stacked = jnp.stack(parts, axis=1)
    return stacked.reshape((stacked.shape[0] * stacked.shape[1], *stacked.shape[2:]))


def _critic_execution_mask(batch: ChunkBatch, config: dict[str, Any]) -> torch.Tensor:
    if bool(config.get("data", {}).get("use_execution_mask", True)):
        return batch.execution_masks
    return torch.ones_like(batch.execution_masks, dtype=torch.bool)


@torch.no_grad()
def _reference_value_mean(
    state: OGPOTrainState,
    batch: ChunkBatch,
    *,
    num_samples: int,
    aggregation_mode: str = "ensemble_mean",
) -> torch.Tensor:
    condition = _policy_condition(state.reference_policy, batch, next_observation=True)
    rollout = state.reference_policy.rollout(condition, group_size=num_samples)
    condition_g = state.reference_policy.repeat_condition(condition, num_samples)
    environment_endpoints = state.reference_policy.flat_actions_to_environment(
        rollout.endpoint,
        condition_g,
    )
    endpoints = environment_endpoints.reshape(batch.batch_size, num_samples, -1)
    chunks = endpoints.reshape(batch.batch_size, num_samples, batch.generated_horizon, batch.action_dim)
    if isinstance(state.target_critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
        features = state.target_critic.encode_state(batch, next_observation=True)
        repeated_features = StateFeatures(
            readout=features.readout.repeat_interleave(num_samples, dim=0)
        )
        flat_chunks = chunks.reshape(batch.batch_size * num_samples, batch.generated_horizon, batch.action_dim)
        flat_masks = batch.execution_masks[:, None, :].expand(
            batch.batch_size, num_samples, batch.generated_horizon
        ).reshape(batch.batch_size * num_samples, batch.generated_horizon)
        q_ref = state.target_critic.q_from_features(repeated_features, flat_chunks, flat_masks)
        q_aggregated, _ = aggregate_value_heads(
            q_ref,
            aggregation_mode,
            generator=state.target_generator,
        )
        return q_aggregated.reshape(batch.batch_size, num_samples).mean(dim=1)
    flat_obs = batch.next_observations[:, None, :].expand(
        batch.batch_size, num_samples, batch.next_observations.shape[-1]
    ).reshape(batch.batch_size * num_samples, -1)
    flat_chunks = chunks.reshape(batch.batch_size * num_samples, batch.generated_horizon, batch.action_dim)
    flat_masks = batch.execution_masks[:, None, :].expand(
        batch.batch_size, num_samples, batch.generated_horizon
    ).reshape(batch.batch_size * num_samples, batch.generated_horizon)
    q_ref = state.target_critic(flat_obs, flat_chunks, flat_masks)
    return q_ref.reshape(state.critic.ensemble_size, batch.batch_size, num_samples).mean(dim=(0, 2))


def _multimodal_scalar_q_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
) -> dict[str, float]:
    """Scalar-Q Origin update with either policy TD or pure MC targets."""
    critic_cfg = config.get("critic", {})
    batch = batch.to(next(state.critic.parameters()).device)
    if zero_grad:
        state.critic_optimizer.zero_grad(set_to_none=True)
    critic_mask = _critic_execution_mask(batch, config)
    features = state.critic.encode_state(batch)
    q_pred = state.critic.q_from_features(features, batch.action_chunks, critic_mask)
    with torch.no_grad():
        target_mode = str(critic_cfg.get("target_mode", "td")).lower()
        if target_mode == "mc_return":
            if batch.mc_returns is None:
                raise ValueError("critic.target_mode=mc_return requires replay mc_returns")
            target = batch.mc_returns
            next_action_samples = 0
            next_q = None
            bootstrap_active = 0.0
            lambda_mc = 1.0
        elif target_mode == "td":
            next_action_samples = int(critic_cfg.get("reference_value_samples", 1))
            if next_action_samples <= 0:
                raise ValueError("TD Origin requires critic.reference_value_samples >= 1")
            bootstrap_mode = str(critic_cfg.get("bootstrap_target", "ensemble_mean"))
            if bootstrap_mode not in {"ensemble_mean", "mean"}:
                raise ValueError("TD Origin requires critic.bootstrap_target=ensemble_mean")
            next_q = _reference_value_mean(
                state,
                batch,
                num_samples=next_action_samples,
                aggregation_mode="ensemble_mean",
            )
            target = batch.chunk_returns + batch.discounts * (1.0 - batch.dones) * next_q
            bootstrap_active = 1.0
            lambda_mc = 0.0
        else:
            raise ValueError(f"unsupported scalar-Q critic.target_mode={target_mode!r}")

    q_error = (q_pred - target.unsqueeze(0)).square()
    q_loss = q_error.mean()
    (q_loss * float(loss_scale)).backward()
    critic_grad_norm = 0.0
    if optimizer_step:
        critic_grad_norm = float(grad_norm(state.critic.parameters()))
        torch.nn.utils.clip_grad_norm_(
            state.critic.parameters(),
            float(critic_cfg.get("max_grad_norm", 1000.0)),
        )
        state.critic_optimizer.step()
    target_update_period = int(critic_cfg.get("target_update_period", 1))
    if target_update_period <= 0:
        raise ValueError("critic.target_update_period must be positive")
    target_updated = optimizer_step and (state.step + 1) % target_update_period == 0
    if target_updated:
        soft_update(
            state.target_critic,
            state.critic,
            float(critic_cfg.get("target_tau", 0.005)),
        )
    if optimizer_step:
        state.step += 1
        state.critic_stage_step += 1
    member_losses = q_error.mean(dim=1)
    metrics = {
        "critic_loss": float(q_loss.detach().item()),
        "q_loss": float(q_loss.detach().item()),
        "q_loss_is_mse": 1.0,
        "divl_loss": 0.0,
        "divl_enabled": 0.0,
        "target_mean": float(target.mean().item()),
        "target_std": float(target.std(unbiased=False).item()),
        "td_error_abs_mean": float(
            (q_pred.detach() - target.unsqueeze(0)).abs().mean().item()
        ),
        "q_mean": float(q_pred.detach().mean().item()),
        "q_std": float(q_pred.detach().std(unbiased=False).item()),
        "v_divl_mean": 0.0,
        "critic_grad_norm": critic_grad_norm,
        "bootstrap_active": bootstrap_active,
        "bootstrap_target_is_min": 0.0,
        "bootstrap_target_is_subsample_min": 0.0,
        "reference_value_samples": float(next_action_samples),
        "lambda_mc": lambda_mc,
        "target_updated": float(target_updated),
        "critic_stage_head_td": float(state.critic_stage == "head_td"),
        "critic_stage_full_td": float(state.critic_stage == "full_td"),
    }
    if next_q is not None:
        metrics["reference_value_mean"] = float(next_q.mean().item())
    for member, member_loss in enumerate(member_losses):
        metrics[f"q_loss_member_{member}"] = float(member_loss.item())
    return metrics


def _multimodal_double_q_divl_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
    same_state_rankq_actions: SameStateRankQActions | None = None,
    same_state_rankq_success_scale: float = 1.0,
    diagnostic_component_gradients: bool = False,
) -> dict[str, float]:
    """Train the clean three-member DIVL clipped-double-Q critic.

    This path is deliberately separate from the historical paired-head path.
    Shapes are explicit: Q predictions are ``[member, double_q, batch]`` and
    each member owns its own categorical V target.  No V or target is averaged
    across members before the TD update.
    """
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        raise TypeError("double-Q DIVL update requires a multimodal critic")
    core = state.critic.core
    if core.q_representation not in {"scalar", "categorical"} or core.q_heads_per_member != 2:
        raise ValueError(
            "double-Q DIVL main path requires scalar/categorical Q and exactly two Q heads per member"
        )
    critic_cfg = config.get("critic", {})
    divl_cfg = config.get("divl", {})
    batch = batch.to(next(state.critic.parameters()).device)
    if zero_grad:
        state.critic_optimizer.zero_grad(set_to_none=True)
    critic_mask = _critic_execution_mask(batch, config)
    features = state.critic.encode_state(batch)
    q_pair_logits = None
    if core.q_representation == "categorical":
        q_pair_logits = state.critic.q_pair_logits_from_features(
            features, batch.action_chunks, critic_mask
        )
        q_pairs = decode_categorical_q(q_pair_logits, core.q_support)
    else:
        q_pairs = state.critic.q_pair_from_features(
            features, batch.action_chunks, critic_mask
        )
    q_clipped = q_pairs.min(dim=1).values

    rankq_settings = same_state_rankq_settings(
        critic_cfg, optimizer_step=state.step
    )
    nested_rankq_enabled = bool(rankq_settings["enabled"])
    if nested_rankq_enabled and bool(critic_cfg.get("enable_rankq", False)):
        raise ValueError("nested critic.rankq and legacy flat RankQ cannot both be enabled")
    if nested_rankq_enabled:
        if same_state_rankq_actions is None:
            action_pool = state.critic.core.action_pool
            same_state_rankq_actions = make_same_state_rankq_actions(
                batch.action_chunks,
                batch.execution_masks,
                action_mean=action_pool.action_mean,
                action_std=action_pool.action_std,
                action_min=action_pool.action_min,
                action_max=action_pool.action_max,
                mild_sigma=float(rankq_settings["mild_sigma"]),
                strong_sigma=float(rankq_settings["strong_sigma"]),
                use_random_negative=bool(rankq_settings["use_random_negative"]),
            )
        else:
            same_state_rankq_actions = same_state_rankq_actions.to(
                batch.action_chunks.device
            )
        variants = [
            ("mild", same_state_rankq_actions.mild),
            ("strong", same_state_rankq_actions.strong),
        ]
        if same_state_rankq_actions.random is not None:
            variants.append(("random", same_state_rankq_actions.random))
        variant_actions = torch.cat([action for _, action in variants], dim=0)
        variant_masks = torch.cat([critic_mask] * len(variants), dim=0)
        variant_features = StateFeatures(
            readout=features.readout.repeat(len(variants), 1)
        )
        variant_raw_q = state.critic.raw_q_ensemble_from_features(
            variant_features,
            variant_actions,
            variant_masks,
        ).reshape(core.num_q_heads, len(variants), batch.batch_size)
        rankq_values = {
            "logged": q_pairs.reshape(core.num_q_heads, batch.batch_size),
            **{
                name: variant_raw_q[:, index]
                for index, (name, _) in enumerate(variants)
            },
        }
        rankq_output = compute_same_state_rankq_loss(
            rankq_values,
            batch.successes,
            use_success_only=bool(rankq_settings["use_success_only"]),
            logged_mild_weight=float(rankq_settings["logged_mild_weight"]),
            mild_strong_weight=float(rankq_settings["mild_strong_weight"]),
            strong_random_weight=float(rankq_settings["strong_random_weight"]),
        )
        rankq_loss = rankq_output.loss
        rankq_metrics = {
            "rankq_enabled": 1.0,
            "rankq_lambda": float(rankq_settings["lambda_rank"]),
            **rankq_output.metrics,
        }
    else:
        # Keep the disabled graph byte-for-byte on the original DIVL path.
        # A detached scalar avoids even a zero-valued auxiliary autograd edge.
        rankq_loss = q_pairs.detach().new_zeros(())
        rankq_metrics = disabled_same_state_rankq_metrics(
            lambda_rank=float(rankq_settings["lambda_rank"]),
            logged_mild_weight=float(rankq_settings["logged_mild_weight"]),
            mild_strong_weight=float(rankq_settings["mild_strong_weight"]),
            strong_random_weight=float(rankq_settings["strong_random_weight"]),
        )
    rankq_metrics.update(
        {
            "lambda_rank_effective": float(
                rankq_settings["lambda_rank_effective"]
            ),
            "lambda_rank_max": float(rankq_settings["lambda_rank_max"]),
            "lambda_rank_schedule_enabled": float(
                bool(rankq_settings["lambda_rank_schedule_enabled"])
            ),
            "lambda_rank_schedule_step": float(state.step),
        }
    )

    with torch.no_grad():
        target_next_features = state.target_critic.encode_state(
            batch, next_observation=True
        )
        next_logits = state.target_critic.value_logits_from_features(
            target_next_features
        )
        next_probs = F.softmax(next_logits, dim=-1)
        next_stats = divl_quantile_values(
            next_probs,
            state.support,
            alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
            alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
            entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
            alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
            use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
            interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            **_double_q_v_tau_kwargs(critic_cfg, divl_cfg),
        )
        # Each member bootstraps from its own V_m quantile.
        v_next = next_stats.quantile_value
        y = (
            batch.chunk_returns.unsqueeze(0)
            + batch.discounts.unsqueeze(0)
            * (1.0 - batch.dones.unsqueeze(0))
            * v_next
        )
        if str(critic_cfg.get("target_mode", "td")).lower() == "mc_return":
            if batch.mc_returns is None:
                raise ValueError("target_mode=mc_return requires replay mc_returns")
            y = batch.mc_returns.unsqueeze(0).expand_as(y)
        elif str(critic_cfg.get("target_mode", "td")).lower() != "td":
            raise ValueError("double-Q DIVL target_mode must be td or mc_return")

        target_features = state.target_critic.encode_state(batch)
        target_pairs = state.target_critic.q_pair_from_features(
            target_features, batch.action_chunks, critic_mask
        )
        target_clipped = target_pairs.min(dim=1).values
        v_q_aggregation = str(critic_cfg.get("v_q_aggregation", "min")).lower()
        v_q_target = aggregate_double_q_for_v(target_pairs, v_q_aggregation)
        z_target = divl_double_q_projection_targets(
            target_pairs,
            state.support,
            aggregation=v_q_aggregation,
        )

    mask = bootstrap_mask(
        state.critic.ensemble_size,
        batch.batch_size,
        float(critic_cfg.get("bootstrap_probability", 1.0)),
        device=q_pairs.device,
        generator=state.target_generator,
    )
    if config.get("training", {}).get("critic_sampling", {}).get("mode") == "member_episode_bootstrap":
        from .episode_bootstrap import member_owner_loss_mask
        mask = member_owner_loss_mask(batch, state.critic.ensemble_size, device=q_pairs.device)
    if config.get("training", {}).get("critic_sampling", {}).get("mode") == "shared_episode_bootstrap":
        mask = torch.tensor([m['bootstrap_inclusion'] for m in batch.behavior_metadata], device=q_pairs.device).T
    if core.q_representation == "categorical":
        assert q_pair_logits is not None and core.q_support is not None
        q_error = one_hot_categorical_cross_entropy(
            q_pair_logits,
            y.unsqueeze(1),
            core.q_support,
        )
        q_entropy = categorical_q_entropy(q_pair_logits.detach())
        q_clip_low = (y < core.q_support[0]).float().mean()
        q_clip_high = (y > core.q_support[-1]).float().mean()
    else:
        q_error = (q_pairs - y.unsqueeze(1)).square()
        q_entropy = q_pairs.new_zeros(q_pairs.shape)
        q_clip_low = q_pairs.new_zeros(())
        q_clip_high = q_pairs.new_zeros(())
    weighted_mask = mask.to(q_error.dtype).unsqueeze(1)
    q_loss = (q_error * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0) / 2.0
    value_logits = state.critic.value_logits_from_features(features)
    value_ce = -(z_target * F.log_softmax(value_logits, dim=-1)).sum(dim=-1)
    divl_loss = (value_ce * mask.to(value_ce.dtype)).sum() / mask.sum().clamp_min(1.0)
    if config.get("training", {}).get("critic_sampling", {}).get("mode") == "shared_episode_bootstrap":
        from .episode_bootstrap import normalize_shared_member_weights
        if 'bootstrap_loss_weights' not in batch.behavior_metadata[0]:
            batch = normalize_shared_member_weights(batch, device=q_pairs.device)
        weights = torch.tensor([m['bootstrap_loss_weights'] for m in batch.behavior_metadata],
                               device=q_pairs.device, dtype=q_error.dtype).T
        q_loss = (q_error * weights[:,None,:]).sum() / (2 * batch.batch_size)
        divl_loss = (value_ce * weights).sum() / batch.batch_size
    divl_objective_loss = (
        q_loss + float(divl_cfg.get("loss_weight", 1.0)) * divl_loss
    )
    rankq_lambda = float(rankq_settings["lambda_rank"])
    rankq_weighted_loss = rankq_lambda * rankq_loss
    loss = (
        divl_objective_loss + rankq_weighted_loss
        if nested_rankq_enabled
        else divl_objective_loss
    )
    if diagnostic_component_gradients:
        if not nested_rankq_enabled:
            raise ValueError("RankQ gradient diagnostic requires active critic.rankq")

        def capture_gradient_statistics(
            *, keep_cpu_copy: bool = False
        ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
            square_parts: dict[str, list[torch.Tensor]] = {
                "total": [],
                "backbone": [],
                "q_head": [],
                "action": [],
                "q_action": [],
            }
            cpu_gradients: dict[str, torch.Tensor] = {}
            for name, parameter in state.critic.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                detached = gradient.detach().float()
                square = detached.square().sum()
                square_parts["total"].append(square)
                if name.startswith("state_encoder."):
                    square_parts["backbone"].append(square)
                if name.startswith("core.q_heads."):
                    square_parts["q_head"].append(square)
                    square_parts["q_action"].append(square)
                if name.startswith("core.action_pool."):
                    square_parts["action"].append(square)
                    square_parts["q_action"].append(square)
                if keep_cpu_copy:
                    cpu_gradients[name] = detached.cpu().clone()

            def norm(parts: list[torch.Tensor]) -> float:
                if not parts:
                    return 0.0
                return float(torch.stack(parts).sum().sqrt().item())

            return (
                    {
                        "total": norm(square_parts["total"]),
                        "backbone": norm(square_parts["backbone"]),
                        "q_head": norm(square_parts["q_head"]),
                        "action": norm(square_parts["action"]),
                        "q_action": norm(square_parts["q_action"]),
                    },
                cpu_gradients,
            )

        state.critic_optimizer.zero_grad(set_to_none=True)
        divl_objective_loss.backward(retain_graph=True)
        divl_gradients, divl_cpu_gradients = capture_gradient_statistics(
            keep_cpu_copy=True
        )
        state.critic_optimizer.zero_grad(set_to_none=True)
        # Probe the unweighted auxiliary itself. Candidate lambda values are
        # applied analytically by the reporting script, not baked into g_R.
        rankq_loss.backward(retain_graph=True)
        rankq_gradients, rankq_cpu_gradients = capture_gradient_statistics(
            keep_cpu_copy=True
        )
        dot = torch.zeros((), dtype=torch.float64)
        for name, parameter in state.critic.named_parameters():
            if parameter.grad is None or name not in divl_cpu_gradients:
                continue
            dot += (
                divl_cpu_gradients[name]
                * parameter.grad.detach().float().cpu()
            ).double().sum()
        gradient_cosine = max(
            -1.0,
            min(
                1.0,
                float(
                    dot.item()
                    / (
                        divl_gradients["total"] * rankq_gradients["total"]
                        + 1.0e-12
                    )
                ),
            ),
        )
        value_grad_norm = float(
            grad_norm(
                parameter
                for head in state.critic.core.value_heads
                for parameter in head.parameters()
            )
        )
        q_head_gradient_count = sum(
            any(
                parameter.grad is not None
                and bool((parameter.grad.detach() != 0).any())
                for parameter in head.parameters()
            )
            for head in state.critic.core.q_heads
        )
        relation_gradient_metrics: dict[str, float] = {}
        relation_cpu_gradients: dict[str, dict[str, torch.Tensor]] = {}
        relation_gradient_norms: dict[str, float] = {}
        relation_items = list(rankq_output.relation_losses.items())
        for relation_index, (relation, relation_loss) in enumerate(relation_items):
            state.critic_optimizer.zero_grad(set_to_none=True)
            relation_loss.backward(
                retain_graph=relation_index < len(relation_items) - 1
            )
            relation_gradients, relation_cpu = capture_gradient_statistics(
                keep_cpu_copy=True
            )
            relation_cpu_gradients[relation] = relation_cpu
            relation_gradient_norms[relation] = relation_gradients["total"]
            relation_dot = torch.zeros((), dtype=torch.float64)
            for name, parameter in state.critic.named_parameters():
                if parameter.grad is None or name not in divl_cpu_gradients:
                    continue
                relation_dot += (
                    divl_cpu_gradients[name]
                    * parameter.grad.detach().float().cpu()
                ).double().sum()
            relation_gradient_metrics.update(
                {
                    f"rankq_{relation}_grad_norm": relation_gradients["total"],
                    f"rankq_{relation}_backbone_grad_norm": relation_gradients[
                        "backbone"
                    ],
                    f"rankq_{relation}_q_head_grad_norm": relation_gradients[
                        "q_head"
                    ],
                    f"rankq_{relation}_action_grad_norm": relation_gradients[
                        "action"
                    ],
                    f"rankq_{relation}_q_action_grad_norm": relation_gradients[
                        "q_action"
                    ],
                    f"rankq_{relation}_divl_gradient_cosine": max(
                        -1.0,
                        min(
                            1.0,
                            float(
                                relation_dot.item()
                                / (
                                    divl_gradients["total"]
                                    * relation_gradients["total"]
                                    + 1.0e-12
                                )
                            ),
                        ),
                    ),
                    f"rankq_{relation}_weighted_individual_grad_norm": float(
                        rankq_output.metrics[f"rankq_{relation}_weight"]
                        * relation_gradients["total"]
                    ),
                }
            )

        def cpu_gradient_cosine(
            left: dict[str, torch.Tensor],
            right: dict[str, torch.Tensor],
            left_norm: float,
            right_norm: float,
        ) -> float:
            pair_dot = torch.zeros((), dtype=torch.float64)
            for name in left.keys() & right.keys():
                pair_dot += (left[name] * right[name]).double().sum()
            return max(
                -1.0,
                min(
                    1.0,
                    float(
                        pair_dot.item() / (left_norm * right_norm + 1.0e-12)
                    ),
                ),
            )

        for relation, relation_cpu in relation_cpu_gradients.items():
            relation_gradient_metrics[
                f"rankq_{relation}_weighted_rankq_gradient_cosine"
            ] = cpu_gradient_cosine(
                relation_cpu,
                rankq_cpu_gradients,
                relation_gradient_norms[relation],
                rankq_gradients["total"],
            )
        relation_names = list(relation_cpu_gradients)
        for left_index, left in enumerate(relation_names):
            for right in relation_names[left_index + 1 :]:
                relation_gradient_metrics[
                    f"rankq_relation_gradient_cosine_{left}__{right}"
                ] = cpu_gradient_cosine(
                    relation_cpu_gradients[left],
                    relation_cpu_gradients[right],
                    relation_gradient_norms[left],
                    relation_gradient_norms[right],
                )
        state.critic_optimizer.zero_grad(set_to_none=True)
        rankq_lambda = float(rankq_settings["lambda_rank"])
        return {
            "divl_grad_norm": divl_gradients["total"],
            "divl_backbone_grad_norm": divl_gradients["backbone"],
            "divl_q_head_grad_norm": divl_gradients["q_head"],
            "rankq_grad_norm": rankq_gradients["total"],
            "rankq_backbone_grad_norm": rankq_gradients["backbone"],
            "rankq_q_head_grad_norm": rankq_gradients["q_head"],
            "rankq_action_grad_norm": rankq_gradients["action"],
            "rankq_q_action_grad_norm": rankq_gradients["q_action"],
            "divl_rankq_gradient_cosine": gradient_cosine,
            "rankq_weighted_grad_norm": rankq_lambda
            * rankq_gradients["total"],
            "rankq_to_divl_grad_norm_ratio": rankq_lambda
            * rankq_gradients["total"]
            / (divl_gradients["total"] + 1e-12),
            "rankq_value_head_grad_norm": value_grad_norm,
            "rankq_q_head_gradient_count": float(q_head_gradient_count),
            "divl_objective_loss": float(divl_objective_loss.detach().item()),
            "rankq_weighted_loss": float(rankq_weighted_loss.detach().item()),
            "total_critic_loss": float(loss.detach().item()),
            **relation_gradient_metrics,
            **rankq_metrics,
        }
    backward_loss = float(loss_scale) * divl_objective_loss
    if nested_rankq_enabled:
        backward_loss = (
            backward_loss
            + float(same_state_rankq_success_scale) * rankq_weighted_loss
        )
    backward_loss.backward()
    critic_grad_norm = 0.0
    critic_grad_norm_postclip = 0.0
    critic_grad_clip_scale = 1.0
    if optimizer_step:
        _all_reduce_gradients(state.critic.parameters())
        critic_grad_norm = float(grad_norm(state.critic.parameters()))
        max_grad_norm = float(critic_cfg.get("max_grad_norm", 10.0))
        torch.nn.utils.clip_grad_norm_(
            state.critic.parameters(), max_grad_norm
        )
        critic_grad_norm_postclip = float(grad_norm(state.critic.parameters()))
        critic_grad_clip_scale = min(
            1.0, max_grad_norm / max(critic_grad_norm, 1.0e-12)
        )
        _apply_critic_lr_schedule(state, config)
        state.critic_optimizer.step()
    target_update_period = int(critic_cfg.get("target_update_period", 1))
    if target_update_period <= 0:
        raise ValueError("critic.target_update_period must be positive")
    target_updated = optimizer_step and (state.step + 1) % target_update_period == 0
    if target_updated:
        soft_update(
            state.target_critic,
            state.critic,
            float(critic_cfg.get("target_tau", 0.005)),
        )
    if optimizer_step:
        state.step += 1
        state.critic_stage_step += 1

    value_probs = value_logits.detach().softmax(dim=-1)
    tau_values = next_stats.tau if next_stats.tau is not None else next_stats.alpha
    value_expected = (
        value_probs * state.support.to(value_probs.device, value_probs.dtype)
    ).sum(dim=-1)
    raw_q = q_pairs.detach().reshape(core.num_q_heads, batch.batch_size)
    metrics: dict[str, float] = {
        "critic_loss": float(loss.detach().item()),
        "total_loss": float(loss.detach().item()),
        "total_critic_loss": float(loss.detach().item()),
        "divl_objective_loss": float(divl_objective_loss.detach().item()),
        "rankq_weighted_loss": float(rankq_weighted_loss.detach().item()),
        "rankq_total_lambda_weighted_loss": float(
            rankq_weighted_loss.detach().item()
        ),
        "q_loss": float(q_loss.detach().item()),
        "q_loss_is_mse": float(core.q_representation == "scalar"),
        "q_representation_is_categorical": float(
            core.q_representation == "categorical"
        ),
        "critic/q_ce_loss": float(q_loss.detach().item())
        if core.q_representation == "categorical"
        else 0.0,
        "critic/q_one_hot_ce_loss": float(q_loss.detach().item())
        if core.q_representation == "categorical"
        else 0.0,
        "critic/q_expected_mse_loss": 0.0,
        "critic/q_target_clip_low_fraction": float(q_clip_low.item()),
        "critic/q_target_clip_high_fraction": float(q_clip_high.item()),
        "critic/q_entropy_mean": float(q_entropy.mean().item()),
        "divl_loss": float(divl_loss.detach().item()),
        "v_loss": float(divl_loss.detach().item()),
        "divl_enabled": 1.0,
        "target_mean": float(y.mean().item()),
        "target_std": float(y.std(unbiased=False).item()),
        "td_error_abs_mean": float((q_clipped.detach() - y).abs().mean().item()),
        "q_mean": float(q_clipped.detach().mean().item()),
        "q_std": float(q_clipped.detach().std(unbiased=False).item()),
        "q_clipped_mean": float(q_clipped.detach().mean().item()),
        "q_ensemble_disagreement": float(q_clipped.detach().std(dim=0, unbiased=False).mean().item()),
        "raw_q_mean": float(raw_q.mean().item()),
        "raw_q_std": float(raw_q.std(unbiased=False).item()),
        "raw_q_ensemble_std": float(
            raw_q.std(dim=0, unbiased=False).mean().item()
        ),
        "mean_pair_disagreement": float(
            (q_pairs.detach()[:, 0] - q_pairs.detach()[:, 1]).abs().mean().item()
        ),
        "v_divl_mean": float(v_next.mean().item()),
        "v_mean": float(v_next.mean().item()),
        "v_std": float(v_next.std(unbiased=False).item()),
        "v_entropy": float(next_stats.entropy.mean().item()),
        "divl_entropy": float(next_stats.entropy.mean().item()),
        "adaptive_alpha": float(tau_values.mean().item()),
        "adaptive_tau_mean": float(tau_values.mean().item()),
        "adaptive_tau_min": float(tau_values.min().item()),
        "adaptive_tau_max": float(tau_values.max().item()),
        "critic_grad_norm": critic_grad_norm,
        "critic_grad_norm_preclip": critic_grad_norm,
        "critic_grad_norm_postclip": critic_grad_norm_postclip,
        "critic_grad_clip_scale": critic_grad_clip_scale,
        "grad_norm_preclip": critic_grad_norm,
        "grad_norm_postclip": critic_grad_norm_postclip,
        "gradient_clipped": float(critic_grad_clip_scale < 1.0),
        "learning_rate": float(state.critic_optimizer.param_groups[0]["lr"]),
        "bootstrap_active": float(mask.float().mean().item()),
        "bootstrap_target_is_memberwise": 1.0,
        "bootstrap_target_is_min": 1.0,
        "v_target_from_member_clipped_q_mean": float(target_clipped.mean().item()),
        "v_q_target_mean": float(v_q_target.mean().item()),
        "v_q_aggregation_is_min": float(v_q_aggregation == "min"),
        "v_q_aggregation_is_mean": float(v_q_aggregation == "mean"),
        "v_tau_config_min": float(
            _double_q_v_tau_kwargs(critic_cfg, divl_cfg)["tau_min"]
        ),
        "v_tau_config_max": float(
            _double_q_v_tau_kwargs(critic_cfg, divl_cfg)["tau_max"]
        ),
        "target_updated": float(target_updated),
        "critic_stage_head_td": float(state.critic_stage == "head_td"),
        "critic_stage_full_td": float(state.critic_stage == "full_td"),
        "categorical_saturation": float(
            ((value_probs[..., 0] + value_probs[..., -1]) > 0.5).float().mean().item()
        ),
        "v_quantile": float(v_next.mean().item()),
        **rankq_metrics,
    }
    if core.num_q_heads == 10:
        metrics["raw_10q_mean"] = metrics["raw_q_mean"]
        metrics["raw_10q_std"] = metrics["raw_q_std"]
    for member in range(core.num_pairs):
        metrics[f"q_loss_member_{member}"] = float(
            q_error.detach()[member].mean().item()
        )
        for q_index in range(2):
            member_error = q_error.detach()[member, q_index]
            metrics[f"q_loss_member_{member}_q{q_index + 1}"] = float(
                member_error.mean().item()
            )
            metrics[f"q_mean_member_{member}_q{q_index + 1}"] = float(
                q_pairs.detach()[member, q_index].mean().item()
            )
            metrics[f"q_std_member_{member}_q{q_index + 1}"] = float(
                q_pairs.detach()[member, q_index].std(unbiased=False).item()
            )
        metrics[f"q_clipped_member_{member}_mean"] = float(
            q_clipped.detach()[member].mean().item()
        )
        metrics[f"q_clipped_member_{member}_std"] = float(
            q_clipped.detach()[member].std(unbiased=False).item()
        )
        metrics[f"v_mean_member_{member}"] = float(v_next.detach()[member].mean().item())
        metrics[f"v_expected_member_{member}"] = float(
            value_expected[member].mean().item()
        )
        metrics[f"v_expected_member_{member}_std"] = float(
            value_expected[member].std(unbiased=False).item()
        )
        metrics[f"v_entropy_member_{member}"] = float(next_stats.entropy[member].mean().item())
        metrics[f"v_tau_member_{member}"] = float(tau_values[member].mean().item())
        metrics[f"v_loss_member_{member}"] = float(value_ce.detach()[member].mean().item())
    correlation_period = int(critic_cfg.get("q_correlation_log_period", 100))
    if correlation_period > 0 and state.step % correlation_period == 0:
        centered = raw_q.float() - raw_q.float().mean(dim=1, keepdim=True)
        norms = centered.square().sum(dim=1).sqrt().clamp_min(1e-12)
        correlation = centered @ centered.t() / (norms[:, None] * norms[None, :])
        off_diagonal = ~torch.eye(
            correlation.shape[0], dtype=torch.bool, device=correlation.device
        )
        metrics["raw_q_pair_correlation"] = float(
            correlation[off_diagonal].mean().item()
        )
    return metrics


def _all_reduce_gradients(
    parameters: Any,
    *,
    bucket_cap_mb: float = 4.0,
) -> None:
    """Average gradients across torch.distributed ranks in bounded buckets."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    world_size = dist.get_world_size()
    cap_bytes = max(1, int(float(bucket_cap_mb) * 1024 * 1024))
    buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}

    def flush(grads: list[torch.Tensor]) -> None:
        if not grads:
            return
        flat = torch.cat([gradient.reshape(-1) for gradient in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for gradient in grads:
            count = gradient.numel()
            gradient.copy_(flat[offset : offset + count].view_as(gradient))
            offset += count

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        key = (gradient.device, gradient.dtype)
        bucket = buckets.setdefault(key, [])
        bucket_bytes = sum(item.numel() * item.element_size() for item in bucket)
        gradient_bytes = gradient.numel() * gradient.element_size()
        if gradient_bytes >= cap_bytes:
            flush(bucket)
            bucket.clear()
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(world_size)
            continue
        if bucket and bucket_bytes + gradient_bytes > cap_bytes:
            flush(bucket)
            bucket.clear()
        bucket.append(gradient)
    for bucket in buckets.values():
        flush(bucket)


def _multimodal_q_predictions(
    state: OGPOTrainState,
    batch: ChunkBatch,
    features: StateFeatures,
    critic_mask: torch.Tensor,
    critic_cfg: dict[str, Any],
    *,
    rankq_actions: RankQActions | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Score positive and ranking actions with one shared state encoding."""
    rank_enabled = bool(critic_cfg.get("rank_consensus_enabled", False))
    rankq_enabled = bool(critic_cfg.get("enable_rankq", False))
    if rank_enabled and rankq_enabled:
        raise ValueError("vanilla RankQ and consensus B2 ranking cannot be enabled together")
    use_strong = rank_enabled and bool(critic_cfg.get("rank_use_strong_noise", True))
    use_random = rank_enabled and bool(critic_cfg.get("rank_use_random_negative", True))
    variants: list[tuple[str, torch.Tensor]] = [("positive", batch.action_chunks)]
    if rankq_enabled:
        if rankq_actions is None:
            action_pool = state.critic.core.action_pool
            rankq_actions = make_rankq_actions(
                batch.action_chunks,
                batch.execution_masks,
                action_mean=action_pool.action_mean,
                action_std=action_pool.action_std,
                action_min=action_pool.action_min,
                action_max=action_pool.action_max,
                noise_sigma=float(critic_cfg.get("rankq_noise_sigma", 0.15)),
            )
        else:
            rankq_actions = rankq_actions.to(batch.action_chunks.device)
        variants.extend(
            [
                ("noisy", rankq_actions.noisy),
                ("very_noisy", rankq_actions.very_noisy),
                ("random", rankq_actions.random),
                ("permuted", rankq_actions.permuted),
            ]
        )
    elif use_strong or use_random:
        action_pool = state.critic.core.action_pool
        strong, random = ranking_action_negatives(
            batch.action_chunks,
            critic_mask,
            action_mean=action_pool.action_mean,
            action_std=action_pool.action_std,
            action_min=action_pool.action_min,
            action_max=action_pool.action_max,
            noise_sigma=float(critic_cfg.get("rank_noise_sigma", 0.15)),
        )
        if use_strong:
            variants.append(("strong", strong))
        if use_random:
            variants.append(("random", random))

    variant_count = len(variants)
    combined_actions = torch.cat([actions for _, actions in variants], dim=0)
    combined_masks = torch.cat([critic_mask] * variant_count, dim=0)
    combined_features = StateFeatures(readout=features.readout.repeat(variant_count, 1))
    if state.critic.core.q_representation == "categorical":
        combined_logits = state.critic.q_logits_from_features(
            combined_features,
            combined_actions,
            combined_masks,
        )
        q_values = decode_categorical_q(combined_logits, state.critic.core.q_support)
        q_logits = combined_logits.reshape(
            state.critic.ensemble_size,
            variant_count,
            batch.batch_size,
            combined_logits.shape[-1],
        )[:, 0]
    else:
        q_values = state.critic.q_from_features(
            combined_features,
            combined_actions,
            combined_masks,
        )
        q_logits = None
    q_values = q_values.reshape(state.critic.ensemble_size, variant_count, batch.batch_size)
    ranking_values = {
        name: q_values[:, index]
        for index, (name, _) in enumerate(variants)
    }
    return q_values[:, 0], q_logits, ranking_values


def _ranking_loss_and_metrics(
    ranking_values: dict[str, torch.Tensor],
    batch: ChunkBatch,
    critic_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    positive = ranking_values["positive"]
    zero = positive.sum().float() * 0.0
    metrics = {
        "critic/rank_loss": 0.0,
        "critic/rank_loss_strong": 0.0,
        "critic/rank_loss_random": 0.0,
        "critic/rank_pair_count": 0.0,
        "critic/rank_d1_mean": 0.0,
        "critic/rank_d2_mean": 0.0,
        "critic/rank_d3_mean": 0.0,
        "critic/rank_worst_margin_mean": 0.0,
    }
    if not bool(critic_cfg.get("rank_consensus_enabled", False)):
        return zero, metrics
    valid = (
        batch.successes.bool()
        if bool(critic_cfg.get("rank_only_success", True))
        else torch.ones_like(batch.successes, dtype=torch.bool)
    )
    q_num_bins = int(critic_cfg.get("q_num_bins", 201))
    if q_num_bins < 2:
        raise ValueError("critic.q_num_bins must be at least 2")
    bin_width = (
        float(critic_cfg.get("q_vmax", 1.1)) - float(critic_cfg.get("q_vmin", -0.1))
    ) / (q_num_bins - 1)
    margin = float(critic_cfg.get("rank_margin_bins", 2.0)) * bin_width
    losses = []
    margin_parts = []
    worst_parts = []
    for name in ("strong", "random"):
        if name not in ranking_values:
            continue
        rank_loss, member_margins, worst_margin = consensus_ranking_loss(
            positive,
            ranking_values[name],
            valid,
            margin=margin,
            softmin_tau=float(critic_cfg.get("rank_softmin_tau", 0.02)),
            temperature=float(critic_cfg.get("rank_temperature", 0.02)),
        )
        losses.append(rank_loss)
        metrics[f"critic/rank_loss_{name}"] = float(rank_loss.detach().item())
        if bool(valid.any()):
            margin_parts.append(member_margins[:, valid])
            worst_parts.append(worst_margin[valid])
    if not losses:
        return zero, metrics
    loss = torch.stack(losses).mean()
    metrics["critic/rank_loss"] = float(loss.detach().item())
    metrics["critic/rank_pair_count"] = float(int(valid.sum().item()) * len(losses))
    if margin_parts:
        all_margins = torch.cat(margin_parts, dim=1).detach()
        all_worst = torch.cat(worst_parts, dim=0).detach()
        for member in range(min(3, all_margins.shape[0])):
            metrics[f"critic/rank_d{member + 1}_mean"] = float(
                all_margins[member].mean().item()
            )
        metrics["critic/rank_worst_margin_mean"] = float(all_worst.mean().item())
    return loss, metrics


def _multimodal_critic_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
    rankq_actions: RankQActions | None = None,
    rankq_success_scale: float = 1.0,
    rankq_failure_scale: float = 1.0,
) -> dict[str, float]:
    critic_cfg = config.get("critic", {})
    divl_cfg = config.get("divl", {})
    batch = batch.to(next(state.critic.parameters()).device)
    if zero_grad:
        state.critic_optimizer.zero_grad(set_to_none=True)
    critic_mask = _critic_execution_mask(batch, config)

    features = state.critic.encode_state(batch)
    q_pred, q_logits, ranking_values = _multimodal_q_predictions(
        state,
        batch,
        features,
        critic_mask,
        critic_cfg,
        rankq_actions=rankq_actions,
    )
    with torch.no_grad():
        target_aggregation = str(critic_cfg.get("bootstrap_target", "ensemble_mean"))
        reference_value_samples = int(critic_cfg.get("reference_value_samples", 0))
        lambda_divl_target = float(critic_cfg.get("lambda_divl_target", 1.0))
        reference_value = None
        if state.critic_stage == "head_mc":
            if batch.mc_returns is None:
                raise ValueError("head_mc critic stage requires replay mc_returns")
            y = batch.mc_returns
            v_next = torch.zeros_like(y)
            next_stats = divl_quantile_values(
                F.softmax(state.critic.value_logits_from_features(features), dim=-1),
                state.support,
                alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
                alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
                entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
                alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
                use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
                interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            )
            q_data_target = batch.mc_returns.unsqueeze(0).expand(
                state.critic.ensemble_size,
                batch.batch_size,
            )
            lambda_mc = 1.0
        else:
            next_features = state.target_critic.encode_state(batch, next_observation=True)
            next_logits = state.target_critic.value_logits_from_features(next_features)
            next_stats = divl_quantile_values(
                F.softmax(next_logits, dim=-1),
                state.support,
                alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
                alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
                entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
                alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
                use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
                interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            )
            v_next, _ = aggregate_value_heads(
                next_stats.quantile_value,
                target_aggregation,
                generator=state.target_generator,
            )
            if not 0.0 <= lambda_divl_target <= 1.0:
                raise ValueError("critic.lambda_divl_target must be in [0, 1]")
            if reference_value_samples > 0:
                reference_value = _reference_value_mean(
                    state,
                    batch,
                    num_samples=reference_value_samples,
                    aggregation_mode=target_aggregation,
                )
                v_bootstrap = (
                    lambda_divl_target * v_next
                    + (1.0 - lambda_divl_target) * reference_value
                )
            else:
                v_bootstrap = v_next
            y_td = batch.chunk_returns + batch.discounts * (1.0 - batch.dones) * v_bootstrap
            lambda_mc = float(critic_cfg.get("lambda_mc", 0.0))
            if not 0.0 <= lambda_mc <= 1.0:
                raise ValueError("critic.lambda_mc must be in [0, 1]")
            if lambda_mc > 0.0 and batch.mc_returns is None:
                raise ValueError("critic.lambda_mc requires replay mc_returns; rebuild the offline dataset")
            y = y_td if lambda_mc == 0.0 else (1.0 - lambda_mc) * y_td + lambda_mc * batch.mc_returns
            target_features = state.target_critic.encode_state(batch)
            q_data_target = state.target_critic.q_from_features(
                target_features,
                batch.action_chunks,
                critic_mask,
            )
        z_target = divl_projection_targets(q_data_target, state.support)

    mask = bootstrap_mask(
        state.critic.ensemble_size,
        batch.batch_size,
        float(critic_cfg.get("bootstrap_probability", 0.8)),
        device=q_pred.device,
    )
    q_representation = state.critic.core.q_representation
    q_target = y.unsqueeze(0).expand_as(q_pred)
    if q_representation == "categorical":
        assert q_logits is not None and state.critic.core.q_support is not None
        q_loss_type = str(
            critic_cfg.get("categorical_q_loss", "hl_gauss_ce")
        ).lower()
        if q_loss_type == "hl_gauss_ce":
            q_target_distribution = hl_gauss_projection(
                y,
                state.critic.core.q_support,
                sigma_bins=float(critic_cfg.get("q_hl_gauss_sigma_bins", 0.75)),
            )
            q_error = -(
                q_target_distribution.unsqueeze(0)
                * F.log_softmax(q_logits.float(), dim=-1)
            ).sum(dim=-1)
        elif q_loss_type == "one_hot_ce":
            q_error = one_hot_categorical_cross_entropy(
                q_logits,
                y,
                state.critic.core.q_support,
            )
        elif q_loss_type == "expected_mse":
            # Preserve the categorical Q parameterization while matching the
            # scalar baseline's regression objective on the decoded mean.
            q_error = (q_pred - q_target).square()
        else:
            raise ValueError(
                "unsupported critic.categorical_q_loss="
                f"{q_loss_type!r}; expected 'hl_gauss_ce', 'one_hot_ce', "
                "or 'expected_mse'"
            )
        q_entropy = categorical_q_entropy(q_logits.detach())
        support = state.critic.core.q_support
        q_clip_low = (y < support[0]).float().mean()
        q_clip_high = (y > support[-1]).float().mean()
    else:
        q_loss_type = str(critic_cfg.get("q_loss", "huber"))
        if q_loss_type == "mse":
            q_error = (q_pred - q_target).square()
        elif q_loss_type == "huber":
            q_error = F.huber_loss(
                q_pred,
                q_target,
                delta=float(critic_cfg.get("huber_delta", 1.0)),
                reduction="none",
            )
        else:
            raise ValueError(f"unsupported critic.q_loss={q_loss_type!r}")
        q_entropy = q_pred.new_zeros(q_pred.shape)
        q_clip_low = q_pred.new_zeros(())
        q_clip_high = q_pred.new_zeros(())
    q_loss = (q_error * mask.to(q_error.dtype)).sum() / mask.sum().clamp_min(1)
    value_logits = state.critic.value_logits_from_features(features)
    divl_loss = -(z_target * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
    rank_loss, rank_metrics = _ranking_loss_and_metrics(
        ranking_values,
        batch,
        critic_cfg,
    )
    original_loss = (
        q_loss
        + float(divl_cfg.get("loss_weight", 1.0)) * divl_loss
        + float(critic_cfg.get("rank_loss_weight", 0.1)) * rank_loss
    )
    rankq_enabled = bool(critic_cfg.get("enable_rankq", False))
    rankq_lambda = float(critic_cfg.get("lambda_rank", 1.0))
    if rankq_enabled and rankq_lambda < 0.0:
        raise ValueError("critic.lambda_rank must be non-negative")
    if rankq_enabled:
        # RankQ operates on the expectation of the categorical Q distribution.
        rankq_output = compute_rankq_loss(ranking_values, batch.successes)
        rankq_backward_loss = (
            float(rankq_success_scale) * rankq_output.success_loss
            + float(rankq_failure_scale) * rankq_output.failure_loss
        )
        loss = original_loss + rankq_lambda * rankq_output.loss
        backward_loss = float(loss_scale) * original_loss + rankq_lambda * rankq_backward_loss
        rankq_metrics = {"rankq/enabled": 1.0, **rankq_output.metrics}
    else:
        loss = original_loss
        backward_loss = float(loss_scale) * original_loss
        rankq_metrics = disabled_rankq_metrics(ensemble_size=state.critic.ensemble_size)
    backward_loss.backward()
    critic_grad_norm = 0.0
    critic_lr_scale = 1.0
    if optimizer_step:
        critic_grad_norm = float(grad_norm(state.critic.parameters()))
        torch.nn.utils.clip_grad_norm_(
            state.critic.parameters(),
            float(critic_cfg.get("max_grad_norm", 10.0)),
        )
        critic_lr_scale = _apply_critic_lr_schedule(state, config)
        state.critic_optimizer.step()

    target_update_period = int(critic_cfg.get("target_update_period", 1))
    target_updated = optimizer_step and (state.step + 1) % target_update_period == 0
    if target_updated:
        soft_update(state.target_critic, state.critic, float(critic_cfg.get("target_tau", 0.005)))
    if optimizer_step:
        state.step += 1
        state.critic_stage_step += 1
    value_probs = F.softmax(value_logits.detach(), dim=-1)
    value_quantiles = next_stats.quantile_value
    value_quantile_min = value_quantiles.min(dim=0).values
    value_quantile_mean = value_quantiles.mean(dim=0)
    value_quantile_max = value_quantiles.max(dim=0).values
    metrics = {
        "critic_loss": float(loss.detach().item()),
        "q_loss": float(q_loss.detach().item()),
        "q_loss_is_mse": float(
            (q_representation == "scalar" and q_loss_type == "mse")
            or (q_representation == "categorical" and q_loss_type == "expected_mse")
        ),
        "q_representation_is_categorical": float(q_representation == "categorical"),
        "divl_loss": float(divl_loss.detach().item()),
        "divl_enabled": 1.0,
        "target_mean": float(y.mean().item()),
        "target_std": float(y.std(unbiased=False).item()),
        "td_error_abs_mean": float((q_pred.detach() - y.unsqueeze(0)).abs().mean().item()),
        "q_mean": float(q_pred.detach().mean().item()),
        "q_std": float(q_pred.detach().std(unbiased=False).item()),
        "v_divl_mean": float(v_next.mean().item()),
        "v_head_min_mean": float(value_quantile_min.mean().item()),
        "v_head_mean_mean": float(value_quantile_mean.mean().item()),
        "v_head_max_mean": float(value_quantile_max.mean().item()),
        "v_head_spread_mean": float(
            (value_quantile_max - value_quantile_min).mean().item()
        ),
        "critic_grad_norm": critic_grad_norm,
        "critic_lr_scale": critic_lr_scale,
        "critic_lr": float(state.critic_optimizer.param_groups[0]["lr"]),
        "bootstrap_active": float(mask.float().mean().item()),
        "divl_entropy": float(next_stats.entropy.mean().item()),
        "adaptive_alpha": float(next_stats.alpha.mean().item()),
        "bootstrap_target_is_min": float(target_aggregation in {"ensemble_min", "min"}),
        "bootstrap_target_is_subsample_min": float(target_aggregation == "subsample_min"),
        "lambda_mc": lambda_mc,
        "categorical_saturation": float(
            ((value_probs[..., 0] + value_probs[..., -1]) > 0.5).float().mean().item()
        ),
        "v_quantile": float(next_stats.quantile_value.mean().item()),
        "target_updated": float(target_updated),
        "critic_stage_head_mc": float(state.critic_stage == "head_mc"),
        "critic_stage_head_td": float(state.critic_stage == "head_td"),
        "critic_stage_gemma_lora_td": float(state.critic_stage == "gemma_lora_td"),
        "critic_stage_full_td": float(state.critic_stage == "full_td"),
        "reference_value_samples": float(reference_value_samples),
        "critic/q_ce_loss": float(q_loss.detach().item())
        if q_representation == "categorical"
        and q_loss_type in {"hl_gauss_ce", "one_hot_ce"}
        else 0.0,
        "critic/q_one_hot_ce_loss": float(q_loss.detach().item())
        if q_representation == "categorical" and q_loss_type == "one_hot_ce"
        else 0.0,
        "critic/q_expected_mse_loss": float(q_loss.detach().item())
        if q_representation == "categorical" and q_loss_type == "expected_mse"
        else 0.0,
        "critic/q_decoded_mean": float(q_pred.detach().mean().item()),
        "critic/q_decoded_std": float(q_pred.detach().std(unbiased=False).item()),
        "critic/q_target_mean": float(y.mean().item()),
        "critic/q_target_std": float(y.std(unbiased=False).item()),
        "critic/q_target_clip_low_fraction": float(q_clip_low.item()),
        "critic/q_target_clip_high_fraction": float(q_clip_high.item()),
        "critic/q_entropy_mean": float(q_entropy.mean().item()),
    }
    metrics.update(rank_metrics)
    metrics.update(rankq_metrics)
    metrics["rankq/lambda"] = rankq_lambda
    if q_representation == "categorical":
        assert q_logits is not None and state.critic.core.q_support is not None
        decoded_q = q_pred.detach().float()
        q_probabilities = F.softmax(q_logits.detach().float(), dim=-1)
        support = state.critic.core.q_support.detach().float()
        metrics.update(
            {
                "categorical_q/q_mean": float(decoded_q.mean().item()),
                "categorical_q/q_min": float(decoded_q.min().item()),
                "categorical_q/q_max": float(decoded_q.max().item()),
                "categorical_q/near_lower_support_fraction": float(
                    (decoded_q < support[0] + 0.02).float().mean().item()
                ),
                "categorical_q/near_upper_support_fraction": float(
                    (decoded_q > support[-1] - 0.02).float().mean().item()
                ),
                "categorical_q/first_atom_mass": float(
                    q_probabilities[..., 0].mean().item()
                ),
                "categorical_q/last_atom_mass": float(
                    q_probabilities[..., -1].mean().item()
                ),
            }
        )
    if reference_value is not None:
        metrics["reference_value_mean"] = float(reference_value.mean().item())
    for member, member_loss in enumerate(q_error.detach().mean(dim=1)):
        metrics[f"q_loss_member_{member}"] = float(member_loss.item())
    return metrics


def critic_update(state: OGPOTrainState, batch: ChunkBatch, config: dict[str, Any]) -> dict[str, float]:
    if not any(parameter.requires_grad for parameter in state.critic.parameters()):
        raise RuntimeError(
            "critic_update called while the offline actor-round critic is frozen; "
            "refresh the critic only in an explicit critic-training phase"
        )
    if isinstance(state.critic, MultiHeadUdivlCritic):
        if (
            state.critic.core.q_heads_per_member == 2
            and bool(config.get("critic", {}).get("double_q_divl", False))
        ):
            # The no-microbatch launcher path still needs the exact same
            # global-valid-count RankQ reduction as gradient accumulation.
            # Delegate one full local batch to that implementation instead of
            # letting DDP equally average per-rank success-only local means.
            # A configured lambda_rank=0 resolves to disabled and deliberately
            # stays on the original pure-DIVL path below (no perturbation or
            # auxiliary raw-Q forward).
            if bool(
                same_state_rankq_settings(
                    config.get("critic", {}), optimizer_step=state.step
                )["enabled"]
            ):
                return accumulated_critic_update(
                    state,
                    batch,
                    config,
                    microbatch_size=batch.batch_size,
                )
            return _multimodal_double_q_divl_update(state, batch, config)
        return _multimodal_critic_update(state, batch, config)
    if isinstance(state.critic, MultiHeadScalarQCritic):
        return _multimodal_scalar_q_update(state, batch, config)
    critic_cfg = config.get("critic", {})
    divl_cfg = config.get("divl", {})
    divl_enabled = bool(divl_cfg.get("enabled", True))
    batch = batch.to(next(state.critic.parameters()).device)
    state.critic_optimizer.zero_grad(set_to_none=True)

    critic_mask = _critic_execution_mask(batch, config)
    q_pred = state.critic(batch.observations, batch.action_chunks, critic_mask)
    with torch.no_grad():
        target_aggregation = str(critic_cfg.get("bootstrap_target", "ensemble_mean"))
        reference_value_samples = int(critic_cfg.get("reference_value_samples", 0))
        lambda_divl_target = float(critic_cfg.get("lambda_divl_target", 1.0))
        reference_value = None
        if divl_enabled:
            next_probs = state.target_divl(batch.next_observations)
            next_stats = divl_quantile_values(
                next_probs,
                state.support,
                alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
                alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
                entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
                alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
                use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
                interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            )
            if target_aggregation == "ensemble_mean":
                v_next = next_stats.quantile_value.mean(dim=0)
            elif target_aggregation == "ensemble_min":
                v_next = next_stats.quantile_value.min(dim=0).values
            else:
                raise ValueError(f"unsupported critic.bootstrap_target={target_aggregation!r}")
            if not 0.0 <= lambda_divl_target <= 1.0:
                raise ValueError("critic.lambda_divl_target must be in [0, 1]")
            if reference_value_samples > 0:
                reference_value = _reference_value_mean(
                    state,
                    batch,
                    num_samples=reference_value_samples,
                )
                v_bootstrap = lambda_divl_target * v_next + (1.0 - lambda_divl_target) * reference_value
            else:
                v_bootstrap = v_next
            y_td = batch.chunk_returns + batch.discounts * (1.0 - batch.dones) * v_bootstrap
            lambda_mc = float(critic_cfg.get("lambda_mc", 0.0))
            if not 0.0 <= lambda_mc <= 1.0:
                raise ValueError("critic.lambda_mc must be in [0, 1]")
            if lambda_mc > 0.0 and batch.mc_returns is None:
                raise ValueError("critic.lambda_mc requires replay mc_returns; rebuild the offline dataset")
            y = y_td if lambda_mc == 0.0 else (1.0 - lambda_mc) * y_td + lambda_mc * batch.mc_returns
            q_data_target = state.target_critic(batch.observations, batch.action_chunks, critic_mask)
            z_target = divl_projection_targets(q_data_target, state.support)
        else:
            if batch.mc_returns is None:
                raise ValueError("divl.enabled=false requires replay mc_returns")
            y = batch.mc_returns
            lambda_mc = 1.0
            v_next = torch.zeros_like(y)
            next_stats = None
            z_target = None

    mask = bootstrap_mask(
        state.critic.ensemble_size,
        batch.batch_size,
        float(critic_cfg.get("bootstrap_probability", 0.8)),
        device=q_pred.device,
    )
    q_error = F.huber_loss(
        q_pred,
        y.unsqueeze(0).expand_as(q_pred),
        delta=float(critic_cfg.get("huber_delta", 1.0)),
        reduction="none",
    )
    q_loss = (q_error * mask.to(q_error.dtype)).sum() / mask.sum().clamp_min(1)
    if divl_enabled:
        assert z_target is not None
        divl_logits = state.divl.logits(batch.observations)
        divl_loss = -(z_target * F.log_softmax(divl_logits, dim=-1)).sum(dim=-1).mean()
    else:
        divl_logits = None
        divl_loss = q_loss.new_zeros(())
    loss = q_loss + float(divl_cfg.get("loss_weight", 1.0)) * divl_loss
    loss.backward()
    critic_params = list(state.critic.parameters()) + list(state.divl.parameters())
    critic_grad_norm = float(grad_norm(critic_params))
    torch.nn.utils.clip_grad_norm_(critic_params, float(critic_cfg.get("max_grad_norm", 10.0)))
    state.critic_optimizer.step()
    target_update_period = int(critic_cfg.get("target_update_period", 1))
    if target_update_period <= 0:
        raise ValueError("critic.target_update_period must be positive")
    target_updated = (state.step + 1) % target_update_period == 0
    if target_updated:
        soft_update(state.target_critic, state.critic, float(critic_cfg.get("target_tau", 0.01)))
        soft_update(state.target_divl, state.divl, float(critic_cfg.get("target_tau", 0.01)))
    state.step += 1
    q_member_denominator = mask.sum(dim=1).clamp_min(1).to(q_error.dtype)
    q_member_losses = (q_error * mask.to(q_error.dtype)).sum(dim=1) / q_member_denominator
    divl_probs = F.softmax(divl_logits.detach(), dim=-1) if divl_logits is not None else None
    metrics = {
        "critic_loss": float(loss.detach().item()),
        "q_loss": float(q_loss.detach().item()),
        "divl_loss": float(divl_loss.detach().item()),
        "divl_enabled": float(divl_enabled),
        "target_mean": float(y.mean().item()),
        "target_std": float(y.std(unbiased=False).item()),
        "td_error_abs_mean": float((q_pred.detach() - y.unsqueeze(0)).abs().mean().item()),
        "q_mean": float(q_pred.detach().mean().item()),
        "q_std": float(q_pred.detach().std(unbiased=False).item()),
        "v_divl_mean": float(v_next.mean().item()),
        "critic_grad_norm": critic_grad_norm,
        "bootstrap_active": float(mask.float().mean().item()),
        "divl_entropy": float(next_stats.entropy.mean().item()) if next_stats is not None else 0.0,
        "adaptive_alpha": float(next_stats.alpha.mean().item()) if next_stats is not None else 0.0,
        "bootstrap_target_is_min": float(target_aggregation == "ensemble_min"),
        "lambda_mc": lambda_mc,
        "categorical_saturation": float(
            ((divl_probs[..., 0] + divl_probs[..., -1]) > 0.5).float().mean().item()
        ) if divl_probs is not None else 0.0,
        "v_quantile": float(next_stats.quantile_value.mean().item()) if next_stats is not None else 0.0,
        "target_updated": float(target_updated),
    }
    for member, member_loss in enumerate(q_member_losses):
        metrics[f"q_loss_member_{member}"] = float(member_loss.item())
    if reference_value is not None:
        metrics["reference_value_mean"] = float(reference_value.mean().item())
        metrics["lambda_divl_target"] = lambda_divl_target
    return metrics


def accumulated_critic_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    microbatch_size: int,
) -> dict[str, float]:
    """Run one multimodal optimizer step over a larger effective batch."""
    if not any(parameter.requires_grad for parameter in state.critic.parameters()):
        raise RuntimeError(
            "accumulated_critic_update called while the offline actor-round "
            "critic is frozen; refresh it only in an explicit critic phase"
        )
    if not isinstance(state.critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
        if microbatch_size < batch.batch_size:
            raise ValueError("critic gradient accumulation is only implemented for multimodal critics")
        return critic_update(state, batch, config)
    microbatch_size = int(microbatch_size)
    if microbatch_size <= 0:
        raise ValueError("critic microbatch_size must be positive")
    if config.get("training", {}).get("critic_sampling", {}).get("mode") == "shared_episode_bootstrap":
        from .episode_bootstrap import normalize_shared_member_weights
        batch = normalize_shared_member_weights(batch, device=next(state.critic.parameters()).device)
    starts = list(range(0, batch.batch_size, microbatch_size))
    total_size = float(batch.batch_size)
    critic_cfg = config.get("critic", {})
    rankq_enabled = bool(critic_cfg.get("enable_rankq", False))
    nested_rankq_settings = same_state_rankq_settings(
        critic_cfg, optimizer_step=state.step
    )
    nested_rankq_enabled = bool(nested_rankq_settings["enabled"])
    if (rankq_enabled or nested_rankq_enabled) and not isinstance(
        state.critic, MultiHeadUdivlCritic
    ):
        raise ValueError("vanilla RankQ requires the multi-head Q-V critic")
    if rankq_enabled and nested_rankq_enabled:
        raise ValueError("nested critic.rankq and legacy flat RankQ cannot both be enabled")
    full_rankq_actions = None
    full_same_state_rankq_actions = None
    total_success = int(batch.successes.bool().sum().item())
    total_failure = batch.batch_size - total_success
    if rankq_enabled:
        if bool(critic_cfg.get("rank_consensus_enabled", False)):
            raise ValueError("vanilla RankQ and consensus B2 ranking cannot be enabled together")
        action_pool = state.critic.core.action_pool
        full_rankq_actions = make_rankq_actions(
            batch.action_chunks,
            batch.execution_masks,
            action_mean=action_pool.action_mean,
            action_std=action_pool.action_std,
            action_min=action_pool.action_min,
            action_max=action_pool.action_max,
            noise_sigma=float(critic_cfg.get("rankq_noise_sigma", 0.15)),
        )
    if nested_rankq_enabled:
        action_pool = state.critic.core.action_pool
        full_same_state_rankq_actions = make_same_state_rankq_actions(
            batch.action_chunks,
            batch.execution_masks,
            action_mean=action_pool.action_mean,
            action_std=action_pool.action_std,
            action_min=action_pool.action_min,
            action_max=action_pool.action_max,
            mild_sigma=float(nested_rankq_settings["mild_sigma"]),
            strong_sigma=float(nested_rankq_settings["strong_sigma"]),
            use_random_negative=bool(
                nested_rankq_settings["use_random_negative"]
            ),
        )
    total_nested_valid = (
        total_success
        if bool(nested_rankq_settings["use_success_only"])
        else batch.batch_size
    )
    total_nested_valid_global = total_nested_valid
    nested_rankq_world_size = 1
    if nested_rankq_enabled and dist.is_available() and dist.is_initialized():
        nested_rankq_world_size = dist.get_world_size()
        global_count = torch.tensor(
            float(total_nested_valid),
            dtype=torch.float64,
            device=next(state.critic.parameters()).device,
        )
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        total_nested_valid_global = int(global_count.item())
    combined: dict[str, float] = {}
    rankq_parts: list[tuple[dict[str, float], int, int]] = []
    nested_rankq_parts: list[tuple[dict[str, float], int]] = []
    for index, start in enumerate(starts):
        stop = min(start + microbatch_size, batch.batch_size)
        indices = torch.arange(start, stop)
        weight = (stop - start) / total_size
        if isinstance(state.critic, MultiHeadScalarQCritic):
            update_fn = _multimodal_scalar_q_update
        elif (
            isinstance(state.critic, MultiHeadUdivlCritic)
            and state.critic.core.q_heads_per_member == 2
            and bool(critic_cfg.get("double_q_divl", False))
        ):
            update_fn = _multimodal_double_q_divl_update
        else:
            update_fn = _multimodal_critic_update
        microbatch = batch.index_select(indices)
        update_kwargs: dict[str, Any] = {}
        if rankq_enabled:
            micro_success = int(microbatch.successes.bool().sum().item())
            micro_failure = microbatch.batch_size - micro_success
            assert full_rankq_actions is not None
            update_kwargs = {
                "rankq_actions": full_rankq_actions.index_select(indices),
                "rankq_success_scale": (
                    micro_success / total_success if total_success else 0.0
                ),
                "rankq_failure_scale": (
                    micro_failure / total_failure if total_failure else 0.0
                ),
            }
        else:
            micro_success = 0
            micro_failure = 0
        if nested_rankq_enabled:
            micro_nested_valid = (
                int(microbatch.successes.bool().sum().item())
                if bool(nested_rankq_settings["use_success_only"])
                else microbatch.batch_size
            )
            assert full_same_state_rankq_actions is not None
            update_kwargs.update(
                {
                    "same_state_rankq_actions": (
                        full_same_state_rankq_actions.index_select(indices)
                    ),
                    "same_state_rankq_success_scale": (
                        ddp_global_valid_mean_scale(
                            micro_nested_valid,
                            total_nested_valid_global,
                            nested_rankq_world_size,
                        )
                    ),
                }
            )
        else:
            micro_nested_valid = 0
        metrics = update_fn(
            state,
            microbatch,
            config,
            zero_grad=index == 0,
            optimizer_step=index == len(starts) - 1,
            loss_scale=weight,
            **update_kwargs,
        )
        if rankq_enabled:
            rankq_parts.append((metrics, micro_success, micro_failure))
        if nested_rankq_enabled:
            nested_rankq_parts.append((metrics, micro_nested_valid))
        for key, value in metrics.items():
            if rankq_enabled and key.startswith("rankq/"):
                continue
            if key == "critic/rank_pair_count":
                combined[key] = combined.get(key, 0.0) + float(value)
            else:
                combined[key] = combined.get(key, 0.0) + weight * float(value)
    if rankq_enabled:
        success_keys = [
            "rankq/success_loss",
            "rankq/margin_pos_noisy",
            "rankq/margin_pos_very_noisy",
            "rankq/margin_pos_random",
            "rankq/margin_pos_permuted",
            "rankq/margin_noisy_very_noisy",
            "rankq/margin_very_noisy_random",
            "rankq/acc_pos_noisy",
            "rankq/acc_pos_very_noisy",
            "rankq/acc_pos_random",
            "rankq/acc_pos_permuted",
            "rankq/acc_noisy_very_noisy",
            "rankq/acc_very_noisy_random",
        ]
        failure_keys = [
            "rankq/failure_loss",
            "rankq/margin_failure_random",
            "rankq/acc_failure_random",
        ]
        for head in range(state.critic.ensemble_size):
            success_keys.append(f"rankq/success_loss_q{head + 1}")
            failure_keys.append(f"rankq/failure_loss_q{head + 1}")
        for key in success_keys:
            combined[key] = sum(
                metrics[key] * (count / total_success if total_success else 0.0)
                for metrics, count, _ in rankq_parts
            )
        for key in failure_keys:
            combined[key] = sum(
                metrics[key] * (count / total_failure if total_failure else 0.0)
                for metrics, _, count in rankq_parts
            )
        for head in range(state.critic.ensemble_size):
            combined[f"rankq/loss_q{head + 1}"] = (
                combined[f"rankq/success_loss_q{head + 1}"]
                + combined[f"rankq/failure_loss_q{head + 1}"]
            )
        combined["rankq/enabled"] = 1.0
        combined["rankq/success_fraction"] = total_success / batch.batch_size
        combined["rankq/failure_fraction"] = total_failure / batch.batch_size
        combined["rankq/success_count"] = float(total_success)
        combined["rankq/failure_count"] = float(total_failure)
        combined["rankq/loss"] = (
            combined["rankq/success_loss"] + combined["rankq/failure_loss"]
        )
        combined["rankq/lambda"] = float(critic_cfg.get("lambda_rank", 1.0))
        combined["critic_loss"] = (
            combined["q_loss"]
            + float(config.get("divl", {}).get("loss_weight", 1.0))
            * combined["divl_loss"]
            + float(critic_cfg.get("rank_loss_weight", 0.1))
            * combined.get("critic/rank_loss", 0.0)
            + combined["rankq/lambda"] * combined["rankq/loss"]
        )
    if nested_rankq_enabled:
        nested_metric_keys = [
            "rankq_loss",
            "rankq_mild_loss",
            "rankq_strong_loss",
            "rankq_random_loss",
            "rankq_total_raw_relation_loss",
            "rankq_total_relation_weighted_loss",
            "rankq_logged_vs_mild_acc",
            "rankq_mild_vs_strong_acc",
            "rankq_strong_vs_random_acc",
            "rankq_raw10_unanimous_logged_vs_mild",
            "rankq_raw10_unanimous_mild_vs_strong",
            "rankq_raw10_unanimous_strong_vs_random",
        ]
        for relation in (
            "logged_vs_mild",
            "mild_vs_strong",
            "strong_vs_random",
        ):
            nested_metric_keys.extend(
                [
                    f"rankq_{relation}_softplus_slope_mean",
                    f"rankq_{relation}_softplus_slope_median",
                    f"rankq_{relation}_softplus_slope_p90",
                    f"rankq_{relation}_softplus_slope_below_1e2",
                    f"rankq_{relation}_weighted_loss",
                ]
            )
        for key in nested_metric_keys:
            local_average = sum(
                part[key]
                * (count / total_nested_valid if total_nested_valid else 0.0)
                for part, count in nested_rankq_parts
            )
            # _distributed_mean_metrics subsequently divides by world size.
            # Pre-scale the local mean by its share of the global valid count
            # so logged metrics and gradients both represent one global
            # success-transition average.
            combined[key] = (
                local_average
                * ddp_global_valid_mean_scale(
                    total_nested_valid,
                    total_nested_valid_global,
                    nested_rankq_world_size,
                )
            )
        combined["rankq_enabled"] = 1.0
        combined["rankq_lambda"] = float(nested_rankq_settings["lambda_rank"])
        for relation, config_key in (
            ("logged_vs_mild", "logged_mild_weight"),
            ("mild_vs_strong", "mild_strong_weight"),
            ("strong_vs_random", "strong_random_weight"),
        ):
            combined[f"rankq_{relation}_weight"] = float(
                nested_rankq_settings[config_key]
            )
        combined["rankq_success_count"] = float(total_nested_valid_global)
        combined["rankq_weighted_loss"] = (
            combined["rankq_lambda"] * combined["rankq_loss"]
        )
        combined["rankq_total_relation_weighted_loss"] = combined["rankq_loss"]
        combined["rankq_total_lambda_weighted_loss"] = combined[
            "rankq_weighted_loss"
        ]
        combined["divl_objective_loss"] = (
            combined["q_loss"]
            + float(config.get("divl", {}).get("loss_weight", 1.0))
            * combined["divl_loss"]
        )
        combined["critic_loss"] = (
            combined["divl_objective_loss"] + combined["rankq_weighted_loss"]
        )
        combined["total_critic_loss"] = combined["critic_loss"]
    combined["critic_grad_norm"] = float(metrics["critic_grad_norm"])
    combined["critic_grad_norm_preclip"] = float(
        metrics.get("critic_grad_norm_preclip", metrics["critic_grad_norm"])
    )
    combined["critic_grad_norm_postclip"] = float(
        metrics.get(
            "critic_grad_norm_postclip",
            min(
                float(metrics["critic_grad_norm"]),
                float(critic_cfg.get("max_grad_norm", 10.0)),
            ),
        )
    )
    combined["critic_grad_clip_scale"] = float(
        metrics.get(
            "critic_grad_clip_scale",
            min(
                1.0,
                float(critic_cfg.get("max_grad_norm", 10.0))
                / max(float(metrics["critic_grad_norm"]), 1.0e-12),
            ),
        )
    )
    combined["grad_norm_preclip"] = combined["critic_grad_norm_preclip"]
    combined["grad_norm_postclip"] = combined["critic_grad_norm_postclip"]
    combined["gradient_clipped"] = float(combined["critic_grad_clip_scale"] < 1.0)
    combined["learning_rate"] = float(state.critic_optimizer.param_groups[0]["lr"])
    combined["total_loss"] = float(combined.get("total_critic_loss", combined["critic_loss"]))
    combined["v_loss"] = float(combined["divl_loss"])
    combined["target_updated"] = float(metrics["target_updated"])
    combined["effective_batch_size"] = float(batch.batch_size)
    combined["microbatch_size"] = float(microbatch_size)
    combined["gradient_accumulation_steps"] = float(len(starts))
    return combined


@torch.no_grad()
def conservative_advantages_for_candidates(
    state: OGPOTrainState,
    observations: torch.Tensor,
    candidate_flat_actions: torch.Tensor,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    chi2_ratio: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    uncertainty_cfg = config.get("uncertainty", {})
    actor_cfg = config.get("actor", {})
    divl_cfg = config.get("divl", {})
    divl_enabled = bool(divl_cfg.get("enabled", True))
    advantage_mode = _effective_advantage_mode(config)
    batch_size, group_size, action_flat = candidate_flat_actions.shape
    chunks = candidate_flat_actions.reshape(batch_size, group_size, batch.generated_horizon, batch.action_dim)
    flat_obs = observations[:, None, :].expand(batch_size, group_size, observations.shape[-1]).reshape(
        batch_size * group_size, -1
    )
    flat_chunks = chunks.reshape(batch_size * group_size, batch.generated_horizon, batch.action_dim)
    critic_mask = _critic_execution_mask(batch, config)
    flat_masks = critic_mask[:, None, :].expand(batch_size, group_size, batch.generated_horizon).reshape(
        batch_size * group_size, batch.generated_horizon
    )
    if isinstance(state.critic, (MultiHeadUdivlCritic, MultiHeadScalarQCritic)):
        features = state.critic.encode_state(batch)
        grouped_features = StateFeatures(
            readout=features.readout.repeat_interleave(group_size, dim=0)
        )
        use_raw_q_ensemble = (
            isinstance(state.critic, MultiHeadUdivlCritic)
            and str(actor_cfg.get("q_ensemble_source", "clipped_pairs")) == "raw_heads"
        )
        if use_raw_q_ensemble:
            q_flat = state.critic.raw_q_ensemble_from_features(
                grouped_features, flat_chunks, flat_masks
            )
            behavior_q_members = state.critic.raw_q_ensemble_from_features(
                features, batch.action_chunks, critic_mask
            )
        else:
            q_flat = state.critic.q_from_features(grouped_features, flat_chunks, flat_masks)
            behavior_q_members = state.critic.q_from_features(
                features, batch.action_chunks, critic_mask
            )
        probs = (
            F.softmax(state.critic.value_logits_from_features(features), dim=-1)
            if isinstance(state.critic, MultiHeadUdivlCritic)
            and not (
                use_raw_q_ensemble
                and advantage_mode in {"conservative", "ca", "chi_po"}
            )
            else None
        )
    else:
        use_raw_q_ensemble = False
        q_flat = state.critic(flat_obs, flat_chunks, flat_masks)
        behavior_q_members = state.critic(observations, batch.action_chunks, critic_mask)
        probs = state.divl(observations) if divl_enabled else None
    q_values = q_flat.reshape(q_flat.shape[0], batch_size, group_size)
    _, q_std = ensemble_mean_std(q_values)
    if divl_enabled and probs is not None:
        divl_stats = divl_quantile_values(
            probs,
            state.support,
            alpha_min=float(divl_cfg.get("alpha_min", 0.5)),
            alpha_max=float(divl_cfg.get("alpha_max", 0.8)),
            entropy_temperature=float(divl_cfg.get("entropy_temperature", 1.0)),
            alpha_mode=str(divl_cfg.get("alpha_mode", "linear")),
            use_adaptive_quantile=bool(divl_cfg.get("use_adaptive_quantile", True)),
            interpolate_quantile=bool(divl_cfg.get("interpolate_quantile", True)),
            adaptive_tau_mode=_divl_adaptive_tau_enabled(divl_cfg),
            tau_base=float(divl_cfg.get("tau_base", 0.6)),
            entropy_coefficient=float(divl_cfg.get("entropy_coefficient", 0.3)),
            tau_min=float(divl_cfg.get("tau_min", 0.0)),
            tau_max=float(divl_cfg.get("tau_max", 1.0)),
        )
        # The clean CA/ChiPO actor path intentionally does not use DIVL V as
        # an actor baseline.  We still evaluate entropy for diagnostics and
        # optional legacy safety gates; only the legacy modes below consume
        # ``value_baselines``.
        value_baselines = (
            None
            if advantage_mode in {"conservative", "ca", "chi_po"}
            else divl_stats.quantile_value
        )
        entropy_state = divl_stats.entropy.mean(dim=0)
    elif divl_enabled:
        # Clean raw-Q OGPO never reads categorical V.  V remains an internal
        # DIVL training head and contributes neither baseline nor ranking.
        value_baselines = None
        entropy_state = observations.new_zeros(batch_size)
    else:
        value_baselines = behavior_q_members
        entropy_state = observations.new_zeros(batch_size)
    chi2_pessimism: Chi2PessimismStats | None = None
    member_advantages: torch.Tensor | None = None
    ca_base_advantage: torch.Tensor | None = None
    ca_positive_ratio = 0.0
    ca_negative_ratio = 0.0
    ca_zero_ratio = 0.0
    if advantage_mode in {"conservative", "ca"}:
        # Main offline OGPO path centers each configured critic signal over
        # the sampled candidate group.  The clean 5-pair config supplies all
        # ten raw heads; legacy configs retain clipped-pair behavior.
        cons, member_advantages, stats = group_relative_conservative_advantage(
            q_values,
            positive_margin=float(uncertainty_cfg.get("positive_margin", 0.0)),
            negative_margin=float(uncertainty_cfg.get("negative_margin", 0.0)),
        )
        positive_ratio = stats.positive_consensus_ratio
        negative_ratio = stats.negative_consensus_ratio
        zero_ratio = stats.zero_ratio
        sign_agreement_ratio = stats.sign_agreement_ratio
        ca_base_advantage = cons.detach().clone()
        ca_positive_ratio = positive_ratio
        ca_negative_ratio = negative_ratio
        ca_zero_ratio = zero_ratio
    elif advantage_mode == "sign_consensus":
        cons, stats = sign_consensus_advantage(
            q_values,
            value_baselines,
            positive_margin=float(uncertainty_cfg.get("positive_margin", 0.0)),
            negative_margin=float(uncertainty_cfg.get("negative_margin", 0.0)),
        )
        positive_ratio = stats.positive_consensus_ratio
        negative_ratio = stats.negative_consensus_ratio
        zero_ratio = stats.zero_ratio
        sign_agreement_ratio = stats.sign_agreement_ratio
    elif advantage_mode == "lcb":
        calibrated_scale = state.conformal_scale if bool(uncertainty_cfg.get("use_conformal", False)) else 1.0
        cons = lcb_advantage(
            q_values,
            value_baselines,
            kappa=float(uncertainty_cfg.get("lcb_kappa", 1.0)),
            calibrated_scale=calibrated_scale,
        )
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    elif advantage_mode == "group_normalization":
        cons = group_normalized_advantage(q_values)
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    elif advantage_mode == "group_mean":
        q_mean = q_values.mean(dim=0)
        cons = q_mean - q_mean.mean(dim=-1, keepdim=True)
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    elif advantage_mode == "chi_po":
        if chi2_ratio is None:
            raise ValueError("Flash-chiPO advantage requires a selected-transition χ² ratio")
        chi2_cfg = _chi2_config(config)
        advantage_clip_value = chi2_cfg.get("advantage_clip")
        cons, chi2_pessimism = chi2_pessimistic_advantage(
            q_values,
            chi2_ratio,
            beta_base=float(chi2_cfg.get("beta_base", chi2_cfg.get("beta", 0.1))),
            q_std_target=float(chi2_cfg.get("q_std_target", 1.0)),
            ensemble_alpha=float(chi2_cfg.get("ensemble_alpha", 5.0)),
            normalize_group=bool(chi2_cfg.get("normalize_group", False)),
            advantage_clip=(
                None if advantage_clip_value is None else float(advantage_clip_value)
            ),
        )
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    elif advantage_mode == "scalar_q":
        behavior_q = behavior_q_members.mean(dim=0)
        cons = q_values.mean(dim=0) - behavior_q.unsqueeze(-1)
        positive_ratio = float((cons > 0).float().mean().item())
        negative_ratio = float((cons < 0).float().mean().item())
        zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - zero_ratio
    else:
        raise ValueError(f"unsupported actor.advantage_mode={advantage_mode!r}")
    final_positive_ratio = positive_ratio
    final_negative_ratio = negative_ratio
    final_zero_ratio = zero_ratio
    advantage_clip = float(actor_cfg.get("advantage_clip", 5.0))
    if advantage_mode in {"group_mean", "chi_po", "conservative", "ca"}:
        normalized = cons
        lambda_abs = 1.0
    elif advantage_mode == "group_normalization":
        normalized = cons.clamp(-advantage_clip, advantage_clip)
        lambda_abs = 0.0
    else:
        state.running_mad.update(cons, ignore_zero=True)
        normalized_abs = state.running_mad.normalize(cons, advantage_clip)
        lambda_abs = scheduled_lambda_abs(
            state.step,
            start=float(actor_cfg.get("lambda_abs_start", 1.0)),
            end=float(actor_cfg.get("lambda_abs_end", 1.0)),
            warmup_steps=int(actor_cfg.get("lambda_abs_warmup_steps", 0)),
        )
        if lambda_abs < 1.0:
            group_advantage = group_normalized_advantage(q_values).clamp(-advantage_clip, advantage_clip)
            normalized = lambda_abs * normalized_abs + (1.0 - lambda_abs) * group_advantage
        else:
            normalized = normalized_abs
    chi2_cfg = _chi2_config(config) if ogpo_variant(config) in {"chi2", "ca_chi2"} else {}
    # ``ca_chi2`` first constructs CA above, then applies the official-style
    # χ² drift shaping.  It must never silently fall back to Q-mean advantage.
    if ogpo_variant(config) == "ca_chi2":
        if chi2_ratio is None:
            raise ValueError("CA+ChiPO requires a full-chain χ² ratio")
        chipo_advantage_fn = apply_chipo_to_ca_advantage
        if actor_cfg.get("role_data_parallel", False):
            from .actor_role_dp import global_chipo
            chipo_advantage_fn = global_chipo
        cons, chi2_pessimism = chipo_advantage_fn(
            cons,
            q_values,
            chi2_ratio,
            beta_base=float(chi2_cfg.get("beta_base", chi2_cfg.get("beta", 0.1))),
            q_std_target=float(chi2_cfg.get("q_std_target", 1.0)),
            ensemble_alpha=float(chi2_cfg.get("ensemble_alpha", 5.0)),
            r_max=float(chi2_cfg.get("r_max", 10.0)),
            normalize_group=bool(chi2_cfg.get("normalize_group", False)),
        )
        normalized = cons
        lambda_abs = 1.0
        final_positive_ratio = float((cons > 0).float().mean().item())
        final_negative_ratio = float((cons < 0).float().mean().item())
        final_zero_ratio = float((cons == 0).float().mean().item())
        sign_agreement_ratio = 1.0 - final_zero_ratio
    apply_project_safety_gates = (
        ogpo_variant(config) not in {"chi2", "ca_chi2"}
        or bool(
        chi2_cfg.get("apply_project_safety_gates", False)
        )
    )
    if apply_project_safety_gates:
        state_weights = state_entropy_weight(
            entropy_state, float(uncertainty_cfg.get("entropy_scale", 0.0))
        ).unsqueeze(-1)
        final_adv = normalized * state_weights
        consensus_per_state = (cons != 0).to(final_adv.dtype).mean(dim=1)
        entropy_skip = entropy_state > float(
            uncertainty_cfg.get("entropy_skip_threshold", 1.1)
        )
        consensus_skip = consensus_per_state < float(
            uncertainty_cfg.get("consensus_skip_threshold", -1.0)
        )
        state_skip = entropy_skip | consensus_skip
        final_adv = torch.where(
            state_skip.unsqueeze(-1), torch.zeros_like(final_adv), final_adv
        )
    else:
        state_weights = torch.ones_like(normalized)
        final_adv = normalized
        entropy_skip = torch.zeros(batch_size, dtype=torch.bool, device=normalized.device)
        consensus_skip = torch.zeros_like(entropy_skip)
        state_skip = torch.zeros_like(entropy_skip)
    support_distance = torch.zeros_like(final_adv)
    support_weights = torch.ones_like(final_adv)
    if apply_project_safety_gates and bool(uncertainty_cfg.get("use_support_weight", False)):
        behavior_flat = batch.action_chunks.reshape(batch_size, -1)[:, None, :]
        support_distance = (candidate_flat_actions - behavior_flat).pow(2).mean(dim=-1).sqrt()
        threshold = uncertainty_cfg.get("support_threshold")
        support_weights = support_weight(
            q_std * (state.conformal_scale if bool(uncertainty_cfg.get("use_conformal", False)) else 1.0),
            support_distance,
            lambda_epi=float(uncertainty_cfg.get("lambda_epi", 0.0)),
            lambda_support=float(uncertainty_cfg.get("lambda_support", 0.0)),
            support_threshold=float(threshold) if threshold is not None else None,
        )
        final_adv = final_adv * support_weights
    diagnostics: dict[str, Any] = {
        "ogpo_variant": ogpo_variant(config),
        "advantage_mode": advantage_mode,
        "positive_consensus_ratio": positive_ratio,
        "negative_consensus_ratio": negative_ratio,
        "zero_disagreement_ratio": zero_ratio,
        "disagreement_zero_advantage_ratio": zero_ratio,
        "sign_agreement_ratio": sign_agreement_ratio,
        "ca_positive_consensus_ratio": ca_positive_ratio,
        "ca_negative_consensus_ratio": ca_negative_ratio,
        "ca_disagreement_zero_ratio": ca_zero_ratio,
        "final_positive_ratio": final_positive_ratio,
        "final_negative_ratio": final_negative_ratio,
        "final_zero_advantage_ratio": final_zero_ratio,
        "final_adv_abs_mean": float(final_adv.abs().mean().item()),
        "final_adv_abs_median": float(torch.quantile(final_adv.abs(), 0.5).item()),
        "advantage_mad": state.running_mad.value,
        "lambda_abs": lambda_abs,
        "advantage_mean": float(final_adv.mean().item()),
        "advantage_std": float(final_adv.std(unbiased=False).item()),
        "advantage_clip_fraction": float(
            (normalized.abs() >= advantage_clip).float().mean().item()
        ),
        "conservative_advantage_abs_mean": float(cons.abs().mean().item()),
        "conservative_advantage_abs_median": float(
            torch.quantile(cons.abs(), 0.5).item()
        ),
        "state_entropy": float(entropy_state.mean().item()),
        "state_entropy_weight": float(state_weights.mean().item()),
        "state_skip_fraction": float(state_skip.float().mean().item()),
        "entropy_skip_fraction": float(entropy_skip.float().mean().item()),
        "consensus_skip_fraction": float(consensus_skip.float().mean().item()),
        "candidate_ensemble_disagreement": float(q_std.mean().item()),
        "q_ensemble_size": float(q_values.shape[0]),
        "q_ensemble_mean": float(q_values.mean().item()),
        "q_ensemble_min": float(q_values.min().item()),
        "q_ensemble_std": float(q_values.std(unbiased=False).item()),
        "support_distance_mean": float(support_distance.mean().item()),
        "support_weight_mean": float(support_weights.mean().item()),
    }
    if use_raw_q_ensemble and q_values.shape[0] % 2 == 0:
        raw_pairs = q_values.reshape(q_values.shape[0] // 2, 2, batch_size, group_size)
        diagnostics["mean_pair_disagreement"] = float(
            (raw_pairs[:, 0] - raw_pairs[:, 1]).abs().mean().item()
        )
    correlation_period = int(actor_cfg.get("q_correlation_log_period", 100))
    if (
        q_values.shape[0] > 1
        and correlation_period > 0
        and state.actor_step % correlation_period == 0
    ):
        matrix = q_values.float().reshape(q_values.shape[0], -1)
        centered = matrix - matrix.mean(dim=1, keepdim=True)
        norms = centered.square().sum(dim=1).sqrt().clamp_min(1e-12)
        correlation = centered @ centered.t() / (norms[:, None] * norms[None, :])
        off_diagonal = ~torch.eye(
            correlation.shape[0], dtype=torch.bool, device=correlation.device
        )
        diagnostics["mean_off_diagonal_q_correlation"] = float(
            correlation[off_diagonal].mean().item()
        )
    if member_advantages is not None:
        for member in range(member_advantages.shape[0]):
            diagnostics[f"member_advantage_{member}_mean"] = float(
                member_advantages[member].mean().item()
            )
            diagnostics[f"member_advantage_{member}_std"] = float(
                member_advantages[member].std(unbiased=False).item()
            )
    ca_for_metrics = ca_base_advantage if ca_base_advantage is not None else cons
    diagnostics["ca_advantage_mean"] = float(ca_for_metrics.mean().item())
    diagnostics["ca_advantage_std"] = float(ca_for_metrics.std(unbiased=False).item())
    diagnostics["ca_advantage_abs_mean"] = float(ca_for_metrics.abs().mean().item())
    diagnostics["ca_advantage_abs_median"] = float(
        torch.quantile(ca_for_metrics.abs(), 0.5).item()
    )
    diagnostics["ca_nonzero_fraction"] = float(
        (ca_for_metrics != 0.0).float().mean().item()
    )
    diagnostics["ca_zero_advantage_fraction"] = float(
        (ca_for_metrics == 0.0).float().mean().item()
    )
    diagnostics["ca_adv_mean"] = diagnostics["ca_advantage_mean"]
    diagnostics["ca_adv_std"] = diagnostics["ca_advantage_std"]
    diagnostics["ca_positive_ratio"] = ca_positive_ratio
    diagnostics["ca_negative_ratio"] = ca_negative_ratio
    diagnostics["ca_disagreement_ratio"] = ca_zero_ratio
    diagnostics["final_actor_advantage_mean"] = float(cons.mean().item())
    diagnostics["final_actor_advantage_std"] = float(cons.std(unbiased=False).item())
    diagnostics["final_adv_mean"] = float(final_adv.mean().item())
    diagnostics["final_adv_std"] = float(final_adv.std(unbiased=False).item())
    diagnostics["ca_chipo_filtered_ca_ratio"] = (
        float(((ca_for_metrics != 0.0) & (cons == 0.0)).float().mean().item())
        if ca_base_advantage is not None and ogpo_variant(config) == "ca_chi2"
        else 0.0
    )
    if chi2_pessimism is not None:
        diagnostics.update(
            {
                "chi2_beta": chi2_pessimism.beta,
                "chi_po_beta": chi2_pessimism.beta,
                "chi2_q_ensemble_std": chi2_pessimism.q_ensemble_std,
                "chi2_pessimism_weight_mean": chi2_pessimism.pessimism_weight_mean,
                "chi2_pessimism_weight_max": chi2_pessimism.pessimism_weight_max,
                "chi_pessimism_weight_min": chi2_pessimism.pessimism_weight_min,
                "chi2_q_target_mean": chi2_pessimism.q_target_mean,
                "chi2_q_target_min": chi2_pessimism.q_target_min,
                "chi2_advantage_penalty_mean": chi2_pessimism.penalty_mean,
                "chi_adv_mean": chi2_pessimism.chi_advantage_mean,
                "chi_adv_std": chi2_pessimism.chi_advantage_std,
                "chi_advantage_mean": chi2_pessimism.chi_advantage_mean,
                "chi_advantage_std": chi2_pessimism.chi_advantage_std,
                "chi_q_mean": chi2_pessimism.q_mean,
                "chi_q_min": chi2_pessimism.q_min,
                "chi_q_target_mean": chi2_pessimism.q_target_mean,
                "chi_q_penalized_mean": chi2_pessimism.q_penalized_mean,
                "chi_penalized_q_mean": chi2_pessimism.q_penalized_mean,
                "chi_pessimism_weight_min": chi2_pessimism.pessimism_weight_min,
                "chi_pessimism_weight_max": chi2_pessimism.pessimism_weight_max,
                "ca_chipo_gate_positive_ratio": chi2_pessimism.gate_positive_ratio,
                "ca_chipo_gate_negative_ratio": chi2_pessimism.gate_negative_ratio,
                "ca_chipo_gate_zero_ratio": chi2_pessimism.gate_zero_ratio,
                "ca_chipo_sign_conflict_ratio": chi2_pessimism.gate_sign_conflict_ratio,
                "chi2_project_safety_gates": float(apply_project_safety_gates),
            }
        )
    return final_adv, diagnostics


def _normalized_state_entropy(state: OGPOTrainState, batch: ChunkBatch) -> torch.Tensor:
    if isinstance(state.critic, MultiHeadUdivlCritic):
        features = state.critic.encode_state(batch)
        probs = F.softmax(state.critic.value_logits_from_features(features), dim=-1)
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        return batch.observations.new_zeros(batch.batch_size)
    else:
        assert state.divl is not None
        probs = state.divl(batch.observations)
    entropy = probs.clamp_min(1e-8).mul(probs.clamp_min(1e-8).log()).sum(-1).neg()
    return entropy / torch.log(torch.tensor(probs.shape[-1], dtype=probs.dtype, device=probs.device))


def _success_subset(batch: ChunkBatch) -> ChunkBatch | None:
    indices = torch.nonzero(batch.successes.bool(), as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    return batch.index_select(indices)


def _actor_regularization_loss(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None = None,
    success_batch: ChunkBatch | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    regularization_cfg = config.get("regularization", {})
    policy_parameter = next(state.policy.parameters())
    zero = policy_parameter.new_zeros(())
    loss = zero
    metrics = {
        "fm_anchor_loss": 0.0,
        "success_buffer_loss": 0.0,
        "action_smoothness": 0.0,
    }

    lambda_fm = float(regularization_cfg.get("lambda_fm", 0.1))
    if lambda_fm > 0.0:
        source = fm_batch if fm_batch is not None else batch
        source = source.to(policy_parameter.device)
        source_condition = _policy_condition(state.policy, source)
        fm = flow_matching_anchor_loss(
            state.policy,
            source_condition,
            state.policy.action_chunks_to_flow(source).reshape(source.batch_size, -1),
        )
        loss = loss + lambda_fm * fm.loss
        metrics["fm_anchor_loss"] = fm.diagnostics["fm_anchor_loss"]

    lambda_success = float(regularization_cfg.get("lambda_success", 0.0))
    if lambda_success > 0.0:
        success_source = success_batch if success_batch is not None else _success_subset(batch)
    else:
        success_source = None
    if success_source is not None:
        success_source = success_source.to(next(state.policy.parameters()).device)
        success_condition = _policy_condition(state.policy, success_source)
        success = success_buffer_loss(
            state.policy,
            success_condition,
            state.policy.action_chunks_to_flow(success_source).reshape(success_source.batch_size, -1),
        )
        loss = loss + lambda_success * success.loss
        metrics["success_buffer_loss"] = success.diagnostics["success_buffer_loss"]
    lambda_smooth = float(regularization_cfg.get("lambda_smooth", 0.0))
    if lambda_smooth > 0.0:
        smooth_condition = _policy_condition(state.policy, batch)
        smooth_rollout = state.policy.rollout(smooth_condition, group_size=1)
        smooth_endpoint = state.policy.flat_actions_to_environment(smooth_rollout.endpoint, smooth_condition)
        smooth_chunks = smooth_endpoint.reshape(batch.batch_size, batch.generated_horizon, batch.action_dim)
        gripper_mask_value = regularization_cfg.get("gripper_mask")
        gripper_mask = None
        if gripper_mask_value is not None:
            gripper_mask = torch.as_tensor(gripper_mask_value, dtype=torch.bool, device=smooth_chunks.device)
        smooth = action_smoothness_loss(
            smooth_chunks,
            gripper_mask=gripper_mask,
            eta=float(regularization_cfg.get("smooth_acceleration_eta", 0.1)),
        )
        loss = loss + lambda_smooth * smooth
        metrics["action_smoothness"] = float(smooth.detach().item())
    return loss, metrics


@torch.no_grad()
def _policy_l2_lag(policy: torch.nn.Module, old_policy: torch.nn.Module) -> float:
    """Cheap sampled trainable-parameter distance for policy diagnostics."""
    total = 0.0
    count = 0
    if isinstance(policy, PI05PytorchFlowPolicy) and isinstance(
        old_policy, PI05PytorchFlowPolicy
    ):
        snapshot = getattr(old_policy, "_backend_parameter_snapshot", None)
        backend_distance_sampled = False
        if snapshot:
            current_backend = dict(policy.backend.named_parameters())
            # A bounded sample avoids scanning multi-billion-parameter
            # backends merely for logging while still tracking real full-
            # finetune drift (residual/std can be disabled in the clean path).
            for name in sorted(snapshot)[:8]:
                current = current_backend[name].detach().reshape(-1)
                reference = snapshot[name].detach().reshape(-1)
                sample_count = min(4096, current.numel())
                diff = current[:sample_count].float() - reference[:sample_count].float()
                total += float(diff.square().sum().item())
                count += sample_count
            backend_distance_sampled = True
        elif old_policy.backend is not policy.backend:
            current_backend = dict(policy.backend.named_parameters())
            old_backend = dict(old_policy.backend.named_parameters())
            for name in sorted(policy._backend_trainable_names)[:8]:
                current = current_backend[name].detach().reshape(-1)
                reference = old_backend[name].detach().reshape(-1)
                sample_count = min(4096, current.numel())
                diff = current[:sample_count].float() - reference[:sample_count].to(
                    device=current.device, dtype=torch.float32
                )
                total += float(diff.square().sum().item())
                count += sample_count
            backend_distance_sampled = True
    else:
        backend_distance_sampled = False
    old_parameters = dict(old_policy.named_parameters())
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if backend_distance_sampled and name.startswith("backend."):
            continue
        old_param = old_parameters.get(name)
        if old_param is None:
            continue
        diff = param.detach() - old_param.detach().to(param.device)
        total += float(diff.pow(2).sum().item())
        count += diff.numel()
    return (total / max(1, count)) ** 0.5


@torch.no_grad()
def _policy_entropy(policy: torch.nn.Module) -> float:
    """Report the diagonal Gaussian transition entropy when available."""
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        return 0.0
    value = log_std.detach().float()
    return float((value + 0.5 * math.log(2.0 * math.pi * math.e)).sum().item())


def _clone_to_cpu(value: Any) -> Any:
    """Deep-copy a training-state payload without retaining accelerator storage."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _clone_to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: _clone_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_device(item, device) for item in value)
    return copy.deepcopy(value)


def _capture_actor_rollback_state(
    state: OGPOTrainState,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Rollback snapshot for a single post-update PyTorch KL transaction."""
    if isinstance(state.policy, PI05PytorchFlowPolicy):
        policy_state = state.policy.adapter_state_dict(
            include_backend=True,
            device=device,
        )
        policy_format = "pi05_pytorch"
    else:
        policy_state = _clone_to_device(state.policy.state_dict(), device)
        policy_format = "state_dict"
    return {
        "policy_format": policy_format,
        "policy": policy_state,
        "optimizer": _clone_to_device(state.actor_optimizer.state_dict(), device),
        "device": str(device),
    }


def _restore_actor_rollback_state(
    state: OGPOTrainState,
    snapshot: dict[str, Any],
) -> None:
    if snapshot["policy_format"] == "pi05_pytorch":
        if not isinstance(state.policy, PI05PytorchFlowPolicy):
            raise TypeError("PI0.5 rollback snapshot requires PI05PytorchFlowPolicy")
        state.policy.load_adapter_state_dict(snapshot["policy"])
    elif snapshot["policy_format"] == "state_dict":
        state.policy.load_state_dict(snapshot["policy"])
    else:
        raise ValueError(f"unknown actor rollback format {snapshot['policy_format']!r}")
    state.actor_optimizer.load_state_dict(snapshot["optimizer"])


@torch.no_grad()
def _selected_transition_reference_kl(
    state: OGPOTrainState,
    *,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    condition: Any,
    microbatch_size: int,
) -> float:
    """Mean current/reference KL over the selected Flash transitions."""
    if microbatch_size <= 0:
        raise ValueError("actor.kl_eval_microbatch_size must be positive")
    total = 0.0
    count = x_t.shape[0]
    for start in range(0, count, microbatch_size):
        stop = min(start + microbatch_size, count)
        indices = torch.arange(start, stop, device=x_t.device)
        micro_condition = _condition_index_select(condition, indices)
        current_mean = state.policy.transition_mean(
            x_t[start:stop], micro_condition, timestep[start:stop]
        )
        current_log_std = state.policy.transition_log_std(
            x_t[start:stop], timestep[start:stop]
        )
        reference_mean = state.reference_policy.transition_mean(
            x_t[start:stop], micro_condition, timestep[start:stop]
        )
        reference_log_std = state.reference_policy.transition_log_std(
            x_t[start:stop], timestep[start:stop]
        )
        total += float(
            gaussian_kl_diag(
                current_mean,
                current_log_std,
                reference_mean,
                reference_log_std,
            ).sum().item()
        )
    return total / max(count, 1)


def _post_update_kl_action(actor_cfg: dict[str, Any]) -> str:
    """Legacy ``reject_update_on_kl`` implies a CPU rollback transaction."""
    action = actor_cfg.get("post_update_kl_action")
    if action is None:
        action = "rollback_cpu" if bool(actor_cfg.get("reject_update_on_kl", False)) else "monitor"
    action = str(action).lower()
    if action not in {"monitor", "stop", "rollback_cpu"}:
        raise ValueError(
            "actor.post_update_kl_action must be monitor, stop, or rollback_cpu"
        )
    return action


def _condition_index_select(condition: Any, indices: torch.Tensor) -> Any:
    if isinstance(condition, PI05FlowCondition):
        return condition.index_select(indices)
    return condition.index_select(0, indices)


def _condition_to_device(condition: Any, device: torch.device | str) -> Any:
    if isinstance(condition, PI05FlowCondition):
        return condition.to(device)
    return condition.to(device)


def _rollout_to_device(rollout: Any, device: torch.device | str) -> Any:
    updates = {}
    for field in fields(rollout):
        value = getattr(rollout, field.name)
        updates[field.name] = value.to(device) if torch.is_tensor(value) else value
    return replace(rollout, **updates)


def _policy_device(policy: OpenPIStochasticFlowPolicy) -> torch.device:
    return next(policy.parameters()).device


def _policy_kl_device_aware(
    policy: OpenPIStochasticFlowPolicy,
    other: OpenPIStochasticFlowPolicy,
    x_t: torch.Tensor,
    condition: Any,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Exact transition KL when frozen policy roles occupy distinct GPUs."""
    policy_device = _policy_device(policy)
    other_device = _policy_device(other)
    mean_p, std_p, mask_p = policy.transition_parameters(
        x_t.to(policy_device),
        _condition_to_device(condition, policy_device),
        timestep.to(policy_device),
    )
    mean_q, std_q, mask_q = other.transition_parameters(
        x_t.to(other_device),
        _condition_to_device(condition, other_device),
        timestep.to(other_device),
    )
    mean_q = mean_q.to(policy_device)
    std_q = std_q.to(policy_device)
    mask_q = mask_q.to(policy_device)
    safe_p = torch.where(std_p > 0.0, std_p, torch.ones_like(std_p))
    safe_q = torch.where(std_q > 0.0, std_q, torch.ones_like(std_q))
    value = gaussian_kl_diag(mean_p, safe_p.log(), mean_q, safe_q.log())
    if policy.sde_mode == "ogpo_constant_corrected" or other.sde_mode == "ogpo_constant_corrected":
        value = value * (mask_p & mask_q).to(value.dtype)
    return value


@torch.no_grad()
def _full_chain_log_probs_no_grad(
    policy: OpenPIStochasticFlowPolicy,
    rollout: Any,
    condition: Any,
    *,
    microbatch_size: int,
) -> torch.Tensor:
    """Re-evaluate all stochastic transitions as ``[trajectory, step]``."""
    total = rollout.states.shape[0]
    parts = []
    for start in range(0, total, microbatch_size):
        stop = min(start + microbatch_size, total)
        indices = torch.arange(start, stop, device=rollout.states.device)
        micro_condition = _condition_index_select(condition, indices)
        step_parts = []
        for flow_step in range(policy.num_steps):
            step_parts.append(
                policy.log_prob(
                    rollout.next_states[start:stop, flow_step],
                    rollout.states[start:stop, flow_step],
                    micro_condition,
                    rollout.timesteps[start:stop, flow_step],
                )
            )
        parts.append(torch.stack(step_parts, dim=1))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _full_chain_policy_kl(
    policy: OpenPIStochasticFlowPolicy,
    other: OpenPIStochasticFlowPolicy,
    rollout: Any,
    condition: Any,
    *,
    microbatch_size: int,
    normalizer: float,
) -> float:
    """Mean normalized chain KL on a fixed denoising trajectory."""
    total = rollout.states.shape[0]
    value = 0.0
    for start in range(0, total, microbatch_size):
        stop = min(start + microbatch_size, total)
        indices = torch.arange(start, stop, device=rollout.states.device)
        micro_condition = _condition_index_select(condition, indices)
        chain_kl = rollout.states.new_zeros(stop - start)
        for flow_step in range(policy.num_steps):
            chain_kl = chain_kl + _policy_kl_device_aware(
                policy,
                other,
                rollout.states[start:stop, flow_step],
                micro_condition,
                rollout.timesteps[start:stop, flow_step],
            )
        value += float((chain_kl / float(normalizer)).sum().item())
    return value / max(total, 1)


@torch.no_grad()
def _sde_rollout_metrics(
    policy: OpenPIStochasticFlowPolicy,
    rollout: Any,
    condition: Any,
) -> dict[str, float]:
    """Summarize the stored rollout without additional actor forwards."""
    del condition
    std_values = []
    stochastic_count = 0.0
    # Official full-chain constant SDE includes one initial N(0,I) prior
    # contribution in addition to K-1 stochastic transition likelihoods.
    logprob_count = 1.0 if policy.sde_mode == "ogpo_constant_corrected" else 0.0
    for step in range(policy.num_steps):
        x_t = rollout.states[:, step]
        timestep = rollout.timesteps[:, step]
        std = policy.transition_std(x_t, timestep)
        std_values.append(std.detach().float())
        stochastic = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
        if policy.sde_mode == "ogpo_constant_corrected":
            stochastic = ~policy._final_transition_mask(timestep)
        # Report a per-trajectory step count (not batch*step elements), so a
        # K-step constant SDE reports K-1 stochastic/logprob transitions.
        stochastic_count += float(stochastic.float().mean().item())
        logprob_count += float(stochastic.float().mean().item())
    std_flat = torch.cat([value.reshape(-1) for value in std_values])
    active_std = std_flat[std_flat > 0.0]
    raw_norms = getattr(rollout, "raw_velocity_norms", None)
    corrected_norms = getattr(rollout, "corrected_drift_norms", None)
    correction_norms = getattr(rollout, "sde_correction_norms", None)
    return {
        "sde_std_mean": float(
            active_std.mean().item() if active_std.numel() else 0.0
        ),
        "sde_std_min": float(std_flat.min().item()),
        "sde_std_max": float(std_flat.max().item()),
        "sde_deterministic_final_fraction": float(
            (std_flat == 0.0).float().mean().item()
        ),
        "num_stochastic_flow_steps": float(stochastic_count),
        "num_logprob_flow_steps": float(logprob_count),
        "sde_correction_norm": float(
            correction_norms.mean().item() if correction_norms is not None else 0.0
        ),
        "raw_velocity_norm": float(
            raw_norms.mean().item() if raw_norms is not None else 0.0
        ),
        "corrected_drift_norm": float(
            corrected_norms.mean().item() if corrected_norms is not None else 0.0
        ),
    }


@torch.no_grad()
def _selected_transition_log_probs(
    policy: OpenPIStochasticFlowPolicy,
    *,
    x_prev: torch.Tensor,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    condition: Any,
    microbatch_size: int,
) -> torch.Tensor:
    """Evaluate one selected Flash transition per candidate without a backward graph."""
    if x_prev.shape != x_t.shape or x_prev.ndim != 2:
        raise ValueError("selected χ² transition states must have shape [N, action_dim]")
    if timestep.shape[0] != x_t.shape[0]:
        raise ValueError("selected χ² transition timesteps must match candidate count")
    if policy.condition_batch_size(condition) != x_t.shape[0]:
        raise ValueError("selected χ² transition condition batch does not match candidates")
    if microbatch_size <= 0:
        raise ValueError("actor.chi2.logprob_microbatch_size must be positive")
    values = []
    for start in range(0, x_t.shape[0], microbatch_size):
        stop = min(start + microbatch_size, x_t.shape[0])
        indices = torch.arange(start, stop, device=x_t.device)
        values.append(
            policy.log_prob(
                x_prev[start:stop],
                x_t[start:stop],
                _condition_index_select(condition, indices),
                timestep[start:stop],
            )
        )
    return torch.cat(values, dim=0)


def _chi2_ratio_metrics(stats: Chi2RatioStats) -> dict[str, float]:
    return {
        "chi2_enabled": 1.0,
        "chi2_selected_log_ratio_mean": stats.log_ratio_mean,
        "chi2_selected_log_ratio_std": stats.log_ratio_std,
        "chi2_selected_log_ratio_min": stats.log_ratio_min,
        "chi2_selected_log_ratio_max": stats.log_ratio_max,
        "chi2_ratio_mean": stats.ratio_mean,
        "chi2_ratio_std": stats.ratio_std,
        "chi2_ratio_min": stats.ratio_min,
        "chi2_ratio_max": stats.ratio_max,
        "chi2_ratio_clipped_fraction": stats.ratio_clipped_fraction,
        "chi2_divergence_proxy": stats.divergence_proxy,
        "chi2_selected_logprob_normalizer": stats.logprob_normalizer,
        "current_slow_logratio_mean": stats.log_ratio_mean,
        "current_slow_logratio_std": stats.log_ratio_std,
        "current_slow_approx_kl": stats.divergence_proxy,
    }


@torch.no_grad()
def _selected_transition_chi2_ratio(
    state: OGPOTrainState,
    *,
    x_prev: torch.Tensor,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    condition: Any,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the Flash selected-transition current/slow ratio for chiPO."""
    if ogpo_variant(config) != "chi2":
        raise ValueError("selected χ² ratio was requested outside actor.ogpo_variant=chi2")
    if state.slow_policy is None:
        raise RuntimeError("OGPO+χ² requires an initialized slow policy")
    chi2_cfg = _chi2_config(config)
    microbatch_size = int(
        chi2_cfg.get("logprob_microbatch_size", x_t.shape[0])
    )
    current_log_prob = _selected_transition_log_probs(
        state.policy,
        x_prev=x_prev,
        x_t=x_t,
        timestep=timestep,
        condition=condition,
        microbatch_size=microbatch_size,
    )
    slow_log_prob = _selected_transition_log_probs(
        state.slow_policy,
        x_prev=x_prev,
        x_t=x_t,
        timestep=timestep,
        condition=condition,
        microbatch_size=microbatch_size,
    )
    ratio, stats = selected_transition_chi2_ratio(
        current_log_prob,
        slow_log_prob,
        event_dim=x_t.shape[-1],
        normalize_action_dim=bool(chi2_cfg.get("normalize_action_dim", True)),
        log_ratio_clip=float(chi2_cfg.get("log_ratio_clip", 20.0)),
        ratio_max=float(chi2_cfg.get("ratio_max", 20.0)),
    )
    return ratio, _chi2_ratio_metrics(stats)


def _select_flash_steps(
    flow_cfg: dict[str, Any],
    *,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    distribution = str(flow_cfg.get("selected_timestep_distribution", "fixed"))
    if distribution == "uniform":
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        return torch.randint(
            num_steps,
            (batch_size,),
            generator=generator,
        ).to(device)
    if distribution == "stratified_uniform":
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        permutations = [
            torch.randperm(num_steps, generator=generator)
            for _ in range(math.ceil(batch_size / num_steps))
        ]
        return torch.cat(permutations)[:batch_size].to(device)
    if distribution == "fixed":
        selected = int(flow_cfg.get("selected_timestep", num_steps // 2))
        if selected < 0 or selected >= num_steps:
            raise ValueError("flow.selected_timestep must be in [0, flow.num_steps)")
        return torch.full((batch_size,), selected, dtype=torch.long, device=device)
    raise ValueError(f"unsupported selected_timestep_distribution={distribution!r}")


def _flash_rectification_weight(
    state: OGPOTrainState,
    flow_cfg: dict[str, Any],
    *,
    timestep: torch.Tensor,
    selected_steps: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    mode = str(flow_cfg.get("temporal_rectification_mode", "analytic"))
    if mode == "none":
        return torch.ones(timestep.shape[0], dtype=timestep.dtype, device=timestep.device)
    if mode == "analytic":
        policy_variance = float(flow_cfg.get("stochastic_variance", 0.04))
        policy_log_std = getattr(state.old_policy, "log_std", None)
        if policy_log_std is not None:
            policy_variance = float(
                policy_log_std.detach().float().exp().square().mean().item()
            )
        return analytic_rectification(
            timestep,
            stochastic_variance=policy_variance,
            sde_mode=str(flow_cfg.get("sde_mode", "gaussian_adapter")),
            clip_min=float(flow_cfg.get("rectification_clip_min", 0.25)),
            clip_max=float(flow_cfg.get("rectification_clip_max", 4.0)),
        )
    if mode == "empirical_ema":
        selected_g = selected_steps.repeat_interleave(group_size)
        weights = [
            state.rectifier.weight(int(step.item()), device=timestep.device)
            for step in selected_g
        ]
        return torch.stack(weights).to(dtype=timestep.dtype, device=timestep.device)
    raise ValueError(f"unsupported temporal_rectification_mode={mode!r}")


def awr_actor_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
) -> dict[str, float]:
    """Scalar-Q AWR baseline implemented as weighted replay flow matching."""
    if ogpo_variant(config) == "chi2":
        raise ValueError("OGPO+χ² requires likelihood-ratio actor updates, not actor.algorithm=awr")
    actor_cfg = config.get("actor", {})
    if state.critic.ensemble_size != 1:
        raise ValueError("AWR baseline requires critic.ensemble_size=1")
    batch = batch.to(next(state.policy.parameters()).device)
    state.critic_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        q_data = state.critic(
            batch.observations,
            batch.action_chunks,
            _critic_execution_mask(batch, config),
        ).squeeze(0)
        advantage = q_data - q_data.mean()
        temperature = float(actor_cfg.get("awr_temperature", 1.0))
        max_weight = float(actor_cfg.get("awr_max_weight", 20.0))
        if temperature <= 0.0 or max_weight < 1.0:
            raise ValueError("AWR requires positive temperature and awr_max_weight >= 1")
        log_weight = (advantage / temperature).clamp(min=-20.0, max=math.log(max_weight))
        weights = log_weight.exp()

    state.actor_optimizer.zero_grad(set_to_none=True)
    condition = _policy_condition(state.policy, batch)
    flow_actions = state.policy.action_chunks_to_flow(batch).reshape(batch.batch_size, -1)
    weighted_fm = weighted_flow_matching_loss(state.policy, condition, flow_actions, weights)
    weighted_fm.loss.backward()
    assert_no_gradients(state.critic, "critic")
    assert_no_gradients(state.reference_policy, "reference_policy")
    actor_grad_norm = float(grad_norm(state.policy.parameters()))
    torch.nn.utils.clip_grad_norm_(state.policy.parameters(), float(actor_cfg.get("max_grad_norm", 1.0)))
    state.actor_optimizer.step()
    return {
        "actor_loss": float(weighted_fm.loss.detach().item()),
        "awr_weighted_fm_loss": weighted_fm.diagnostics["weighted_fm_loss"],
        "awr_weight_mean": weighted_fm.diagnostics["awr_weight_mean"],
        "awr_weight_max": weighted_fm.diagnostics["awr_weight_max"],
        "awr_advantage_mean": float(advantage.mean().item()),
        "awr_advantage_std": float(advantage.std(unbiased=False).item()),
        "actor_grad_norm": actor_grad_norm,
        "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
    }


_ACTION_EXPERT_PARAMETER_PREFIXES = (
    "backend.paligemma_with_expert.gemma_expert.",
    "backend.action_in_proj.",
    "backend.action_out_proj.",
    "backend.time_mlp_in.",
    "backend.time_mlp_out.",
    "backend.state_proj.",
    "backend.action_time_mlp_in.",
    "backend.action_time_mlp_out.",
)


def _actor_gradient_group(parameter_name: str) -> str:
    if parameter_name.startswith(_ACTION_EXPERT_PARAMETER_PREFIXES):
        return "action_expert"
    if parameter_name.startswith("backend."):
        return "backbone"
    return "other"


def _gradient_group_norms(policy: nn.Module) -> dict[str, float]:
    squares: dict[str, torch.Tensor] = {}
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        group = _actor_gradient_group(name)
        square = parameter.grad.detach().float().square().sum()
        squares[group] = square if group not in squares else squares[group] + square
    return {
        group: float(square.sqrt().item()) for group, square in squares.items()
    }


def full_actor_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None = None,
    success_batch: ChunkBatch | None = None,
    diagnostic_no_optimizer_step: bool = False,
) -> dict[str, float]:
    actor_cfg = config.get("actor", {})
    role_dp = False
    if actor_cfg.get("role_data_parallel", False):
        from .actor_role_dp import validate_config
        role_dp = validate_config(config)
    training_cfg = config.get("training", {})
    offload_frozen_critic = bool(
        training_cfg.get("offload_frozen_critic_during_actor_optimization", False)
    )
    if bool(config.get("training", {}).get("critic_frozen_during_actor", False)):
        critic_modules = [state.critic, state.target_critic]
        if state.divl is not None:
            critic_modules.extend([state.divl, state.target_divl])
        if any(
            parameter.requires_grad
            for module in critic_modules
            if module is not None
            for parameter in module.parameters()
        ):
            raise RuntimeError(
                "critic_frozen_during_actor requires freeze_critic_for_actor() "
                "before the actor update"
            )
    if (
        not _using_jax_actor(state.policy)
        and ogpo_variant(config) in {"chi2", "ca_chi2"}
        and str(actor_cfg.get("chi2", {}).get("ratio_scope", "full_chain"))
        != "full_chain"
    ):
        raise ValueError(
            "full-chain ChiPO requires actor.chi2.ratio_scope=full_chain"
        )
    if _using_jax_actor(state.policy) and ogpo_variant(config) in {"chi2", "ca_chi2"}:
        raise ValueError(
            "full-chain ChiPO composition is implemented only for the PyTorch actor path"
        )
    regularization_cfg = config.get("regularization", {})
    uncertainty_cfg = config.get("uncertainty", {})
    policy_device = next(state.policy.parameters()).device
    batch = batch.to(policy_device)
    if offload_frozen_critic:
        if not bool(training_cfg.get("critic_frozen_during_actor", False)):
            raise ValueError(
                "critic offload requires training.critic_frozen_during_actor=true"
            )
        if next(state.critic.parameters()).device != policy_device:
            state.critic.to(policy_device)
    group_size = int(actor_cfg.get("group_size", 4))
    actor_epochs = max(1, int(actor_cfg.get("actor_epochs_per_rollout", 1)))
    if diagnostic_no_optimizer_step and actor_epochs != 1:
        raise ValueError("pure component-gradient diagnostics require one actor epoch")
    if diagnostic_no_optimizer_step and not bool(
        config.get("diagnostics", {}).get("actor_component_grad_norms", False)
    ):
        raise ValueError(
            "diagnostic_no_optimizer_step requires "
            "diagnostics.actor_component_grad_norms=true"
        )
    if diagnostic_no_optimizer_step and (
        float(regularization_cfg.get("lambda_fm", 0.0)) != 0.0
        or float(regularization_cfg.get("lambda_smooth", 0.0)) != 0.0
    ):
        raise ValueError(
            "pure PPO/BC diagnostics require lambda_fm=lambda_smooth=0"
        )
    state.critic_optimizer.zero_grad(set_to_none=True)
    condition = _policy_condition(state.policy, batch)
    condition_g = state.policy.repeat_condition(condition, group_size)

    if _using_jax_actor(state.policy):
        assert isinstance(state.policy, PI05JaxFlowPolicy)
        assert isinstance(state.old_policy, PI05JaxFlowPolicy)
        assert isinstance(state.reference_policy, PI05JaxFlowPolicy)
        flow_spec = OpenPIJaxFlowSpec(state.policy.num_steps)
        jax_observation = policy_observation_to_jax(condition.observation)
        old_actor = nnx.merge(state.old_policy.actor_graphdef, state.old_policy.actor_state)
        reference_actor = nnx.merge(state.reference_policy.actor_graphdef, state.reference_policy.actor_state)

        with torch.no_grad():
            rng = jax.random.PRNGKey(int(state.step) + 31)
            old_rollout_j = jax_rollout(
                actor=old_actor,
                flow_spec=flow_spec,
                observation=jax_observation,
                group_size=group_size,
                rng=rng,
                sde_mode=state.old_policy.sde_mode,
            )
            old_endpoint_torch = torch.as_tensor(np.asarray(old_rollout_j.endpoint).copy(), device=batch.observations.device, dtype=batch.action_chunks.dtype)
            environment_endpoint = state.old_policy.flat_actions_to_environment(old_endpoint_torch, condition_g)
            endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
            advantages, adv_diag = conservative_advantages_for_candidates(
                state, batch.observations, endpoint, batch, config
            )
            entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
            eps_clip = actor_clip_for_uncertainty(
                entropy_norm,
                actor_cfg,
                uncertainty_cfg,
            ).repeat_interleave(group_size)
        jax_regularization = _prepare_jax_regularization(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
        )
        beta_base = float(regularization_cfg.get("beta_kl", 0.01))
        adapt_kl_beta = bool(uncertainty_cfg.get("adapt_kl_beta", False))
        uncertainty_scale = float(regularization_cfg.get("kl_uncertainty_scale", 1.0))
        normalize_logprob_by_action_dim = bool(
            actor_cfg.get("normalize_logprob_by_action_dim", False)
        )
        logprob_normalizer = float(state.policy.action_dim) if normalize_logprob_by_action_dim else 1.0
        old_states = old_rollout_j.states
        old_next_states = old_rollout_j.next_states
        old_timesteps = old_rollout_j.timesteps
        advantages_j = jnp.asarray(advantages.reshape(-1).detach().cpu().numpy())
        eps_clip_j = jnp.asarray(eps_clip.detach().cpu().numpy())
        entropy_norm_j = jnp.asarray(entropy_norm.detach().cpu().numpy())
        full_ratio_mode = str(actor_cfg.get("full_ratio_mode", "per_transition"))
        normalize_logprob_by_denoising_steps = bool(
            actor_cfg.get("normalize_logprob_by_denoising_steps", False)
        )
        chain_logprob_normalizer = logprob_normalizer
        if full_ratio_mode == "ais_joint" and normalize_logprob_by_denoising_steps:
            chain_logprob_normalizer *= float(state.policy.num_steps)
        gradient_microbatch_size = int(
            actor_cfg.get("gradient_microbatch_size", advantages_j.shape[0])
        )
        if gradient_microbatch_size <= 0:
            raise ValueError("actor.gradient_microbatch_size must be positive")
        loss_total = 0.0
        ppo_total = 0.0
        kl_total = 0.0
        grad_total = 0.0

        if gradient_microbatch_size < advantages_j.shape[0]:
            if beta_base != 0.0 or adapt_kl_beta:
                raise ValueError(
                    "full-chain gradient microbatching currently requires beta_kl=0 "
                    "and uncertainty.adapt_kl_beta=false"
                )
            if (
                float(jax_regularization["lambda_fm"]) != 0.0
                or float(jax_regularization["lambda_success"]) != 0.0
            ):
                raise ValueError(
                    "full-chain gradient microbatching currently requires "
                    "lambda_fm=lambda_success=0"
                )
            jax_observation_g = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, group_size, axis=0) if x is not None else None,
                jax_observation,
            )

            if bool(actor_cfg.get("full_chain_streaming_backward", False)):
                policy_graphdef = state.policy.actor_graphdef
                old_graphdef = state.old_policy.actor_graphdef

                @jax.jit
                def _stream_current_log_prob(
                    actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    actor = nnx.merge(policy_graphdef, actor_state)
                    return jax_transition_log_prob(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_prev=x_prev,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    ) / chain_logprob_normalizer

                @jax.jit
                def _stream_old_log_prob(
                    old_actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    actor = nnx.merge(old_graphdef, old_actor_state)
                    return jax_transition_log_prob(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_prev=x_prev,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.old_policy.sde_mode,
                    ) / chain_logprob_normalizer

                @jax.jit
                @jax.value_and_grad
                def _stream_weighted_log_prob(
                    actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                    coefficient,
                ):
                    log_prob = _stream_current_log_prob(
                        actor_state,
                        x_prev,
                        x_t,
                        observation,
                        timestep,
                    )
                    return jnp.mean(jax.lax.stop_gradient(coefficient) * log_prob)

                torch.cuda.empty_cache()
                ratio = np.ones(int(advantages_j.shape[0]), dtype=np.float32)
                clipped_ratio = ratio.copy()
                for _ in range(actor_epochs):
                    host_grads = None
                    epoch_loss = 0.0
                    ratio_parts = []
                    clipped_parts = []
                    total_candidates = int(advantages_j.shape[0])
                    for start in range(0, total_candidates, gradient_microbatch_size):
                        stop = min(start + gradient_microbatch_size, total_candidates)
                        sample_weight = (stop - start) / total_candidates
                        observation_mb = _slice_jax_batch(jax_observation_g, start, stop)
                        new_step_log_probs = []
                        old_step_log_probs = []
                        for flow_step in range(state.policy.num_steps):
                            args = (
                                old_next_states[start:stop, flow_step],
                                old_states[start:stop, flow_step],
                                observation_mb,
                                old_timesteps[start:stop, flow_step],
                            )
                            new_step_log_probs.append(
                                np.asarray(
                                    _stream_current_log_prob(
                                        state.policy.actor_state,
                                        *args,
                                    )
                                )
                            )
                            old_step_log_probs.append(
                                np.asarray(
                                    _stream_old_log_prob(
                                        state.old_policy.actor_state,
                                        *args,
                                    )
                                )
                            )
                            jax.effects_barrier()

                        log_ratio = np.sum(
                            np.stack(new_step_log_probs, axis=1)
                            - np.stack(old_step_log_probs, axis=1),
                            axis=1,
                        )
                        ratio_mb = np.exp(np.clip(log_ratio, -20.0, 20.0))
                        eps_mb = np.full_like(
                            ratio_mb,
                            _ppo_chain_clip(actor_cfg),
                        )
                        clipped_mb = np.clip(ratio_mb, 1.0 - eps_mb, 1.0 + eps_mb)
                        advantage_mb = np.asarray(advantages_j[start:stop])
                        objective_mb = np.minimum(
                            ratio_mb * advantage_mb,
                            clipped_mb * advantage_mb,
                        )
                        epoch_loss += sample_weight * float(-objective_mb.mean())
                        # Exact derivative of the clipped surrogate with
                        # respect to the normalized joint chain log-ratio.
                        active = np.where(
                            advantage_mb >= 0.0,
                            ratio_mb <= 1.0 + eps_mb,
                            ratio_mb >= 1.0 - eps_mb,
                        )
                        coefficient = np.where(
                            active,
                            -ratio_mb * advantage_mb,
                            0.0,
                        ).astype(np.float32)

                        for flow_step in range(state.policy.num_steps):
                            _, grads_step = _stream_weighted_log_prob(
                                state.policy.actor_state,
                                old_next_states[start:stop, flow_step],
                                old_states[start:stop, flow_step],
                                observation_mb,
                                old_timesteps[start:stop, flow_step],
                                jnp.asarray(coefficient),
                            )
                            host_grads = _accumulate_jax_grads_on_host(
                                host_grads,
                                grads_step,
                                weight=sample_weight,
                            )
                            del grads_step
                            jax.effects_barrier()
                            gc.collect()
                        ratio_parts.append(ratio_mb)
                        clipped_parts.append(clipped_mb)

                    assert host_grads is not None
                    actor_grad_norm = _jax_tree_l2_norm(host_grads)
                    state.policy.apply_actor_gradients(host_grads)
                    loss_total += epoch_loss
                    ppo_total += epoch_loss
                    grad_total += actor_grad_norm
                    ratio = np.concatenate(ratio_parts)
                    clipped_ratio = np.concatenate(clipped_parts)
                return {
                    "actor_loss": loss_total / actor_epochs,
                    "full_ppo_loss": ppo_total / actor_epochs,
                    "reference_kl": 0.0,
                    "reference_kl_beta": 0.0,
                    "ustate_adapt_ppo_clip": 0.0,
                    "ustate_adapt_kl_beta": 0.0,
                    "actor_epochs": float(actor_epochs),
                    "importance_ratio_mean": float(ratio.mean()),
                    "importance_ratio_std": float(ratio.std()),
                    "importance_ratio_min": float(ratio.min()),
                    "importance_ratio_max": float(ratio.max()),
                    "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
                    "actor_grad_norm": grad_total / actor_epochs,
                    "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
                    "fm_anchor_loss": 0.0,
                    "success_buffer_loss": 0.0,
                    "action_smoothness": 0.0,
                    "gradient_microbatch_size": float(gradient_microbatch_size),
                    "full_chain_streaming_backward": 1.0,
                    **adv_diag,
                }

            def _origin_microbatch_loss(
                actor_state,
                states,
                next_states,
                timesteps,
                observation,
                candidate_advantages,
                candidate_eps,
            ):
                actor = nnx.merge(state.policy.actor_graphdef, actor_state)
                current_log_probs = []
                frozen_log_probs = []
                for flow_step in range(state.policy.num_steps):
                    current_log_probs.append(
                        jax_transition_log_prob(
                            actor=actor,
                            flow_spec=flow_spec,
                            x_prev=next_states[:, flow_step],
                            x_t=states[:, flow_step],
                            observation=observation,
                            timestep=timesteps[:, flow_step],
                            sde_mode=state.policy.sde_mode,
                        ) / chain_logprob_normalizer
                    )
                    frozen_log_probs.append(
                        jax.lax.stop_gradient(
                            jax_transition_log_prob(
                                actor=old_actor,
                                flow_spec=flow_spec,
                                x_prev=next_states[:, flow_step],
                                x_t=states[:, flow_step],
                                observation=observation,
                                timestep=timesteps[:, flow_step],
                                sde_mode=state.old_policy.sde_mode,
                            ) / chain_logprob_normalizer
                        )
                    )
                current = jnp.stack(current_log_probs, axis=1)
                frozen = jnp.stack(frozen_log_probs, axis=1)
                if full_ratio_mode != "ais_joint":
                    raise ValueError(
                        "OGPO-origin microbatching requires actor.full_ratio_mode=ais_joint"
                    )
                ppo = jax_full_chain_ais_ppo_loss(
                    current,
                    frozen,
                    candidate_advantages,
                    clip_eps=candidate_eps,
                )
                return ppo.loss, {
                    "ratio": ppo.ratio,
                    "clipped_ratio": ppo.clipped_ratio,
                }

            torch.cuda.empty_cache()
            ratio = np.ones(int(advantages_j.shape[0]), dtype=np.float32)
            clipped_ratio = ratio.copy()
            for _ in range(actor_epochs):
                host_grads = None
                epoch_loss = 0.0
                ratio_parts = []
                clipped_parts = []
                total_candidates = int(advantages_j.shape[0])
                for start in range(0, total_candidates, gradient_microbatch_size):
                    stop = min(start + gradient_microbatch_size, total_candidates)
                    sample_weight = (stop - start) / total_candidates
                    (loss_mb, aux_mb), grads_mb = jax.value_and_grad(
                        _origin_microbatch_loss,
                        has_aux=True,
                    )(
                        state.policy.actor_state,
                        old_states[start:stop],
                        old_next_states[start:stop],
                        old_timesteps[start:stop],
                        _slice_jax_batch(jax_observation_g, start, stop),
                        advantages_j[start:stop],
                        jnp.full(
                            (stop - start,),
                            _ppo_chain_clip(actor_cfg),
                            dtype=advantages_j.dtype,
                        ),
                    )
                    host_grads = _accumulate_jax_grads_on_host(
                        host_grads,
                        grads_mb,
                        weight=sample_weight,
                    )
                    epoch_loss += sample_weight * float(loss_mb)
                    ratio_parts.append(np.asarray(aux_mb["ratio"]))
                    clipped_parts.append(np.asarray(aux_mb["clipped_ratio"]))
                    del grads_mb
                    jax.effects_barrier()
                    gc.collect()
                assert host_grads is not None
                actor_grad_norm = _jax_tree_l2_norm(host_grads)
                state.policy.apply_actor_gradients(host_grads)
                loss_total += epoch_loss
                ppo_total += epoch_loss
                grad_total += actor_grad_norm
                ratio = np.concatenate(ratio_parts)
                clipped_ratio = np.concatenate(clipped_parts)
            return {
                "actor_loss": loss_total / actor_epochs,
                "full_ppo_loss": ppo_total / actor_epochs,
                "reference_kl": 0.0,
                "reference_kl_beta": 0.0,
                "ustate_adapt_ppo_clip": 0.0,
                "ustate_adapt_kl_beta": 0.0,
                "actor_epochs": float(actor_epochs),
                "importance_ratio_mean": float(ratio.mean()),
                "importance_ratio_std": float(ratio.std()),
                "importance_ratio_min": float(ratio.min()),
                "importance_ratio_max": float(ratio.max()),
                "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
                "actor_grad_norm": grad_total / actor_epochs,
                "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
                "fm_anchor_loss": 0.0,
                "success_buffer_loss": 0.0,
                "action_smoothness": 0.0,
                "gradient_microbatch_size": float(gradient_microbatch_size),
                **adv_diag,
            }

        def _loss_fn(actor_state):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            step_log_probs = []
            canonical_old_log_probs = []
            for step in range(state.policy.num_steps):
                step_observation = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x, group_size, axis=0) if x is not None else None,
                    jax_observation,
                )
                step_log_probs.append(
                    jax_transition_log_prob(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_prev=old_next_states[:, step],
                        x_t=old_states[:, step],
                        observation=step_observation,
                        timestep=old_timesteps[:, step],
                        sde_mode=state.policy.sde_mode,
                    )
                )
                canonical_old_log_probs.append(
                    jax.lax.stop_gradient(
                        jax_transition_log_prob(
                            actor=old_actor,
                            flow_spec=flow_spec,
                            x_prev=old_next_states[:, step],
                            x_t=old_states[:, step],
                            observation=step_observation,
                            timestep=old_timesteps[:, step],
                            sde_mode=state.old_policy.sde_mode,
                        )
                    )
                )
            new_log_probs = jnp.stack(step_log_probs, axis=1) / chain_logprob_normalizer
            old_log_probs = jnp.stack(canonical_old_log_probs, axis=1) / chain_logprob_normalizer
            if full_ratio_mode == "ais_joint":
                ppo = jax_full_chain_ais_ppo_loss(
                    new_log_probs,
                    old_log_probs,
                    advantages_j,
                    clip_eps=jnp.full_like(advantages_j, _ppo_chain_clip(actor_cfg)),
                )
            elif full_ratio_mode == "per_transition":
                ppo = jax_full_chain_ppo_loss(
                    new_log_probs,
                    old_log_probs,
                    advantages_j,
                    clip_eps=eps_clip_j,
                )
            else:
                raise ValueError(f"unsupported actor.full_ratio_mode={full_ratio_mode!r}")
            transition_ref_kl = jax_transition_kl(
                actor=actor,
                other_actor=reference_actor,
                flow_spec=flow_spec,
                x_t=old_states[:, 0],
                observation=jax.tree_util.tree_map(lambda x: jnp.repeat(x, group_size, axis=0) if x is not None else None, jax_observation),
                timestep=old_timesteps[:, 0],
                sde_mode=state.policy.sde_mode,
            )
            kl_penalty, ref_kl, beta = jax_state_adaptive_kl_penalty(
                transition_ref_kl,
                entropy_norm_j,
                group_size=group_size,
                beta_base=beta_base,
                adapt_kl_beta=adapt_kl_beta,
                uncertainty_scale=uncertainty_scale,
            )
            regularization_loss, fm_loss, success_loss = _jax_regularization_loss(
                actor,
                jax_regularization,
            )
            total = ppo.loss + kl_penalty + regularization_loss
            return total, {
                "ppo_loss": ppo.loss,
                "ref_kl": ref_kl,
                "beta": beta,
                "fm_loss": fm_loss,
                "success_loss": success_loss,
                "ratio": ppo.ratio,
                "clipped_ratio": ppo.clipped_ratio,
            }

        torch.cuda.empty_cache()
        for _ in range(actor_epochs):
            (total_loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.policy.actor_state)
            grad_leaves = jax.tree_util.tree_leaves(grads)
            grad_norm_sq = 0.0
            for leaf in grad_leaves:
                if leaf is not None:
                    grad_norm_sq += float(jnp.sum(jnp.square(leaf)))
            actor_grad_norm = grad_norm_sq ** 0.5
            state.policy.apply_actor_gradients(grads)
            loss_total += float(total_loss)
            ppo_total += float(aux["ppo_loss"])
            kl_total += float(aux["ref_kl"])
            grad_total += actor_grad_norm
        ratio = np.asarray(aux["ratio"])
        clipped_ratio = np.asarray(aux["clipped_ratio"])
        return {
            "actor_loss": loss_total / actor_epochs,
            "full_ppo_loss": ppo_total / actor_epochs,
            "reference_kl": kl_total / actor_epochs,
            "reference_kl_beta": float(aux["beta"]),
            "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
            "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
            "actor_epochs": float(actor_epochs),
            "importance_ratio_mean": float(ratio.mean()),
            "importance_ratio_std": float(ratio.std()),
            "importance_ratio_min": float(ratio.min()),
            "importance_ratio_max": float(ratio.max()),
            "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
            "actor_grad_norm": grad_total / actor_epochs,
            "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
            "fm_anchor_loss": float(aux["fm_loss"]),
            "success_buffer_loss": float(aux["success_loss"]),
            "action_smoothness": 0.0,
            **adv_diag,
        }

    with torch.no_grad():
        old_device = _policy_device(state.old_policy)
        old_condition = (
            _policy_condition(state.old_policy, batch)
            if old_device != policy_device
            else condition
        )
        old_condition_g = state.old_policy.repeat_condition(old_condition, group_size)
        role_old_rollout = state.old_policy.rollout(
            old_condition, group_size=group_size
        )
        sde_diag = _sde_rollout_metrics(
            state.old_policy, role_old_rollout, old_condition_g
        )
        environment_endpoint = state.old_policy.flat_actions_to_environment(
            role_old_rollout.endpoint, old_condition_g
        ).to(policy_device)
        old_rollout = _rollout_to_device(role_old_rollout, policy_device)
        endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
        chi2_ratio = None
        chi2_diag: dict[str, float] = {"chi2_enabled": 0.0}
        if ogpo_variant(config) in {"chi2", "ca_chi2"}:
            if state.slow_policy is None:
                raise RuntimeError("full-chain ChiPO requires an independent slow policy")
            current_chain_log_probs = []
            slow_chain_log_probs = []
            slow_device = _policy_device(state.slow_policy)
            slow_condition = (
                _policy_condition(state.slow_policy, batch)
                if slow_device != policy_device
                else condition
            )
            slow_condition_g = state.slow_policy.repeat_condition(
                slow_condition, group_size
            )
            for flow_step in range(state.policy.num_steps):
                x_t_step = old_rollout.states[:, flow_step]
                x_prev_step = old_rollout.next_states[:, flow_step]
                timestep_step = old_rollout.timesteps[:, flow_step]
                current_chain_log_probs.append(
                    state.policy.log_prob(
                        x_prev_step, x_t_step, condition_g, timestep_step
                    )
                )
                slow_chain_log_probs.append(
                    state.slow_policy.log_prob(
                        x_prev_step.to(slow_device),
                        x_t_step.to(slow_device),
                        slow_condition_g,
                        timestep_step.to(slow_device),
                    ).to(policy_device)
                )
            current_chain_log_probs_t = torch.stack(current_chain_log_probs, dim=1)
            slow_chain_log_probs_t = torch.stack(slow_chain_log_probs, dim=1)
            chi2_ratio_flat, chi2_stats = full_chain_chipo_ratio(
                current_chain_log_probs_t,
                slow_chain_log_probs_t,
                action_dim=int(state.policy.action_dim),
                normalize_logprob_by_action_dim=bool(
                    actor_cfg.get("normalize_logprob_by_action_dim", True)
                ),
                normalize_logprob_by_denoising_steps=bool(
                    actor_cfg.get("normalize_logprob_by_denoising_steps", True)
                ),
                # Official ChiPO uses exp(current-slow) directly. Legacy
                # experiments retain their explicit numerical ratio clamps.
                log_ratio_clip=(
                    None
                    if bool(config.get("training", {}).get("clean_pytorch_main", False))
                    else float(actor_cfg.get("chi2", {}).get("log_ratio_clip", 20.0))
                ),
                ratio_max=(
                    None
                    if bool(config.get("training", {}).get("clean_pytorch_main", False))
                    else float(actor_cfg.get("chi2", {}).get("ratio_max", 20.0))
                ),
            )
            chi2_ratio = chi2_ratio_flat.reshape(batch.batch_size, group_size)
            chi2_diag = _chi2_ratio_metrics(chi2_stats)
        advantages, adv_diag = conservative_advantages_for_candidates(
            state,
            batch.observations,
            endpoint,
            batch,
            config,
            chi2_ratio=chi2_ratio,
        )

    with torch.no_grad():
        entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
        entropy_norm_grouped = entropy_norm.repeat_interleave(group_size)
        eps_clip = actor_clip_for_uncertainty(
            entropy_norm,
            actor_cfg,
            uncertainty_cfg,
        ).repeat_interleave(group_size)
    if offload_frozen_critic:
        # Q scoring and optional entropy diagnostics are complete.  The
        # frozen critic does not participate in PPO/KL/backward, so release
        # its GPU storage until the next transaction's scoring phase.
        state.critic.to("cpu")
        if policy_device.type == "cuda":
            torch.cuda.empty_cache()
    elif (
        isinstance(state.policy, PI05PytorchFlowPolicy)
        and bool(config.get("flow", {}).get("pytorch_role_devices", {}))
        and policy_device.type == "cuda"
    ):
        # Candidate rollout, 10-Q scoring, and frozen slow-policy forwards
        # are all no-grad. Drop their temporary allocations before reserving
        # the exact rollback snapshot on its dedicated GPU role.
        torch.cuda.empty_cache()
    loss_total = 0.0
    ppo_total = 0.0
    kl_total = 0.0
    grad_total = 0.0
    chi2_loss_total = 0.0
    chain_logprob_normalizer = 1.0
    if bool(actor_cfg.get("normalize_logprob_by_action_dim", False)):
        chain_logprob_normalizer *= float(state.policy.action_dim)
    if bool(actor_cfg.get("normalize_logprob_by_denoising_steps", False)):
        # OGPO's compute_flow_log_prob counts the initial N(0,I) prior plus
        # the K-1 stochastic transitions when the final constant-SDE step is
        # deterministic: (1 + (K-1)) == K.  The prior cancels in current/old
        # and current/slow differences, but remains part of this denominator.
        chain_logprob_normalizer *= float(state.policy.num_steps)
    gradient_microbatch_size = min(
        max(1, int(actor_cfg.get("gradient_microbatch_size", advantages.numel()))),
        int(advantages.numel()),
    )
    full_ratio_mode = str(actor_cfg.get("full_ratio_mode", "per_transition"))
    if full_ratio_mode not in {"ais_joint", "per_transition"}:
        raise ValueError(f"unsupported actor.full_ratio_mode={full_ratio_mode!r}")
    if full_ratio_mode != "ais_joint" and (
        bool(actor_cfg.get("normalize_logprob_by_action_dim", False))
        or bool(actor_cfg.get("normalize_logprob_by_denoising_steps", False))
    ):
        raise ValueError("log-prob normalization requires actor.full_ratio_mode=ais_joint")
    chi2_upper_ratio_bound = None
    chi2_regularization_beta = 0.0
    if ogpo_variant(config) in {"chi2", "ca_chi2"}:
        chi2_regularization_beta = float(
            actor_cfg.get("chi2", {}).get(
                "regularization_beta",
                actor_cfg.get("chi2", {}).get("beta_base", 0.1),
            )
        )
        chi2_upper_ratio_bound = chi2_ppo_upper_bound(
            eps_clip,
            beta=float(adv_diag.get("chi2_beta", 0.0)),
            r_max=float(actor_cfg.get("chi2", {}).get("r_max", 10.0)),
        )
    last_ppo: Any | None = None
    last_ratio = None
    last_clipped_ratio = None
    last_log_ratio = None
    last_reg_diag: dict[str, float] = {}
    last_kl_beta = entropy_norm.new_tensor(float(regularization_cfg.get("beta_kl", 0.01)))
    post_update_action = _post_update_kl_action(actor_cfg)
    post_update_kl_limit = float(actor_cfg.get("max_policy_reference_kl", float("inf")))
    transaction_snapshot = (
        _capture_actor_rollback_state(
            state,
            device=actor_cfg.get("rollback_snapshot_device", "cpu"),
        )
        if post_update_action == "rollback_cpu" and not diagnostic_no_optimizer_step
        else None
    )
    accepted_actor_epochs = 0
    rejected_actor_epochs = 0
    component_grad_diagnostics = bool(
        config.get("diagnostics", {}).get("actor_component_grad_norms", False)
    )
    ppo_component_grad_total = 0.0
    bc_component_grad_total = 0.0
    ppo_bc_grad_cosine_total = 0.0
    last_ppo_group_norms: dict[str, float] = {}
    last_bc_group_norms: dict[str, float] = {}
    last_group_cosines: dict[str, float] = {}
    transaction_rejected = False
    post_update_reference_kl = 0.0
    post_update_validation_failed = False
    for _ in range(actor_epochs):
        state.actor_optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_ppo = 0.0
        epoch_kl = 0.0
        ratio_parts: list[torch.Tensor] = []
        clipped_parts: list[torch.Tensor] = []
        log_ratio_parts: list[torch.Tensor] = []
        for start in range(0, int(advantages.numel()), gradient_microbatch_size):
            stop = min(start + gradient_microbatch_size, int(advantages.numel()))
            indices = torch.arange(start, stop, device=old_rollout.states.device)
            micro_condition = _condition_index_select(condition_g, indices)
            new_log_probs = []
            for flow_step in range(state.policy.num_steps):
                new_log_probs.append(
                    state.policy.log_prob(
                        old_rollout.next_states[start:stop, flow_step],
                        old_rollout.states[start:stop, flow_step],
                        micro_condition,
                        old_rollout.timesteps[start:stop, flow_step],
                    )
                )
            new_log_probs_tensor = torch.stack(new_log_probs, dim=1)
            sample_fraction = (stop - start) / max(1, int(advantages.numel()))
            if full_ratio_mode == "ais_joint":
                if ogpo_variant(config) in {"chi2", "ca_chi2"}:
                    ppo = full_chain_chipo_ppo_loss(
                        new_log_probs_tensor,
                        old_rollout.log_probs[start:stop],
                        advantages.reshape(-1)[start:stop],
                        clip_eps=_ppo_chain_clip(actor_cfg),
                        beta=float(adv_diag.get("chi2_beta", 0.0)),
                        r_max=float(actor_cfg.get("chi2", {}).get("r_max", 10.0)),
                        logprob_normalizer=chain_logprob_normalizer,
                        log_ratio_clip=float(actor_cfg.get("full_chain_log_ratio_clip", 20.0)),
                    )
                else:
                    ppo = full_chain_ais_ppo_loss(
                        new_log_probs_tensor,
                        old_rollout.log_probs[start:stop],
                        advantages.reshape(-1)[start:stop],
                        clip_eps=_ppo_chain_clip(actor_cfg),
                        logprob_normalizer=chain_logprob_normalizer,
                        upper_ratio_bound=(
                            None
                            if chi2_upper_ratio_bound is None
                            else chi2_upper_ratio_bound[start:stop]
                        ),
                        log_ratio_clip=float(actor_cfg.get("full_chain_log_ratio_clip", 20.0)),
                    )
            else:
                ppo = full_chain_ppo_loss(
                    new_log_probs_tensor,
                    old_rollout.log_probs[start:stop],
                    advantages.reshape(-1)[start:stop],
                    clip_eps=eps_clip[start:stop],
                )
            transition_ref_kl = _policy_kl_device_aware(
                state.policy,
                state.reference_policy,
                old_rollout.states[start:stop, 0],
                micro_condition,
                old_rollout.timesteps[start:stop, 0],
            )
            beta = (
                float(regularization_cfg.get("beta_kl", 0.01))
                * (
                    1.0
                    + kl_uncertainty_scale(regularization_cfg, uncertainty_cfg)
                        * entropy_norm_grouped[start:stop].clamp(0.0, 1.0)
                )
            )
            kl_penalty = (beta * transition_ref_kl).sum() / max(1, int(advantages.numel()))
            chi_loss = new_log_probs_tensor.new_zeros(())
            legacy_extra_chi2_loss = bool(
                actor_cfg.get(
                    "legacy_extra_chi2_loss",
                    actor_cfg.get("chi2", {}).get("legacy_extra_chi2_loss", False),
                )
            )
            if ogpo_variant(config) in {"chi2", "ca_chi2"} and legacy_extra_chi2_loss:
                chi_loss, _ = full_chain_chipo_regularization_loss(
                    new_log_probs_tensor,
                    slow_chain_log_probs_t[start:stop],
                    action_dim=int(state.policy.action_dim),
                    beta=chi2_regularization_beta,
                    normalize_logprob_by_action_dim=bool(
                        actor_cfg.get("normalize_logprob_by_action_dim", True)
                    ),
                    normalize_logprob_by_denoising_steps=bool(
                        actor_cfg.get("normalize_logprob_by_denoising_steps", True)
                    ),
                    log_ratio_clip=float(
                        actor_cfg.get("chi2", {}).get("log_ratio_clip", 20.0)
                    ),
                    ratio_max=float(actor_cfg.get("chi2", {}).get("ratio_max", 20.0)),
                )
            micro_loss = ppo.loss * sample_fraction + kl_penalty + chi_loss * sample_fraction
            micro_loss.backward()
            epoch_loss += float(micro_loss.detach().item())
            epoch_ppo += float(ppo.loss.detach().item()) * sample_fraction
            epoch_kl += float(transition_ref_kl.detach().sum().item()) / max(1, int(advantages.numel()))
            chi2_loss_total += float(chi_loss.detach().item()) * sample_fraction
            ratio_parts.append(ppo.ratio.detach())
            clipped_parts.append(ppo.clipped_ratio.detach())
            log_ratio_parts.append(ppo.log_ratio.detach())
            last_kl_beta = beta.detach().mean()
            del new_log_probs_tensor, micro_loss, transition_ref_kl

        # Success BC/FM is a separate forward/backward component.  It is
        # accumulated once per optimizer transaction, so its coefficient does
        # not get multiplied by the number of PPO gradient microbatches.
        ppo_component_grad_norm = (
            float(grad_norm(state.policy.parameters()))
            if component_grad_diagnostics
            else 0.0
        )
        ppo_group_norms = (
            _gradient_group_norms(state.policy)
            if component_grad_diagnostics
            else {}
        )
        reg_loss, reg_diag = _actor_regularization_loss(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
        )
        component_accumulator: dict[str, torch.Tensor | None] = {
            "bc_square": None,
            "ppo_bc_dot": None,
        }
        component_group_squares: dict[str, torch.Tensor] = {}
        component_group_dots: dict[str, torch.Tensor] = {}
        component_hooks = []
        if component_grad_diagnostics and reg_loss.requires_grad:
            for parameter_name, parameter in state.policy.named_parameters():
                if not parameter.requires_grad:
                    continue
                parameter_group = _actor_gradient_group(parameter_name)

                def capture_component_gradient(
                    gradient,
                    *,
                    parameter=parameter,
                    parameter_group=parameter_group,
                ):
                    detached = gradient.detach().float()
                    square = detached.square().sum()
                    previous_square = component_accumulator["bc_square"]
                    component_accumulator["bc_square"] = (
                        square if previous_square is None else previous_square + square
                    )
                    component_group_squares[parameter_group] = (
                        square
                        if parameter_group not in component_group_squares
                        else component_group_squares[parameter_group] + square
                    )
                    if parameter.grad is not None:
                        dot = (
                            detached
                            * parameter.grad.detach().to(
                                device=detached.device,
                                dtype=detached.dtype,
                            )
                        ).sum()
                        previous_dot = component_accumulator["ppo_bc_dot"]
                        component_accumulator["ppo_bc_dot"] = (
                            dot if previous_dot is None else previous_dot + dot
                        )
                        component_group_dots[parameter_group] = (
                            dot
                            if parameter_group not in component_group_dots
                            else component_group_dots[parameter_group] + dot
                        )
                    return gradient

                component_hooks.append(parameter.register_hook(capture_component_gradient))
        try:
            if reg_loss.requires_grad:
                reg_loss.backward()
        finally:
            for hook in component_hooks:
                hook.remove()
        bc_square = component_accumulator["bc_square"]
        bc_component_grad_norm = (
            float(bc_square.sqrt().item()) if bc_square is not None else 0.0
        )
        ppo_bc_dot_tensor = component_accumulator["ppo_bc_dot"]
        ppo_bc_dot = (
            float(ppo_bc_dot_tensor.item()) if ppo_bc_dot_tensor is not None else 0.0
        )
        ppo_bc_grad_cosine = (
            ppo_bc_dot / (ppo_component_grad_norm * bc_component_grad_norm)
            if ppo_component_grad_norm > 0.0 and bc_component_grad_norm > 0.0
            else 0.0
        )
        bc_group_norms = {
            group: float(square.sqrt().item())
            for group, square in component_group_squares.items()
        }
        group_cosines = {}
        for group in set(ppo_group_norms) | set(bc_group_norms):
            ppo_norm = float(ppo_group_norms.get(group, 0.0))
            bc_norm = float(bc_group_norms.get(group, 0.0))
            dot = component_group_dots.get(group)
            group_cosines[group] = (
                float(dot.item()) / (ppo_norm * bc_norm)
                if dot is not None and ppo_norm > 0.0 and bc_norm > 0.0
                else 0.0
            )
        last_ppo_group_norms = ppo_group_norms
        last_bc_group_norms = bc_group_norms
        last_group_cosines = group_cosines
        ppo_component_grad_total += ppo_component_grad_norm
        bc_component_grad_total += bc_component_grad_norm
        ppo_bc_grad_cosine_total += ppo_bc_grad_cosine
        epoch_loss += float(reg_loss.detach().item())
        assert_no_gradients(state.critic, "critic")
        if state.divl is not None:
            assert_no_gradients(state.divl, "divl")
        assert_no_gradients(state.reference_policy, "reference_policy")
        if state.slow_policy is not None:
            assert_no_gradients(state.slow_policy, "slow_policy")
        if role_dp:
            from .actor_role_dp import average_gradients
            average_gradients(state.policy.parameters())
        actor_grad_norm = float(grad_norm(state.policy.parameters()))
        if diagnostic_no_optimizer_step:
            lambda_success = float(regularization_cfg.get("lambda_success", 0.0))
            if lambda_success <= 0.0:
                raise ValueError("pure PPO/BC diagnostics require lambda_success > 0")
            bc_unweighted_norm = bc_component_grad_norm / lambda_success
            bc_0p1_norm = 0.1 * bc_unweighted_norm
            max_grad_norm = float(actor_cfg.get("max_grad_norm", 1.0))
            result: dict[str, float] = {
                "actor_loss": float(epoch_loss),
                "full_ppo_loss": float(epoch_ppo),
                "success_buffer_loss": float(reg_diag.get("success_buffer_loss", 0.0)),
                "success_bc_weighted_loss": float(
                    lambda_success * float(reg_diag.get("success_buffer_loss", 0.0))
                ),
                "ppo_component_grad_norm": ppo_component_grad_norm,
                "bc_component_grad_norm": bc_component_grad_norm,
                "bc_unweighted_component_grad_norm": bc_unweighted_norm,
                "bc_0p1_component_grad_norm": bc_0p1_norm,
                "bc0p1_to_ppo_grad_norm_ratio": bc_0p1_norm
                / (ppo_component_grad_norm + 1e-12),
                "ppo_bc_grad_cosine": ppo_bc_grad_cosine,
                "combined_grad_norm": actor_grad_norm,
                "gradient_clip_threshold": max_grad_norm,
                "would_gradient_clip": float(actor_grad_norm > max_grad_norm),
                "ppo_zero_gradient": float(ppo_component_grad_norm <= 1e-12),
                "component_grad_norm_diagnostics": 1.0,
                **{
                    key: float(value)
                    for key, value in adv_diag.items()
                    if isinstance(value, (int, float))
                },
            }
            for group in ("backbone", "action_expert", "other"):
                ppo_group = float(last_ppo_group_norms.get(group, 0.0))
                bc_weighted_group = float(last_bc_group_norms.get(group, 0.0))
                bc_unweighted_group = bc_weighted_group / lambda_success
                bc_0p1_group = 0.1 * bc_unweighted_group
                result.update(
                    {
                        f"{group}_ppo_grad_norm": ppo_group,
                        f"{group}_bc_unweighted_grad_norm": bc_unweighted_group,
                        f"{group}_bc_0p1_grad_norm": bc_0p1_group,
                        f"{group}_bc0p1_to_ppo_grad_norm_ratio": bc_0p1_group
                        / (ppo_group + 1e-12),
                        f"{group}_ppo_bc_grad_cosine": float(
                            last_group_cosines.get(group, 0.0)
                        ),
                    }
                )
            state.actor_optimizer.zero_grad(set_to_none=True)
            return result
        torch.nn.utils.clip_grad_norm_(
            state.policy.parameters(), float(actor_cfg.get("max_grad_norm", 1.0))
        )
        loss_total += epoch_loss
        ppo_total += epoch_ppo
        kl_total += epoch_kl
        grad_total += actor_grad_norm
        last_ratio = torch.cat(ratio_parts)
        last_clipped_ratio = torch.cat(clipped_parts)
        last_log_ratio = torch.cat(log_ratio_parts)
        last_reg_diag = reg_diag
        last_ppo = ppo
        state.actor_optimizer.step()
        post_update_reference_kl = _full_chain_policy_kl(
            state.policy,
            state.reference_policy,
            old_rollout,
            condition_g,
            microbatch_size=gradient_microbatch_size,
            normalizer=chain_logprob_normalizer,
        )
        dp_invalid = False
        if role_dp:
            from .actor_role_dp import global_validation
            post_update_reference_kl, dp_invalid = global_validation(
                post_update_reference_kl,
                not all(math.isfinite(value) for value in (epoch_loss, epoch_ppo, epoch_kl, actor_grad_norm)),
                policy_device,
            )
        violates_post_update_kl = (
            not math.isfinite(post_update_reference_kl)
            or post_update_reference_kl > post_update_kl_limit
        )
        post_update_validation_failed = dp_invalid or violates_post_update_kl or not all(
            math.isfinite(value)
            for value in (epoch_loss, epoch_ppo, epoch_kl, actor_grad_norm)
        )
        if post_update_validation_failed and post_update_action == "rollback_cpu":
            if transaction_snapshot is None:
                raise AssertionError("rollback action requires an actor snapshot")
            _restore_actor_rollback_state(state, transaction_snapshot)
            rejected_actor_epochs += 1
            transaction_rejected = True
            accepted_actor_epochs = 0
            break
        accepted_actor_epochs += 1
        if violates_post_update_kl and post_update_action == "stop":
            break
    if last_ppo is None or last_ratio is None or last_clipped_ratio is None or last_log_ratio is None:
        raise AssertionError("full-chain actor update did not execute an epoch")
    transaction_snapshot = None
    # Re-evaluate after the optimizer transaction.  These are distinct from
    # the pre-update on-policy ratios used by PPO and expose real policy drift.
    post_log_probs = _full_chain_log_probs_no_grad(
        state.policy,
        old_rollout,
        condition_g,
        microbatch_size=gradient_microbatch_size,
    )
    post_ratio, post_ratio_stats = full_chain_chipo_ratio(
        post_log_probs,
        old_rollout.log_probs,
        action_dim=int(state.policy.action_dim),
        normalize_logprob_by_action_dim=bool(actor_cfg.get("normalize_logprob_by_action_dim", False)),
        normalize_logprob_by_denoising_steps=bool(actor_cfg.get("normalize_logprob_by_denoising_steps", False)),
        log_ratio_clip=float(actor_cfg.get("full_chain_log_ratio_clip", 20.0)),
        ratio_max=float(actor_cfg.get("post_ratio_max", 1.0e6)),
    )
    post_old_kl = float(
        ((post_ratio - 1.0) - post_log_probs.sub(old_rollout.log_probs).sum(dim=1) / chain_logprob_normalizer)
        .mean()
        .item()
    )
    if state.old_policy is not state.policy:
        # When a separate old-policy module exists, report the exact Gaussian
        # chain KL; full PI0.5 finetune uses a functional on-policy snapshot,
        # for which the ratio-based approximation above is the non-duplicating
        # equivalent.
        post_old_kl = _full_chain_policy_kl(
            state.policy,
            state.old_policy,
            old_rollout,
            condition_g,
            microbatch_size=gradient_microbatch_size,
            normalizer=chain_logprob_normalizer,
        )
    slow_kl = 0.0
    if state.slow_policy is not None:
        slow_kl = _full_chain_policy_kl(
            state.policy,
            state.slow_policy,
            old_rollout,
            condition_g,
            microbatch_size=gradient_microbatch_size,
            normalizer=chain_logprob_normalizer,
        )
    post_clip_eps = _ppo_chain_clip(actor_cfg)
    post_upper = torch.full_like(post_ratio, 1.0 + post_clip_eps)
    if chi2_upper_ratio_bound is not None:
        # The upper bound is candidate-specific but is constant across a
        # denoising chain; it is already aligned with rollout candidate order.
        post_upper = torch.minimum(post_upper, chi2_upper_ratio_bound.detach())
    post_clipped = torch.maximum(
        torch.minimum(post_ratio, post_upper),
        torch.full_like(post_ratio, 1.0 - post_clip_eps),
    )
    # Validation and any rollback have completed above.  At this point an
    # accepted transaction is exactly one whose updated actor parameters were
    # retained; a rejected transaction has already restored current+optimizer.
    update_accepted = bool(accepted_actor_epochs > 0 and not transaction_rejected)
    result = {
        "actor_loss": loss_total / actor_epochs,
        "full_ppo_loss": ppo_total / actor_epochs,
        "reference_kl": kl_total / actor_epochs,
        "post_update_reference_kl": post_update_reference_kl,
        "post_update_kl_exceeded": float(
            (not math.isfinite(post_update_reference_kl))
            or post_update_reference_kl > post_update_kl_limit
        ),
        "post_update_validation_failed": float(post_update_validation_failed),
        "post_update_kl_action_rollback_cpu": float(post_update_action == "rollback_cpu"),
        "post_update_kl_action_stop": float(post_update_action == "stop"),
        "policy_reference_kl_hard_limit": post_update_kl_limit,
        "policy_reference_kl_hard_limit_exceeded": float(
            (not math.isfinite(post_update_reference_kl))
            or post_update_reference_kl > post_update_kl_limit
        ),
        "policy_reference_kl_utilization": (
            post_update_reference_kl / post_update_kl_limit
            if math.isfinite(post_update_kl_limit) and post_update_kl_limit > 0.0
            else 0.0
        ),
        "actor_update_accepted": float(update_accepted),
        "actor_update_rejected": float(
            accepted_actor_epochs == 0 and rejected_actor_epochs > 0
        ),
        "accepted_actor_epochs": float(accepted_actor_epochs),
        "rejected_actor_epochs": float(rejected_actor_epochs),
        "reference_kl_beta": float(last_kl_beta.detach().item()),
        "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
        "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
        "actor_epochs": float(actor_epochs),
        "importance_ratio_mean": float(last_ratio.mean().item()),
        "importance_ratio_std": float(last_ratio.std(unbiased=False).item()),
        "importance_ratio_min": float(last_ratio.min().item()),
        "importance_ratio_max": float(last_ratio.max().item()),
        "ppo_clip_fraction": float(last_ratio.ne(last_clipped_ratio).float().mean().item()),
        "importance_ratio_p5": float(torch.quantile(last_ratio, 0.05).item()),
        "importance_ratio_p50": float(torch.quantile(last_ratio, 0.50).item()),
        "importance_ratio_p95": float(torch.quantile(last_ratio, 0.95).item()),
        "full_chain_log_ratio_mean": float(last_log_ratio.mean().item()),
        "full_chain_log_ratio_std": float(last_log_ratio.std(unbiased=False).item()),
        "full_chain_log_ratio_min": float(last_log_ratio.min().item()),
        "full_chain_log_ratio_max": float(last_log_ratio.max().item()),
        "full_chain_logprob_normalizer": chain_logprob_normalizer,
        "full_chain_joint_ratio": 1.0,
        "full_chain_one_clip": 1.0,
        "ppo_logratio_mean": float(last_log_ratio.mean().item()),
        "ppo_logratio_std": float(last_log_ratio.std(unbiased=False).item()),
        "chi2_upper_ratio_bound_mean": (
            float(chi2_upper_ratio_bound.mean().item())
            if chi2_upper_ratio_bound is not None
            else 0.0
        ),
        "actor_grad_norm": grad_total / actor_epochs,
        "actor_parameter_update_norm_estimate": float(
            (grad_total / max(1, actor_epochs))
            * float(actor_cfg.get("learning_rate", 1.0e-4))
        ),
        # Record the effective production update geometry.  These diagnostics
        # make it possible to distinguish state batching from candidate
        # grouping and to verify gradient accumulation from the run artifact.
        "state_batch_size": float(batch.batch_size),
        "candidate_group_size": float(group_size),
        "logical_candidate_batch_size": float(advantages.numel()),
        "gradient_microbatch_size": float(gradient_microbatch_size),
        "gradient_accumulation_steps": float(
            math.ceil(int(advantages.numel()) / gradient_microbatch_size)
        ),
        "policy_entropy": _policy_entropy(state.policy),
        "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
        **last_reg_diag,
        **adv_diag,
        **chi2_diag,
        **sde_diag,
        "post_update_importance_ratio_mean": post_ratio_stats.ratio_mean,
        "post_update_importance_ratio_std": post_ratio_stats.ratio_std,
        "post_update_importance_ratio_p5": float(torch.quantile(post_ratio, 0.05).item()),
        "post_update_importance_ratio_p50": float(torch.quantile(post_ratio, 0.50).item()),
        "post_update_importance_ratio_p95": float(torch.quantile(post_ratio, 0.95).item()),
        "post_update_importance_ratio_min": post_ratio_stats.ratio_min,
        "post_update_importance_ratio_max": post_ratio_stats.ratio_max,
        "post_update_full_chain_log_ratio_mean": post_ratio_stats.log_ratio_mean,
        "post_update_full_chain_log_ratio_std": post_ratio_stats.log_ratio_std,
        "post_update_full_chain_log_ratio_min": post_ratio_stats.log_ratio_min,
        "post_update_full_chain_log_ratio_max": post_ratio_stats.log_ratio_max,
        "post_update_ppo_clip_fraction": float(
            post_ratio.ne(post_clipped).float().mean().item()
        ),
        "current_old_policy_approx_kl": post_old_kl,
        "current_old_policy_kl": post_old_kl,
        "ppo_approx_kl": post_old_kl,
        "current_slow_policy_kl": slow_kl,
        "chi2_current_slow_policy_kl": slow_kl,
        "current_slow_logratio_mean": float(
            chi2_diag.get("chi2_selected_log_ratio_mean", 0.0)
        ),
        "current_slow_logratio_std": float(
            chi2_diag.get("chi2_selected_log_ratio_std", 0.0)
        ),
        "current_slow_approx_kl": slow_kl,
        "success_bc_weighted_loss": float(
            float(regularization_cfg.get("lambda_success", 0.0))
            * float(last_reg_diag.get("success_buffer_loss", 0.0))
        ),
        "total_actor_loss": loss_total / actor_epochs,
        "critic_offloaded_during_actor_optimization": float(offload_frozen_critic),
    }
    result["chi2_advantage_penalty_mean"] = float(
        adv_diag.get("chi2_advantage_penalty_mean", 0.0)
    )
    result["chi2_loss"] = float(chi2_loss_total / max(1, actor_epochs))
    result["chi2_regularization_loss"] = result["chi2_loss"]
    result["chi2_regularization_beta"] = chi2_regularization_beta
    if component_grad_diagnostics:
        ppo_component = ppo_component_grad_total / actor_epochs
        bc_component = bc_component_grad_total / actor_epochs
        result.update(
            {
                "ppo_component_grad_norm": ppo_component,
                "bc_component_grad_norm": bc_component,
                "bc_to_ppo_grad_norm_ratio": bc_component / (ppo_component + 1e-12),
                "ppo_bc_grad_cosine": ppo_bc_grad_cosine_total / actor_epochs,
                "component_grad_norm_diagnostics": 1.0,
            }
        )
    lifecycle_metrics = finalize_actor_update_transaction(
        state,
        accepted=update_accepted,
        config=config,
    )
    result.update(lifecycle_metrics)
    return result


def flash_actor_update(
    state: OGPOTrainState,
    batch: ChunkBatch,
    config: dict[str, Any],
    *,
    fm_batch: ChunkBatch | None = None,
    success_batch: ChunkBatch | None = None,
    actor_step: int | None = None,
) -> dict[str, float]:
    actor_cfg = config.get("actor", {})
    flow_cfg = config.get("flow", {})
    regularization_cfg = config.get("regularization", {})
    uncertainty_cfg = config.get("uncertainty", {})
    batch = batch.to(next(state.policy.parameters()).device)
    group_size = int(actor_cfg.get("candidate_group_size", actor_cfg.get("group_size", 4)))
    gradient_microbatch_size = int(actor_cfg.get("gradient_microbatch_size", group_size))
    kl_eval_microbatch_size = int(
        actor_cfg.get("kl_eval_microbatch_size", gradient_microbatch_size)
    )
    data_parallel_devices = int(actor_cfg.get("data_parallel_devices", 1))
    distributed_gradient_reduction = str(
        actor_cfg.get("distributed_gradient_reduction", "manual")
    ).lower()
    rollout_state_microbatch_size = int(
        actor_cfg.get("rollout_state_microbatch_size", batch.batch_size)
    )
    success_update_period = int(regularization_cfg.get("success_update_period", 1))
    if (
        group_size <= 0
        or gradient_microbatch_size <= 0
        or kl_eval_microbatch_size <= 0
        or rollout_state_microbatch_size <= 0
        or data_parallel_devices <= 0
        or success_update_period <= 0
    ):
        raise ValueError(
            "candidate_group_size, gradient_microbatch_size, "
            "kl_eval_microbatch_size, rollout_state_microbatch_size, "
            "data_parallel_devices, and success_update_period must be positive"
        )
    if distributed_gradient_reduction not in {"manual", "pmean"}:
        raise ValueError(
            "actor.distributed_gradient_reduction must be 'manual' or 'pmean'; "
            f"got {distributed_gradient_reduction!r}"
        )
    success_update_due = (
        actor_step is None or int(actor_step) % success_update_period == 0
    )
    sampling_step = int(state.step if actor_step is None else actor_step)
    sampling_seed = int(config.get("training", {}).get("seed", 0)) + sampling_step
    actor_epochs = max(1, int(actor_cfg.get("actor_epochs_per_rollout", 1)))
    stabilize_on_policy_statistics = bool(
        actor_cfg.get("stabilize_single_epoch_bf16_statistics", False)
    )
    if stabilize_on_policy_statistics:
        sync_period = int(actor_cfg.get("old_policy_sync_period", 1))
        old_policy_ema = float(actor_cfg.get("old_policy_ema", 0.0))
        if sync_period != 1 or old_policy_ema != 0.0:
            raise ValueError(
                "actor.stabilize_single_epoch_bf16_statistics requires "
                "old_policy_sync_period=1 and old_policy_ema=0"
            )
    regularization_epochs = int(
        regularization_cfg.get("actor_regularization_epochs_per_rollout", 1)
    )
    if regularization_epochs not in {0, 1}:
        raise ValueError(
            "regularization.actor_regularization_epochs_per_rollout currently "
            "supports only 0 or 1"
        )
    selected_steps = _select_flash_steps(
        flow_cfg,
        batch_size=batch.batch_size,
        num_steps=state.policy.num_steps,
        device=batch.observations.device,
        seed=sampling_seed + 101,
    )
    state.critic_optimizer.zero_grad(set_to_none=True)
    condition = _policy_condition(state.policy, batch)
    condition_g = state.policy.repeat_condition(condition, group_size)

    if _using_jax_actor(state.policy):
        assert isinstance(state.policy, PI05JaxFlowPolicy)
        assert isinstance(state.old_policy, PI05JaxFlowPolicy)
        assert isinstance(state.reference_policy, PI05JaxFlowPolicy)
        reference_kl_action_horizon = int(
            actor_cfg.get(
                "reference_kl_action_horizon",
                state.policy.model_horizon,
            )
        )
        if not 1 <= reference_kl_action_horizon <= state.policy.model_horizon:
            raise ValueError(
                "actor.reference_kl_action_horizon must be in "
                f"[1, {state.policy.model_horizon}], got "
                f"{reference_kl_action_horizon}"
            )
        reference_kl_event_dim = (
            reference_kl_action_horizon * state.policy.environment_action_dim
        )
        ppo_action_horizon = int(
            actor_cfg.get("ppo_action_horizon", state.policy.model_horizon)
        )
        if not 1 <= ppo_action_horizon <= state.policy.model_horizon:
            raise ValueError(
                "actor.ppo_action_horizon must be in "
                f"[1, {state.policy.model_horizon}], got {ppo_action_horizon}"
            )
        ppo_event_dim = ppo_action_horizon * state.policy.environment_action_dim
        full_reference_kl_event_dim = (
            state.policy.model_horizon * state.policy.flow_action_dim
        )

        def _select_environment_prefix(value, action_horizon):
            return value.reshape(
                value.shape[0],
                state.policy.model_horizon,
                state.policy.flow_action_dim,
            )[:, :action_horizon, : state.policy.environment_action_dim].reshape(
                value.shape[0], -1
            )

        def _select_environment_prefix_numpy(value, action_horizon):
            return value.reshape(
                value.shape[0],
                state.policy.model_horizon,
                state.policy.flow_action_dim,
            )[:, :action_horizon, : state.policy.environment_action_dim].reshape(
                value.shape[0], -1
            )

        def _ppo_log_prob(x, mean, log_std):
            return jax_gaussian_log_prob(
                _select_environment_prefix(x, ppo_action_horizon),
                _select_environment_prefix(mean, ppo_action_horizon),
                _select_environment_prefix(log_std, ppo_action_horizon),
            )

        def _prefix_reference_kl(
            mean_p,
            log_std_p,
            mean_q,
            log_std_q,
        ):
            return jax_gaussian_kl_diag(
                _select_environment_prefix(mean_p, reference_kl_action_horizon),
                _select_environment_prefix(log_std_p, reference_kl_action_horizon),
                _select_environment_prefix(mean_q, reference_kl_action_horizon),
                _select_environment_prefix(log_std_q, reference_kl_action_horizon),
            )

        flow_spec = OpenPIJaxFlowSpec(state.policy.num_steps)
        jax_observation = policy_observation_to_jax(condition.observation)
        jax_observation_g = policy_observation_to_jax(condition_g.observation)
        local_devices = jax.local_devices()
        if data_parallel_devices > len(local_devices):
            raise RuntimeError(
                f"actor.data_parallel_devices={data_parallel_devices}, but JAX sees only "
                f"{len(local_devices)} local devices: {local_devices}"
            )
        actor_devices = local_devices[:data_parallel_devices]
        parallel_frozen_statistics = bool(
            actor_cfg.get("parallel_frozen_statistics", False)
        )
        old_statistics_device_index = int(
            actor_cfg.get("old_statistics_device_index", 0)
        )
        reference_statistics_device_index = int(
            actor_cfg.get(
                "reference_statistics_device_index",
                1 if parallel_frozen_statistics and data_parallel_devices > 1 else 0,
            )
        )
        regularization_device_index = int(
            actor_cfg.get(
                "regularization_device_index",
                2 if parallel_frozen_statistics and data_parallel_devices > 2 else 0,
            )
        )
        statistics_device_indices = (
            old_statistics_device_index,
            reference_statistics_device_index,
            regularization_device_index,
        )
        if any(
            index < 0 or index >= data_parallel_devices
            for index in statistics_device_indices
        ):
            raise ValueError(
                "old/reference statistics and regularization device indices must "
                f"be in [0, {data_parallel_devices}); got {statistics_device_indices}"
            )
        if parallel_frozen_statistics and data_parallel_devices < 3:
            raise ValueError(
                "actor.parallel_frozen_statistics requires at least three JAX devices"
            )
        total_candidates = batch.batch_size * group_size
        if total_candidates % data_parallel_devices:
            raise ValueError(
                f"effective candidate batch {total_candidates} must be divisible by "
                f"actor.data_parallel_devices={data_parallel_devices}"
            )
        if data_parallel_devices > 1:
            # Orbax restores arrays with the sharding saved in the checkpoint
            # (e.g. 4-way sharded for a 4-GPU run). On a different topology
            # (8 GPUs) may_alias=True device_put aliases one shard instead of
            # consolidating, so pmap broadcast arguments land on the wrong
            # device. Force a full copy to the target device.
            state.policy.actor_state = _jax_tree_copy_to_device(
                state.policy.actor_state,
                actor_devices[0],
            )
            state.old_policy.actor_state = _jax_tree_copy_to_device(
                state.old_policy.actor_state,
                actor_devices[old_statistics_device_index],
            )
            reference_statistics_storage_index = (
                old_statistics_device_index
                if stabilize_on_policy_statistics
                else reference_statistics_device_index
            )
            state.reference_policy.actor_state = _jax_tree_copy_to_device(
                state.reference_policy.actor_state,
                actor_devices[reference_statistics_storage_index],
            )
            state.policy.actor_opt_state = _jax_tree_copy_to_device(
                state.policy.actor_opt_state,
                actor_devices[0],
            )
            jax.effects_barrier()
            gc.collect()

        with torch.no_grad():
            rng = jax.random.PRNGKey(sampling_seed + 17)
            selected_steps_jax = jnp.asarray(selected_steps.detach().cpu().numpy())
            selected_steps_grouped_jax = jnp.repeat(selected_steps_jax, group_size)
            if data_parallel_devices > 1:
                rollout_cache_key = (
                    data_parallel_devices,
                    state.old_policy.num_steps,
                    state.old_policy.sde_mode,
                )
                rollout_cache = getattr(state.policy, "_flash_dp_rollout_cache", None)
                if rollout_cache is None or rollout_cache[0] != rollout_cache_key:
                    old_graphdef = state.old_policy.actor_graphdef

                    def _distributed_rollout(actor_state, observation, selected_step, device_rng):
                        actor = nnx.merge(old_graphdef, actor_state)
                        rollout = sample_jax_flash_rollout(
                            actor=actor,
                            flow_spec=OpenPIJaxFlowSpec(state.old_policy.num_steps),
                            observation=observation,
                            group_size=1,
                            selected_step=selected_step,
                            rng=device_rng,
                            sde_mode=state.old_policy.sde_mode,
                        )
                        return rollout.x_t, rollout.x_prev, rollout.timestep, rollout.endpoint

                    distributed_rollout = jax.pmap(
                        _distributed_rollout,
                        in_axes=(None, 0, 0, 0),
                        devices=actor_devices,
                    )
                    state.policy._flash_dp_rollout_cache = (
                        rollout_cache_key,
                        distributed_rollout,
                    )
                else:
                    distributed_rollout = rollout_cache[1]
                rollout_result = distributed_rollout(
                    state.old_policy.actor_state,
                    _shard_jax_batch(jax_observation_g, data_parallel_devices),
                    _shard_jax_batch(selected_steps_grouped_jax, data_parallel_devices),
                    jax.random.split(rng, data_parallel_devices),
                )
                old_x_t, old_x_prev, old_timestep, old_endpoint = (
                    value.reshape(total_candidates, *value.shape[2:])
                    for value in rollout_result
                )
            else:
                candidate_rngs = jax.random.split(rng, group_size)
                old_actor = nnx.merge(
                    state.old_policy.actor_graphdef,
                    state.old_policy.actor_state,
                )
                rollout_parts = []
                for candidate in range(group_size):
                    state_parts = [
                        _sample_frozen_jax_flash_rollout(
                            old_actor,
                            _slice_jax_batch(jax_observation, start, stop),
                            selected_step=selected_steps_jax[start:stop],
                            rng=jax.random.fold_in(candidate_rngs[candidate], start),
                            num_steps=state.old_policy.num_steps,
                            sde_mode=state.old_policy.sde_mode,
                            group_size=1,
                        )
                        for start in range(0, batch.batch_size, rollout_state_microbatch_size)
                        for stop in [
                            min(start + rollout_state_microbatch_size, batch.batch_size)
                        ]
                    ]
                    rollout_parts.append(
                        tuple(
                            jnp.concatenate(
                                [part[component] for part in state_parts],
                                axis=0,
                            )
                            for component in range(4)
                        )
                    )
                old_x_t, old_x_prev, old_timestep, old_endpoint = (
                    _stack_candidate_rollouts([part[index] for part in rollout_parts])
                    for index in range(4)
                )
            old_endpoint_torch = torch.as_tensor(
                np.asarray(old_endpoint).copy(),
                device=batch.observations.device,
                dtype=batch.action_chunks.dtype,
            )
            environment_endpoint = state.old_policy.flat_actions_to_environment(old_endpoint_torch, condition_g)
            endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
            advantages, adv_diag = conservative_advantages_for_candidates(
                state, batch.observations, endpoint, batch, config
            )
            entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
            eps_clip = (
                actor_clip_for_uncertainty(entropy_norm, actor_cfg, uncertainty_cfg)
                .unsqueeze(-1)
                .expand(batch.batch_size, group_size)
                .reshape(-1)
            )
        selected_counts = torch.bincount(selected_steps.detach().cpu(), minlength=state.policy.num_steps)
        loss_total = 0.0
        flash_total = 0.0
        raw_flash_total = 0.0
        kl_total = 0.0
        rectification_total = 0.0
        raw_loss_by_step = [0.0] * state.policy.num_steps
        raw_grad_by_step = [0.0] * state.policy.num_steps
        rectified_grad_by_step = [0.0] * state.policy.num_steps
        selected_steps_grouped = selected_steps.repeat_interleave(group_size)
        rectification = _flash_rectification_weight(
            state,
            flow_cfg,
            timestep=torch.as_tensor(
                np.asarray(old_timestep).copy(),
                device=batch.observations.device,
                dtype=batch.action_chunks.dtype,
            ),
            selected_steps=selected_steps,
            group_size=group_size,
        )
        jax_regularization = _prepare_jax_regularization(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
            enable_success=success_update_due,
            seed_step=sampling_seed,
        )
        beta_base = float(regularization_cfg.get("beta_kl", 0.01))
        adapt_kl_beta = bool(uncertainty_cfg.get("adapt_kl_beta", False))
        uncertainty_scale = float(regularization_cfg.get("kl_uncertainty_scale", 1.0))
        normalize_logprob_by_action_dim = bool(
            actor_cfg.get("normalize_logprob_by_action_dim", False)
        )
        logprob_normalizer = (
            float(ppo_event_dim) if normalize_logprob_by_action_dim else 1.0
        )
        old_timestep = old_timestep.reshape(-1, 1)
        advantages_j = jnp.asarray(advantages.reshape(-1).detach().cpu().numpy())
        eps_clip_j = jnp.asarray(eps_clip.detach().cpu().numpy())
        rectification_j = jnp.asarray(rectification.detach().cpu().numpy())
        entropy_norm_j = jnp.repeat(
            jnp.asarray(entropy_norm.detach().cpu().numpy()),
            group_size,
        )
        num_candidates = int(advantages_j.shape[0])

        def _policy_loss_fn(
            actor_state,
            old_actor_state,
            reference_actor_state,
            x_prev,
            x_t,
            observation,
            timestep,
            candidate_advantages,
            candidate_eps_clip,
            candidate_rectification,
            candidate_entropy,
            fixed_old_log_prob=None,
            fixed_old_mean=None,
            fixed_old_log_std=None,
            fixed_reference_mean=None,
            fixed_reference_log_std=None,
            anchor_on_policy_statistics=0.0,
        ):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            loss_old_actor = nnx.merge(
                state.old_policy.actor_graphdef,
                old_actor_state,
            )
            loss_reference_actor = nnx.merge(
                state.reference_policy.actor_graphdef,
                reference_actor_state,
            )
            # Compute old/current probabilities in the same transformed trace.
            # PI0.5's bf16 fusion otherwise produces a large spurious ratio
            # offset when summing over the 660-dimensional action chunk.
            old_mean = jax_transition_mean(
                actor=loss_old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            old_log_std = jax_transition_log_std(
                actor=loss_old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            old_log_prob = jax.lax.stop_gradient(
                _ppo_log_prob(x_prev, old_mean, old_log_std)
            ) / logprob_normalizer
            current_mean = jax_transition_mean(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            current_log_std = jax_transition_log_std(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            new_log_prob = _ppo_log_prob(
                x_prev,
                current_mean,
                current_log_std,
            ) / logprob_normalizer
            if stabilize_on_policy_statistics:
                if any(
                    value is None
                    for value in (
                        fixed_old_log_prob,
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                    )
                ):
                    raise ValueError("stabilized on-policy statistics were not supplied")
                old_log_prob = fixed_old_log_prob
                old_mean = fixed_old_mean
                old_log_std = fixed_old_log_std
                ppo_new_log_prob = _conditionally_anchor_current_to_old_value(
                    new_log_prob,
                    old_log_prob,
                    anchor_on_policy_statistics,
                )
                kl_current_mean = _conditionally_anchor_current_to_old_value(
                    current_mean,
                    old_mean,
                    anchor_on_policy_statistics,
                )
                kl_current_log_std = _conditionally_anchor_current_to_old_value(
                    current_log_std,
                    old_log_std,
                    anchor_on_policy_statistics,
                )
            else:
                ppo_new_log_prob = new_log_prob
                kl_current_mean = current_mean
                kl_current_log_std = current_log_std
            raw_flash = jax_flash_ppo_loss(
                ppo_new_log_prob,
                old_log_prob,
                candidate_advantages,
                clip_eps=candidate_eps_clip,
                rectification_weight=jnp.ones_like(candidate_rectification),
            )
            flash = jax_flash_ppo_loss(
                ppo_new_log_prob,
                old_log_prob,
                candidate_advantages,
                clip_eps=candidate_eps_clip,
                rectification_weight=candidate_rectification,
            )
            reference_mean = jax_transition_mean(
                actor=loss_reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            reference_log_std = jax_transition_log_std(
                actor=loss_reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            if stabilize_on_policy_statistics:
                reference_mean = fixed_reference_mean
                reference_log_std = fixed_reference_log_std
            transition_ref_kl = _prefix_reference_kl(
                kl_current_mean,
                kl_current_log_std,
                reference_mean,
                reference_log_std,
            )
            kl_penalty, ref_kl, beta = jax_state_adaptive_kl_penalty(
                transition_ref_kl,
                candidate_entropy,
                group_size=1,
                beta_base=beta_base,
                adapt_kl_beta=adapt_kl_beta,
                uncertainty_scale=uncertainty_scale,
            )
            total = flash.loss + kl_penalty
            aux = {
                "flash_loss": flash.loss,
                "raw_flash_loss": raw_flash.loss,
                "ref_kl": ref_kl,
                "beta": beta,
                "ratio": flash.ratio,
                "clipped_ratio": flash.clipped_ratio,
                "per_sample_loss": raw_flash.per_sample_loss,
            }
            return total, aux

        def _reference_kl_fn(
            actor_state,
            old_actor_state,
            reference_actor_state,
            x_t,
            observation,
            timestep,
        ):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            old_actor = nnx.merge(state.old_policy.actor_graphdef, old_actor_state)
            reference_actor = nnx.merge(
                state.reference_policy.actor_graphdef,
                reference_actor_state,
            )
            current_mean = jax_transition_mean(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            current_log_std = jax_transition_log_std(
                actor=actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.policy.sde_mode,
            )
            reference_mean = jax_transition_mean(
                actor=reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            reference_log_std = jax_transition_log_std(
                actor=reference_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.reference_policy.sde_mode,
            )
            old_mean = jax_transition_mean(
                actor=old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                observation=observation,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            old_log_std = jax_transition_log_std(
                actor=old_actor,
                flow_spec=flow_spec,
                x_t=x_t,
                timestep=timestep,
                sde_mode=state.old_policy.sde_mode,
            )
            return (
                jnp.mean(
                    _prefix_reference_kl(
                        current_mean,
                        current_log_std,
                        reference_mean,
                        reference_log_std,
                    )
                ),
                jnp.mean(
                    jax_gaussian_kl_diag(
                        current_mean,
                        current_log_std,
                        reference_mean,
                        reference_log_std,
                    )
                ),
                jnp.mean(
                    jax_gaussian_kl_diag(
                        _select_environment_prefix(current_mean, ppo_action_horizon),
                        _select_environment_prefix(current_log_std, ppo_action_horizon),
                        _select_environment_prefix(old_mean, ppo_action_horizon),
                        _select_environment_prefix(old_log_std, ppo_action_horizon),
                    )
                ),
            )

        def _flow_matching_component(actor_state, inputs):
            actor = nnx.merge(state.policy.actor_graphdef, actor_state)
            return jax_flow_matching_loss(actor=actor, **inputs)

        if data_parallel_devices > 1:
            dp_cache_key = (
                data_parallel_devices,
                state.policy.num_steps,
                state.policy.sde_mode,
                state.reference_policy.sde_mode,
                beta_base,
                adapt_kl_beta,
                uncertainty_scale,
                normalize_logprob_by_action_dim,
                parallel_frozen_statistics,
                old_statistics_device_index,
                reference_statistics_device_index,
                regularization_device_index,
                distributed_gradient_reduction,
                stabilize_on_policy_statistics,
                reference_kl_event_dim,
                ppo_event_dim,
            )
            dp_cache = getattr(state.policy, "_flash_dp_train_cache", None)
            if dp_cache is None or dp_cache[0] != dp_cache_key:
                policy_graphdef = state.policy.actor_graphdef
                old_graphdef = state.old_policy.actor_graphdef
                reference_graphdef = state.reference_policy.actor_graphdef

                def _frozen_policy_statistics(
                    frozen_actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    frozen_actor = nnx.merge(old_graphdef, frozen_actor_state)
                    frozen_mean = jax_transition_mean(
                        actor=frozen_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.old_policy.sde_mode,
                    )
                    frozen_log_std = jax_transition_log_std(
                        actor=frozen_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.old_policy.sde_mode,
                    )
                    frozen_log_prob = (
                        _ppo_log_prob(x_prev, frozen_mean, frozen_log_std)
                        / logprob_normalizer
                    )
                    return frozen_log_prob, frozen_mean, frozen_log_std

                def _reference_policy_statistics(
                    reference_actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                ):
                    reference_actor = nnx.merge(reference_graphdef, reference_actor_state)
                    reference_mean = jax_transition_mean(
                        actor=reference_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.reference_policy.sde_mode,
                    )
                    reference_log_std = jax_transition_log_std(
                        actor=reference_actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.reference_policy.sde_mode,
                    )
                    reference_log_prob = (
                        _ppo_log_prob(
                            x_prev,
                            reference_mean,
                            reference_log_std,
                        )
                        / logprob_normalizer
                    )
                    return reference_log_prob, reference_mean, reference_log_std

                def _distributed_policy_loss_fn(
                    actor_state,
                    x_prev,
                    x_t,
                    observation,
                    timestep,
                    candidate_advantages,
                    candidate_eps_clip,
                    candidate_rectification,
                    candidate_entropy,
                    old_log_prob,
                    old_mean,
                    old_log_std,
                    reference_mean,
                    reference_log_std,
                    log_prob_calibration,
                    mean_calibration,
                    log_std_calibration,
                    anchor_on_policy_statistics,
                ):
                    actor = nnx.merge(policy_graphdef, actor_state)
                    current_mean = jax_transition_mean(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    current_log_std = jax_transition_log_std(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    new_log_prob = _ppo_log_prob(
                        x_prev,
                        current_mean,
                        current_log_std,
                    ) / logprob_normalizer
                    if stabilize_on_policy_statistics:
                        calibrated_new_log_prob = new_log_prob + jax.lax.stop_gradient(
                            log_prob_calibration
                        )
                        calibrated_current_mean = current_mean + jax.lax.stop_gradient(
                            mean_calibration
                        )
                        calibrated_current_log_std = (
                            current_log_std
                            + jax.lax.stop_gradient(log_std_calibration)
                        )
                        ppo_new_log_prob = _conditionally_anchor_current_to_old_value(
                            calibrated_new_log_prob,
                            old_log_prob,
                            anchor_on_policy_statistics,
                        )
                        kl_current_mean = _conditionally_anchor_current_to_old_value(
                            calibrated_current_mean,
                            old_mean,
                            anchor_on_policy_statistics,
                        )
                        kl_current_log_std = _conditionally_anchor_current_to_old_value(
                            calibrated_current_log_std,
                            old_log_std,
                            anchor_on_policy_statistics,
                        )
                    else:
                        ppo_new_log_prob = new_log_prob
                        kl_current_mean = current_mean
                        kl_current_log_std = current_log_std
                    raw_flash = jax_flash_ppo_loss(
                        ppo_new_log_prob,
                        old_log_prob,
                        candidate_advantages,
                        clip_eps=candidate_eps_clip,
                        rectification_weight=jnp.ones_like(candidate_rectification),
                    )
                    flash = jax_flash_ppo_loss(
                        ppo_new_log_prob,
                        old_log_prob,
                        candidate_advantages,
                        clip_eps=candidate_eps_clip,
                        rectification_weight=candidate_rectification,
                    )
                    transition_ref_kl = _prefix_reference_kl(
                        kl_current_mean,
                        kl_current_log_std,
                        reference_mean,
                        reference_log_std,
                    )
                    kl_penalty, ref_kl, beta = jax_state_adaptive_kl_penalty(
                        transition_ref_kl,
                        candidate_entropy,
                        group_size=1,
                        beta_base=beta_base,
                        adapt_kl_beta=adapt_kl_beta,
                        uncertainty_scale=uncertainty_scale,
                    )
                    total = flash.loss + kl_penalty
                    return total, {
                        "flash_loss": flash.loss,
                        "raw_flash_loss": raw_flash.loss,
                        "ref_kl": ref_kl,
                        "beta": beta,
                        "ratio": flash.ratio,
                        "clipped_ratio": flash.clipped_ratio,
                        "per_sample_loss": raw_flash.per_sample_loss,
                        "preupdate_mean_abs_diff": jnp.mean(jnp.abs(current_mean - old_mean)),
                        "preupdate_log_prob_abs_diff": jnp.mean(jnp.abs(new_log_prob - old_log_prob)),
                        "surrogate_log_prob_abs_diff": jnp.mean(
                            jnp.abs(ppo_new_log_prob - old_log_prob)
                        ),
                        "next_log_prob_calibration": old_log_prob - new_log_prob,
                        "next_mean_calibration": old_mean - current_mean,
                        "next_log_std_calibration": old_log_std - current_log_std,
                    }

                def _distributed_policy_loss_and_grad(*args):
                    (loss, aux), grads = jax.value_and_grad(
                        _distributed_policy_loss_fn,
                        has_aux=True,
                    )(*args)
                    if distributed_gradient_reduction == "pmean":
                        grads = jax.lax.pmean(grads, axis_name="actor_data")
                    return (loss, aux), grads

                def _distributed_reference_kl(
                    actor_state,
                    x_t,
                    observation,
                    timestep,
                    old_mean,
                    old_log_std,
                    reference_mean,
                    reference_log_std,
                ):
                    actor = nnx.merge(policy_graphdef, actor_state)
                    current_mean = jax_transition_mean(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        observation=observation,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    current_log_std = jax_transition_log_std(
                        actor=actor,
                        flow_spec=flow_spec,
                        x_t=x_t,
                        timestep=timestep,
                        sde_mode=state.policy.sde_mode,
                    )
                    prefix_kl = jnp.mean(
                        _prefix_reference_kl(
                            current_mean,
                            current_log_std,
                            reference_mean,
                            reference_log_std,
                        )
                    )
                    full_kl = jnp.mean(
                        jax_gaussian_kl_diag(
                            current_mean,
                            current_log_std,
                            reference_mean,
                            reference_log_std,
                        )
                    )
                    old_policy_kl = jnp.mean(
                        jax_gaussian_kl_diag(
                            _select_environment_prefix(current_mean, ppo_action_horizon),
                            _select_environment_prefix(current_log_std, ppo_action_horizon),
                            _select_environment_prefix(old_mean, ppo_action_horizon),
                            _select_environment_prefix(old_log_std, ppo_action_horizon),
                        )
                    )
                    return prefix_kl, full_kl, old_policy_kl

                # Frozen-policy forward passes and success BC are independent.
                # Pinning each to a configured device lets JAX enqueue them
                # concurrently while retaining one globally equivalent loss.
                distributed_old_statistics = jax.jit(
                    _frozen_policy_statistics,
                    device=actor_devices[old_statistics_device_index],
                )
                if stabilize_on_policy_statistics:
                    # Equal BF16 policies must use one executable and device.
                    # Cross-trace fourth-decimal errors become O(0.1) after the
                    # 660-dimensional transition KL is summed.
                    distributed_reference_statistics = distributed_old_statistics
                else:
                    distributed_reference_statistics = jax.jit(
                        _reference_policy_statistics,
                        device=actor_devices[reference_statistics_device_index],
                    )
                distributed_regularization_loss_and_grad = jax.jit(
                    jax.value_and_grad(_flow_matching_component),
                    device=actor_devices[regularization_device_index],
                )
                distributed_policy_loss_and_grad = jax.pmap(
                    _distributed_policy_loss_and_grad,
                    axis_name="actor_data",
                    in_axes=(
                        None,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        None,
                    ),
                    devices=actor_devices,
                )
                distributed_reference_kl = jax.pmap(
                    _distributed_reference_kl,
                    axis_name="actor_data",
                    in_axes=(None, 0, 0, 0, 0, 0, 0, 0),
                    devices=actor_devices,
                )
                state.policy._flash_dp_train_cache = (
                    dp_cache_key,
                    distributed_old_statistics,
                    distributed_reference_statistics,
                    distributed_regularization_loss_and_grad,
                    distributed_policy_loss_and_grad,
                    distributed_reference_kl,
                )
            else:
                distributed_old_statistics = dp_cache[1]
                distributed_reference_statistics = dp_cache[2]
                distributed_regularization_loss_and_grad = dp_cache[3]
                distributed_policy_loss_and_grad = dp_cache[4]
                distributed_reference_kl = dp_cache[5]
            fixed_old_log_prob = None
            fixed_old_mean = None
            fixed_old_log_std = None
            fixed_reference_mean = None
            fixed_reference_log_std = None
            reference_kl_eval = None
        else:
            distributed_old_statistics = None
            distributed_reference_statistics = None
            distributed_regularization_loss_and_grad = None
            distributed_policy_loss_and_grad = None
            distributed_reference_kl = None
            fixed_old_log_prob = None
            fixed_old_mean = None
            fixed_old_log_std = None
            fixed_reference_mean = None
            fixed_reference_log_std = None
            reference_kl_eval = jax.jit(_reference_kl_fn)

        # Enqueue success/FM backward before the two frozen-policy forwards.
        # On the accelerated layout all three execute on distinct GPUs. Their
        # gradients are still summed exactly as before.
        precomputed_regularization: dict[str, tuple[Any, Any]] = {}
        if data_parallel_devices > 1:
            assert distributed_regularization_loss_and_grad is not None
            for name in ("fm", "success"):
                inputs = jax_regularization[name]
                weight = float(jax_regularization[f"lambda_{name}"])
                if inputs is None or weight == 0.0:
                    continue
                precomputed_regularization[name] = (
                    distributed_regularization_loss_and_grad(
                        state.policy.actor_state,
                        inputs,
                    )
                )
            assert distributed_old_statistics is not None
            assert distributed_reference_statistics is not None
            fixed_statistics_microbatch_size = int(
                actor_cfg.get(
                    "fixed_statistics_microbatch_size",
                    max(1, num_candidates // data_parallel_devices),
                )
            )
            if fixed_statistics_microbatch_size <= 0:
                raise ValueError("fixed_statistics_microbatch_size must be positive")
            old_statistics_parts: list[tuple[Any, Any, Any]] = []
            reference_statistics_parts: list[tuple[Any, Any, Any]] = []
            for start in range(0, num_candidates, fixed_statistics_microbatch_size):
                stop = min(start + fixed_statistics_microbatch_size, num_candidates)
                observation_part = _slice_jax_batch(jax_observation_g, start, stop)
                old_statistics_parts.append(
                    distributed_old_statistics(
                        state.old_policy.actor_state,
                        old_x_prev[start:stop],
                        old_x_t[start:stop],
                        observation_part,
                        old_timestep[start:stop],
                    )
                )
                reference_statistics_parts.append(
                    distributed_reference_statistics(
                        state.reference_policy.actor_state,
                        old_x_prev[start:stop],
                        old_x_t[start:stop],
                        observation_part,
                        old_timestep[start:stop],
                    )
                )
            # These outputs are tiny compared with PI0.5 parameters and
            # gradients. Concatenate on host to avoid JAX 0.5.3 reusing a
            # same-shape concatenate executable compiled for another GPU.
            # The two frozen forwards and success BC still execute in parallel.
            jax.effects_barrier()
            old_statistics_host = tuple(
                np.concatenate(
                    [np.asarray(jax.device_get(part[index])) for part in old_statistics_parts],
                    axis=0,
                )
                for index in range(3)
            )
            reference_statistics_host = tuple(
                np.concatenate(
                    [
                        np.asarray(jax.device_get(part[index]))
                        for part in reference_statistics_parts
                    ],
                    axis=0,
                )
                for index in range(3)
            )
            old_log_prob_host, old_mean_host, old_log_std_host = old_statistics_host
            _, reference_mean_host, reference_log_std_host = reference_statistics_host
            fixed_old_log_prob, fixed_old_mean, fixed_old_log_std = (
                _shard_jax_batch(jax.device_put(value, actor_devices[0]), data_parallel_devices)
                for value in (old_log_prob_host, old_mean_host, old_log_std_host)
            )
            fixed_reference_mean, fixed_reference_log_std = (
                _shard_jax_batch(jax.device_put(value, actor_devices[0]), data_parallel_devices)
                for value in (reference_mean_host, reference_log_std_host)
            )
            del (
                old_statistics_parts,
                reference_statistics_parts,
                old_statistics_host,
                reference_statistics_host,
            )
            jax.effects_barrier()

            # Do not carry a full auxiliary gradient on its worker GPU into
            # pmap. Move it to the optimizer device and release the source
            # tree before the distributed PPO backward starts.
            for name, (component_loss, component_grads) in tuple(
                precomputed_regularization.items()
            ):
                component_grads_on_optimizer = _jax_tree_copy_to_device(
                    component_grads,
                    actor_devices[0],
                )
                precomputed_regularization[name] = (
                    jax.device_put(component_loss, actor_devices[0]),
                    component_grads_on_optimizer,
                )
                del component_grads
            jax.effects_barrier()
            gc.collect()

        # Release PyTorch's critic cache so JAX can use its full memory cap for
        # the full-finetune value_and_grad backward through PI0.5.
        torch.cuda.empty_cache()
        if data_parallel_devices > 1:
            log_prob_calibration = jnp.zeros_like(fixed_old_log_prob)
            mean_calibration = jnp.zeros_like(fixed_old_mean)
            log_std_calibration = jnp.zeros_like(fixed_old_log_std)
        else:
            log_prob_calibration = None
            mean_calibration = None
            log_std_calibration = None
        post_update_kl_total = 0.0
        post_update_full_kl_total = 0.0
        rejected_updates = 0
        accepted_updates = 0
        attempted_epochs = 0
        success_grad_norm_total = 0.0
        success_grad_scale_total = 0.0
        fm_loss_total = 0.0
        success_loss_total = 0.0
        actor_grad_norms: list[float] = []
        epoch_metrics: dict[str, float] = {}
        final_post_update_kl: float | None = None
        final_post_update_full_kl: float | None = None
        final_post_update_old_policy_kl = 0.0
        ratio = np.ones(num_candidates, dtype=np.float32)
        clipped_ratio = ratio.copy()
        for epoch_index in range(actor_epochs):
            attempted_epochs += 1
            anchor_on_policy_statistics = float(
                stabilize_on_policy_statistics and epoch_index == 0
            )
            host_grads = None
            device_grads = None
            policy_loss_value = 0.0
            flash_loss_value = 0.0
            raw_flash_loss_value = 0.0
            reference_kl_value = 0.0
            beta_value = beta_base
            ratio_parts = []
            clipped_ratio_parts = []
            per_sample_loss_parts = []
            preupdate_mean_diff_parts = []
            preupdate_log_prob_diff_parts = []
            surrogate_log_prob_diff_parts = []
            if data_parallel_devices > 1:
                assert distributed_policy_loss_and_grad is not None
                (policy_loss_devices, aux_devices), policy_grads = (
                    distributed_policy_loss_and_grad(
                        state.policy.actor_state,
                        _shard_jax_batch(old_x_prev, data_parallel_devices),
                        _shard_jax_batch(old_x_t, data_parallel_devices),
                        _shard_jax_batch(jax_observation_g, data_parallel_devices),
                        _shard_jax_batch(old_timestep, data_parallel_devices),
                        _shard_jax_batch(advantages_j, data_parallel_devices),
                        _shard_jax_batch(eps_clip_j, data_parallel_devices),
                        _shard_jax_batch(rectification_j, data_parallel_devices),
                        _shard_jax_batch(entropy_norm_j, data_parallel_devices),
                        fixed_old_log_prob,
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                        log_prob_calibration,
                        mean_calibration,
                        log_std_calibration,
                        anchor_on_policy_statistics,
                    )
                )
                if stabilize_on_policy_statistics and epoch_index == 0:
                    log_prob_calibration = jax.lax.stop_gradient(
                        aux_devices["next_log_prob_calibration"]
                    )
                    mean_calibration = jax.lax.stop_gradient(
                        aux_devices["next_mean_calibration"]
                    )
                    log_std_calibration = jax.lax.stop_gradient(
                        aux_devices["next_log_std_calibration"]
                    )
                if distributed_gradient_reduction == "pmean":
                    # pmean produced the same average on every replica. Copy
                    # replica zero into an independently owned optimizer tree
                    # so the remaining seven output shards can be released.
                    device_grads = _jax_tree_copy_to_device(
                        _first_pmap_replica(policy_grads),
                        actor_devices[0],
                    )
                else:
                    device_grads = _mean_pmap_gradients_on_device(
                        policy_grads,
                        actor_devices[0],
                    )
                del policy_grads
                policy_loss_value = float(np.asarray(policy_loss_devices).mean())
                flash_loss_value = float(np.asarray(aux_devices["flash_loss"]).mean())
                raw_flash_loss_value = float(
                    np.asarray(aux_devices["raw_flash_loss"]).mean()
                )
                reference_kl_value = float(np.asarray(aux_devices["ref_kl"]).mean())
                beta_value = float(np.asarray(aux_devices["beta"]).mean())
                ratio_parts.append(np.asarray(aux_devices["ratio"]).reshape(-1))
                clipped_ratio_parts.append(
                    np.asarray(aux_devices["clipped_ratio"]).reshape(-1)
                )
                per_sample_loss_parts.append(
                    np.asarray(aux_devices["per_sample_loss"]).reshape(-1)
                )
                preupdate_mean_diff_parts.append(
                    np.asarray(aux_devices["preupdate_mean_abs_diff"]).reshape(-1)
                )
                preupdate_log_prob_diff_parts.append(
                    np.asarray(aux_devices["preupdate_log_prob_abs_diff"]).reshape(-1)
                )
                surrogate_log_prob_diff_parts.append(
                    np.asarray(aux_devices["surrogate_log_prob_abs_diff"]).reshape(-1)
                )
                jax.effects_barrier()
                gc.collect()
            else:
                for start in range(0, num_candidates, gradient_microbatch_size):
                    stop = min(start + gradient_microbatch_size, num_candidates)
                    sample_weight = (stop - start) / num_candidates
                    observation_mb = _slice_jax_batch(jax_observation_g, start, stop)
                    (policy_loss_mb, aux_mb), policy_grads = jax.value_and_grad(
                        _policy_loss_fn,
                        has_aux=True,
                    )(
                        state.policy.actor_state,
                        state.old_policy.actor_state,
                        state.reference_policy.actor_state,
                        old_x_prev[start:stop],
                        old_x_t[start:stop],
                        observation_mb,
                        old_timestep[start:stop],
                        advantages_j[start:stop],
                        eps_clip_j[start:stop],
                        rectification_j[start:stop],
                        entropy_norm_j[start:stop],
                        fixed_old_log_prob,
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                        anchor_on_policy_statistics,
                    )
                    host_grads = _accumulate_jax_grads_on_host(
                        host_grads,
                        policy_grads,
                        weight=sample_weight,
                    )
                    del policy_grads
                    policy_loss_value += sample_weight * float(policy_loss_mb)
                    flash_loss_value += sample_weight * float(aux_mb["flash_loss"])
                    raw_flash_loss_value += sample_weight * float(aux_mb["raw_flash_loss"])
                    reference_kl_value += sample_weight * float(aux_mb["ref_kl"])
                    beta_value = float(aux_mb["beta"])
                    ratio_parts.append(np.asarray(aux_mb["ratio"]))
                    clipped_ratio_parts.append(np.asarray(aux_mb["clipped_ratio"]))
                    per_sample_loss_parts.append(np.asarray(aux_mb["per_sample_loss"]))
                    jax.effects_barrier()
                    gc.collect()

            fm_loss = jnp.asarray(0.0, dtype=jnp.float32)
            success_loss = jnp.asarray(0.0, dtype=jnp.float32)
            success_grad_norm = 0.0
            success_grad_scale = 0.0
            for name in ("fm", "success") if epoch_index < regularization_epochs else ():
                inputs = jax_regularization[name]
                weight = float(jax_regularization[f"lambda_{name}"])
                if inputs is None or weight == 0.0:
                    continue
                if data_parallel_devices > 1:
                    # Pop before donating component_grads to the fused tree
                    # addition. Keeping the tuple in the dict prevents XLA
                    # from reusing a multi-GiB gradient buffer.
                    component_loss, component_grads = precomputed_regularization.pop(name)
                else:
                    component_loss, component_grads = jax.value_and_grad(
                        _flow_matching_component,
                    )(state.policy.actor_state, inputs)
                effective_weight = weight
                if name == "success":
                    success_grad_scale = 1.0
                    success_grad_norm = (
                        float(_jax_tree_l2_norm_device(component_grads))
                        if data_parallel_devices > 1
                        else _jax_tree_l2_norm(component_grads)
                    )
                    weighted_norm = abs(weight) * success_grad_norm
                    max_norm = float(
                        regularization_cfg.get(
                            "success_weighted_grad_max_norm",
                            float("inf"),
                        )
                    )
                    if weighted_norm > max_norm:
                        success_grad_scale = max_norm / max(weighted_norm, 1e-12)
                        effective_weight *= success_grad_scale
                if data_parallel_devices > 1:
                    assert device_grads is not None
                    device_grads = _jax_tree_add_scaled(
                        device_grads,
                        component_grads,
                        other_weight=effective_weight,
                    )
                else:
                    host_grads = _accumulate_jax_grads_on_host(
                        host_grads,
                        component_grads,
                        weight=effective_weight,
                    )
                if name == "fm":
                    fm_loss = component_loss
                else:
                    success_loss = component_loss
                del component_grads
                jax.effects_barrier()
                gc.collect()

            fm_loss_total += float(fm_loss)
            success_loss_total += float(success_loss)

            total_loss = (
                policy_loss_value
                + float(jax_regularization["lambda_fm"]) * fm_loss
                + float(jax_regularization["lambda_success"]) * success_loss
            )
            actor_grad_norm = (
                float(_jax_tree_l2_norm_device(device_grads))
                if data_parallel_devices > 1
                else _jax_tree_l2_norm(host_grads)
            )
            actor_grad_norms.append(actor_grad_norm)
            actor_state_before = state.policy.actor_state
            optimizer_state_before = state.policy.actor_opt_state
            if data_parallel_devices > 1:
                grads = device_grads
                device_grads = None
            else:
                grads = jax.device_put(host_grads)
                del host_grads
            state.policy.apply_actor_gradients(grads)
            del grads
            jax.effects_barrier()
            gc.collect()

            if data_parallel_devices > 1 and stabilize_on_policy_statistics:
                assert distributed_old_statistics is not None
                post_statistics_parts = []
                for start in range(0, num_candidates, fixed_statistics_microbatch_size):
                    stop = min(start + fixed_statistics_microbatch_size, num_candidates)
                    post_statistics_parts.append(
                        distributed_old_statistics(
                            state.policy.actor_state,
                            old_x_prev[start:stop],
                            old_x_t[start:stop],
                            _slice_jax_batch(jax_observation_g, start, stop),
                            old_timestep[start:stop],
                        )
                    )
                jax.effects_barrier()
                post_mean_host = np.concatenate(
                    [
                        np.asarray(jax.device_get(part[1]))
                        for part in post_statistics_parts
                    ],
                    axis=0,
                )
                post_log_std_host = np.concatenate(
                    [
                        np.asarray(jax.device_get(part[2]))
                        for part in post_statistics_parts
                    ],
                    axis=0,
                )
                post_update_kl = float(
                    _numpy_gaussian_kl_diag(
                        _select_environment_prefix_numpy(
                            post_mean_host, reference_kl_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            post_log_std_host, reference_kl_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            reference_mean_host, reference_kl_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            reference_log_std_host, reference_kl_action_horizon
                        ),
                    ).mean()
                )
                post_update_full_kl = float(
                    _numpy_gaussian_kl_diag(
                        post_mean_host,
                        post_log_std_host,
                        reference_mean_host,
                        reference_log_std_host,
                    ).mean()
                )
                post_update_old_policy_kl = float(
                    _numpy_gaussian_kl_diag(
                        _select_environment_prefix_numpy(
                            post_mean_host, ppo_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            post_log_std_host, ppo_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            old_mean_host, ppo_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            old_log_std_host, ppo_action_horizon
                        ),
                    ).mean()
                )
                del post_statistics_parts, post_mean_host, post_log_std_host
            elif data_parallel_devices > 1:
                assert distributed_reference_kl is not None
                (
                    post_update_kl_devices,
                    post_update_full_kl_devices,
                    post_update_old_policy_kl_devices,
                ) = (
                    distributed_reference_kl(
                        state.policy.actor_state,
                        _shard_jax_batch(old_x_t, data_parallel_devices),
                        _shard_jax_batch(jax_observation_g, data_parallel_devices),
                        _shard_jax_batch(old_timestep, data_parallel_devices),
                        fixed_old_mean,
                        fixed_old_log_std,
                        fixed_reference_mean,
                        fixed_reference_log_std,
                    )
                )
                post_update_kl = float(np.asarray(post_update_kl_devices).mean())
                post_update_full_kl = float(
                    np.asarray(post_update_full_kl_devices).mean()
                )
                post_update_old_policy_kl = float(
                    np.asarray(post_update_old_policy_kl_devices).mean()
                )
            else:
                assert reference_kl_eval is not None
                post_update_kl = 0.0
                post_update_full_kl = 0.0
                post_update_old_policy_kl = 0.0
                for start in range(0, num_candidates, kl_eval_microbatch_size):
                    stop = min(start + kl_eval_microbatch_size, num_candidates)
                    sample_weight = (stop - start) / num_candidates
                    (
                        post_update_kl_part,
                        post_update_full_kl_part,
                        post_update_old_policy_kl_part,
                    ) = (
                        reference_kl_eval(
                            state.policy.actor_state,
                            state.old_policy.actor_state,
                            state.reference_policy.actor_state,
                            old_x_t[start:stop],
                            _slice_jax_batch(jax_observation_g, start, stop),
                            old_timestep[start:stop],
                        )
                    )
                    post_update_kl += sample_weight * float(post_update_kl_part)
                    post_update_full_kl += sample_weight * float(
                        post_update_full_kl_part
                    )
                    post_update_old_policy_kl += sample_weight * float(
                        post_update_old_policy_kl_part
                    )
            reject_update = bool(actor_cfg.get("reject_update_on_kl", False)) and (
                not math.isfinite(post_update_kl)
                or post_update_kl
                > float(actor_cfg.get("max_policy_reference_kl", float("inf")))
            )
            if reject_update:
                # Keep the accepted state live while explicitly dropping the
                # rejected full-model and Adafactor pytrees. Without this,
                # their multi-GiB JAX buffers can survive until a later Python
                # collection and overlap with the next PI0.5 backward.
                rejected_actor_state = state.policy.actor_state
                rejected_optimizer_state = state.policy.actor_opt_state
                state.policy.actor_state = actor_state_before
                state.policy.actor_opt_state = optimizer_state_before
                state.policy._sync_torch_adapter_from_jax()
                rejected_updates += 1
                del rejected_actor_state, rejected_optimizer_state
                jax.effects_barrier()
                if bool(actor_cfg.get("jax_clear_caches_after_kl_rejection", True)):
                    jax.clear_caches()
                gc.collect()
            else:
                # The accepted proposal is now authoritative. Release the
                # rollback-only references before the next actor iteration.
                del actor_state_before, optimizer_state_before
                accepted_updates += 1
                final_post_update_kl = post_update_kl
                final_post_update_full_kl = post_update_full_kl
                final_post_update_old_policy_kl = post_update_old_policy_kl
                gc.collect()

            loss_total += float(total_loss)
            flash_total += flash_loss_value
            raw_flash_total += raw_flash_loss_value
            kl_total += reference_kl_value
            post_update_kl_total += post_update_kl
            post_update_full_kl_total += post_update_full_kl
            success_grad_norm_total += success_grad_norm
            success_grad_scale_total += success_grad_scale
            rectification_total += float(rectification.mean().item())
            ratio = np.concatenate(ratio_parts)
            clipped_ratio = np.concatenate(clipped_ratio_parts)
            per_sample_loss = np.concatenate(per_sample_loss_parts)
            epoch_prefix = f"ppo_epoch_{epoch_index + 1}"
            epoch_metrics.update(
                {
                    f"{epoch_prefix}_importance_ratio_mean": float(ratio.mean()),
                    f"{epoch_prefix}_importance_ratio_std": float(ratio.std()),
                    f"{epoch_prefix}_importance_ratio_min": float(ratio.min()),
                    f"{epoch_prefix}_importance_ratio_max": float(ratio.max()),
                    f"{epoch_prefix}_clip_fraction": float(
                        (ratio != clipped_ratio).mean()
                    ),
                    f"{epoch_prefix}_actor_grad_norm": actor_grad_norm,
                    f"{epoch_prefix}_actor_grad_clip_scale": min(
                        1.0,
                        float(actor_cfg.get("max_grad_norm", 1.0))
                        / max(actor_grad_norm, 1e-12),
                    ),
                    f"{epoch_prefix}_post_reference_kl": post_update_kl,
                    f"{epoch_prefix}_post_full_reference_kl": post_update_full_kl,
                    f"{epoch_prefix}_post_old_policy_kl": post_update_old_policy_kl,
                    f"{epoch_prefix}_accepted": float(not reject_update),
                    f"{epoch_prefix}_on_policy_value_anchor": anchor_on_policy_statistics,
                }
            )
            for step_idx, count in enumerate(selected_counts.tolist()):
                if not count:
                    continue
                step_mask = selected_steps_grouped == step_idx
                step_raw_loss = float(
                    per_sample_loss[step_mask.detach().cpu().numpy()].mean()
                )
                raw_loss_by_step[step_idx] += step_raw_loss
                raw_grad_by_step[step_idx] += actor_grad_norm
                rectified_grad_by_step[step_idx] += actor_grad_norm * float(rectification[step_mask].mean().item())
                if str(flow_cfg.get("temporal_rectification_mode", "analytic")) == "empirical_ema":
                    state.rectifier.update(step_idx, actor_grad_norm, count=count)
            if reject_update:
                # PPO epochs form one transaction over fixed rollout data. If
                # an epoch is rejected, later epochs must not build on a state
                # that the optimizer never accepted.
                break

        # The precomputed success/FM entries retain a full PI0.5 gradient tree.
        # Drop those references before returning so a following distributed
        # backward does not overlap with an auxiliary gradient from this step.
        precomputed_regularization.clear()
        jax.effects_barrier()
        gc.collect()
        if final_post_update_kl is None:
            if data_parallel_devices > 1:
                final_post_update_kl = float(
                    _numpy_gaussian_kl_diag(
                        _select_environment_prefix_numpy(
                            old_mean_host, reference_kl_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            old_log_std_host, reference_kl_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            reference_mean_host, reference_kl_action_horizon
                        ),
                        _select_environment_prefix_numpy(
                            reference_log_std_host, reference_kl_action_horizon
                        ),
                    ).mean()
                )
                final_post_update_full_kl = float(
                    _numpy_gaussian_kl_diag(
                        old_mean_host,
                        old_log_std_host,
                        reference_mean_host,
                        reference_log_std_host,
                    ).mean()
                )
            else:
                assert reference_kl_eval is not None
                final_post_update_kl = 0.0
                final_post_update_full_kl = 0.0
                for start in range(0, num_candidates, kl_eval_microbatch_size):
                    stop = min(start + kl_eval_microbatch_size, num_candidates)
                    sample_weight = (stop - start) / num_candidates
                    final_kl_part, final_full_kl_part, _ = reference_kl_eval(
                        state.policy.actor_state,
                        state.old_policy.actor_state,
                        state.reference_policy.actor_state,
                        old_x_t[start:stop],
                        _slice_jax_batch(jax_observation_g, start, stop),
                        old_timestep[start:stop],
                    )
                    final_post_update_kl += sample_weight * float(final_kl_part)
                    final_post_update_full_kl += sample_weight * float(
                        final_full_kl_part
                    )
        assert final_post_update_full_kl is not None
        epoch_divisor = max(1, attempted_epochs)
        metrics = {
            "actor_loss": loss_total / epoch_divisor,
            "flash_ppo_loss": flash_total / epoch_divisor,
            "flash_raw_ppo_loss": raw_flash_total / epoch_divisor,
            "reference_kl": kl_total / epoch_divisor,
            "selected_step_kl": kl_total / epoch_divisor,
            "post_update_reference_kl": final_post_update_kl,
            "post_update_full_reference_kl": final_post_update_full_kl,
            "post_update_old_policy_kl": final_post_update_old_policy_kl,
            "reference_kl_action_horizon": float(reference_kl_action_horizon),
            "reference_kl_event_dim": float(reference_kl_event_dim),
            "reference_kl_full_event_dim": float(full_reference_kl_event_dim),
            "reference_kl_uses_action_prefix": float(
                reference_kl_action_horizon < state.policy.model_horizon
            ),
            "ppo_action_horizon": float(ppo_action_horizon),
            "ppo_event_dim": float(ppo_event_dim),
            "ppo_uses_action_prefix": float(
                ppo_action_horizon < state.policy.model_horizon
            ),
            "actor_update_rejected": float(accepted_updates == 0),
            "actor_update_accepted": float(accepted_updates > 0),
            "actor_update_partially_rejected": float(
                accepted_updates > 0 and rejected_updates > 0
            ),
            "jax_rejection_cleanup_applied": float(rejected_updates > 0),
            "rejected_update_count": float(rejected_updates),
            "accepted_actor_epochs": float(accepted_updates),
            "attempted_actor_epochs": float(attempted_epochs),
            "actor_sampling_seed": float(sampling_seed),
            "normalize_logprob_by_action_dim": float(normalize_logprob_by_action_dim),
            "logprob_normalizer": logprob_normalizer,
            "reference_kl_beta": beta_value,
            "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
            "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
            "actor_epochs": float(actor_epochs),
            "actor_regularization_epochs_per_rollout": float(regularization_epochs),
            "selected_step": float(selected_steps.float().mean().item()),
            "selected_step_min": float(selected_steps.min().item()),
            "selected_step_max": float(selected_steps.max().item()),
            "rectification_weight": rectification_total / epoch_divisor,
            "rectification_weight_min": float(rectification.min().item()),
            "rectification_weight_max": float(rectification.max().item()),
            "rectification_weight_std": float(rectification.float().std(unbiased=False).item()),
            "importance_ratio_mean": float(ratio.mean()),
            "importance_ratio_std": float(ratio.std()),
            "importance_ratio_min": float(ratio.min()),
            "importance_ratio_max": float(ratio.max()),
            "preupdate_transition_mean_abs_diff": float(np.mean(np.concatenate(preupdate_mean_diff_parts))) if preupdate_mean_diff_parts else 0.0,
            "preupdate_log_prob_abs_diff": float(np.mean(np.concatenate(preupdate_log_prob_diff_parts))) if preupdate_log_prob_diff_parts else 0.0,
            "surrogate_log_prob_abs_diff": float(
                np.mean(np.concatenate(surrogate_log_prob_diff_parts))
            ) if surrogate_log_prob_diff_parts else 0.0,
            "single_epoch_bf16_statistics_stabilized": float(
                stabilize_on_policy_statistics
            ),
            "on_policy_bf16_statistics_stabilized": float(
                stabilize_on_policy_statistics
            ),
            "ppo_clip_fraction": float((ratio != clipped_ratio).mean()),
            "actor_grad_norm": actor_grad_norm,
            "actor_grad_norm_mean": float(np.mean(actor_grad_norms)),
            "actor_grad_clip_scale": min(
                1.0,
                float(actor_cfg.get("max_grad_norm", 1.0)) / max(actor_grad_norm, 1e-12),
            ),
            "actor_grad_was_clipped": float(
                actor_grad_norm > float(actor_cfg.get("max_grad_norm", 1.0))
            ),
            "success_grad_norm": success_grad_norm_total,
            "success_grad_scale": success_grad_scale_total,
            "success_update_applied": float(success_update_due),
            "success_update_period": float(success_update_period),
            "candidate_group_size": float(group_size),
            "gradient_microbatch_size": float(gradient_microbatch_size),
            "rollout_state_microbatch_size": float(rollout_state_microbatch_size),
            "kl_eval_microbatch_size": float(kl_eval_microbatch_size),
            "actor_data_parallel_devices": float(data_parallel_devices),
            "distributed_gradient_reduction_pmean": float(
                distributed_gradient_reduction == "pmean"
            ),
            "parallel_frozen_statistics": float(parallel_frozen_statistics),
            "old_statistics_device_index": float(old_statistics_device_index),
            "reference_statistics_device_index": float(reference_statistics_device_index),
            "regularization_device_index": float(regularization_device_index),
            "actor_candidates_per_device": float(num_candidates / data_parallel_devices),
            "distributed_backward_calls": float(
                attempted_epochs
                * (
                    1
                    if data_parallel_devices > 1
                    else math.ceil(num_candidates / gradient_microbatch_size)
                )
            ),
            "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
            "fm_anchor_loss": fm_loss_total,
            "success_buffer_loss": success_loss_total,
            "action_smoothness": 0.0,
            **epoch_metrics,
            **adv_diag,
        }
        post_update_kl_mean = final_post_update_kl
        kl_target = float(actor_cfg.get("target_policy_reference_kl", float("inf")))
        kl_hard_limit = float(actor_cfg.get("max_policy_reference_kl", float("inf")))
        metrics.update(
            {
                "policy_reference_kl_target": kl_target,
                "policy_reference_kl_hard_limit": kl_hard_limit,
                "policy_reference_kl_target_exceeded": float(post_update_kl_mean > kl_target),
                "policy_reference_kl_hard_limit_exceeded": float(
                    post_update_kl_mean > kl_hard_limit
                ),
                "policy_reference_kl_nonfinite": float(not math.isfinite(post_update_kl_mean)),
                "policy_reference_kl_utilization": (
                    post_update_kl_mean / kl_hard_limit
                    if math.isfinite(kl_hard_limit) and kl_hard_limit > 0.0
                    else 0.0
                ),
            }
        )
        for step_idx, count in enumerate(selected_counts.tolist()):
            metrics[f"selected_step_count_{step_idx}"] = float(count)
            metrics[f"rectifier_count_{step_idx}"] = float(state.rectifier.counts[step_idx].item())
            metrics[f"rectifier_grad_ema_{step_idx}"] = float(state.rectifier.grad_ema[step_idx].item())
            metrics[f"flash_raw_loss_step_{step_idx}"] = raw_loss_by_step[step_idx] / epoch_divisor
            metrics[f"flash_raw_grad_norm_step_{step_idx}"] = raw_grad_by_step[step_idx] / epoch_divisor
            metrics[f"flash_rectified_grad_norm_step_{step_idx}"] = rectified_grad_by_step[step_idx] / epoch_divisor
        return metrics

    with torch.no_grad():
        old_rollout = sample_flash_rollout(
            state.old_policy,
            condition,
            group_size=group_size,
            selected_step=selected_steps,
        )
        environment_endpoint = state.old_policy.flat_actions_to_environment(old_rollout.endpoint, condition_g)
        endpoint = environment_endpoint.reshape(batch.batch_size, group_size, -1)
        chi2_diag: dict[str, float] = {"chi2_enabled": 0.0}
        chi2_ratio = None
        if ogpo_variant(config) == "chi2":
            chi2_ratio_flat, chi2_diag = _selected_transition_chi2_ratio(
                state,
                x_prev=old_rollout.x_prev,
                x_t=old_rollout.x_t,
                timestep=old_rollout.timestep,
                condition=condition_g,
                config=config,
            )
            chi2_ratio = chi2_ratio_flat.reshape(batch.batch_size, group_size)
        advantages, adv_diag = conservative_advantages_for_candidates(
            state,
            batch.observations,
            endpoint,
            batch,
            config,
            chi2_ratio=chi2_ratio,
        )
        entropy_norm = _normalized_state_entropy(state, batch).mean(dim=0)
        eps_clip = (
            actor_clip_for_uncertainty(entropy_norm, actor_cfg, uncertainty_cfg)
            .unsqueeze(-1)
            .expand(batch.batch_size, group_size)
            .reshape(-1)
        )
    selected_counts = torch.bincount(selected_steps.detach().cpu(), minlength=state.policy.num_steps)
    loss_total = 0.0
    flash_total = 0.0
    raw_flash_total = 0.0
    kl_total = 0.0
    grad_total = 0.0
    rectification_total = 0.0
    raw_loss_by_step = [0.0] * state.policy.num_steps
    raw_grad_by_step = [0.0] * state.policy.num_steps
    rectified_grad_by_step = [0.0] * state.policy.num_steps
    selected_steps_grouped = selected_steps.repeat_interleave(group_size)
    trainable_actor_parameters = [parameter for parameter in state.policy.parameters() if parameter.requires_grad]
    compute_step_grad_diagnostics = bool(
        actor_cfg.get("compute_step_grad_diagnostics", True)
    )
    num_candidates = batch.batch_size * group_size
    gradient_microbatch_size = min(
        int(actor_cfg.get("gradient_microbatch_size", num_candidates)),
        num_candidates,
    )
    if compute_step_grad_diagnostics and gradient_microbatch_size < num_candidates:
        raise ValueError(
            "actor.compute_step_grad_diagnostics must be false when the PyTorch "
            "full actor uses gradient microbatching"
        )
    rectification = _flash_rectification_weight(
        state,
        flow_cfg,
        timestep=old_rollout.timestep,
        selected_steps=selected_steps,
        group_size=group_size,
    )
    ratio_values: list[torch.Tensor] = []
    clipped_ratio_values: list[torch.Tensor] = []
    raw_per_sample_values: list[torch.Tensor] = []
    kl_beta = entropy_norm.new_tensor(float(regularization_cfg.get("beta_kl", 0.01)))
    reg_diag: dict[str, float] = {}
    post_update_action = _post_update_kl_action(actor_cfg)
    post_update_kl_limit = float(
        actor_cfg.get("max_policy_reference_kl", float("inf"))
    )
    post_update_kl_total = 0.0
    post_update_kl_last = 0.0
    post_update_kl_exceeded = False
    accepted_actor_epochs = 0
    rejected_actor_epochs = 0
    actor_grad_clip_scale_total = 0.0
    actor_advantages = advantages.reshape(-1)
    chi2_logprob_normalizer = float(
        chi2_diag.get("chi2_selected_logprob_normalizer", 1.0)
    )
    chi2_upper_ratio_bound = None
    if ogpo_variant(config) == "chi2":
        chi2_upper_ratio_bound = chi2_ppo_upper_bound(
            eps_clip,
            beta=float(adv_diag["chi2_beta"]),
            r_max=float(_chi2_config(config).get("r_max", 10.0)),
        )
    for _ in range(actor_epochs):
        state.actor_optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_flash = 0.0
        epoch_raw_flash = 0.0
        epoch_ref_kl = 0.0
        epoch_ratios: list[torch.Tensor] = []
        epoch_clipped_ratios: list[torch.Tensor] = []
        epoch_raw_per_sample: list[torch.Tensor] = []
        candidate_beta = (
            float(regularization_cfg.get("beta_kl", 0.01))
            * (
                1.0
                + kl_uncertainty_scale(regularization_cfg, uncertainty_cfg)
                * entropy_norm.clamp(0.0, 1.0)
            )
        ).repeat_interleave(group_size)
        kl_beta = candidate_beta.reshape(batch.batch_size, group_size).mean(dim=1).mean()
        for start in range(0, num_candidates, gradient_microbatch_size):
            stop = min(start + gradient_microbatch_size, num_candidates)
            indices = torch.arange(start, stop, device=old_rollout.x_t.device)
            micro_condition = (
                condition_g.index_select(indices)
                if isinstance(condition_g, PI05FlowCondition)
                else condition_g.index_select(0, indices)
            )
            x_t = old_rollout.x_t[start:stop]
            timestep = old_rollout.timestep[start:stop]
            current_mean = state.policy.transition_mean(x_t, micro_condition, timestep)
            current_log_std = state.policy.transition_log_std(x_t, timestep)
            new_log_prob = gaussian_log_prob(
                old_rollout.x_prev[start:stop], current_mean, current_log_std
            )
            ppo_new_log_prob = new_log_prob / chi2_logprob_normalizer
            ppo_old_log_prob = (
                old_rollout.old_log_prob[start:stop] / chi2_logprob_normalizer
            )
            raw_flash = flash_ppo_loss(
                ppo_new_log_prob,
                ppo_old_log_prob,
                actor_advantages[start:stop],
                clip_eps=eps_clip[start:stop],
                upper_ratio_bound=(
                    None
                    if chi2_upper_ratio_bound is None
                    else chi2_upper_ratio_bound[start:stop]
                ),
                rectification_weight=1.0,
            )
            flash = flash_ppo_loss(
                ppo_new_log_prob,
                ppo_old_log_prob,
                actor_advantages[start:stop],
                clip_eps=eps_clip[start:stop],
                upper_ratio_bound=(
                    None
                    if chi2_upper_ratio_bound is None
                    else chi2_upper_ratio_bound[start:stop]
                ),
                rectification_weight=rectification[start:stop],
            )
            with torch.no_grad():
                reference_mean = state.reference_policy.transition_mean(
                    x_t, micro_condition, timestep
                )
                reference_log_std = state.reference_policy.transition_log_std(
                    x_t, timestep
                )
            transition_ref_kl = gaussian_kl_diag(
                current_mean, current_log_std, reference_mean, reference_log_std
            )
            sample_fraction = (stop - start) / num_candidates
            kl_penalty = (
                candidate_beta[start:stop] * transition_ref_kl
            ).sum() / num_candidates
            micro_loss = flash.loss * sample_fraction + kl_penalty
            if compute_step_grad_diagnostics:
                for step_idx, count in enumerate(selected_counts.tolist()):
                    if not count:
                        continue
                    step_mask = selected_steps_grouped == step_idx
                    step_raw_loss = raw_flash.per_sample_loss[step_mask].mean()
                    step_gradients = torch.autograd.grad(
                        step_raw_loss,
                        trainable_actor_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    finite_gradients = [
                        gradient.detach().norm(2)
                        for gradient in step_gradients
                        if gradient is not None
                    ]
                    raw_grad = (
                        float(torch.norm(torch.stack(finite_gradients), 2).item())
                        if finite_gradients
                        else 0.0
                    )
                    step_rectification = float(
                        rectification[step_mask].detach().mean().item()
                    )
                    raw_loss_by_step[step_idx] += float(step_raw_loss.detach().item())
                    raw_grad_by_step[step_idx] += raw_grad
                    rectified_grad_by_step[step_idx] += raw_grad * step_rectification
                    if (
                        str(flow_cfg.get("temporal_rectification_mode", "analytic"))
                        == "empirical_ema"
                    ):
                        state.rectifier.update(step_idx, raw_grad, count=count)
            micro_loss.backward()
            epoch_loss += float(micro_loss.detach().item())
            epoch_flash += float(flash.loss.detach().item()) * sample_fraction
            epoch_raw_flash += float(raw_flash.loss.detach().item()) * sample_fraction
            epoch_ref_kl += float(transition_ref_kl.detach().sum().item()) / num_candidates
            log_ratio = (
                ppo_new_log_prob.detach() - ppo_old_log_prob.detach()
            ).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            clip = eps_clip[start:stop]
            upper = (
                1.0 + clip
                if chi2_upper_ratio_bound is None
                else chi2_upper_ratio_bound[start:stop]
            )
            clipped_ratio = torch.maximum(
                torch.minimum(ratio, upper), 1.0 - clip
            )
            epoch_ratios.append(ratio.cpu())
            epoch_clipped_ratios.append(clipped_ratio.cpu())
            epoch_raw_per_sample.append(raw_flash.per_sample_loss.detach().cpu())
            del current_mean, new_log_prob, ppo_new_log_prob, reference_mean, transition_ref_kl, micro_loss
        reg_loss, reg_diag = _actor_regularization_loss(
            state,
            batch,
            config,
            fm_batch=fm_batch,
            success_batch=success_batch,
        )
        if reg_loss.requires_grad:
            reg_loss.backward()
        epoch_loss += float(reg_loss.detach().item())
        assert_no_gradients(state.critic, "critic")
        if state.divl is not None:
            assert_no_gradients(state.divl, "divl")
        assert_no_gradients(state.reference_policy, "reference_policy")
        if state.slow_policy is not None:
            assert_no_gradients(state.slow_policy, "slow_policy")
        actor_grad_norm = float(grad_norm(state.policy.parameters()))
        actor_grad_clip_scale = min(
            1.0,
            float(actor_cfg.get("max_grad_norm", 1.0)) / max(actor_grad_norm, 1e-12),
        )
        rollback_snapshot = None
        if post_update_action == "rollback_cpu":
            rollback_snapshot = _capture_actor_rollback_state(state)
        torch.nn.utils.clip_grad_norm_(state.policy.parameters(), float(actor_cfg.get("max_grad_norm", 1.0)))
        state.actor_optimizer.step()
        post_update_kl = _selected_transition_reference_kl(
            state,
            x_t=old_rollout.x_t,
            timestep=old_rollout.timestep,
            condition=condition_g,
            microbatch_size=int(
                actor_cfg.get("kl_eval_microbatch_size", gradient_microbatch_size)
            ),
        )
        post_update_kl_last = post_update_kl
        violates_post_update_kl = (
            not math.isfinite(post_update_kl)
            or post_update_kl > post_update_kl_limit
        )
        post_update_kl_exceeded = post_update_kl_exceeded or violates_post_update_kl
        if violates_post_update_kl and post_update_action == "rollback_cpu":
            assert rollback_snapshot is not None
            _restore_actor_rollback_state(state, rollback_snapshot)
            rejected_actor_epochs += 1
            del rollback_snapshot
            break
        del rollback_snapshot
        accepted_actor_epochs += 1
        epoch_ratio = torch.cat(epoch_ratios)
        epoch_clipped_ratio = torch.cat(epoch_clipped_ratios)
        epoch_raw_per_sample_tensor = torch.cat(epoch_raw_per_sample)
        ratio_values.append(epoch_ratio)
        clipped_ratio_values.append(epoch_clipped_ratio)
        raw_per_sample_values.append(epoch_raw_per_sample_tensor)
        if not compute_step_grad_diagnostics:
            for step_idx, count in enumerate(selected_counts.tolist()):
                if not count:
                    continue
                step_mask = (selected_steps_grouped == step_idx).detach().cpu()
                step_raw_loss = epoch_raw_per_sample_tensor[step_mask].mean()
                raw_loss_by_step[step_idx] += float(step_raw_loss.item())
        loss_total += epoch_loss
        flash_total += epoch_flash
        raw_flash_total += epoch_raw_flash
        kl_total += epoch_ref_kl
        grad_total += actor_grad_norm
        actor_grad_clip_scale_total += actor_grad_clip_scale
        post_update_kl_total += post_update_kl
        rectification_total += float(rectification.detach().mean().item())
        if violates_post_update_kl and post_update_action == "stop":
            break
    if ratio_values:
        ratio_all = torch.cat(ratio_values)
        clipped_ratio_all = torch.cat(clipped_ratio_values)
    else:
        ratio_all = torch.ones(1, device=old_rollout.x_t.device)
        clipped_ratio_all = ratio_all
    epoch_divisor = max(1, accepted_actor_epochs)
    metrics = {
        "actor_loss": loss_total / epoch_divisor,
        "flash_ppo_loss": flash_total / epoch_divisor,
        "flash_raw_ppo_loss": raw_flash_total / epoch_divisor,
        "reference_kl": kl_total / epoch_divisor,
        "selected_step_kl": kl_total / epoch_divisor,
        "post_update_reference_kl": post_update_kl_last,
        "post_update_reference_kl_mean": post_update_kl_total / epoch_divisor,
        "post_update_kl_exceeded": float(post_update_kl_exceeded),
        "post_update_kl_action_rollback_cpu": float(post_update_action == "rollback_cpu"),
        "post_update_kl_action_stop": float(post_update_action == "stop"),
        "actor_update_rejected": float(accepted_actor_epochs == 0 and rejected_actor_epochs > 0),
        "actor_update_accepted": float(accepted_actor_epochs > 0),
        "actor_update_partially_rejected": float(
            accepted_actor_epochs > 0 and rejected_actor_epochs > 0
        ),
        "accepted_actor_epochs": float(accepted_actor_epochs),
        "rejected_actor_epochs": float(rejected_actor_epochs),
        "reference_kl_beta": float(kl_beta.detach().item()),
        "ustate_adapt_ppo_clip": float(bool(uncertainty_cfg.get("adapt_ppo_clip", False))),
        "ustate_adapt_kl_beta": float(bool(uncertainty_cfg.get("adapt_kl_beta", False))),
        "actor_epochs": float(actor_epochs),
        "selected_step": float(selected_steps.float().mean().item()),
        "selected_step_min": float(selected_steps.min().item()),
        "selected_step_max": float(selected_steps.max().item()),
        "rectification_weight": rectification_total / epoch_divisor,
        "rectification_weight_min": float(rectification.min().item()),
        "rectification_weight_max": float(rectification.max().item()),
        "rectification_weight_std": float(rectification.float().std(unbiased=False).item()),
        "importance_ratio_mean": float(ratio_all.mean().item()),
        "importance_ratio_std": float(ratio_all.std(unbiased=False).item()),
        "importance_ratio_min": float(ratio_all.min().item()),
        "importance_ratio_max": float(ratio_all.max().item()),
        "policy_entropy": _policy_entropy(state.policy),
        "ppo_clip_fraction": float(ratio_all.ne(clipped_ratio_all).float().mean().item()),
        "actor_grad_norm": grad_total / epoch_divisor,
        "actor_grad_clip_scale": actor_grad_clip_scale_total / epoch_divisor,
        "actor_grad_was_clipped": float(actor_grad_clip_scale_total < accepted_actor_epochs),
        "candidate_group_size": float(group_size),
        "gradient_microbatch_size": float(gradient_microbatch_size),
        "logical_candidate_batch_size": float(num_candidates),
        "step_grad_diagnostics_enabled": float(compute_step_grad_diagnostics),
        "old_policy_lag": _policy_l2_lag(state.policy, state.old_policy),
        "chi2_ppo_upper_bound_mean": (
            1.0 + float(eps_clip.mean().item())
            if chi2_upper_ratio_bound is None
            else float(chi2_upper_ratio_bound.mean().item())
        ),
        **reg_diag,
        **adv_diag,
        **chi2_diag,
    }
    for step_idx, count in enumerate(selected_counts.tolist()):
        metrics[f"selected_step_count_{step_idx}"] = float(count)
        metrics[f"rectifier_count_{step_idx}"] = float(state.rectifier.counts[step_idx].item())
        metrics[f"rectifier_grad_ema_{step_idx}"] = float(state.rectifier.grad_ema[step_idx].item())
        metrics[f"flash_raw_loss_step_{step_idx}"] = raw_loss_by_step[step_idx] / epoch_divisor
        metrics[f"flash_raw_grad_norm_step_{step_idx}"] = raw_grad_by_step[step_idx] / epoch_divisor
        metrics[f"flash_rectified_grad_norm_step_{step_idx}"] = (
            rectified_grad_by_step[step_idx] / epoch_divisor
        )
    return metrics


@torch.no_grad()
def sync_old_policy(state: OGPOTrainState, *, ema: float = 0.0) -> None:
    if not 0.0 <= float(ema) < 1.0:
        raise ValueError("old-policy EMA must be in [0, 1)")
    if state.old_policy is state.policy:
        return
    if float(ema) == 0.0:
        if isinstance(state.policy, PI05JaxFlowPolicy):
            assert isinstance(state.old_policy, PI05JaxFlowPolicy)
            state.old_policy.actor_state = jax.tree.map(lambda value: value, state.policy.actor_state)
            state.old_policy._sync_torch_adapter_from_jax()
        elif isinstance(state.policy, PI05PytorchFlowPolicy):
            assert isinstance(state.old_policy, PI05PytorchFlowPolicy)
            state.old_policy.load_adapter_state_dict(state.policy.adapter_state_dict())
        else:
            state.old_policy.load_state_dict(state.policy.state_dict())
        return
    if isinstance(state.policy, PI05JaxFlowPolicy):
        assert isinstance(state.old_policy, PI05JaxFlowPolicy)
        mixed_state = ema_actor_state(
            state.old_policy.actor_state,
            state.policy.actor_state,
            ema=float(ema),
        )
        state.old_policy._replace_actor_state(mixed_state)
        return
    if isinstance(state.policy, PI05PytorchFlowPolicy):
        if not isinstance(state.old_policy, PI05PytorchFlowPolicy):
            raise TypeError("PyTorch PI0.5 policy requires a PyTorch old-policy snapshot")
        # The frozen snapshot intentionally does not register the shared
        # backend as a submodule, so parameter-list zip/EMA would omit or
        # misalign full-backend tensors. Update adapter and functional backend
        # snapshots by name instead.
        state.old_policy.sync_adapter_snapshot_from(
            state.policy,
            ema=float(ema),
        )
        return
    for old_param, param in zip(state.old_policy.parameters(), state.policy.parameters(), strict=True):
        if not param.requires_grad:
            continue
        old_param.mul_(float(ema)).add_(param.detach(), alpha=1.0 - float(ema))


@torch.no_grad()
def sync_slow_policy(state: OGPOTrainState, *, ema: float) -> None:
    """Update the independent χ² slow policy after an accepted actor update.

    Callers intentionally invoke this only after post-update KL/rollback
    acceptance.  A rejected transaction must leave this snapshot untouched.
    ``ema`` is the retention coefficient for the old slow state; the official
    upstream new/current mixing coefficient is ``tau_slow = 1 - ema``.
    """
    slow_policy = state.slow_policy
    if slow_policy is None:
        return
    if not 0.0 <= float(ema) < 1.0:
        raise ValueError("slow-policy EMA must be in [0, 1)")
    if isinstance(state.policy, PI05PytorchFlowPolicy):
        if not isinstance(slow_policy, PI05PytorchFlowPolicy):
            raise TypeError("PyTorch PI0.5 policy requires a PyTorch slow policy")
        slow_policy.sync_adapter_snapshot_from(state.policy, ema=float(ema))
        return
    if isinstance(state.policy, PI05JaxFlowPolicy):
        if not isinstance(slow_policy, PI05JaxFlowPolicy):
            raise TypeError("JAX PI0.5 policy requires a JAX slow policy")
        if float(ema) == 0.0:
            slow_policy.actor_state = jax.tree.map(lambda value: value, state.policy.actor_state)
            slow_policy._sync_torch_adapter_from_jax()
        else:
            slow_policy._replace_actor_state(
                ema_actor_state(
                    slow_policy.actor_state,
                    state.policy.actor_state,
                    ema=float(ema),
                )
            )
        return
    if float(ema) == 0.0:
        slow_policy.load_state_dict(state.policy.state_dict())
        return
    source_parameters = dict(state.policy.named_parameters())
    for name, parameter in slow_policy.named_parameters():
        source = source_parameters.get(name)
        if source is not None:
            parameter.mul_(float(ema)).add_(source.detach(), alpha=1.0 - float(ema))
    source_buffers = dict(state.policy.named_buffers())
    for name, buffer in slow_policy.named_buffers():
        source = source_buffers.get(name)
        if source is not None:
            buffer.mul_(float(ema)).add_(source.detach(), alpha=1.0 - float(ema))


@torch.no_grad()
def finalize_actor_update_transaction(
    state: OGPOTrainState,
    *,
    accepted: bool,
    config: dict[str, Any],
) -> dict[str, float]:
    """Finalize current/old/slow lifecycle after validation and any rollback.

    The caller must invoke this exactly once per completed actor transaction,
    after a rejected current actor has already been restored. This helper does
    not compute losses, step an optimizer, validate KL, or perform rollback.
    """
    if state.old_policy is state.policy:
        raise RuntimeError("PPO old policy must not alias current policy")
    if state.slow_policy is not None and (
        state.slow_policy is state.policy or state.slow_policy is state.old_policy
    ):
        raise RuntimeError("ChiPO slow policy must be independent of current and old")

    actor_cfg = config.get("actor", {})
    old_period = int(actor_cfg.get("old_policy_sync_period", 1))
    if old_period < 0:
        raise ValueError("actor.old_policy_sync_period must be non-negative")
    slow_cfg = _chi2_config(config)
    slow_period = int(slow_cfg.get("slow_policy_update_period", 1))
    if slow_period <= 0:
        raise ValueError("actor.chi2.slow_policy_update_period must be positive")

    old_sync_applied = False
    slow_sync_applied = False
    if bool(accepted):
        state.accepted_actor_updates += 1
        accepted_count = state.accepted_actor_updates
        if old_period > 0 and accepted_count % old_period == 0:
            sync_old_policy(
                state,
                ema=float(actor_cfg.get("old_policy_ema", 0.0)),
            )
            old_sync_applied = True
        if state.slow_policy is not None and slow_policy_update_due(
            accepted_actor_updates=accepted_count,
            update_period=slow_period,
            update_accepted=True,
        ):
            sync_slow_policy(state, ema=chi2_slow_policy_ema(config))
            slow_sync_applied = True

    # Attempt count advances for accepted and rejected transactions alike.
    state.actor_step += 1
    slow_tau = 1.0 - chi2_slow_policy_ema(config) if state.slow_policy is not None else 0.0
    return {
        "actor_update_accepted": float(bool(accepted)),
        "actor_step": float(state.actor_step),
        "accepted_actor_updates": float(state.accepted_actor_updates),
        "old_policy_sync_applied": float(old_sync_applied),
        "slow_policy_sync_applied": float(slow_sync_applied),
        "slow_policy_update_applied": float(slow_sync_applied),
        # Compatibility aliases used by existing dashboards.
        "chi2_slow_policy_sync_applied": float(slow_sync_applied),
        "old_policy_sync_period": float(old_period),
        "slow_policy_update_period": float(slow_period),
        "slow_tau": float(slow_tau),
        "slow_policy_ema": float(1.0 - slow_tau) if state.slow_policy is not None else 0.0,
        "chi2_slow_policy_tau": float(slow_tau),
    }


def actor_guard_reason(metrics: dict[str, float], config: dict[str, Any]) -> str | None:
    actor_cfg = config.get("actor", {})
    if bool(metrics.get("post_update_kl_exceeded", 0.0)) and _post_update_kl_action(
        actor_cfg
    ) == "stop":
        return "post_update_policy_reference_kl_exceeded"
    reference_kl = float(metrics.get("reference_kl", 0.0))
    if (
        not bool(actor_cfg.get("reject_update_on_kl", False))
        and reference_kl > float(actor_cfg.get("max_policy_reference_kl", float("inf")))
    ):
        return "policy_reference_kl_exceeded"
    disagreement = float(metrics.get("candidate_ensemble_disagreement", 0.0))
    if disagreement > float(actor_cfg.get("max_critic_disagreement", float("inf"))):
        return "critic_disagreement_exceeded"
    support_distance = float(metrics.get("support_distance_mean", 0.0))
    if support_distance > float(actor_cfg.get("max_support_distance", float("inf"))):
        return "support_distance_exceeded"
    if float(metrics.get("consecutive_kl_rejections", 0.0)) >= float(
        actor_cfg.get("max_consecutive_kl_rejections", float("inf"))
    ):
        return "repeated_policy_reference_kl_rejections"
    for key in ("actor_loss", "importance_ratio_mean", "importance_ratio_std"):
        value = torch.tensor(float(metrics.get(key, 0.0)))
        if not torch.isfinite(value):
            return f"nonfinite_{key}"
    return None


def actor_delay_active(step: int, config: dict[str, Any]) -> bool:
    return int(step) < int(config.get("actor", {}).get("actor_delay", 0))


def freeze_critic_for_actor(state: OGPOTrainState) -> None:
    """Make the offline actor-round critic invariant explicit."""
    state.critic.requires_grad_(False)
    state.target_critic.requires_grad_(False)
    if state.divl is not None:
        state.divl.requires_grad_(False)
    if state.target_divl is not None:
        state.target_divl.requires_grad_(False)
    state.critic.eval()
    state.target_critic.eval()
    if state.divl is not None:
        state.divl.eval()
    if state.target_divl is not None:
        state.target_divl.eval()


def actor_start_gate(
    state: OGPOTrainState,
    validation_batch: ChunkBatch,
    config: dict[str, Any],
    *,
    outer_step: int,
) -> tuple[str | None, dict[str, float]]:
    """Return a Phase-B gate reason and validation diagnostics."""
    if actor_delay_active(outer_step, config):
        return "actor_delay", {"critic_training_step": float(state.step)}
    critic_cfg = config.get("critic", {})
    if bool(critic_cfg.get("force_actor", False)):
        return None, {"critic_training_step": float(state.step), "actor_gate_forced": 1.0}
    if state.step < int(critic_cfg.get("warmup_steps", 0)):
        return "critic_warmup", {"critic_training_step": float(state.step)}

    from .evaluator import offline_calibration_metrics  # noqa: PLC0415

    metrics = offline_calibration_metrics(
        state.critic,
        validation_batch,
        divl=state.divl if bool(config.get("divl", {}).get("enabled", True)) else None,
        conformal_scale=state.conformal_scale,
        inference_batch_size=config.get("evaluation", {}).get(
            "actor_gate_inference_batch_size"
        ),
    )
    metrics["critic_training_step"] = float(state.step)
    metrics["critic_gate_sample_count"] = float(validation_batch.batch_size)
    metrics["critic_gate_forced"] = 0.0
    min_ranking = float(
        critic_cfg.get("min_ranking_accuracy", critic_cfg.get("critic_min_ranking_accuracy", float("-inf")))
    )
    metrics["critic_gate_min_ranking_accuracy"] = min_ranking
    metrics["critic_gate_ranking_margin"] = metrics["pairwise_ranking_accuracy"] - min_ranking
    if metrics["pairwise_ranking_accuracy"] < min_ranking:
        metrics["critic_gate_passed"] = 0.0
        return "critic_ranking_accuracy_below_min", metrics
    min_rank_correlation = float(critic_cfg.get("min_q_rank_correlation", float("-inf")))
    metrics["critic_gate_min_q_rank_correlation"] = min_rank_correlation
    metrics["critic_gate_rank_correlation_margin"] = (
        metrics["q_rank_correlation"] - min_rank_correlation
    )
    if metrics["q_rank_correlation"] < min_rank_correlation:
        metrics["critic_gate_passed"] = 0.0
        return "critic_rank_correlation_below_min", metrics
    max_exploitation_gap = float(
        critic_cfg.get("max_abs_q_exploitation_gap", float("inf"))
    )
    metrics["critic_gate_max_abs_q_exploitation_gap"] = max_exploitation_gap
    metrics["critic_gate_exploitation_gap_margin"] = (
        max_exploitation_gap - abs(metrics["q_exploitation_gap"])
    )
    if abs(metrics["q_exploitation_gap"]) > max_exploitation_gap:
        metrics["critic_gate_passed"] = 0.0
        return "critic_q_exploitation_gap_above_max", metrics
    min_coverage = float(critic_cfg.get("min_coverage", critic_cfg.get("critic_min_coverage", 0.0)))
    metrics["critic_gate_min_coverage"] = min_coverage
    metrics["critic_gate_coverage_margin"] = metrics["interval_coverage"] - min_coverage
    if metrics["interval_coverage"] < min_coverage:
        metrics["critic_gate_passed"] = 0.0
        return "critic_coverage_below_min", metrics
    entropy = metrics.get("categorical_entropy", 0.5)
    if entropy < float(critic_cfg.get("min_divl_entropy", 0.0)):
        metrics["critic_gate_passed"] = 0.0
        return "critic_divl_entropy_too_low", metrics
    if entropy > float(critic_cfg.get("max_divl_entropy", 1.0)):
        metrics["critic_gate_passed"] = 0.0
        return "critic_divl_entropy_too_high", metrics
    max_saturation = float(critic_cfg.get("max_categorical_saturation", float("inf")))
    metrics["critic_gate_max_categorical_saturation"] = max_saturation
    metrics["critic_gate_categorical_saturation_margin"] = (
        max_saturation - metrics.get("categorical_saturation", 0.0)
    )
    if metrics.get("categorical_saturation", 0.0) > max_saturation:
        metrics["critic_gate_passed"] = 0.0
        return "critic_categorical_saturation_above_max", metrics
    metrics["critic_gate_passed"] = 1.0
    return None, metrics


def _policy_checkpoint_state(
    policy: OpenPIStochasticFlowPolicy,
    *,
    jax_sidecar: str | None = None,
    jax_sidecar_has_old_policy: bool = True,
    include_pytorch_backend: bool = True,
) -> dict[str, Any]:
    if isinstance(policy, PI05JaxFlowPolicy) and jax_sidecar is not None:
        return {
            "format": "pi05_jax_full_finetune",
            "state": policy.adapter_state_dict(),
            "environment_action_dim": policy.environment_action_dim,
            "flow_action_dim": policy.flow_action_dim,
            "jax_sidecar": jax_sidecar,
            "jax_sidecar_has_old_policy": jax_sidecar_has_old_policy,
        }
    if isinstance(policy, PI05PytorchFlowPolicy):
        return {
            "format": (
                "pi05_pytorch_full_finetune"
                if policy.backend_train_mode != "none"
                else "pi05_residual_adapter"
            ),
            "state": policy.adapter_state_dict(include_backend=include_pytorch_backend),
            "environment_action_dim": policy.environment_action_dim,
            "flow_action_dim": policy.environment_action_dim,
            "backend_train_mode": policy.backend_train_mode,
            "residual_enabled": bool(getattr(policy, "residual_enabled", True)),
            "sde_mode": policy.sde_mode,
            "constant_noise_std": float(getattr(policy, "constant_noise_std", 0.0)),
            "learn_sde_std": bool(getattr(policy, "learn_sde_std", True)),
            "randn_clip_value": float(getattr(policy, "randn_clip_value", 3.0)),
        }
    if isinstance(policy, PI05JaxFlowPolicy):
        return {
            "format": "pi05_residual_adapter",
            "state": policy.adapter_state_dict(),
            "environment_action_dim": policy.environment_action_dim,
            "flow_action_dim": getattr(
                policy,
                "flow_action_dim",
                policy.environment_action_dim,
            ),
        }
    return {"format": "full_state_dict", "state": policy.state_dict()}


def _load_policy_checkpoint_state(policy: OpenPIStochasticFlowPolicy, payload: dict[str, Any]) -> None:
    if "format" not in payload:
        policy.load_state_dict(payload)
    elif payload["format"] in {
        "pi05_residual_adapter",
        "pi05_jax_full_finetune",
        "pi05_pytorch_full_finetune",
    }:
        if not isinstance(policy, (PI05PytorchFlowPolicy, PI05JaxFlowPolicy)):
            raise TypeError("PI0.5 residual checkpoint requires a PI0.5 flow policy (pytorch or jax)")
        if payload["format"] == "pi05_jax_full_finetune" and not isinstance(policy, PI05JaxFlowPolicy):
            raise TypeError("full-finetune JAX checkpoint requires PI05JaxFlowPolicy")
        if payload["format"] == "pi05_pytorch_full_finetune" and not isinstance(
            policy, PI05PytorchFlowPolicy
        ):
            raise TypeError("full-finetune PyTorch checkpoint requires PI05PytorchFlowPolicy")
        policy.load_adapter_state_dict(payload["state"])
    elif payload["format"] == "full_state_dict":
        policy.load_state_dict(payload["state"])
    else:
        raise ValueError(f"unknown policy checkpoint format: {payload['format']!r}")


def _load_q_ensemble_state(module: ScalarQEnsemble, payload: dict[str, torch.Tensor]) -> None:
    incompatible = module.load_state_dict(payload, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_prior_missing = [key for key in incompatible.missing_keys if ".prior." not in key]
    if unexpected or non_prior_missing:
        raise RuntimeError(
            f"incompatible critic checkpoint: missing={non_prior_missing}, unexpected={unexpected}"
        )


def _prepare_restored_critic_stage(state: OGPOTrainState, payload: dict[str, Any]) -> None:
    restored_stage = str(payload.get("critic_stage", state.critic_stage))
    if isinstance(state.critic, MultiHeadUdivlCritic):
        configure_critic_stage(state.critic, restored_stage)
        state.critic_optimizer = _make_critic_optimizer(
            state.critic,
            state.divl,
            payload.get("config", {}).get("critic", {}),
        )
    state.critic_stage = restored_stage
    state.critic_stage_step = int(payload.get("critic_stage_step", 0))


def save_checkpoint(state: OGPOTrainState, config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jax_sidecar_name = None
    checkpoint_old_policy = True
    if isinstance(state.policy, PI05JaxFlowPolicy):
        if not isinstance(state.old_policy, PI05JaxFlowPolicy):
            raise TypeError("JAX current policy requires a JAX old policy")
        jax_sidecar_path = Path(f"{path}.jax")
        checkpoint_old_policy = bool(
            config.get("actor", {}).get("checkpoint_old_policy", True)
        )
        state.policy.save_training_checkpoint(
            jax_sidecar_path,
            old_policy=state.old_policy if checkpoint_old_policy else None,
        )
        jax_sidecar_name = jax_sidecar_path.name
    payload = {
            "policy": _policy_checkpoint_state(
                state.policy,
                jax_sidecar=jax_sidecar_name,
                jax_sidecar_has_old_policy=checkpoint_old_policy,
            ),
            "old_policy": _policy_checkpoint_state(
                state.old_policy,
                include_pytorch_backend=state.old_policy is not state.policy,
            ),
            "slow_policy": (
                None
                if state.slow_policy is None
                else _policy_checkpoint_state(
                    state.slow_policy,
                    include_pytorch_backend=True,
                )
            ),
            "chi2_metadata": {
                "ogpo_variant": ogpo_variant(config),
                "slow_policy_present": state.slow_policy is not None,
                "ratio_scope": config.get("actor", {}).get("chi2", {}).get(
                    "ratio_scope"
                ),
            },
            "offline_metadata": {
                "critic_frozen_during_actor": bool(
                    config.get("training", {}).get(
                        "critic_frozen_during_actor", False
                    )
                ),
                "critic_update_during_actor": bool(
                    config.get("offline", {}).get(
                        "critic_update_during_actor", False
                    )
                ),
            },
            "actor_metadata": {
                "flash_enabled": bool(
                    config.get("actor", {}).get("flash_enabled", False)
                ),
                "full_ratio_mode": config.get("actor", {}).get(
                    "full_ratio_mode"
                ),
                "normalize_logprob_by_action_dim": bool(
                    config.get("actor", {}).get(
                        "normalize_logprob_by_action_dim", False
                    )
                ),
                "normalize_logprob_by_denoising_steps": bool(
                    config.get("actor", {}).get(
                        "normalize_logprob_by_denoising_steps", False
                    )
                ),
            },
            "reference_metadata": {
                "type": type(state.reference_policy).__name__,
                "flow_convention": "openpi_pi05_euler",
                "checkpoint_dir": getattr(state.reference_policy, "checkpoint_dir", None),
                "train_config": getattr(state.reference_policy, "train_config_name", None),
            },
            "support": state.support.detach().cpu(),
            "critic_optimizer": state.critic_optimizer.state_dict(),
            "actor_optimizer": state.actor_optimizer.state_dict(),
            "schedulers": {},
            "running_mad": state.running_mad.value,
            "rectifier_grad_ema": state.rectifier.grad_ema,
            "rectifier_counts": state.rectifier.counts,
            "conformal_scale": state.conformal_scale,
            "training_step": state.step,
            "actor_step": state.actor_step,
            "accepted_actor_updates": state.accepted_actor_updates,
            "critic_stage": state.critic_stage,
            "critic_stage_step": state.critic_stage_step,
            "target_generator_state": (
                None if state.target_generator is None else state.target_generator.get_state()
            ),
            "config": config,
    }
    if isinstance(state.critic, MultiHeadUdivlCritic):
        payload.update(
            {
                "critic_format": "gemma_siglip_multihead",
                "multimodal_critic": state.critic.state_dict(),
                "target_multimodal_critic": state.target_critic.state_dict(),
                "critic_metadata": {
                    **getattr(state.critic, "model_metadata", {}),
                    "action_mean": state.critic.core.action_pool.action_mean.detach().cpu(),
                    "action_std": state.critic.core.action_pool.action_std.detach().cpu(),
                    "q_representation": state.critic.core.q_representation,
                    "num_pairs": state.critic.core.num_pairs,
                    "num_raw_q_heads": state.critic.core.num_q_heads,
                    "q_heads_per_member": state.critic.core.q_heads_per_member,
                    "double_q_divl": bool(
                        config.get("critic", {}).get("double_q_divl", False)
                    ),
                    "q_support": (
                        None
                        if state.critic.core.q_support is None
                        else state.critic.core.q_support.detach().cpu()
                    ),
                    "q_hl_gauss_sigma_bins": config.get("critic", {}).get(
                        "q_hl_gauss_sigma_bins"
                    ),
                    "categorical_q_loss": config.get("critic", {}).get(
                        "categorical_q_loss", "hl_gauss_ce"
                    ),
                    "rank_consensus": {
                        key: config.get("critic", {}).get(key)
                        for key in (
                            "rank_consensus_enabled",
                            "rank_loss_weight",
                            "rank_margin_bins",
                            "rank_softmin_tau",
                            "rank_temperature",
                            "rank_noise_sigma",
                            "rank_use_strong_noise",
                            "rank_use_random_negative",
                            "rank_only_success",
                        )
                    },
                    "rankq": {
                        key: config.get("critic", {}).get(key)
                        for key in (
                            "enable_rankq",
                            "lambda_rank",
                            "rankq_noise_sigma",
                        )
                    },
                },
            }
        )
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        payload.update(
            {
                "critic_format": "gemma_siglip_scalar_q",
                "multimodal_critic": state.critic.state_dict(),
                "target_multimodal_critic": state.target_critic.state_dict(),
                "critic_metadata": {
                    **getattr(state.critic, "model_metadata", {}),
                    "action_mean": state.critic.core.action_pool.action_mean.detach().cpu(),
                    "action_std": state.critic.core.action_pool.action_std.detach().cpu(),
                },
            }
        )
    else:
        assert state.divl is not None and state.target_divl is not None
        payload.update(
            {
                "critic_format": "mlp",
                "critic_ensemble": state.critic.state_dict(),
                "target_critics": state.target_critic.state_dict(),
                "divl": state.divl.state_dict(),
                "target_divl": state.target_divl.state_dict(),
            }
        )
    torch.save(payload, path)


def _validate_multimodal_checkpoint_q_structure(
    payload: dict[str, Any],
    critic: MultiHeadUdivlCritic,
) -> None:
    """Fail clearly before loading a mismatched pair/raw-Q checkpoint."""
    metadata = payload.get("critic_metadata", {})
    source_config = payload.get("config", {}).get("critic", {})
    source_pairs = int(
        metadata.get(
            "num_pairs",
            source_config.get("ensemble_size", 0),
        )
    )
    source_raw_q = int(metadata.get("num_raw_q_heads", 0))
    if source_raw_q <= 0:
        q_head_indices = {
            int(key.split(".")[2])
            for key in payload.get("multimodal_critic", {})
            if key.startswith("core.q_heads.") and key.split(".")[2].isdigit()
        }
        source_raw_q = len(q_head_indices)
    if source_pairs <= 0:
        heads_per_pair = int(
            metadata.get(
                "q_heads_per_member",
                source_config.get("q_heads_per_member", 1),
            )
        )
        if source_raw_q > 0 and heads_per_pair > 0:
            source_pairs = source_raw_q // heads_per_pair
    expected_pairs = critic.core.num_pairs
    expected_raw_q = critic.core.num_q_heads
    if source_pairs != expected_pairs or source_raw_q != expected_raw_q:
        raise ValueError(
            "Expected a "
            f"{expected_pairs}-pair / {expected_raw_q}-Q critic for OGPO, "
            "but loaded checkpoint contains "
            f"{source_pairs} pairs / {source_raw_q} raw Q heads."
        )


def load_checkpoint(
    path: str | Path,
    state: OGPOTrainState,
    *,
    restore_actor_optimizer: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=next(state.policy.parameters()).device, weights_only=False)
    _load_policy_checkpoint_state(state.policy, payload["policy"])
    old_payload = payload.get("old_policy")
    if old_payload is None:
        warnings.warn(
            "legacy checkpoint has no old_policy; initializing PPO old from current",
            RuntimeWarning,
            stacklevel=2,
        )
        sync_old_policy(state, ema=0.0)
        payload["ppo_old_policy_restored"] = False
    else:
        _load_policy_checkpoint_state(state.old_policy, old_payload)
        payload["ppo_old_policy_restored"] = True
    slow_payload = payload.get("slow_policy")
    if state.slow_policy is not None:
        if slow_payload is None:
            # A CA checkpoint can seed a new χ² run.  It has no historical
            # slow policy, so the only correct deterministic initialization is
            # the restored current actor (before the next EMA update).
            sync_slow_policy(state, ema=0.0)
            warnings.warn(
                "legacy checkpoint has no slow_policy; initializing ChiPO slow from current",
                RuntimeWarning,
                stacklevel=2,
            )
            payload["chi2_slow_policy_restored"] = False
        else:
            _load_policy_checkpoint_state(state.slow_policy, slow_payload)
            payload["chi2_slow_policy_restored"] = True
    if payload["policy"].get("format") == "pi05_jax_full_finetune":
        if not isinstance(state.policy, PI05JaxFlowPolicy) or not isinstance(
            state.old_policy,
            PI05JaxFlowPolicy,
        ):
            raise TypeError("full-finetune JAX checkpoint requires JAX current and old policies")
        sidecar = path.parent / payload["policy"]["jax_sidecar"]
        sidecar_has_old_policy = bool(
            payload["policy"].get("jax_sidecar_has_old_policy", True)
        )
        state.policy.restore_training_checkpoint(
            sidecar,
            old_policy=state.old_policy if sidecar_has_old_policy else None,
            restore_optimizer=restore_actor_optimizer,
        )
        if not sidecar_has_old_policy:
            state.old_policy.actor_state = jax.tree.map(
                lambda value: value,
                state.policy.actor_state,
            )
            state.old_policy._sync_torch_adapter_from_jax()
    if isinstance(state.critic, MultiHeadUdivlCritic):
        if payload.get("critic_format") != "gemma_siglip_multihead":
            raise ValueError("checkpoint does not contain a multimodal critic")
        _validate_multimodal_checkpoint_q_structure(payload, state.critic)
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        if payload.get("critic_format") != "gemma_siglip_scalar_q":
            raise ValueError("checkpoint does not contain an OGPO-origin scalar Q critic")
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    else:
        _load_q_ensemble_state(state.critic, payload["critic_ensemble"])
        _load_q_ensemble_state(state.target_critic, payload["target_critics"])
        assert state.divl is not None and state.target_divl is not None
        state.divl.load_state_dict(payload["divl"])
        state.target_divl.load_state_dict(payload["target_divl"])
    if "support" in payload:
        state.support = payload["support"].to(next(state.critic.parameters()).device)
    _prepare_restored_critic_stage(state, payload)
    state.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    if restore_actor_optimizer:
        state.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    state.running_mad.value = float(payload["running_mad"])
    state.conformal_scale = float(payload.get("conformal_scale", 1.0))
    if "rectifier_grad_ema" in payload:
        state.rectifier.grad_ema = payload["rectifier_grad_ema"].detach().cpu()
    if "rectifier_counts" in payload:
        state.rectifier.counts = payload["rectifier_counts"].detach().cpu()
    state.step = int(payload["training_step"])
    if "actor_step" not in payload:
        warnings.warn(
            "legacy checkpoint has no actor_step; defaulting actor attempts to 0",
            RuntimeWarning,
            stacklevel=2,
        )
    if "accepted_actor_updates" not in payload:
        warnings.warn(
            "legacy checkpoint has no accepted_actor_updates; defaulting to 0",
            RuntimeWarning,
            stacklevel=2,
        )
    state.actor_step = int(payload.get("actor_step", 0))
    state.accepted_actor_updates = int(payload.get("accepted_actor_updates", 0))
    if state.target_generator is not None and payload.get("target_generator_state") is not None:
        state.target_generator.set_state(payload["target_generator_state"].cpu())
    return payload


def load_critic_checkpoint(
    path: str | Path,
    state: OGPOTrainState,
    *,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    """Restore outer-MDP value state without requiring the same actor type."""
    payload = torch.load(path, map_location=next(state.critic.parameters()).device, weights_only=False)
    if isinstance(state.critic, MultiHeadUdivlCritic):
        if payload.get("critic_format") != "gemma_siglip_multihead":
            raise ValueError("checkpoint does not contain a multimodal critic")
        _validate_multimodal_checkpoint_q_structure(payload, state.critic)
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    elif isinstance(state.critic, MultiHeadScalarQCritic):
        if payload.get("critic_format") != "gemma_siglip_scalar_q":
            raise ValueError("checkpoint does not contain an OGPO-origin scalar Q critic")
        state.critic.load_state_dict(payload["multimodal_critic"])
        state.target_critic.load_state_dict(payload["target_multimodal_critic"])
    else:
        _load_q_ensemble_state(state.critic, payload["critic_ensemble"])
        _load_q_ensemble_state(state.target_critic, payload["target_critics"])
        assert state.divl is not None and state.target_divl is not None
        state.divl.load_state_dict(payload["divl"])
        state.target_divl.load_state_dict(payload["target_divl"])
    if "support" in payload:
        state.support = payload["support"].to(next(state.critic.parameters()).device)
    _prepare_restored_critic_stage(state, payload)
    if load_optimizer and "critic_optimizer" in payload:
        state.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    state.running_mad.value = float(payload.get("running_mad", state.running_mad.value))
    state.conformal_scale = float(payload.get("conformal_scale", 1.0))
    state.step = int(payload.get("training_step", 0))
    if state.target_generator is not None and payload.get("target_generator_state") is not None:
        state.target_generator.set_state(payload["target_generator_state"].cpu())
    return payload


def initialize_critic_from_checkpoint(
    path: str | Path,
    state: OGPOTrainState,
    *,
    affine_rebase_action_normalizer: bool = False,
) -> dict[str, Any]:
    """Initialize a critic while keeping the current run config and optimizer fresh.

    This is intentionally distinct from resume: it supports scalar-Q to
    categorical-Q initialization and does not restore steps, optimizer state,
    training stage, support, or early-stopping state. When requested, the
    action projection is rebased so the destination replay's normalization
    statistics are retained without changing the initialized projection's
    output for any raw action.
    """
    path = Path(path)
    payload = torch.load(
        path,
        map_location=next(state.critic.parameters()).device,
        weights_only=False,
    )
    if not isinstance(state.critic, MultiHeadUdivlCritic):
        raise TypeError("partial critic initialization requires MultiHeadUdivlCritic")
    if payload.get("critic_format") != "gemma_siglip_multihead":
        raise ValueError("initial checkpoint does not contain a multimodal U-DIVL critic")

    destination_action_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if affine_rebase_action_normalizer:
        for name, module in (
            ("online", state.critic),
            ("target", state.target_critic),
        ):
            action_pool = module.core.action_pool
            destination_action_stats[name] = (
                action_pool.action_mean.detach().clone(),
                action_pool.action_std.detach().clone(),
            )

    def load_compatible(
        module: MultiHeadUdivlCritic,
        source: dict[str, torch.Tensor],
        *,
        skip_q_heads: bool,
    ) -> tuple[list[str], list[str]]:
        destination = module.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for name, value in source.items():
            if skip_q_heads and (name.startswith("core.q_heads.") or name == "core.q_support"):
                skipped.append(name)
                continue
            if name not in destination or destination[name].shape != value.shape:
                skipped.append(name)
                continue
            compatible[name] = value
        module.load_state_dict(compatible, strict=False)
        return sorted(compatible), sorted(skipped)

    source_online = payload["multimodal_critic"]
    source_target = payload["target_multimodal_critic"]
    destination_categorical = state.critic.core.q_representation == "categorical"
    source_categorical = "core.q_support" in source_online
    scalar_to_categorical = destination_categorical and not source_categorical
    online_loaded, online_skipped = load_compatible(
        state.critic,
        source_online,
        skip_q_heads=scalar_to_categorical,
    )
    target_loaded, target_skipped = load_compatible(
        state.target_critic,
        source_target,
        skip_q_heads=scalar_to_categorical,
    )
    if affine_rebase_action_normalizer:
        required_action_parameters = {
            "core.action_pool.action_mean",
            "core.action_pool.action_std",
            "core.action_pool.action_projection.weight",
            "core.action_pool.action_projection.bias",
        }

        def rebase_action_normalizer(
            module: MultiHeadUdivlCritic,
            loaded: list[str],
            destination_mean: torch.Tensor,
            destination_std: torch.Tensor,
            *,
            module_name: str,
        ) -> None:
            missing = required_action_parameters.difference(loaded)
            if missing:
                raise ValueError(
                    f"cannot affine-rebase {module_name} action normalizer; "
                    f"checkpoint did not load: {', '.join(sorted(missing))}"
                )
            action_pool = module.core.action_pool
            projection = action_pool.action_projection
            if projection.bias is None:
                raise ValueError(
                    f"cannot affine-rebase {module_name} action normalizer without a bias"
                )
            source_mean = action_pool.action_mean.detach().clone()
            source_std = action_pool.action_std.detach().clone()
            if not all(
                bool(torch.isfinite(value).all())
                for value in (source_mean, source_std, destination_mean, destination_std)
            ):
                raise ValueError(
                    f"cannot affine-rebase {module_name} action normalizer with non-finite statistics"
                )
            if bool((source_std <= 0).any()) or bool((destination_std <= 0).any()):
                raise ValueError(
                    f"cannot affine-rebase {module_name} action normalizer with non-positive std"
                )

            projection_destination_mean = destination_mean.to(
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
            projection_destination_std = destination_std.to(
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
            projection_source_mean = source_mean.to(
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
            projection_source_std = source_std.to(
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
            source_weight = projection.weight.detach().clone()
            source_bias = projection.bias.detach().clone()
            source_scale = projection_source_std.clamp_min(1e-6)
            destination_scale = projection_destination_std.clamp_min(1e-6)
            scale = destination_scale / source_scale
            shift = (projection_destination_mean - projection_source_mean) / source_scale
            with torch.no_grad():
                projection.weight.copy_(source_weight * scale.unsqueeze(0))
                projection.bias.copy_(source_bias + source_weight @ shift)
                action_pool.action_mean.copy_(destination_mean)
                action_pool.action_std.copy_(destination_std)

        rebase_action_normalizer(
            state.critic,
            online_loaded,
            *destination_action_stats["online"],
            module_name="online critic",
        )
        rebase_action_normalizer(
            state.target_critic,
            target_loaded,
            *destination_action_stats["target"],
            module_name="target critic",
        )
        print(
            "[critic-init] affine-rebased online/target action projections and retained "
            "destination replay normalization statistics",
            flush=True,
        )
    if scalar_to_categorical:
        state.target_critic.core.q_heads.load_state_dict(
            state.critic.core.q_heads.state_dict()
        )

    q_reinitialized = sorted(
        name
        for name in state.critic.state_dict()
        if name.startswith("core.q_heads.") or name == "core.q_support"
    ) if scalar_to_categorical else []
    print(
        "[critic-init] loaded parameters: "
        f"online={len(online_loaded)} target={len(target_loaded)} from {path}",
        flush=True,
    )
    print(
        "[critic-init] reinitialized categorical Q parameters: "
        + (", ".join(q_reinitialized) if q_reinitialized else "none"),
        flush=True,
    )
    print(
        "[critic-init] skipped incompatible source parameters: "
        f"online={len(online_skipped)} target={len(target_skipped)}",
        flush=True,
    )
    print("[critic-init] skipped incompatible optimizer states: fresh optimizer", flush=True)
    return payload
