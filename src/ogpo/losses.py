from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .openpi_flow_spec import OpenPIFlowSpec


@dataclass(frozen=True)
class LossWithDiagnostics:
    loss: torch.Tensor
    diagnostics: dict[str, float]


def flow_matching_anchor_loss(
    policy,
    condition: torch.Tensor,
    action_endpoint: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
) -> LossWithDiagnostics:
    """OpenPI PI0/PI0.5 linear-interpolation flow-matching anchor."""
    batch, action_dim = action_endpoint.shape
    if noise is None:
        noise = torch.randn(
            batch,
            action_dim,
            dtype=action_endpoint.dtype,
            device=action_endpoint.device,
            generator=generator,
        )
    if timestep is None:
        flow_spec = getattr(policy, "flow_spec", OpenPIFlowSpec())
        timestep = flow_spec.sample_training_time(
            (batch,),
            device=action_endpoint.device,
            dtype=action_endpoint.dtype,
            generator=generator,
        ).unsqueeze(-1)
    flow_spec = getattr(policy, "flow_spec", OpenPIFlowSpec())
    x_tau, target_velocity = flow_spec.training_pair(action_endpoint, noise, timestep)
    pred_velocity = policy.predict_velocity(x_tau, condition, timestep)
    loss = F.mse_loss(pred_velocity, target_velocity)
    return LossWithDiagnostics(loss=loss, diagnostics={"fm_anchor_loss": float(loss.detach().item())})


def weighted_flow_matching_loss(
    policy,
    condition: torch.Tensor,
    action_endpoint: torch.Tensor,
    weights: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
) -> LossWithDiagnostics:
    """AWR-style per-sample weighted OpenPI flow-matching objective."""
    batch, action_dim = action_endpoint.shape
    if weights.shape != (batch,):
        raise ValueError("flow-matching weights must have shape [batch]")
    if (weights < 0).any():
        raise ValueError("flow-matching weights must be non-negative")
    if noise is None:
        noise = torch.randn(
            batch,
            action_dim,
            dtype=action_endpoint.dtype,
            device=action_endpoint.device,
            generator=generator,
        )
    flow_spec = getattr(policy, "flow_spec", OpenPIFlowSpec())
    if timestep is None:
        timestep = flow_spec.sample_training_time(
            (batch,),
            device=action_endpoint.device,
            dtype=action_endpoint.dtype,
            generator=generator,
        ).unsqueeze(-1)
    x_tau, target_velocity = flow_spec.training_pair(action_endpoint, noise, timestep)
    pred_velocity = policy.predict_velocity(x_tau, condition, timestep)
    per_sample = (pred_velocity - target_velocity).pow(2).reshape(batch, -1).mean(dim=1)
    detached_weights = weights.detach().to(dtype=per_sample.dtype, device=per_sample.device)
    loss = (per_sample * detached_weights).sum() / detached_weights.sum().clamp_min(1e-8)
    return LossWithDiagnostics(
        loss=loss,
        diagnostics={
            "weighted_fm_loss": float(loss.detach().item()),
            "awr_weight_mean": float(detached_weights.mean().item()),
            "awr_weight_max": float(detached_weights.max().item()),
        },
    )


def success_buffer_loss(policy, condition: torch.Tensor, action_endpoint: torch.Tensor) -> LossWithDiagnostics:
    result = flow_matching_anchor_loss(policy, condition, action_endpoint)
    return LossWithDiagnostics(
        loss=result.loss,
        diagnostics={"success_buffer_loss": result.diagnostics["fm_anchor_loss"]},
    )


def action_smoothness_loss(action_chunks: torch.Tensor, gripper_mask: torch.Tensor | None = None, eta: float = 0.1) -> torch.Tensor:
    assert action_chunks.ndim == 3
    actions = action_chunks
    if gripper_mask is not None:
        keep = (~gripper_mask.bool()).to(actions.dtype).view(1, 1, -1)
        actions = actions * keep
    vel = actions[:, 1:] - actions[:, :-1]
    acc = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2] if actions.shape[1] >= 3 else torch.zeros_like(vel[:, :0])
    return vel.pow(2).mean() + float(eta) * (acc.pow(2).mean() if acc.numel() else vel.new_tensor(0.0))
