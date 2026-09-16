from __future__ import annotations

import torch


def analytic_rectification(
    timestep: torch.Tensor,
    *,
    stochastic_variance: float = 1.0,
    sde_mode: str = "ogpo_corrected",
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Rectify the timestep-dependent velocity score scale.

    The OGPO-corrected PI0.5 transition has standard deviation
    ``sigma * sqrt(t)`` and velocity coefficient
    ``|dt| * (1 + 0.5 * sigma**2 * (1 - t))``.  Flash-GRPO therefore
    weights its selected-step gradient by the reciprocal of their ratio.
    Constants shared by every sample cancel under batch-mean normalization.
    """
    flat_t = timestep.detach().reshape(-1).float()
    if sde_mode == "gaussian_adapter":
        return torch.ones_like(flat_t)
    if sde_mode != "ogpo_corrected":
        raise ValueError(f"unsupported SDE mode for analytic rectification: {sde_mode!r}")
    if stochastic_variance <= 0.0:
        raise ValueError("stochastic_variance must be positive")
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError("rectification clipping bounds must satisfy 0 < min <= max")

    variance = flat_t.new_tensor(float(stochastic_variance))
    t = flat_t.clamp_min(torch.finfo(flat_t.dtype).eps)
    inverse_scale = torch.sqrt(t) / (1.0 + 0.5 * variance * (1.0 - t))
    weights = inverse_scale / inverse_scale.mean().clamp_min(torch.finfo(flat_t.dtype).eps)
    return weights.clamp(float(clip_min), float(clip_max))


class EmpiricalGradientRectifier:
    def __init__(self, num_steps: int, momentum: float = 0.95, clip_min: float = 0.25, clip_max: float = 4.0):
        self.num_steps = int(num_steps)
        self.momentum = float(momentum)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.grad_ema = torch.ones(self.num_steps)
        self.counts = torch.zeros(self.num_steps, dtype=torch.long)

    def update(self, step_index: int, raw_grad_norm: float, *, count: int = 1) -> None:
        idx = int(step_index)
        if idx < 0 or idx >= self.num_steps:
            raise ValueError("step_index outside rectifier range")
        value = torch.tensor(float(raw_grad_norm)).clamp_min(1e-8)
        self.grad_ema[idx] = self.momentum * self.grad_ema[idx] + (1.0 - self.momentum) * value
        self.counts[idx] += int(count)

    def weight(self, step_index: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        idx = int(step_index)
        global_ema = self.grad_ema.mean()
        weight = global_ema / self.grad_ema[idx].clamp_min(1e-8)
        return weight.clamp(self.clip_min, self.clip_max).to(device) if device is not None else weight
