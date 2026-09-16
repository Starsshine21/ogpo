from __future__ import annotations

import torch


def aggregate_value_heads(
    values: torch.Tensor,
    mode: str,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Aggregate [heads, batch] values and expose sampled indices for diagnostics."""
    if values.ndim != 2:
        raise ValueError("values must have shape [heads, batch]")
    heads, batch = values.shape
    normalized_mode = {"ensemble_mean": "mean", "ensemble_min": "min"}.get(mode, mode)
    if normalized_mode == "mean":
        return values.mean(dim=0), None
    if normalized_mode == "min":
        return values.min(dim=0).values, None
    if normalized_mode != "subsample_min":
        raise ValueError(f"unsupported value aggregation mode={mode!r}")
    if heads < 2:
        raise ValueError("subsample_min requires at least two heads")

    first = torch.randint(heads, (batch,), device=values.device, generator=generator)
    offset = torch.randint(1, heads, (batch,), device=values.device, generator=generator)
    second = (first + offset) % heads
    indices = torch.stack([first, second], dim=0)
    first_values = values.gather(0, first.unsqueeze(0)).squeeze(0)
    second_values = values.gather(0, second.unsqueeze(0)).squeeze(0)
    return torch.minimum(first_values, second_values), indices

