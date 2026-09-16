from __future__ import annotations

import torch


def bootstrap_mask(
    ensemble_size: int,
    batch_size: int,
    probability: float,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 < probability <= 1.0:
        raise ValueError("bootstrap probability must be in (0, 1]")
    probs = torch.full((ensemble_size, batch_size), probability, device=device)
    return torch.bernoulli(probs, generator=generator).bool()


def ensemble_mean_std(values: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean/std over ensemble dimension 0."""
    assert values.ndim >= 2
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std


def sign_agreement(values: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """True when all ensemble members agree on a strict sign."""
    all_pos = torch.min(values, dim=0).values > margin
    all_neg = torch.max(values, dim=0).values < -margin
    return all_pos | all_neg
