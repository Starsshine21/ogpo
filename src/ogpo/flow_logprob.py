from __future__ import annotations

import math

import torch


def gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian log probability summed over the last dimension."""
    var = torch.exp(2.0 * log_std)
    log_prob = -0.5 * (((value - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))
    return log_prob.flatten(start_dim=-1).sum(dim=-1)


def gaussian_kl_diag(
    mean_p: torch.Tensor,
    log_std_p: torch.Tensor,
    mean_q: torch.Tensor,
    log_std_q: torch.Tensor,
) -> torch.Tensor:
    var_p = torch.exp(2.0 * log_std_p)
    var_q = torch.exp(2.0 * log_std_q)
    kl = log_std_q - log_std_p + (var_p + (mean_p - mean_q).pow(2)) / (2.0 * var_q) - 0.5
    return kl.flatten(start_dim=-1).sum(dim=-1)
