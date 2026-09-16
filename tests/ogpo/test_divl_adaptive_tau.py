import pytest
import torch

from ogpo.divl import divl_quantile_values


def _stats(probs):
    return divl_quantile_values(
        probs,
        torch.linspace(-0.1, 1.1, 201),
        alpha_min=0.0,
        alpha_max=1.0,
        adaptive_tau_mode=True,
        tau_base=0.6,
        entropy_coefficient=0.3,
        tau_min=0.3,
        tau_max=0.6,
        interpolate_quantile=False,
    )


def test_adaptive_tau_uses_entropy_normalized_by_log_201():
    low = torch.zeros(1, 1, 201)
    low[..., 100] = 1.0
    middle = torch.full((1, 1, 201), 0.2 / 200)
    middle[..., 100] = 0.8
    uniform = torch.full((1, 1, 201), 1.0 / 201)
    low_stats, middle_stats, high_stats = map(_stats, (low, middle, uniform))
    print(
        "entropy/tau",
        (low_stats.entropy.item(), low_stats.tau.item()),
        (middle_stats.entropy.item(), middle_stats.tau.item()),
        (high_stats.entropy.item(), high_stats.tau.item()),
    )
    assert low_stats.entropy.item() == pytest.approx(0.0, abs=1e-5)
    assert low_stats.tau.item() == pytest.approx(0.6, abs=1e-5)
    assert high_stats.entropy.item() == pytest.approx(1.0, abs=1e-5)
    assert high_stats.tau.item() == pytest.approx(0.3, abs=1e-5)
    assert low_stats.entropy < middle_stats.entropy < high_stats.entropy
    assert low_stats.tau > middle_stats.tau > high_stats.tau
