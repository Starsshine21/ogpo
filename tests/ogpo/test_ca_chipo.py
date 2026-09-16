import torch

from ogpo.chi2_regularization import (
    chi2_pessimistic_advantage,
    compute_chipo_beta,
    sign_safe_advantage_intersection,
)
from ogpo.conservative_advantage import _safe_max


def test_upstream_safe_max_exact_sign_cases():
    values = torch.tensor([[1.0, -1.0, 1.0], [2.0, -2.0, -1.0], [3.0, -3.0, 2.0]])
    assert torch.equal(_safe_max(values, dim=0), torch.tensor([1.0, -1.0, 0.0]))


def test_official_chipo_numerical_parity():
    q = torch.tensor(
        [[[1.0, 3.0]], [[0.0, 4.0]], [[2.0, 2.0]]]
    )  # [member, batch, candidate]
    ratio = torch.tensor([[1.2, 0.8]])
    beta, q_std = compute_chipo_beta(q, beta_base=0.1, q_std_target=1.0)
    q_mean = q.mean(0)
    q_min = q.min(0).values
    weight = torch.sigmoid(5.0 * (ratio - 1.0))
    q_target = (1.0 - weight) * q_mean + weight * q_min
    penalized = q_target - beta * ratio
    expected = penalized - penalized.mean(-1, keepdim=True)
    actual, stats = chi2_pessimistic_advantage(
        q,
        ratio,
        beta_base=0.1,
        q_std_target=1.0,
        ensemble_alpha=5.0,
        normalize_group=False,
        advantage_clip=None,
    )
    assert torch.allclose(q_std, q.std(0, unbiased=False).mean())
    assert torch.allclose(actual, expected)
    assert stats.q_target_mean == float(q_target.mean())


def test_ca_zero_and_opposite_sign_are_always_filtered():
    ca = torch.tensor([[0.0, 1.0, 1.0, -2.0]])
    chi = torch.tensor([[3.0, -1.0, 0.5, -1.0]])
    final, _ = sign_safe_advantage_intersection(ca, chi)
    assert torch.equal(final, torch.tensor([[0.0, 0.0, 0.5, -1.0]]))
