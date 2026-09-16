import torch

from ogpo.divl import divl_quantile_values
from ogpo.distributional_value import (
    adaptive_alpha,
    categorical_entropy,
    categorical_projection,
    categorical_quantile,
    make_support,
    make_support_from_targets,
)


def test_fixed_divl_quantile_disables_entropy_adaptation():
    support = torch.linspace(-1.0, 1.0, 5)
    probs = torch.tensor(
        [[[0.0, 0.0, 1.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]]]
    )
    stats = divl_quantile_values(
        probs,
        support,
        alpha_min=0.4,
        alpha_max=0.8,
        use_adaptive_quantile=False,
    )

    assert torch.equal(stats.alpha, torch.full((1, 2), 0.8))


def test_projection_probability_sum_clip_and_single_point():
    support = make_support(-1.0, 1.0, 5)
    targets = torch.tensor([-2.0, 0.0, 2.0])
    probs = categorical_projection(targets, support)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))
    assert torch.isfinite(probs).all()
    assert probs[0, 0] == 1.0
    assert probs[1, 2] == 1.0
    assert probs[2, -1] == 1.0


def test_support_can_be_estimated_from_fixed_replay_returns():
    support = make_support_from_targets(
        torch.tensor([0.0, 0.25, 1.0]),
        num_atoms=5,
        margin_fraction=0.1,
    )

    assert torch.allclose(support, torch.linspace(-0.1, 1.1, 5))


def test_quantile_monotonic_entropy_alpha_range():
    support = make_support(0.0, 4.0, 5)
    probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])
    q25 = categorical_quantile(probs, support, torch.tensor([0.25]))
    q75 = categorical_quantile(probs, support, torch.tensor([0.75]))
    assert q75 >= q25
    entropy = categorical_entropy(probs, normalized=True)
    assert torch.all((entropy >= 0.0) & (entropy <= 1.0))
    alpha = adaptive_alpha(entropy, 0.4, 0.8)
    assert torch.all((alpha >= 0.4) & (alpha <= 0.8))


def test_lwd_quantile_selects_first_atom_crossing_cdf():
    support = torch.tensor([0.0, 1.0, 2.0])
    probs = torch.tensor([[0.2, 0.5, 0.3]])

    quantile = categorical_quantile(
        probs,
        support,
        torch.tensor([0.5]),
        interpolate=False,
    )

    assert torch.equal(quantile, torch.tensor([1.0]))


def test_lwd_adaptive_alpha_uses_base_minus_entropy_sensitivity():
    entropy = torch.tensor([0.0, 0.2, 1.0])

    alpha = adaptive_alpha(
        entropy,
        alpha_min=0.5,
        alpha_max=0.6,
        temperature=0.3,
        mode="lwd_linear",
    )

    assert torch.allclose(alpha, torch.tensor([0.6, 0.54, 0.5]))
