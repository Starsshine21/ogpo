import torch

from ogpo.conservative_advantage import (
    RunningMAD,
    group_normalized_advantage,
    scheduled_lambda_abs,
    sign_consensus_advantage,
)


def test_sign_consensus_positive_negative_disagreement_and_margin():
    baseline = torch.zeros(3, 4)
    q = torch.tensor(
        [
            [1.0, -3.0, 1.0, 0.05],
            [2.0, -2.0, -1.0, 0.06],
            [3.0, -1.0, 2.0, 0.07],
        ]
    )
    adv, stats = sign_consensus_advantage(q, baseline, positive_margin=0.1, negative_margin=0.1)
    assert adv[0] == 1.0
    assert adv[1] == -1.0
    assert adv[2] == 0.0
    assert adv[3] == 0.0
    assert stats.positive_consensus_ratio == 0.25
    assert stats.negative_consensus_ratio == 0.25


def test_uniformly_poor_group_does_not_make_its_least_bad_candidate_positive():
    value = torch.tensor([[5.0], [4.5], [5.5]])
    q = torch.tensor(
        [
            [[1.0, 3.0, 2.0]],
            [[0.5, 4.0, 1.5]],
            [[2.0, 5.0, 3.0]],
        ]
    )

    advantage, _ = sign_consensus_advantage(q, value)

    assert torch.equal(advantage, torch.tensor([[-3.5, -0.5, -2.5]]))
    assert advantage.max() <= 0.0


def test_running_mad_ignores_zero_samples():
    mad = RunningMAD(momentum=0.0)
    scale = mad.update(torch.tensor([0.0, 0.0, 2.0, 4.0]), ignore_zero=True)
    assert scale > 0.0
    before = mad.value
    after = mad.update(torch.zeros(10), ignore_zero=True)
    assert after == before


def test_group_normalization_ablation_is_separate():
    q = torch.randn(3, 2, 4)
    group = group_normalized_advantage(q)
    assert group.shape == (2, 4)
    assert torch.allclose(group.mean(dim=-1), torch.zeros(2), atol=1e-5)


def test_absolute_advantage_mix_schedule_reaches_one():
    assert scheduled_lambda_abs(0, start=0.25, end=1.0, warmup_steps=100) == 0.25
    assert scheduled_lambda_abs(50, start=0.25, end=1.0, warmup_steps=100) == 0.625
    assert scheduled_lambda_abs(100, start=0.25, end=1.0, warmup_steps=100) == 1.0
    assert scheduled_lambda_abs(200, start=0.25, end=1.0, warmup_steps=100) == 1.0
