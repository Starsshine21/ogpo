from __future__ import annotations

import torch

from evaluate_ogpo_critic_v2 import advantage_metrics, candidate_ranking_metrics


def test_unanimous_candidate_ranking_is_consistent() -> None:
    values = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(1, 4, 1).expand(3, 4, 10)
    metrics = candidate_ranking_metrics(values)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["pairwise_ranking_consistency"] == 1.0
    assert metrics["candidate_disagreement"] == 0.0


def test_split_top_choice_has_half_top1_agreement() -> None:
    values = torch.zeros(1, 4, 10)
    values[0, 0, :5] = 2.0
    values[0, 1, 5:] = 2.0
    metrics = candidate_ranking_metrics(values)
    assert metrics["top1_agreement"] == 0.5


def test_advantage_metrics_use_production_consensus() -> None:
    values = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(1, 4, 1).expand(2, 4, 10)
    metrics = advantage_metrics(values)
    assert metrics["ca_nonzero_fraction"] == 1.0
    assert metrics["median_abs_advantage"] > 0.0
