from __future__ import annotations

import pytest
import torch

from ogpo.chi2_regularization import apply_chipo_to_ca_advantage
from ogpo.conservative_advantage import group_relative_conservative_advantage
from ogpo.critic_raw10_evaluator import (
    actor_signal_metrics,
    fixed_validation_indices,
    focused_return_metrics,
    off_diagonal_q_correlation_metrics,
    rank_checkpoints,
    raw10_fidelity_metrics,
    raw10_from_q_pairs,
    same_state_action_ranking_metrics,
    validate_candidate_cache,
    validate_selection_split,
)


CHIPO = {
    "beta_base": 0.1,
    "q_std_target": 1.0,
    "ensemble_alpha": 5.0,
    "r_max": 10.0,
}


def test_five_by_two_pairs_flatten_to_ten_raw_q_in_member_major_order():
    pairs = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    raw = raw10_from_q_pairs(pairs)
    assert raw.shape == (10, 3)
    assert torch.equal(raw[0], pairs[0, 0])
    assert torch.equal(raw[1], pairs[0, 1])
    assert torch.equal(raw[8], pairs[4, 0])
    assert torch.equal(raw[9], pairs[4, 1])


def test_ogpo_fidelity_rejects_pair_min_five_head_input():
    pair_min = torch.randn(5, 8)
    target = torch.randn(8)
    try:
        raw10_fidelity_metrics(pair_min, target)
    except ValueError as error:
        assert "[10,N]" in str(error)
    else:
        raise AssertionError("pair-min values must not be accepted as OGPO raw10 scores")


def test_same_state_action_ranking_reports_ten_head_unanimity():
    positive = torch.ones(10, 2)
    negative = torch.zeros(3, 10, 2)
    metrics = same_state_action_ranking_metrics(positive, negative)
    assert metrics["raw10_action_rank_mean"] == 1.0
    assert metrics["raw10_action_rank_unanimous"] == 1.0
    assert metrics["raw10_min_action_margin"] == 1.0
    assert metrics["raw10_member_action_rank_mean"] == 1.0
    assert metrics["raw10_member_action_rank_min"] == 1.0


def test_mild_strong_random_action_metrics_remain_separable():
    positive = torch.ones(10, 2)
    mild = torch.full((1, 10, 2), 0.5)
    strong = torch.full((1, 10, 2), 0.0)
    random = torch.full((1, 10, 2), 2.0)
    metrics = {
        name: same_state_action_ranking_metrics(positive, negative)
        for name, negative in (("mild", mild), ("strong", strong), ("random", random))
    }
    assert metrics["mild"]["raw10_action_rank_unanimous"] == 1.0
    assert metrics["strong"]["raw10_action_rank_unanimous"] == 1.0
    assert metrics["random"]["raw10_action_rank_unanimous"] == 0.0


def test_member_action_rank_min_captures_one_veto_head():
    positive = torch.ones(10, 4)
    negative = torch.zeros(1, 10, 4)
    negative[:, 9, :3] = 2.0
    metrics = same_state_action_ranking_metrics(positive, negative)
    assert metrics["raw10_member_action_rank_mean"] == pytest.approx(0.925)
    assert metrics["raw10_member_action_rank_min"] == 0.25
    assert metrics["raw10_action_rank_unanimous"] == 0.25


def test_ten_positive_members_produce_positive_ca_candidate():
    q_values = torch.tensor([[[1.0, -1.0, 0.0, 0.0]]]).expand(10, -1, -1).clone()
    ca, _, _ = group_relative_conservative_advantage(q_values)
    metrics = actor_signal_metrics(q_values, chi2_config=CHIPO)
    assert ca[0, 0] > 0
    assert metrics["ca_positive_ratio"] == 0.25
    assert metrics["ca_nonzero_ratio"] == 0.5


def test_nine_positive_one_negative_member_zeros_ca_candidate():
    positive = torch.tensor([1.0, -1.0, 0.0, 0.0])
    negative = torch.tensor([-1.0, 1.0, 0.0, 0.0])
    q_values = torch.stack([positive] * 9 + [negative]).unsqueeze(1)
    ca, _, _ = group_relative_conservative_advantage(q_values)
    assert ca[0, 0] == 0


def test_fixed_128_candidate_cache_selection_is_reproducible(tmp_path):
    first = fixed_validation_indices(256, 128, 27)
    second = fixed_validation_indices(256, 128, 27)
    assert torch.equal(first, second)
    payload = {
        "schema_version": 2,
        "split": "validation",
        "seed": 27,
        "group_size": 4,
        "candidate_state_count": 128,
        "validation_indices": first,
        "episode_ids": torch.arange(128),
        "timesteps": torch.arange(128),
        "batch": {},
        "candidate_actions": torch.zeros(128, 4, 50, 14),
        "base_actor_checkpoint": "model.safetensors",
        "base_actor_identity": {"size": 1},
        "validation_replay_identity": {"size": 2},
    }
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    restored = torch.load(path, weights_only=False)
    validate_candidate_cache(restored, expected_count=128, expected_seed=27)
    assert torch.equal(restored["validation_indices"], first)
    assert torch.equal(restored["candidate_actions"], payload["candidate_actions"])


def test_candidate_cache_count_identity_mismatch_is_rejected():
    payload = {
        "schema_version": 2,
        "split": "validation",
        "seed": 27,
        "group_size": 4,
        "candidate_state_count": 32,
        "validation_indices": torch.arange(32),
        "episode_ids": torch.arange(32),
        "timesteps": torch.arange(32),
        "batch": {},
        "candidate_actions": torch.zeros(32, 4, 2, 2),
        "base_actor_checkpoint": "model.safetensors",
        "base_actor_identity": {"size": 1},
        "validation_replay_identity": {"size": 2},
    }
    try:
        validate_candidate_cache(payload, expected_count=128, expected_seed=27)
    except ValueError as error:
        assert "count mismatch" in str(error)
    else:
        raise AssertionError("a 32-state cache must not satisfy a 128-state request")


def test_chipo_ratio_one_proxy_calls_production_composition():
    q_values = torch.tensor([[[1.0, -1.0, 0.5, -0.5]]]).expand(10, -1, -1).clone()
    ca, _, _ = group_relative_conservative_advantage(q_values)
    expected, _ = apply_chipo_to_ca_advantage(
        ca,
        q_values,
        torch.ones_like(ca),
        beta_base=0.1,
        q_std_target=1.0,
        ensemble_alpha=5.0,
        r_max=10.0,
    )
    metrics = actor_signal_metrics(q_values, chi2_config=CHIPO)
    assert metrics["chipo_ratio_mean"] == 1.0
    assert metrics["final_nonzero_ratio"] == float((expected != 0).float().mean())
    assert metrics["final_adv_std"] == float(expected.std(unbiased=False))
    assert metrics["ca_adv_std"] == float(ca.std(unbiased=False))
    assert metrics["chipo_adv_scale_ratio"] == pytest.approx(
        float(expected.abs().mean() / (ca.abs().mean() + 1e-12))
    )


def test_disagreement_error_spearman_detects_meaningful_uncertainty():
    target = torch.zeros(8)
    scale = torch.arange(8, dtype=torch.float32)
    coefficients = torch.linspace(-0.5, 0.5, 10).unsqueeze(1)
    raw_q = scale.unsqueeze(0) + coefficients * scale.unsqueeze(0)
    metrics = raw10_fidelity_metrics(raw_q, target)
    assert metrics["error_disagreement_spearman"] > 0.99
    assert metrics["raw10_ensemble_std_mean"] > 0.0


def test_off_diagonal_q_correlation_reports_mean_min_and_max():
    base = torch.arange(8, dtype=torch.float32)
    raw_q = torch.stack([base, base, -base] + [base * (index + 1) for index in range(7)])
    metrics = off_diagonal_q_correlation_metrics(raw_q)
    assert metrics["max_off_diagonal_q_correlation"] == pytest.approx(1.0)
    assert metrics["min_off_diagonal_q_correlation"] == pytest.approx(-1.0)
    assert -1.0 <= metrics["mean_off_diagonal_q_correlation"] <= 1.0


def test_focused_return_metrics_are_episodewise_and_terminal_aware():
    target = torch.tensor([0.0, 0.5, 0.9, 1.0, 0.0, 0.6, 0.85, 1.0])
    mean_q = torch.tensor([0.0, 0.4, 0.7, 0.8, 0.0, 0.4, 0.6, 0.7])
    raw_q = mean_q.unsqueeze(0).expand(10, -1).clone()
    episode_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    successes = torch.ones(8)
    dones = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1])
    metrics = focused_return_metrics(
        raw_q, target, episode_ids, successes, dones
    )
    assert metrics["success_episode_mc_spearman_count"] == 2
    assert metrics["success_episode_mc_spearman_median"] == pytest.approx(1.0)
    assert metrics["success_episode_mc_spearman_p10"] == pytest.approx(1.0)
    assert metrics["high_return_count"] == 4
    assert metrics["high_return_mae"] == pytest.approx(0.2375)
    assert metrics["high_return_bias"] == pytest.approx(-0.2375)
    assert metrics["terminal_success_count"] == 2
    assert metrics["terminal_success_abs_q_minus_1_mean"] == pytest.approx(0.25)


def _ranking_row(**updates):
    row = {
        "step": 1000,
        "mild_raw10_action_rank_unanimous": 0.5,
        "raw10_action_rank_unanimous": 0.5,
        "raw10_mc_spearman": 0.5,
        "raw10_pairwise_rank_acc": 0.7,
        "success_episode_mc_spearman_median": 0.5,
        "success_episode_mc_spearman_p10": 0.3,
        "raw10_member_rank_min": 0.2,
        "raw10_member_action_rank_min": 0.6,
        "error_disagreement_spearman": 0.3,
        "ca_nonzero_ratio": 0.5,
        "ca_adv_std": 0.1,
        "final_adv_std": 0.1,
        "final_nonzero_ratio": 0.5,
        "final_zero_ratio": 0.5,
        "raw10_rmse": 0.2,
        "raw10_abs_bias": 0.1,
        "high_return_mae": 0.2,
        "high_return_bias": -0.1,
        "terminal_success_abs_q_minus_1_mean": 0.2,
        "raw10_ensemble_std_mean": 0.05,
        "mean_off_diagonal_q_correlation": 0.9,
    }
    row.update(updates)
    return row


def test_composite_does_not_double_count_final_nonzero_ratio():
    first = _ranking_row(step=1000, final_nonzero_ratio=0.1)
    second = _ranking_row(step=2000, final_nonzero_ratio=0.9)
    ranked = rank_checkpoints([first, second])
    scores = {row["step"]: row["composite_score"] for row in ranked}
    assert scores[1000] == scores[2000]


def test_ca_nonzero_is_diagnostic_only_not_monotonic_score():
    first = _ranking_row(step=1000, ca_nonzero_ratio=0.1)
    second = _ranking_row(step=2000, ca_nonzero_ratio=0.9)
    ranked = rank_checkpoints([first, second])
    scores = {row["step"]: row["composite_score"] for row in ranked}
    assert scores[1000] == scores[2000]
    assert all(row["ca_nonzero_diagnostic_only"] for row in ranked)


def test_selection_sweep_requires_validation_and_rejects_heldout():
    validate_selection_split("validation")
    for split, heldout in (("heldout", None), ("validation", "heldout.pt")):
        try:
            validate_selection_split(split, heldout)
        except ValueError:
            pass
        else:
            raise AssertionError("heldout must not participate in checkpoint selection")
