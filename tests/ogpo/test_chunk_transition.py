from dataclasses import replace

import pytest
import torch

from ogpo.chunk_transition import (
    assert_suffix_invariant,
    compute_chunk_return,
    compute_transition_discount,
    flatten_masked_action,
    make_execution_mask,
    mask_action_suffix,
)
from ogpo.replay import (
    BalancedCriticReplay,
    OutcomeBalancedCriticReplay,
    TaskBalancedCriticReplay,
    TaskOutcomeBalancedCriticReplay,
    add_monte_carlo_returns,
    make_n_step_replay,
    make_synthetic_replay,
    prepare_replay_from_config,
    rebase_sparse_binary_replay_gamma,
    split_replay,
    split_success_buffers,
)


def test_balanced_critic_replay_includes_success_terminal_and_failure_strata():
    batch = make_synthetic_replay(
        num_samples=12,
        generated_horizon=3,
        executed_horizon=1,
        action_dim=2,
    )
    episode_ids = torch.arange(4).repeat_interleave(3)
    successes = (episode_ids < 2).float()
    dones = torch.zeros(12)
    dones[2::3] = 1.0
    batch = replace(
        batch,
        episode_ids=episode_ids,
        timesteps=torch.arange(3).repeat(4),
        successes=successes,
        dones=dones,
    )
    replay = BalancedCriticReplay(batch)

    sample = replay.sample(8, generator=torch.Generator().manual_seed(5))

    assert sample.batch_size == 8
    assert int(sample.successes.sum()) >= 3
    assert int((~sample.successes.bool()).sum()) >= 1
    assert bool((sample.successes.bool() & sample.dones.bool()).any())


def test_task_balanced_replay_is_independent_of_task_transition_count():
    batch = make_synthetic_replay(num_samples=100, generated_horizon=3, executed_horizon=1)
    task_ids = ["short"] * 10 + ["long"] * 90
    batch = replace(batch, task_ids=task_ids)
    replay = TaskBalancedCriticReplay(batch, task_names=("short", "long"))
    sample = replay.sample(10000, generator=torch.Generator().manual_seed(19))
    short_fraction = sample.task_ids.count("short") / sample.batch_size
    assert short_fraction == pytest.approx(0.5, abs=0.02)


def test_outcome_balanced_replay_ignores_transition_level_imbalance():
    batch = make_synthetic_replay(num_samples=100, generated_horizon=3, executed_horizon=1)
    batch = replace(batch, successes=torch.tensor([1.0] * 10 + [0.0] * 90))
    replay = OutcomeBalancedCriticReplay(batch, success_probability=0.5)
    sample = replay.sample(10000, generator=torch.Generator().manual_seed(23))
    assert float(sample.successes.float().mean()) == pytest.approx(0.5, abs=0.02)


def test_task_outcome_balanced_replay_balances_both_axes():
    batch = make_synthetic_replay(num_samples=200, generated_horizon=3, executed_horizon=1)
    task_ids = ["small"] * 20 + ["large"] * 180
    successes = torch.tensor(
        [1.0] * 2 + [0.0] * 18 + [1.0] * 162 + [0.0] * 18
    )
    batch = replace(batch, task_ids=task_ids, successes=successes)
    replay = TaskOutcomeBalancedCriticReplay(
        batch,
        task_names=("small", "large"),
        success_probability=0.5,
    )
    sample = replay.sample(20000, generator=torch.Generator().manual_seed(29))
    assert sample.task_ids.count("small") / sample.batch_size == pytest.approx(0.5, abs=0.02)
    assert float(sample.successes.float().mean()) == pytest.approx(0.5, abs=0.02)
    for task in ("small", "large"):
        mask = torch.tensor([value == task for value in sample.task_ids])
        assert float(sample.successes[mask].float().mean()) == pytest.approx(0.5, abs=0.03)


def test_execution_mask_and_suffix_invariant():
    action = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    mask = make_execution_mask(2, 5)
    masked = mask_action_suffix(action, mask)
    assert masked[:2].sum() == action[:2].sum()
    assert masked[2:].sum() == 0
    assert_suffix_invariant(action, mask)
    flat_a = flatten_masked_action(action, mask)
    mutated = action.clone()
    mutated[2:] = 999
    flat_b = flatten_masked_action(mutated, mask)
    assert torch.equal(flat_a, flat_b)


def test_chunk_return_and_discount():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    value = compute_chunk_return(rewards, gamma=0.5, executed_length=2)
    assert torch.allclose(value, torch.tensor(2.0))
    discount = compute_transition_discount(0.5, 2)
    assert torch.allclose(discount, torch.tensor(0.25))


def test_success_buffers():
    batch = make_synthetic_replay(num_samples=12)
    buffers = split_success_buffers(batch)
    assert "success" in buffers
    assert "failure" in buffers
    assert buffers["success"].successes.bool().all()
    assert (~buffers["failure"].successes.bool()).all()


def test_near_success_buffer_requires_informative_failure_returns():
    batch = make_synthetic_replay(num_samples=12)
    batch = replace(
        batch,
        successes=torch.zeros(12),
        chunk_returns=torch.zeros(12),
    )

    buffers = split_success_buffers(batch)

    assert "near_success" not in buffers


def test_replay_splits_keep_episodes_disjoint():
    batch = make_synthetic_replay(num_samples=24)

    splits = split_replay(batch, train_ratio=0.6, validation_ratio=0.2, seed=3)
    episode_sets = [set(split.episode_ids.tolist()) for split in splits.values()]

    assert set(splits) == {"train", "validation", "heldout"}
    assert all(left.isdisjoint(right) for i, left in enumerate(episode_sets) for right in episode_sets[i + 1 :])


def test_n_step_replay_accumulates_returns_discount_and_next_state():
    batch = make_synthetic_replay(
        num_samples=4,
        generated_horizon=3,
        executed_horizon=1,
        action_dim=2,
    )
    batch = replace(
        batch,
        episode_ids=torch.zeros(4, dtype=torch.long),
        timesteps=torch.arange(4, dtype=torch.long),
        chunk_returns=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        discounts=torch.full((4,), 0.5),
        dones=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        next_observations=torch.arange(4 * batch.obs_dim, dtype=torch.float32).reshape(4, batch.obs_dim),
    )

    result = make_n_step_replay(batch, n_step=2)

    assert torch.allclose(result.chunk_returns[:3], torch.tensor([2.0, 3.5, 5.0]))
    assert torch.allclose(result.discounts[:3], torch.tensor([0.25, 0.25, 0.25]))
    assert torch.equal(result.next_observations[0], batch.next_observations[1])
    assert result.behavior_metadata[0]["n_step"] == 2


def test_monte_carlo_returns_follow_episode_discounts():
    batch = make_synthetic_replay(num_samples=4, generated_horizon=3, executed_horizon=1, action_dim=2)
    batch = replace(
        batch,
        episode_ids=torch.zeros(4, dtype=torch.long),
        timesteps=torch.arange(4, dtype=torch.long),
        chunk_returns=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        discounts=torch.full((4,), 0.5),
        dones=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    result = add_monte_carlo_returns(batch)

    assert result.mc_returns is not None
    assert torch.allclose(result.mc_returns, torch.tensor([3.25, 4.5, 5.0, 4.0]))


def test_overlapping_dense_windows_follow_timestep_successors():
    gamma = 0.9
    batch = make_synthetic_replay(
        num_samples=6,
        generated_horizon=3,
        executed_horizon=3,
        action_dim=2,
    )
    batch = replace(
        batch,
        episode_ids=torch.zeros(6, dtype=torch.long),
        timesteps=torch.arange(6, dtype=torch.long),
        executed_lengths=torch.tensor([3, 3, 3, 3, 2, 1]),
        chunk_returns=torch.tensor([0.0, 0.0, 0.0, gamma**2, gamma, 1.0]),
        discounts=torch.tensor([gamma**3] * 4 + [gamma**2, gamma]),
        dones=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        next_observations=torch.arange(
            6 * batch.obs_dim, dtype=torch.float32
        ).reshape(6, batch.obs_dim),
    )

    n_step = make_n_step_replay(batch, n_step=2)
    with_mc = add_monte_carlo_returns(batch)

    assert n_step.behavior_metadata[0]["n_step"] == 2
    assert torch.equal(n_step.next_observations[0], batch.next_observations[3])
    assert torch.allclose(n_step.chunk_returns[0], torch.tensor(gamma**5))
    assert with_mc.mc_returns is not None
    assert torch.allclose(
        with_mc.mc_returns,
        torch.tensor([gamma**5, gamma**4, gamma**3, gamma**2, gamma, 1.0]),
    )


def test_sparse_binary_gamma_rebase_updates_reward_discount_and_mc_return():
    old_gamma = 0.9999
    new_gamma = 0.999
    batch = make_synthetic_replay(
        num_samples=3,
        generated_horizon=4,
        executed_horizon=4,
        action_dim=2,
    )
    batch = replace(
        batch,
        episode_ids=torch.zeros(3, dtype=torch.long),
        timesteps=torch.tensor([0, 4, 8]),
        executed_lengths=torch.full((3,), 4, dtype=torch.long),
        chunk_returns=torch.tensor([0.0, 0.0, old_gamma**2]),
        discounts=torch.full((3,), old_gamma**4),
        dones=torch.tensor([0.0, 0.0, 1.0]),
    )

    result = rebase_sparse_binary_replay_gamma(
        batch,
        old_gamma=old_gamma,
        new_gamma=new_gamma,
    )

    assert torch.allclose(result.discounts, torch.full((3,), new_gamma**4))
    assert torch.allclose(
        result.chunk_returns,
        torch.tensor([0.0, 0.0, new_gamma**2]),
    )
    assert result.mc_returns is not None
    assert torch.allclose(
        result.mc_returns,
        torch.tensor([new_gamma**10, new_gamma**6, new_gamma**2]),
        atol=1e-6,
    )


def test_gamma_rebase_is_disabled_by_default():
    batch = make_synthetic_replay(num_samples=4)
    assert prepare_replay_from_config(batch, {"gamma": 0.999}) is batch


def test_sparse_binary_gamma_rebase_rejects_nonbinary_chunk_return():
    batch = make_synthetic_replay(
        num_samples=2,
        generated_horizon=4,
        executed_horizon=4,
        action_dim=2,
    )
    batch = replace(
        batch,
        chunk_returns=torch.tensor([0.0, 0.5]),
        discounts=torch.full((2,), 0.9999**4),
    )

    try:
        rebase_sparse_binary_replay_gamma(
            batch,
            old_gamma=0.9999,
            new_gamma=0.999,
        )
    except ValueError as exc:
        assert "not a single sparse binary reward" in str(exc)
    else:
        raise AssertionError("expected nonbinary chunk return to be rejected")
