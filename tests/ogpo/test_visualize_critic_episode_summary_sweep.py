from __future__ import annotations

import torch

from ogpo.replay import make_synthetic_replay
from visualize_critic_episode_curves import episode_catalog
from visualize_critic_episode_summary_sweep import (
    _parse_steps,
    choose_new_episode,
)


def test_selector_excludes_prior_episode_and_preserves_outcome():
    batch = make_synthetic_replay(num_samples=20)
    successes = torch.zeros_like(batch.successes)
    successes[batch.episode_ids == 1] = 1.0
    successes[batch.episode_ids == 2] = 1.0
    batch = type(batch)(**{**batch.__dict__, "successes": successes})
    episodes = episode_catalog(batch)
    success = choose_new_episode(episodes, success=True, excluded_id=1)
    failure = choose_new_episode(episodes, success=False, excluded_id=0)
    assert success.episode_id == 2
    assert success.success is True
    assert failure.episode_id != 0
    assert failure.success is False


def test_default_sweep_steps_are_all_milestones():
    assert _parse_steps("1000,2000,3000,4000,5000,6000,7000,8000") == list(
        range(1000, 8001, 1000)
    )
