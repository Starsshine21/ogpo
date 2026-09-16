from __future__ import annotations

import numpy as np
import pytest
import torch

from ogpo.replay import make_synthetic_replay
from visualize_critic_episode_curves import (
    aggregate_progress,
    choose_episode,
    episode_catalog,
    interpolate_episode,
    nearest_progress_indices,
)


def _label_episodes(batch):
    successes = torch.zeros_like(batch.successes)
    successes[batch.episode_ids == 1] = 1.0
    return type(batch)(
        **{
            **batch.__dict__,
            "successes": successes,
            "mc_returns": batch.chunk_returns.clone(),
        }
    )


def test_auto_episode_selection_is_median_length_then_episode_id():
    batch = _label_episodes(make_synthetic_replay(num_samples=16))
    episodes = episode_catalog(batch)
    success = choose_episode(episodes, success=True, requested_id=None)
    failure = choose_episode(episodes, success=False, requested_id=None)
    assert success.episode_id == 1
    assert failure.episode_id == 0
    assert success.success is True
    assert failure.success is False


def test_requested_episode_must_match_requested_outcome():
    batch = _label_episodes(make_synthetic_replay(num_samples=16))
    episodes = episode_catalog(batch)
    with pytest.raises(ValueError, match="not labelled success"):
        choose_episode(episodes, success=True, requested_id=0)


def test_episode_catalog_sorts_timesteps():
    batch = _label_episodes(make_synthetic_replay(num_samples=16))
    permutation = torch.tensor([3, 1, 2, 0] + list(range(4, 16)))
    shuffled = batch.index_select(permutation)
    episode = episode_catalog(shuffled)[0]
    assert episode.timesteps.tolist() == [0, 1, 2, 3]


def test_progress_indices_use_nearest_valid_timestep():
    batch = _label_episodes(make_synthetic_replay(num_samples=16))
    episode = episode_catalog(batch)[0]
    selected = nearest_progress_indices(episode)
    assert int(batch.timesteps[selected["t020"]]) == 1
    assert int(batch.timesteps[selected["t050"]]) in {1, 2}
    assert int(batch.timesteps[selected["t095"]]) == 3


def test_interpolation_and_aggregate_use_raw10_mean_and_std():
    batch = _label_episodes(make_synthetic_replay(num_samples=16))
    episodes = episode_catalog(batch)
    base = torch.arange(batch.batch_size, dtype=torch.float32)
    offsets = torch.linspace(-0.5, 0.5, 10).unsqueeze(1)
    raw_q = base.unsqueeze(0) + offsets
    arrays = {
        "progress": np.array([0.0, 0.5, 1.0]),
        "q_mean": np.array([0.0, 1.0, 2.0]),
        "q_std": np.array([1.0, 2.0, 3.0]),
    }
    q, disagreement = interpolate_episode(arrays, np.linspace(0.0, 1.0, 5))
    assert q.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert disagreement.tolist() == [1.0, 1.5, 2.0, 2.5, 3.0]
    aggregate = aggregate_progress(
        episodes,
        batch,
        raw_q,
        max_per_class=10,
        bins=100,
    )
    assert aggregate["progress"].shape == (100,)
    assert aggregate["success"]["q_mean"].shape == (100,)
    assert aggregate["failure"]["disagreement_mean"].shape == (100,)
