from visualize_critic_success10_with_frames import (
    key_frame_local_indices,
    representative_indices,
)


def test_representative_indices_are_deterministic_unique_and_cover_endpoints():
    indices = representative_indices(12, 10)
    assert len(indices) == len(set(indices)) == 10
    assert indices[0] == 0
    assert indices[-1] == 11
    assert indices == representative_indices(12, 10)


def test_key_frames_cover_episode_progress():
    assert key_frame_local_indices(101) == [0, 25, 50, 75, 100]
    assert key_frame_local_indices(5) == [0, 1, 2, 3, 4]
