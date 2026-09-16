import numpy as np
import torch

from ogpo.replay import load_replay, save_replay, save_replay_metadata, split_replay
from ogpo.zarr_replay import (
    ZarrConversionConfig,
    concat_chunk_batches,
    convert_replay_paths,
    find_replay_paths,
)


def test_npz_episode_conversion_to_chunk_batch(tmp_path):
    path = tmp_path / "episode_success.npz"
    states = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    actions = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    rewards = np.ones(6, dtype=np.float32)
    done = np.zeros(6, dtype=np.bool_)
    done[-1] = True
    success = np.zeros(6, dtype=np.bool_)
    success[-1] = True
    np.savez(path, state=states, action_rotvec=actions, reward=rewards, done=done, success=success)

    paths = find_replay_paths(tmp_path)
    batch = convert_replay_paths(
        paths,
        ZarrConversionConfig(
            generated_horizon=4,
            executed_horizon=2,
            gamma=0.5,
            task_id="fixture_task",
            language="fixture language",
            behavior_policy="fixture_policy",
        ),
    )
    assert batch.batch_size == 3
    assert batch.action_chunks.shape == (3, 4, 3)
    assert batch.execution_masks[:, :2].all()
    assert (~batch.execution_masks[:, 2:]).all()
    assert torch.allclose(batch.chunk_returns[:2], torch.tensor([1.5, 1.5]))
    assert batch.successes.bool().all()
    assert batch.task_ids == ["fixture_task"] * 3
    assert batch.behavior_metadata[0]["reward_fallback"] is False


def test_split_and_metadata_write(tmp_path):
    path = tmp_path / "episode_success.npz"
    np.savez(
        path,
        state=np.random.randn(8, 3).astype("float32"),
        action=np.random.randn(8, 2).astype("float32"),
        reward=np.zeros(8, dtype="float32"),
        done=np.r_[np.zeros(7, dtype=bool), True],
        success=np.r_[np.zeros(7, dtype=bool), True],
    )
    batch = convert_replay_paths(
        [path],
        ZarrConversionConfig(generated_horizon=3, executed_horizon=1, gamma=0.9),
    )
    splits = split_replay(batch, train_ratio=0.6, validation_ratio=0.2, seed=1)
    assert "train" in splits
    output = tmp_path / "replay.pt"
    save_replay(batch, output)
    save_replay_metadata(output, {"samples": batch.batch_size})
    assert output.exists()
    assert output.with_suffix(".pt.json").exists()


def test_conversion_keeps_terminal_partial_chunk_and_reward(tmp_path):
    path = tmp_path / "episode_terminal_partial.npz"
    np.savez(
        path,
        state=np.zeros((5, 3), dtype=np.float32),
        action=np.zeros((5, 2), dtype=np.float32),
        reward=np.asarray([0, 0, 0, 0, 1], dtype=np.float32),
        done=np.asarray([0, 0, 0, 0, 1], dtype=bool),
        success=np.asarray([0, 0, 0, 0, 1], dtype=bool),
    )

    batch = convert_replay_paths(
        [path],
        ZarrConversionConfig(
            generated_horizon=3,
            executed_horizon=2,
            stride=2,
            gamma=0.9,
        ),
    )

    assert batch.batch_size == 3
    assert batch.executed_lengths.tolist() == [2, 2, 1]
    assert torch.allclose(batch.chunk_returns, torch.tensor([0.0, 0.0, 1.0]))
    assert batch.dones.tolist() == [0.0, 0.0, 1.0]


def test_rgb_observations_survive_conversion_batch_ops_and_roundtrip(tmp_path):
    path = tmp_path / "episode_rgb.npz"
    steps = 5
    base_rgb = np.arange(steps * 4 * 6 * 3, dtype=np.uint8).reshape(steps, 4, 6, 3)
    wrist_rgb = np.flip(base_rgb, axis=2).copy()
    np.savez(
        path,
        state=np.random.randn(steps, 3).astype("float32"),
        action=np.random.randn(steps, 2).astype("float32"),
        base_rgb=base_rgb,
        wrist_rgb=wrist_rgb,
    )
    cfg = ZarrConversionConfig(
        generated_horizon=3,
        executed_horizon=2,
        gamma=0.9,
        image_keys=("base_rgb", "wrist_rgb"),
    )

    batch = convert_replay_paths([path], cfg)

    assert batch.images is not None and batch.next_images is not None
    assert set(batch.images) == {"base_rgb", "wrist_rgb"}
    assert batch.images["base_rgb"].dtype == torch.uint8
    assert torch.equal(batch.images["base_rgb"][0], torch.from_numpy(base_rgb[0]))
    assert torch.equal(batch.next_images["base_rgb"][0], torch.from_numpy(base_rgb[2]))

    selected = batch.index_select(torch.tensor([1, 0])).to("cpu")
    combined = concat_chunk_batches([selected, selected])
    assert combined.images is not None
    assert combined.images["wrist_rgb"].shape[0] == 4

    output = tmp_path / "rgb_replay.pt"
    save_replay(combined, output)
    restored = load_replay(output)
    assert restored.next_images is not None
    assert torch.equal(restored.next_images["base_rgb"], combined.next_images["base_rgb"])
