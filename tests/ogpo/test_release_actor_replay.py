from pathlib import Path
import sys

import torch

from ogpo.replay import CompositeChunkBatch, make_synthetic_replay, save_replay

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from train_full_ogpo import IndexedActorReplay, TaskCycleActorReplay, load_actor_training_batch


def test_prepared_actor_shards_remain_composite_and_success_only(tmp_path):
    first = make_synthetic_replay(num_samples=8, seed=1)
    second = make_synthetic_replay(num_samples=8, seed=2)
    save_replay(first, tmp_path / "first.pt")
    save_replay(second, tmp_path / "second.pt")
    batch = load_actor_training_batch(
        tmp_path,
        {"dataset_paths": ["first.pt", "second.pt"], "dataset_preprocessed": True},
    )
    assert isinstance(batch, CompositeChunkBatch)
    assert batch.batch_size == 16
    task_replay = TaskCycleActorReplay(batch, ["synthetic_task"])
    assert task_replay.sample_task(
        "synthetic_task", 4, generator=torch.Generator().manual_seed(3)
    ).batch_size == 4
    success_indices = torch.nonzero(batch.successes.bool(), as_tuple=False).flatten()
    success_replay = IndexedActorReplay(batch, success_indices)
    assert success_replay.sample(
        4, generator=torch.Generator().manual_seed(4)
    ).successes.bool().all()
