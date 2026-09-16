from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import json
import torch
from torch.utils.data import Dataset

from .chunk_transition import compute_chunk_return, compute_transition_discount, make_execution_mask
from .types import ChunkBatch


class OfflineChunkReplay(Dataset):
    """Fixed offline replay dataset for chunk transitions."""

    def __init__(self, batch: ChunkBatch):
        self.batch = batch

    def __len__(self) -> int:
        return self.batch.batch_size

    def __getitem__(self, index: int) -> ChunkBatch:
        return self.batch.index_select(torch.tensor([index], dtype=torch.long))

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ChunkBatch:
        indices = torch.randint(self.batch.batch_size, (batch_size,), generator=generator)
        sample = self.batch.index_select(indices)
        return sample.to(device) if device is not None else sample


class TaskBalancedCriticReplay:
    """Uniformly choose a task, then uniformly choose a transition in it."""

    def __init__(self, batch: ChunkBatch, *, task_names: list[str] | tuple[str, ...] | None = None):
        self.batch = batch
        available = sorted(set(str(task) for task in batch.task_ids))
        expected = available if task_names is None else [str(task) for task in task_names]
        if len(expected) == 0 or len(set(expected)) != len(expected):
            raise ValueError("task-balanced replay requires unique non-empty task names")
        missing = sorted(set(expected) - set(available))
        unexpected = sorted(set(available) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"task-balanced replay task mismatch: missing={missing} unexpected={unexpected}"
            )
        self.task_names = tuple(expected)
        self._indices = {
            task: torch.tensor(
                [index for index, value in enumerate(batch.task_ids) if str(value) == task],
                dtype=torch.long,
            )
            for task in self.task_names
        }
        if any(indices.numel() == 0 for indices in self._indices.values()):
            raise ValueError("task-balanced replay contains an empty task pool")

    def __len__(self) -> int:
        return self.batch.batch_size

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ChunkBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_tasks = torch.randint(len(self.task_names), (batch_size,), generator=generator)
        selected = []
        for task_index in selected_tasks.tolist():
            pool = self._indices[self.task_names[task_index]]
            offset = int(torch.randint(pool.numel(), (1,), generator=generator).item())
            selected.append(pool[offset])
        indices = torch.stack(selected)
        sample = self.batch.index_select(indices)
        return sample.to(device) if device is not None else sample


class OutcomeBalancedCriticReplay:
    """Bernoulli-select success/failure, then sample a transition uniformly."""

    def __init__(self, batch: ChunkBatch, *, success_probability: float = 0.5):
        self.batch = batch
        self.success_probability = float(success_probability)
        if not 0.0 < self.success_probability < 1.0:
            raise ValueError("outcome-balanced success_probability must be in (0,1)")
        success = batch.successes.bool().cpu()
        self._success = torch.nonzero(success, as_tuple=False).flatten()
        self._failure = torch.nonzero(~success, as_tuple=False).flatten()
        if self._success.numel() == 0 or self._failure.numel() == 0:
            raise ValueError("outcome-balanced replay requires non-empty success and failure pools")

    def __len__(self) -> int:
        return self.batch.batch_size

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ChunkBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        choose_success = torch.rand(batch_size, generator=generator) < self.success_probability
        selected = torch.empty(batch_size, dtype=torch.long)
        for outcome, pool in ((True, self._success), (False, self._failure)):
            positions = torch.nonzero(choose_success == outcome, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            offsets = torch.randint(pool.numel(), (positions.numel(),), generator=generator)
            selected[positions] = pool.index_select(0, offsets)
        sample = self.batch.index_select(selected)
        return sample.to(device) if device is not None else sample


class TaskOutcomeBalancedCriticReplay:
    """Uniformly choose a task, then an outcome, then a transition in that pool."""

    def __init__(
        self,
        batch: ChunkBatch,
        *,
        task_names: list[str] | tuple[str, ...] | None = None,
        success_probability: float = 0.5,
    ):
        self.batch = batch
        self.success_probability = float(success_probability)
        if not 0.0 < self.success_probability < 1.0:
            raise ValueError("task-outcome-balanced success_probability must be in (0,1)")
        available = sorted(set(str(task) for task in batch.task_ids))
        expected = available if task_names is None else [str(task) for task in task_names]
        if len(expected) == 0 or len(set(expected)) != len(expected):
            raise ValueError("task-outcome-balanced replay requires unique non-empty task names")
        missing = sorted(set(expected) - set(available))
        unexpected = sorted(set(available) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"task-outcome-balanced replay task mismatch: missing={missing} unexpected={unexpected}"
            )
        self.task_names = tuple(expected)
        successes = batch.successes.bool().cpu()
        self._indices: dict[str, dict[bool, torch.Tensor]] = {}
        for task in self.task_names:
            task_mask = torch.tensor([str(value) == task for value in batch.task_ids])
            pools = {
                outcome: torch.nonzero(task_mask & (successes == outcome), as_tuple=False).flatten()
                for outcome in (True, False)
            }
            for outcome, pool in pools.items():
                if pool.numel() == 0:
                    name = "success" if outcome else "failure"
                    raise ValueError(f"task-outcome-balanced replay has empty {name} pool for {task!r}")
            self._indices[task] = pools

    def __len__(self) -> int:
        return self.batch.batch_size

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ChunkBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_tasks = torch.randint(len(self.task_names), (batch_size,), generator=generator)
        choose_success = torch.rand(batch_size, generator=generator) < self.success_probability
        selected = []
        for task_index, outcome in zip(selected_tasks.tolist(), choose_success.tolist(), strict=True):
            pool = self._indices[self.task_names[task_index]][bool(outcome)]
            offset = int(torch.randint(pool.numel(), (1,), generator=generator).item())
            selected.append(pool[offset])
        sample = self.batch.index_select(torch.stack(selected))
        return sample.to(device) if device is not None else sample


class BalancedCriticReplay:
    """Episode-balanced replay with explicit success and reward-anchor strata."""

    def __init__(
        self,
        batch: ChunkBatch,
        *,
        uniform_fraction: float = 0.5,
        success_fraction: float = 0.25,
        terminal_success_fraction: float = 0.125,
        failure_fraction: float = 0.125,
    ):
        fractions = {
            "uniform": float(uniform_fraction),
            "success": float(success_fraction),
            "terminal_success": float(terminal_success_fraction),
            "failure": float(failure_fraction),
        }
        if any(value < 0.0 for value in fractions.values()):
            raise ValueError("balanced critic replay fractions must be non-negative")
        if abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise ValueError("balanced critic replay fractions must sum to 1")
        self.batch = batch
        self.fractions = fractions
        self._all = self._group_indices(torch.ones(batch.batch_size, dtype=torch.bool))
        success = batch.successes.bool().cpu()
        self._success = self._group_indices(success)
        self._failure = self._group_indices(~success)
        terminal_success = success & batch.dones.bool().cpu()
        self._terminal_success = self._group_indices(terminal_success)

    def _group_indices(self, mask: torch.Tensor) -> list[torch.Tensor]:
        groups = []
        episode_ids = self.batch.episode_ids.cpu()
        for episode_id in torch.unique(episode_ids[mask]):
            indices = torch.nonzero(mask & (episode_ids == episode_id), as_tuple=False).flatten()
            if indices.numel():
                groups.append(indices)
        return groups

    @staticmethod
    def _counts(batch_size: int, fractions: dict[str, float]) -> dict[str, int]:
        raw = {name: batch_size * value for name, value in fractions.items()}
        counts = {name: int(value) for name, value in raw.items()}
        remaining = batch_size - sum(counts.values())
        order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
        for name in order[:remaining]:
            counts[name] += 1
        return counts

    def _sample_groups(
        self,
        groups: list[torch.Tensor],
        count: int,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.long)
        if not groups:
            groups = self._all
        selected_groups = torch.randint(len(groups), (count,), generator=generator)
        selected = []
        for group_index in selected_groups.tolist():
            group = groups[group_index]
            offset = torch.randint(group.numel(), (1,), generator=generator).item()
            selected.append(group[offset])
        return torch.stack(selected)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ChunkBatch:
        counts = self._counts(int(batch_size), self.fractions)
        indices = torch.cat(
            [
                self._sample_groups(self._all, counts["uniform"], generator=generator),
                self._sample_groups(self._success, counts["success"], generator=generator),
                self._sample_groups(
                    self._terminal_success,
                    counts["terminal_success"],
                    generator=generator,
                ),
                self._sample_groups(self._failure, counts["failure"], generator=generator),
            ]
        )
        order = torch.randperm(indices.numel(), generator=generator)
        sample = self.batch.index_select(indices[order])
        return sample.to(device) if device is not None else sample


def save_replay(batch: ChunkBatch, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(batch), path)


def save_replay_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    meta_path = Path(path).with_suffix(Path(path).suffix + ".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_replay(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
) -> ChunkBatch:
    payload: dict[str, Any] = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
        mmap=bool(mmap),
    )
    return ChunkBatch(**payload)


def split_success_buffers(batch: ChunkBatch) -> dict[str, ChunkBatch]:
    success_mask = batch.successes.bool()
    failure_mask = ~success_mask
    success_indices = torch.nonzero(success_mask, as_tuple=False).flatten()
    failure_indices = torch.nonzero(failure_mask, as_tuple=False).flatten()
    buffers = {"all": batch}
    if success_indices.numel() > 0:
        buffers["success"] = batch.index_select(success_indices)
    if failure_indices.numel() > 0:
        buffers["failure"] = batch.index_select(failure_indices)
    if failure_indices.numel() > 0:
        returns = batch.chunk_returns.index_select(0, failure_indices)
        if bool(returns.max() > returns.min()):
            cutoff = torch.quantile(returns, 0.75)
            near = failure_indices[returns >= cutoff]
            if near.numel() > 0:
                buffers["near_success"] = batch.index_select(near)
    return buffers


def split_replay(
    batch: ChunkBatch,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 0,
) -> dict[str, ChunkBatch]:
    """Create train/validation/held-out splits without episode leakage."""
    if train_ratio <= 0.0 or validation_ratio < 0.0 or train_ratio + validation_ratio >= 1.0:
        raise ValueError("expected train_ratio > 0, validation_ratio >= 0, and train+validation < 1")
    generator = torch.Generator().manual_seed(seed)
    episodes = torch.unique(batch.episode_ids)
    if episodes.numel() >= 3:
        episodes = episodes[torch.randperm(episodes.numel(), generator=generator)]
        train_episode_count = min(max(1, int(episodes.numel() * train_ratio)), episodes.numel() - 2)
        validation_episode_count = min(
            max(1, int(episodes.numel() * validation_ratio)),
            episodes.numel() - train_episode_count - 1,
        )
        train_episodes = episodes[:train_episode_count]
        validation_episodes = episodes[
            train_episode_count : train_episode_count + validation_episode_count
        ]
        heldout_episodes = episodes[train_episode_count + validation_episode_count :]

        def indices_for(selected: torch.Tensor) -> torch.Tensor:
            mask = (batch.episode_ids[:, None] == selected[None, :]).any(dim=1)
            return torch.nonzero(mask, as_tuple=False).flatten()

        return {
            "train": batch.index_select(indices_for(train_episodes)),
            "validation": batch.index_select(indices_for(validation_episodes)),
            "heldout": batch.index_select(indices_for(heldout_episodes)),
        }

    # A one- or two-episode smoke dataset cannot form three disjoint trajectory
    # splits, so retain deterministic transition-level splits for that case.
    perm = torch.randperm(batch.batch_size, generator=generator)
    train_end = max(1, int(batch.batch_size * train_ratio))
    val_end = max(train_end + 1, int(batch.batch_size * (train_ratio + validation_ratio)))
    val_end = min(val_end, batch.batch_size)
    splits = {"train": batch.index_select(perm[:train_end])}
    if val_end > train_end:
        splits["validation"] = batch.index_select(perm[train_end:val_end])
    if batch.batch_size > val_end:
        splits["heldout"] = batch.index_select(perm[val_end:])
    return splits


def make_n_step_replay(batch: ChunkBatch, *, n_step: int) -> ChunkBatch:
    """Fold consecutive chunk transitions into n-step outer-MDP targets."""
    n_step = int(n_step)
    if n_step <= 0:
        raise ValueError("n_step must be positive")
    if n_step == 1:
        return batch

    lookup: dict[tuple[int, int], int] = {}
    for index, (episode_id, timestep) in enumerate(
        zip(batch.episode_ids.tolist(), batch.timesteps.tolist(), strict=True)
    ):
        key = (int(episode_id), int(timestep))
        if key in lookup:
            raise ValueError(f"duplicate replay transition key {key}")
        lookup[key] = index

    returns = []
    discounts = []
    dones = []
    successes = []
    next_indices = []
    metadata = []
    for start in range(batch.batch_size):
        total = batch.chunk_returns.new_tensor(0.0)
        cumulative_discount = batch.discounts.new_tensor(1.0)
        last = start
        used = 0
        index = start
        visited_indices = []
        for offset in range(n_step):
            if offset > 0:
                previous = last
                successor_key = (
                    int(batch.episode_ids[previous].item()),
                    int(
                        batch.timesteps[previous].item()
                        + batch.executed_lengths[previous].item()
                    ),
                )
                successor = lookup.get(successor_key)
                if successor is None:
                    break
                index = successor
            total = total + cumulative_discount * batch.chunk_returns[index]
            cumulative_discount = cumulative_discount * batch.discounts[index]
            last = index
            used += 1
            visited_indices.append(index)
            if bool(batch.dones[index].item()):
                break
        returns.append(total)
        discounts.append(cumulative_discount)
        dones.append(batch.dones[last])
        # The replay may be shuffled, concatenated from multiple episodes, or
        # contain overlapping action-chunk windows.  A contiguous tensor slice
        # between ``start`` and ``last`` is not the n-step trajectory and can
        # include transitions from unrelated episodes.  Aggregate only the
        # indices actually traversed through the (episode_id, timestep) lookup.
        if not visited_indices:
            raise AssertionError("n-step traversal must visit at least one transition")
        visited_tensor = torch.tensor(
            visited_indices,
            dtype=torch.long,
            device=batch.successes.device,
        )
        successes.append(batch.successes.index_select(0, visited_tensor).max())
        next_indices.append(last)
        item_metadata = dict(batch.behavior_metadata[start])
        item_metadata["n_step"] = used
        item_metadata["n_step_visited_indices"] = [int(value) for value in visited_indices]
        item_metadata["n_step_visited_keys"] = [
            {
                "episode_id": int(batch.episode_ids[value].item()),
                "timestep": int(batch.timesteps[value].item()),
            }
            for value in visited_indices
        ]
        metadata.append(item_metadata)

    index_tensor = torch.tensor(next_indices, dtype=torch.long, device=batch.next_observations.device)
    nested = {}
    if batch.next_images is not None:
        nested["next_images"] = {
            key: value.index_select(0, index_tensor.to(value.device))
            for key, value in batch.next_images.items()
        }
    if batch.next_critic_features is not None:
        nested["next_critic_features"] = batch.next_critic_features.index_select(
            0, index_tensor.to(batch.next_critic_features.device)
        )
    return replace(
        batch,
        chunk_returns=torch.stack(returns),
        discounts=torch.stack(discounts),
        next_observations=batch.next_observations.index_select(0, index_tensor),
        next_proprioceptions=batch.next_proprioceptions.index_select(
            0, index_tensor.to(batch.next_proprioceptions.device)
        ),
        dones=torch.stack(dones),
        successes=torch.stack(successes),
        behavior_metadata=metadata,
        **nested,
    )


def add_monte_carlo_returns(batch: ChunkBatch) -> ChunkBatch:
    """Compute discounted return-to-go over contiguous outer-MDP chunks."""
    mc_returns = torch.empty_like(batch.chunk_returns)
    keys = [
        (int(episode_id), int(timestep))
        for episode_id, timestep in zip(
            batch.episode_ids.tolist(), batch.timesteps.tolist(), strict=True
        )
    ]
    lookup: dict[tuple[int, int], int] = {}
    for index, key in enumerate(keys):
        if key in lookup:
            raise ValueError(f"duplicate replay transition key {key}")
        lookup[key] = index
    order = sorted(
        range(batch.batch_size),
        key=lambda index: keys[index],
        reverse=True,
    )
    for index in order:
        successor_key = (
            keys[index][0],
            keys[index][1] + int(batch.executed_lengths[index].item()),
        )
        next_index = None if bool(batch.dones[index].item()) else lookup.get(successor_key)
        if next_index is None:
            mc_returns[index] = batch.chunk_returns[index]
        else:
            mc_returns[index] = (
                batch.chunk_returns[index]
                + batch.discounts[index] * mc_returns[next_index]
            )
    return replace(batch, mc_returns=mc_returns)


def rebase_sparse_binary_replay_gamma(
    batch: ChunkBatch,
    *,
    old_gamma: float,
    new_gamma: float,
    atol: float = 2e-5,
) -> ChunkBatch:
    """Recompute discounts/returns when sparse chunks contain at most one unit reward.

    The source per-step rewards are not stored in ChunkBatch. This helper therefore
    validates the current chunk return against ``old_gamma ** reward_offset`` and
    refuses data that cannot be represented as zero or one binary reward.
    """
    if not 0.0 < old_gamma <= 1.0 or not 0.0 < new_gamma <= 1.0:
        raise ValueError("old_gamma and new_gamma must lie in (0, 1]")
    lengths = batch.executed_lengths.long()
    expected_old_discounts = torch.pow(
        batch.discounts.new_tensor(float(old_gamma)),
        lengths.to(batch.discounts.dtype),
    )
    if not torch.allclose(
        batch.discounts,
        expected_old_discounts,
        atol=float(atol),
        rtol=0.0,
    ):
        error = (batch.discounts - expected_old_discounts).abs().max().item()
        raise ValueError(
            "replay discounts are inconsistent with configured old_gamma; "
            f"max_abs_error={error:.3e}"
        )
    if bool((batch.chunk_returns < -float(atol)).any()):
        raise ValueError("sparse binary gamma rebasing does not support negative rewards")

    new_returns = torch.zeros_like(batch.chunk_returns)
    positive = torch.nonzero(batch.chunk_returns > float(atol), as_tuple=False).flatten()
    for index in positive.tolist():
        length = int(lengths[index].item())
        offsets = torch.arange(
            length,
            device=batch.chunk_returns.device,
            dtype=batch.chunk_returns.dtype,
        )
        candidates = torch.pow(
            batch.chunk_returns.new_tensor(float(old_gamma)),
            offsets,
        )
        errors = (candidates - batch.chunk_returns[index]).abs()
        offset = int(errors.argmin().item())
        if float(errors[offset].item()) > float(atol):
            raise ValueError(
                "chunk return is not a single sparse binary reward under old_gamma: "
                f"index={index} return={float(batch.chunk_returns[index]):.8f} "
                f"executed_length={length} min_error={float(errors[offset]):.3e}"
            )
        new_returns[index] = float(new_gamma) ** offset

    new_discounts = torch.pow(
        batch.discounts.new_tensor(float(new_gamma)),
        lengths.to(batch.discounts.dtype),
    )
    rebased = replace(
        batch,
        chunk_returns=new_returns,
        discounts=new_discounts,
        mc_returns=None,
    )
    return add_monte_carlo_returns(rebased)


def rebase_replay_gamma_from_config(
    batch: ChunkBatch,
    data_config: dict[str, Any],
) -> ChunkBatch:
    """Apply the configured opt-in replay gamma transform."""
    gamma_rebase = data_config.get("gamma_rebase", {})
    if bool(gamma_rebase.get("enabled", False)):
        batch = rebase_sparse_binary_replay_gamma(
            batch,
            old_gamma=float(gamma_rebase["source_gamma"]),
            new_gamma=float(data_config["gamma"]),
            atol=float(gamma_rebase.get("atol", 2e-5)),
        )
    return batch


def prepare_replay_from_config(batch: ChunkBatch, data_config: dict[str, Any]) -> ChunkBatch:
    """Apply opt-in reward/discount transforms followed by n-step folding."""
    batch = rebase_replay_gamma_from_config(batch, data_config)
    n_step = int(data_config.get("n_step", 1))
    if bool(data_config.get("apply_n_step_on_load", False)) and n_step > 1:
        batch = make_n_step_replay(batch, n_step=n_step)
    return batch


def make_synthetic_replay(
    *,
    num_samples: int = 64,
    obs_dim: int = 12,
    proprio_dim: int = 4,
    generated_horizon: int = 6,
    action_dim: int = 3,
    executed_horizon: int = 3,
    gamma: float = 0.97,
    seed: int = 7,
) -> ChunkBatch:
    """Create deterministic toy replay for unit and smoke tests."""
    if executed_horizon <= 0 or executed_horizon > generated_horizon:
        raise ValueError("executed_horizon must be in [1, generated_horizon]")

    gen = torch.Generator().manual_seed(seed)
    observations = torch.randn(num_samples, obs_dim, generator=gen)
    proprioceptions = torch.randn(num_samples, proprio_dim, generator=gen)
    action_chunks = torch.randn(num_samples, generated_horizon, action_dim, generator=gen)
    execution_masks = make_execution_mask(
        torch.full((num_samples,), executed_horizon), generated_horizon
    )
    executed_lengths = torch.full((num_samples,), executed_horizon, dtype=torch.long)

    prefix = action_chunks[:, :executed_horizon]
    target_direction = torch.tanh(observations[:, :action_dim]).unsqueeze(1)
    per_step_rewards = 1.0 - (prefix - target_direction).pow(2).mean(dim=2)
    chunk_returns = compute_chunk_return(per_step_rewards, gamma, executed_lengths)
    discounts = compute_transition_discount(gamma, executed_lengths)
    next_observations = observations + 0.05 * torch.randn(num_samples, obs_dim, generator=gen)
    next_proprioceptions = proprioceptions + 0.05 * torch.randn(
        num_samples, proprio_dim, generator=gen
    )
    dones = torch.zeros(num_samples)
    successes = (chunk_returns > torch.median(chunk_returns)).float()
    episode_ids = torch.arange(num_samples, dtype=torch.long) // 4
    timesteps = torch.arange(num_samples, dtype=torch.long) % 4
    task_ids = ["synthetic_task"] * num_samples
    languages = ["move toward the synthetic target"] * num_samples
    behavior_metadata = [{"policy": "synthetic_behavior", "seed": seed} for _ in range(num_samples)]

    return ChunkBatch(
        observations=observations,
        proprioceptions=proprioceptions,
        action_chunks=action_chunks,
        execution_masks=execution_masks,
        executed_lengths=executed_lengths,
        chunk_returns=chunk_returns,
        discounts=discounts,
        next_observations=next_observations,
        next_proprioceptions=next_proprioceptions,
        dones=dones,
        successes=successes,
        episode_ids=episode_ids,
        timesteps=timesteps,
        task_ids=task_ids,
        languages=languages,
        behavior_metadata=behavior_metadata,
    )
