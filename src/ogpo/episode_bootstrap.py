"""Persistent member-level episode bootstrap with balanced member sampling."""
from dataclasses import replace

import torch

from .replay import TaskOutcomeBalancedCriticReplay


class MemberEpisodeBootstrapReplay(TaskOutcomeBalancedCriticReplay):
    """Each member receives batch_size examples from its own fixed inclusion pool.

    Replay tensors are not copied: only transition indices are filtered. Both Q
    heads and V of the owning member use the same observations and loss mask.
    """

    def __init__(self, batch, *, masks, task_names, success_probability=0.5):
        super().__init__(batch, task_names=task_names,
                         success_probability=success_probability)
        self.member_indices = []
        ids = batch.episode_ids.cpu().tolist()
        self.coverage = {}
        for member in range(5):
            member_pools = {}
            self.coverage[str(member)] = {}
            for task in self.task_names:
                allowed = set(masks["members"][str(member)][task])
                member_pools[task] = {}
                for outcome, pool in self._indices[task].items():
                    filtered = torch.tensor([i for i in pool.tolist() if ids[i] in allowed], dtype=torch.long)
                    if not filtered.numel():
                        raise ValueError(f"empty bootstrap pool member={member} task={task} outcome={outcome}")
                    member_pools[task][outcome] = filtered
                self.coverage[str(member)][task] = len({ids[i] for pool in member_pools[task].values() for i in pool.tolist()})
            self.member_indices.append(member_pools)

    def sample(self, batch_size, *, generator=None, device=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected, owners = [], []
        for member, pools in enumerate(self.member_indices):
            tasks = torch.randint(len(self.task_names), (batch_size,), generator=generator)
            outcomes = torch.rand(batch_size, generator=generator) < self.success_probability
            for task, outcome in zip(tasks.tolist(), outcomes.tolist(), strict=True):
                pool = pools[self.task_names[task]][outcome]
                selected.append(int(pool[torch.randint(len(pool), (1,), generator=generator)].item()))
                owners.append(member)
        sample = self.batch.index_select(torch.tensor(selected))
        sample = replace(sample, behavior_metadata=[dict(meta, bootstrap_owner=owner)
                         for meta, owner in zip(sample.behavior_metadata, owners, strict=True)])
        return sample.to(device) if device is not None else sample


def member_owner_loss_mask(batch, members, *, device):
    owners = torch.tensor([meta["bootstrap_owner"] for meta in batch.behavior_metadata], device=device)
    if ((owners < 0) | (owners >= members)).any():
        raise ValueError("invalid bootstrap owner")
    return torch.arange(members, device=device)[:, None] == owners[None, :]


class SharedEpisodeBootstrapReplay(TaskOutcomeBalancedCriticReplay):
    """One shared balanced batch; persistent masks gate each member's Q pair/V."""
    def __init__(self, batch, *, masks, task_names, success_probability=0.5):
        super().__init__(batch, task_names=task_names, success_probability=success_probability)
        self.included = [{task: set(ids) for task, ids in masks['members'][str(m)].items()} for m in range(5)]

    def sample(self, batch_size, *, generator=None, device=None):
        sample = super().sample(batch_size, generator=generator)
        sample = replace(sample, behavior_metadata=[dict(meta, bootstrap_inclusion=[
            int(eid) in member[str(task)] for member in self.included])
            for meta,task,eid in zip(sample.behavior_metadata,sample.task_ids,sample.episode_ids.tolist(),strict=True)])
        return sample.to(device) if device is not None else sample


def normalize_shared_member_weights(batch, *, device):
    """Normalize once over the full global batch, NOT once per microbatch.

    Accumulation averages over local batch and DDP averages ranks. We compensate
    both so each head pair/V receives sum(valid loss)/global_member_count/M.
    A member with zero global samples contributes zero, without NaN.
    """
    mask = torch.tensor([m['bootstrap_inclusion'] for m in batch.behavior_metadata], device=device, dtype=torch.float32)
    counts = mask.sum(0)
    world = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(counts)
    weights = mask * (batch.batch_size * world / mask.shape[1]) / counts.clamp_min(1)[None,:]
    return replace(batch, behavior_metadata=[dict(meta, bootstrap_loss_weights=w)
        for meta,w in zip(batch.behavior_metadata,weights.cpu().tolist(),strict=True)])
