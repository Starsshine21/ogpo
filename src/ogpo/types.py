from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass(frozen=True)
class ChunkTransition:
    """One outer-MDP action-chunk transition.

    The critic target is defined on the executed prefix:
    `(s_t, A_{t,0:m}, R_t^(m), s_{t+m}, done)`.
    """

    observation: torch.Tensor
    language: str
    proprioception: torch.Tensor
    action_chunk: torch.Tensor
    execution_mask: torch.Tensor
    executed_length: int
    chunk_return: torch.Tensor
    discount: torch.Tensor
    next_observation: torch.Tensor
    next_proprioception: torch.Tensor
    done: torch.Tensor
    success: torch.Tensor
    episode_id: int
    timestep: int
    task_id: str
    behavior_metadata: dict[str, Any] | None = None

    def to_batch(self) -> "ChunkBatch":
        return ChunkBatch(
            observations=self.observation.unsqueeze(0),
            proprioceptions=self.proprioception.unsqueeze(0),
            action_chunks=self.action_chunk.unsqueeze(0),
            execution_masks=self.execution_mask.unsqueeze(0),
            executed_lengths=torch.tensor([self.executed_length], dtype=torch.long),
            chunk_returns=self.chunk_return.reshape(1),
            discounts=self.discount.reshape(1),
            next_observations=self.next_observation.unsqueeze(0),
            next_proprioceptions=self.next_proprioception.unsqueeze(0),
            dones=self.done.reshape(1),
            successes=self.success.reshape(1),
            episode_ids=torch.tensor([self.episode_id], dtype=torch.long),
            timesteps=torch.tensor([self.timestep], dtype=torch.long),
            task_ids=[self.task_id],
            languages=[self.language],
            behavior_metadata=[self.behavior_metadata or {}],
        )


@dataclass(frozen=True)
class ChunkBatch:
    """Batched chunk transitions.

    Shapes:
      observations: [B, obs_dim]
      action_chunks: [B, H, D]
      execution_masks: [B, H]
      chunk_returns: [B]
      discounts: [B], already equal to gamma ** executed_length
    """

    observations: torch.Tensor
    proprioceptions: torch.Tensor
    action_chunks: torch.Tensor
    execution_masks: torch.Tensor
    executed_lengths: torch.Tensor
    chunk_returns: torch.Tensor
    discounts: torch.Tensor
    next_observations: torch.Tensor
    next_proprioceptions: torch.Tensor
    dones: torch.Tensor
    successes: torch.Tensor
    episode_ids: torch.Tensor
    timesteps: torch.Tensor
    task_ids: list[str]
    languages: list[str]
    behavior_metadata: list[dict[str, Any]]
    images: dict[str, torch.Tensor] | None = None
    next_images: dict[str, torch.Tensor] | None = None
    mc_returns: torch.Tensor | None = None
    critic_features: torch.Tensor | None = None
    next_critic_features: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert self.observations.ndim == 2
        assert self.action_chunks.ndim == 3
        assert self.execution_masks.shape == self.action_chunks.shape[:2]
        assert self.executed_lengths.shape == self.chunk_returns.shape
        assert self.discounts.shape == self.chunk_returns.shape
        assert self.next_observations.shape == self.observations.shape
        assert self.dones.shape == self.chunk_returns.shape
        assert self.successes.shape == self.chunk_returns.shape
        if self.mc_returns is not None and self.mc_returns.shape != self.chunk_returns.shape:
            raise ValueError("mc_returns must match chunk_returns shape")
        if (self.critic_features is None) != (self.next_critic_features is None):
            raise ValueError(
                "critic_features and next_critic_features must either both be present or absent"
            )
        if self.critic_features is not None:
            if self.critic_features.ndim != 2:
                raise ValueError("critic_features must have shape [batch, feature_dim]")
            if self.critic_features.shape != self.next_critic_features.shape:
                raise ValueError("current and next critic feature shapes must match")
            if self.critic_features.shape[0] != self.batch_size:
                raise ValueError("critic feature batch size does not match replay")
        if (self.images is None) != (self.next_images is None):
            raise ValueError("images and next_images must either both be present or both be absent")
        if self.images is not None and self.next_images is not None:
            if set(self.images) != set(self.next_images):
                raise ValueError("images and next_images must have identical camera keys")
            for key in self.images:
                if self.images[key].shape[0] != self.batch_size:
                    raise ValueError(f"image batch for {key!r} has the wrong leading dimension")
                if self.next_images[key].shape != self.images[key].shape:
                    raise ValueError(f"next image shape for {key!r} does not match current image shape")

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def generated_horizon(self) -> int:
        return int(self.action_chunks.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action_chunks.shape[2])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    def to(self, device: torch.device | str) -> "ChunkBatch":
        tensor_fields = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        nested = {}
        if self.images is not None:
            nested["images"] = {key: value.to(device) for key, value in self.images.items()}
            nested["next_images"] = {key: value.to(device) for key, value in self.next_images.items()}
        return replace(self, **tensor_fields, **nested)

    def index_select(self, indices: torch.Tensor) -> "ChunkBatch":
        idx_list = indices.detach().cpu().tolist()
        tensor_fields = {
            name: value.index_select(0, indices.to(value.device))
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        nested = {}
        if self.images is not None:
            nested["images"] = {
                key: value.index_select(0, indices.to(value.device)) for key, value in self.images.items()
            }
            nested["next_images"] = {
                key: value.index_select(0, indices.to(value.device)) for key, value in self.next_images.items()
            }
        return replace(
            self,
            **tensor_fields,
            **nested,
            task_ids=[self.task_ids[i] for i in idx_list],
            languages=[self.languages[i] for i in idx_list],
            behavior_metadata=[self.behavior_metadata[i] for i in idx_list],
        )
