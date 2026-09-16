from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .multimodal_critic import MultiHeadScalarQCritic
from .types import ChunkBatch


def attach_origin_feature_cache(
    batch: ChunkBatch,
    payload: dict[str, torch.Tensor],
) -> ChunkBatch:
    current = payload["critic_features"]
    next_features = payload["next_critic_features"]
    if current.shape[0] != batch.batch_size:
        raise ValueError(
            f"origin cache has {current.shape[0]} rows but replay has {batch.batch_size}"
        )
    return replace(
        batch,
        critic_features=current,
        next_critic_features=next_features,
    )


@torch.no_grad()
def build_origin_feature_cache(
    state: Any,
    batch: ChunkBatch,
    *,
    inference_batch_size: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(state.critic, MultiHeadScalarQCritic):
        raise TypeError("origin feature cache requires MultiHeadScalarQCritic")
    if any(parameter.requires_grad for parameter in state.critic.state_encoder.parameters()):
        raise ValueError("origin feature cache requires a fully frozen state encoder")
    batch_size = max(1, int(inference_batch_size))
    device = next(state.critic.parameters()).device
    was_training = state.critic.training
    state.critic.eval()
    current_parts = []
    next_parts = []
    for start in range(0, batch.batch_size, batch_size):
        stop = min(start + batch_size, batch.batch_size)
        indices = torch.arange(start, stop)
        sample = batch.index_select(indices).to(device)
        current_parts.append(
            state.critic.state_encoder(sample, next_observation=False)
            .detach()
            .to(device="cpu", dtype=torch.bfloat16)
        )
        next_parts.append(
            state.critic.state_encoder(sample, next_observation=True)
            .detach()
            .to(device="cpu", dtype=torch.bfloat16)
        )
    state.critic.train(was_training)
    return {
        "critic_features": torch.cat(current_parts, dim=0),
        "next_critic_features": torch.cat(next_parts, dim=0),
    }


def load_or_build_origin_feature_cache(
    state: Any,
    batch: ChunkBatch,
    path: str | Path,
    *,
    inference_batch_size: int,
) -> ChunkBatch:
    path = Path(path)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[origin-cache] loaded {path}", flush=True)
    else:
        payload = build_origin_feature_cache(
            state,
            batch,
            inference_batch_size=inference_batch_size,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"[origin-cache] saved {path}", flush=True)
    return attach_origin_feature_cache(batch, payload)
