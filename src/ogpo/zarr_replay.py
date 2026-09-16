from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .chunk_transition import compute_chunk_return, compute_transition_discount, make_execution_mask
from .types import ChunkBatch


@dataclass(frozen=True)
class ZarrConversionConfig:
    generated_horizon: int
    executed_horizon: int
    gamma: float
    action_key: str = "action_rotvec"
    fallback_action_key: str = "action"
    state_key: str = "state"
    reward_key: str = "reward"
    done_key: str = "done"
    success_key: str = "success"
    stride: int | None = None
    terminal_success_reward: float = 1.0
    task_id: str = "dexjoco"
    language: str = ""
    behavior_policy: str = "unknown"
    success_from_path: bool = True
    image_keys: tuple[str, ...] | None = None


def _import_zarr():
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "zarr is required for replay.zarr conversion. Use an environment with zarr "
            "installed, e.g. ../pi05/.conda-pi05-openpi-final, or install zarr into "
            "the active OGPO environment."
        ) from exc
    return zarr


def _read_episode_arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    zarr = _import_zarr()
    root = zarr.open(str(path), mode="r")
    data_group = root["data"] if "data" in root else root
    return {key: np.asarray(data_group[key]) for key in data_group.keys()}


def _select_action(arrays: dict[str, np.ndarray], cfg: ZarrConversionConfig) -> np.ndarray:
    if cfg.action_key in arrays:
        return np.asarray(arrays[cfg.action_key], dtype=np.float32)
    if cfg.fallback_action_key in arrays:
        return np.asarray(arrays[cfg.fallback_action_key], dtype=np.float32)
    raise KeyError(f"missing action key {cfg.action_key!r} and fallback {cfg.fallback_action_key!r}")


def _infer_rewards(arrays: dict[str, np.ndarray], length: int, cfg: ZarrConversionConfig, success: bool) -> np.ndarray:
    if cfg.reward_key in arrays:
        return np.asarray(arrays[cfg.reward_key], dtype=np.float32).reshape(length)
    rewards = np.zeros((length,), dtype=np.float32)
    if success and length > 0:
        rewards[-1] = float(cfg.terminal_success_reward)
    return rewards


def _infer_done(arrays: dict[str, np.ndarray], length: int, cfg: ZarrConversionConfig) -> np.ndarray:
    if cfg.done_key in arrays:
        return np.asarray(arrays[cfg.done_key], dtype=np.float32).reshape(length)
    done = np.zeros((length,), dtype=np.float32)
    if length > 0:
        done[-1] = 1.0
    return done


def _infer_success(arrays: dict[str, np.ndarray], path: Path, length: int, cfg: ZarrConversionConfig) -> np.ndarray:
    if cfg.success_key in arrays:
        success = np.asarray(arrays[cfg.success_key], dtype=np.float32).reshape(length)
        return success
    inferred = False
    if cfg.success_from_path:
        inferred = "success" in str(path).lower() or "_demo_" in str(path).lower()
    success = np.zeros((length,), dtype=np.float32)
    if inferred and length > 0:
        success[-1] = 1.0
    return success


def _pad_chunk(actions: np.ndarray, start: int, horizon: int) -> tuple[np.ndarray, int]:
    available = max(0, min(horizon, actions.shape[0] - start))
    if available <= 0:
        raise ValueError("cannot build chunk with no available actions")
    chunk = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    chunk[:available] = actions[start : start + available]
    return chunk, available


def episode_arrays_to_chunk_batch(
    arrays: dict[str, np.ndarray],
    *,
    path: Path,
    episode_id: int,
    cfg: ZarrConversionConfig,
) -> ChunkBatch:
    actions = _select_action(arrays, cfg)
    if actions.ndim != 2:
        raise ValueError(f"expected action array [T, D], got {actions.shape}")
    if cfg.state_key not in arrays:
        raise KeyError(f"missing required state key {cfg.state_key!r}")
    states = np.asarray(arrays[cfg.state_key], dtype=np.float32)
    if states.ndim < 2 or states.shape[0] != actions.shape[0]:
        raise ValueError(f"state shape {states.shape} incompatible with action shape {actions.shape}")
    states = states.reshape(states.shape[0], -1)
    length = actions.shape[0]
    success_arr = _infer_success(arrays, path, length, cfg)
    episode_success = bool(success_arr.max() > 0)
    rewards = _infer_rewards(arrays, length, cfg, episode_success)
    dones = _infer_done(arrays, length, cfg)
    if cfg.image_keys is None:
        image_keys = tuple(
            key
            for key, value in arrays.items()
            if value.ndim == 4 and value.shape[0] == length and value.shape[-1] in (1, 3, 4)
        )
    else:
        missing = set(cfg.image_keys) - set(arrays)
        if missing:
            raise KeyError(f"missing configured image arrays: {sorted(missing)}")
        image_keys = cfg.image_keys
    images = {key: np.asarray(arrays[key]) for key in image_keys}

    stride = cfg.stride if cfg.stride is not None else cfg.executed_horizon
    rows: list[dict[str, Any]] = []
    for start in range(0, length, stride):
        action_chunk, available = _pad_chunk(actions, start, cfg.generated_horizon)
        executed = min(cfg.executed_horizon, available)
        if executed <= 0:
            continue
        next_index = min(start + executed, length - 1)
        reward_prefix = torch.from_numpy(rewards[start : start + executed])
        rows.append(
            {
                "observation": states[start],
                "proprioception": states[start],
                "action_chunk": action_chunk,
                "execution_mask": make_execution_mask(executed, cfg.generated_horizon).cpu().numpy(),
                "executed_length": executed,
                "chunk_return": compute_chunk_return(reward_prefix, cfg.gamma, executed).item(),
                "discount": compute_transition_discount(cfg.gamma, executed).item(),
                "next_observation": states[next_index],
                "next_proprioception": states[next_index],
                "images": {key: value[start] for key, value in images.items()},
                "next_images": {key: value[next_index] for key, value in images.items()},
                "done": float(dones[start : start + executed].max() > 0),
                "success": float(episode_success),
                "episode_id": episode_id,
                "timestep": start,
                "task_id": cfg.task_id,
                "language": cfg.language,
                "behavior_metadata": {
                    "behavior_policy": cfg.behavior_policy,
                    "source_path": str(path),
                    "action_key": cfg.action_key if cfg.action_key in arrays else cfg.fallback_action_key,
                    "reward_fallback": cfg.reward_key not in arrays,
                    "success_fallback": cfg.success_key not in arrays,
                    "transition_success": bool(success_arr[start : next_index + 1].max() > 0),
                },
            }
        )
    return rows_to_chunk_batch(rows)


def rows_to_chunk_batch(rows: list[dict[str, Any]]) -> ChunkBatch:
    if not rows:
        raise ValueError("no chunk transitions were produced")
    image_keys = tuple(rows[0].get("images", {}))
    return ChunkBatch(
        observations=torch.tensor(np.stack([r["observation"] for r in rows]), dtype=torch.float32),
        proprioceptions=torch.tensor(np.stack([r["proprioception"] for r in rows]), dtype=torch.float32),
        action_chunks=torch.tensor(np.stack([r["action_chunk"] for r in rows]), dtype=torch.float32),
        execution_masks=torch.tensor(np.stack([r["execution_mask"] for r in rows]), dtype=torch.bool),
        executed_lengths=torch.tensor([r["executed_length"] for r in rows], dtype=torch.long),
        chunk_returns=torch.tensor([r["chunk_return"] for r in rows], dtype=torch.float32),
        discounts=torch.tensor([r["discount"] for r in rows], dtype=torch.float32),
        next_observations=torch.tensor(np.stack([r["next_observation"] for r in rows]), dtype=torch.float32),
        next_proprioceptions=torch.tensor(np.stack([r["next_proprioception"] for r in rows]), dtype=torch.float32),
        dones=torch.tensor([r["done"] for r in rows], dtype=torch.float32),
        successes=torch.tensor([r["success"] for r in rows], dtype=torch.float32),
        episode_ids=torch.tensor([r["episode_id"] for r in rows], dtype=torch.long),
        timesteps=torch.tensor([r["timestep"] for r in rows], dtype=torch.long),
        task_ids=[r["task_id"] for r in rows],
        languages=[r["language"] for r in rows],
        behavior_metadata=[r["behavior_metadata"] for r in rows],
        images={key: torch.from_numpy(np.stack([r["images"][key] for r in rows])) for key in image_keys}
        if image_keys
        else None,
        next_images={key: torch.from_numpy(np.stack([r["next_images"][key] for r in rows])) for key in image_keys}
        if image_keys
        else None,
    )


def concat_chunk_batches(batches: list[ChunkBatch]) -> ChunkBatch:
    if not batches:
        raise ValueError("expected at least one batch")
    image_key_sets = [set(batch.images or {}) for batch in batches]
    if any(keys != image_key_sets[0] for keys in image_key_sets[1:]):
        raise ValueError("all replay batches must have identical camera keys")
    image_keys = sorted(image_key_sets[0])
    return ChunkBatch(
        observations=torch.cat([b.observations for b in batches], dim=0),
        proprioceptions=torch.cat([b.proprioceptions for b in batches], dim=0),
        action_chunks=torch.cat([b.action_chunks for b in batches], dim=0),
        execution_masks=torch.cat([b.execution_masks for b in batches], dim=0),
        executed_lengths=torch.cat([b.executed_lengths for b in batches], dim=0),
        chunk_returns=torch.cat([b.chunk_returns for b in batches], dim=0),
        discounts=torch.cat([b.discounts for b in batches], dim=0),
        next_observations=torch.cat([b.next_observations for b in batches], dim=0),
        next_proprioceptions=torch.cat([b.next_proprioceptions for b in batches], dim=0),
        dones=torch.cat([b.dones for b in batches], dim=0),
        successes=torch.cat([b.successes for b in batches], dim=0),
        episode_ids=torch.cat([b.episode_ids for b in batches], dim=0),
        timesteps=torch.cat([b.timesteps for b in batches], dim=0),
        task_ids=[task for b in batches for task in b.task_ids],
        languages=[lang for b in batches for lang in b.languages],
        behavior_metadata=[meta for b in batches for meta in b.behavior_metadata],
        images={key: torch.cat([b.images[key] for b in batches], dim=0) for key in image_keys}
        if image_keys
        else None,
        next_images={key: torch.cat([b.next_images[key] for b in batches], dim=0) for key in image_keys}
        if image_keys
        else None,
        mc_returns=torch.cat([b.mc_returns for b in batches], dim=0)
        if all(b.mc_returns is not None for b in batches)
        else None,
    )


def find_replay_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file() and root.suffix == ".npz":
        return [root]
    if root.name == "replay.zarr":
        return [root]
    paths = list(root.rglob("replay.zarr"))
    for npz_path in root.rglob("*.npz"):
        stem = npz_path.stem.lower()
        if stem.endswith("_depth"):
            continue
        if stem.startswith("episode") or stem.startswith("replay"):
            paths.append(npz_path)
    return sorted(paths)


def convert_replay_paths(paths: list[Path], cfg: ZarrConversionConfig) -> ChunkBatch:
    batches = []
    for episode_id, path in enumerate(paths):
        arrays = _read_episode_arrays(path)
        batches.append(episode_arrays_to_chunk_batch(arrays, path=path, episode_id=episode_id, cfg=cfg))
    return concat_chunk_batches(batches)
