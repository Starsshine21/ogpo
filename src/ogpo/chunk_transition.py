from __future__ import annotations

import torch


def make_execution_mask(
    executed_length: int | torch.Tensor,
    generated_horizon: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a boolean mask where only the executed prefix is true."""
    if isinstance(executed_length, torch.Tensor):
        if executed_length.ndim == 0:
            length = int(executed_length.item())
        else:
            arange = torch.arange(generated_horizon, device=executed_length.device)
            return arange.unsqueeze(0) < executed_length.long().unsqueeze(1)
    else:
        length = int(executed_length)
    if length < 0 or length > generated_horizon:
        raise ValueError(f"executed_length={length} outside [0, {generated_horizon}]")
    return torch.arange(generated_horizon, device=device) < length


def mask_action_suffix(action_chunk: torch.Tensor, execution_mask: torch.Tensor) -> torch.Tensor:
    """Zero unexecuted suffix actions without changing the tensor shape."""
    assert action_chunk.ndim in (2, 3)
    if action_chunk.ndim == 2:
        assert execution_mask.shape == action_chunk.shape[:1]
        return action_chunk * execution_mask.to(action_chunk.dtype).unsqueeze(-1)
    assert execution_mask.shape == action_chunk.shape[:2]
    return action_chunk * execution_mask.to(action_chunk.dtype).unsqueeze(-1)


def flatten_masked_action(action_chunk: torch.Tensor, execution_mask: torch.Tensor) -> torch.Tensor:
    """Flatten action chunks after masking unexecuted suffix values."""
    masked = mask_action_suffix(action_chunk, execution_mask)
    if masked.ndim == 2:
        return masked.reshape(-1)
    return masked.reshape(masked.shape[0], -1)


def compute_chunk_return(
    rewards: torch.Tensor,
    gamma: float,
    executed_length: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute sum_j gamma^j r_j over the executed prefix."""
    if rewards.ndim == 1:
        length = int(executed_length) if executed_length is not None else int(rewards.shape[0])
        prefix = rewards[:length]
        powers = torch.arange(length, device=rewards.device, dtype=rewards.dtype)
        return torch.sum(prefix * (float(gamma) ** powers))

    assert rewards.ndim == 2
    batch, horizon = rewards.shape
    if executed_length is None:
        mask = torch.ones(batch, horizon, dtype=torch.bool, device=rewards.device)
    else:
        length_tensor = torch.as_tensor(executed_length, device=rewards.device, dtype=torch.long)
        if length_tensor.ndim == 0:
            length_tensor = length_tensor.expand(batch)
        mask = make_execution_mask(length_tensor, horizon)
    powers = torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    discounts = (float(gamma) ** powers).unsqueeze(0)
    return torch.sum(rewards * discounts * mask.to(rewards.dtype), dim=1)


def compute_transition_discount(
    gamma: float,
    executed_length: int | torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    length = torch.as_tensor(executed_length, dtype=torch.float32, device=device)
    return torch.pow(torch.tensor(float(gamma), dtype=torch.float32, device=length.device), length)


def assert_suffix_invariant(
    action_chunk: torch.Tensor,
    execution_mask: torch.Tensor,
    mutate_value: float = 123.0,
) -> None:
    """Raise if changing the unexecuted suffix changes the effective action."""
    baseline = flatten_masked_action(action_chunk, execution_mask)
    mutated = action_chunk.clone()
    if action_chunk.ndim == 2:
        mutated[~execution_mask.bool()] = mutate_value
    else:
        mutated[~execution_mask.bool()] = mutate_value
    after = flatten_masked_action(mutated, execution_mask)
    if not torch.equal(baseline, after):
        raise AssertionError("unexecuted suffix changed effective action semantics")
