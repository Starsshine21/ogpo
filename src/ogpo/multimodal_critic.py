from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .categorical_q import decode_categorical_q
from .value_critic_protocol import StateFeatures


class MaskedTemporalActionPool(nn.Module):
    """Encode only the executed action prefix into one fixed-width token."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        max_horizon: int,
        num_attention_heads: int,
    ):
        super().__init__()
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        if hidden_dim % num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = int(max_horizon)
        self.action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.temporal_positions = nn.Parameter(torch.zeros(self.max_horizon, self.hidden_dim))
        self.query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_attention_heads),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        # Dataset bounds are rebuilt from replay and intentionally excluded
        # from state_dict so legacy scalar checkpoints remain strictly loadable.
        self.register_buffer("action_min", torch.full((self.action_dim,), -1.0), persistent=False)
        self.register_buffer("action_max", torch.full((self.action_dim,), 1.0), persistent=False)
        nn.init.normal_(self.temporal_positions, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        if action_chunks.ndim != 3:
            raise ValueError("action_chunks must have shape [batch, horizon, action_dim]")
        batch, horizon, action_dim = action_chunks.shape
        if action_dim != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {action_dim}")
        if execution_masks.shape != (batch, horizon):
            raise ValueError("execution_masks must match action chunk batch and horizon")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon={horizon} exceeds max_horizon={self.max_horizon}")
        mask = execution_masks.bool()
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("each sample must contain at least one executed action")

        normalized = (action_chunks - self.action_mean) / self.action_std.clamp_min(1e-6)
        tokens = self.action_projection(normalized)
        tokens = tokens + self.temporal_positions[:horizon].unsqueeze(0)
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attention(
            query,
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return self.output_norm(pooled[:, 0])


def _prediction_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class MultiHeadUdivlCore(nn.Module):
    """Shared action representation with corresponding Q and Value heads."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        max_horizon: int,
        action_hidden_dim: int,
        head_hidden_dim: int,
        num_attention_heads: int,
        num_value_atoms: int,
        num_pairs: int = 3,
        q_representation: str = "scalar",
        q_num_bins: int = 201,
        q_vmin: float = -0.1,
        q_vmax: float = 1.1,
        q_heads_per_member: int = 1,
    ):
        super().__init__()
        if num_pairs <= 1:
            raise ValueError("num_pairs must be greater than one")
        self.state_dim = int(state_dim)
        self.num_pairs = int(num_pairs)
        self.q_heads_per_member = int(q_heads_per_member)
        if self.q_heads_per_member <= 0:
            raise ValueError("q_heads_per_member must be positive")
        self.num_value_atoms = int(num_value_atoms)
        self.q_representation = str(q_representation).lower()
        if self.q_representation not in {"scalar", "categorical"}:
            raise ValueError("q_representation must be 'scalar' or 'categorical'")
        if self.q_representation == "categorical":
            if int(q_num_bins) < 2 or not float(q_vmin) < float(q_vmax):
                raise ValueError("categorical Q requires valid bins and support bounds")
            self.register_buffer(
                "q_support",
                torch.linspace(float(q_vmin), float(q_vmax), int(q_num_bins)),
            )
            q_output_dim = int(q_num_bins)
        else:
            self.q_support = None
            q_output_dim = 1
        self.action_pool = MaskedTemporalActionPool(
            action_dim,
            action_hidden_dim,
            max_horizon,
            num_attention_heads,
        )
        q_input_dim = self.state_dim + int(action_hidden_dim)
        self.q_heads = nn.ModuleList(
            [
                _prediction_head(q_input_dim, head_hidden_dim, q_output_dim)
                for _ in range(self.num_pairs * self.q_heads_per_member)
            ]
        )
        self.value_heads = nn.ModuleList(
            [
                _prediction_head(self.state_dim, head_hidden_dim, self.num_value_atoms)
                for _ in range(self.num_pairs)
            ]
        )

    @property
    def ensemble_size(self) -> int:
        return self.num_pairs

    @property
    def member_count(self) -> int:
        return self.num_pairs

    @property
    def num_q_heads(self) -> int:
        return self.num_pairs * self.q_heads_per_member

    @property
    def member_q_heads(self) -> tuple[tuple[nn.Module, ...], ...]:
        """View the flattened state-dict ModuleList as member-local Q pairs."""
        return tuple(
            tuple(
                self.q_heads[
                    member * self.q_heads_per_member + offset
                ]
                for offset in range(self.q_heads_per_member)
            )
            for member in range(self.num_pairs)
        )

    def _q_head_outputs_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.state_dim:
            raise ValueError(f"readout must have shape [batch, {self.state_dim}]")
        head_parameter = next(self.q_heads[0].parameters())
        readout = readout.to(device=head_parameter.device, dtype=head_parameter.dtype)
        action_features = self.action_pool(action_chunks, execution_masks)
        action_features = action_features.to(device=head_parameter.device, dtype=head_parameter.dtype)
        if readout.shape[0] != action_features.shape[0]:
            raise ValueError("state and action batch sizes must match")
        fused = torch.cat([readout, action_features], dim=-1)
        outputs = torch.stack([head(fused) for head in self.q_heads], dim=0)
        # Expose the mathematically meaningful member/head axes while keeping
        # the flattened ModuleList state-dict layout stable for legacy models.
        return outputs.reshape(
            self.num_pairs,
            self.q_heads_per_member,
            outputs.shape[1],
            outputs.shape[2],
        )

    def q_logits_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_representation != "categorical":
            raise RuntimeError("q_logits are only available for categorical Q")
        outputs = self._q_head_outputs_from_readout(readout, action_chunks, execution_masks)
        return outputs[:, 0] if self.q_heads_per_member == 1 else outputs

    def q_pair_logits_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return categorical Q logits as ``[member, double_q, batch, atoms]``."""
        if self.q_representation != "categorical":
            raise RuntimeError("q_pair_logits are only available for categorical Q")
        return self._q_head_outputs_from_readout(readout, action_chunks, execution_masks)

    def q_pair_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return Q values as ``[member, double_q, batch]``."""
        outputs = self._q_head_outputs_from_readout(readout, action_chunks, execution_masks)
        if self.q_representation == "scalar":
            return outputs.squeeze(-1)
        assert self.q_support is not None
        return decode_categorical_q(outputs, self.q_support)

    def raw_q_ensemble_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return every raw Q head as ``[num_pairs * heads_per_pair, batch]``.

        The stable order is member-major: Q11, Q12, Q21, Q22, ... .  DIVL
        training deliberately uses ``clipped_q_from_readout`` instead; this
        interface is the explicit OGPO actor boundary.
        """
        pairs = self.q_pair_from_readout(readout, action_chunks, execution_masks)
        return pairs.reshape(self.num_q_heads, pairs.shape[-1])

    def clipped_q_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Member-wise clipped double-Q value ``min(Q_{m,1}, Q_{m,2})``."""
        pairs = self.q_pair_from_readout(readout, action_chunks, execution_masks)
        if self.q_heads_per_member == 1:
            return pairs[:, 0]
        return pairs.min(dim=1).values

    def q_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        # Legacy callers consume one Q value per ensemble member.  For a
        # double-Q member, return the clipped value; new callers can request
        # the explicit pair tensor through ``q_pair_from_readout``.
        return self.clipped_q_from_readout(readout, action_chunks, execution_masks)

    def value_logits_from_readout(self, readout: torch.Tensor) -> torch.Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.state_dim:
            raise ValueError(f"readout must have shape [batch, {self.state_dim}]")
        head_parameter = next(self.value_heads[0].parameters())
        readout = readout.to(device=head_parameter.device, dtype=head_parameter.dtype)
        return torch.stack([head(readout) for head in self.value_heads], dim=0)


class MultiHeadUdivlCritic(nn.Module):
    """Compose one state encoder with the shared multi-head U-DIVL core."""

    def __init__(self, state_encoder: nn.Module, core: MultiHeadUdivlCore):
        super().__init__()
        self.state_encoder = state_encoder
        self.core = core

    @property
    def ensemble_size(self) -> int:
        return self.core.ensemble_size

    @property
    def num_raw_q_heads(self) -> int:
        return self.core.num_q_heads

    def encode_state(self, batch: Any, *, next_observation: bool = False) -> StateFeatures:
        readout = self.state_encoder(batch, next_observation=next_observation)
        return StateFeatures(readout=readout)

    def q_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.q_from_readout(features.readout, action_chunks, execution_masks)

    def q_pair_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.q_pair_from_readout(features.readout, action_chunks, execution_masks)

    def raw_q_ensemble_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.raw_q_ensemble_from_readout(
            features.readout, action_chunks, execution_masks
        )

    def clipped_q_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.clipped_q_from_readout(features.readout, action_chunks, execution_masks)

    def q_logits_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.q_logits_from_readout(features.readout, action_chunks, execution_masks)

    def q_pair_logits_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.q_pair_logits_from_readout(features.readout, action_chunks, execution_masks)

    def value_logits_from_features(self, features: StateFeatures) -> torch.Tensor:
        return self.core.value_logits_from_readout(features.readout)

    def forward(
        self,
        batch: Any,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
        *,
        next_observation: bool = False,
    ) -> torch.Tensor:
        features = self.encode_state(batch, next_observation=next_observation)
        return self.q_from_features(features, action_chunks, execution_masks)


class MultiHeadScalarQCore(nn.Module):
    """Shared action representation with independent scalar Q heads only."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        max_horizon: int,
        action_hidden_dim: int,
        head_hidden_dim: int,
        num_attention_heads: int,
        num_heads: int,
    ):
        super().__init__()
        if num_heads <= 1:
            raise ValueError("num_heads must be greater than one")
        self.state_dim = int(state_dim)
        self.num_heads = int(num_heads)
        self.action_pool = MaskedTemporalActionPool(
            action_dim,
            action_hidden_dim,
            max_horizon,
            num_attention_heads,
        )
        q_input_dim = self.state_dim + int(action_hidden_dim)
        self.q_heads = nn.ModuleList(
            [_prediction_head(q_input_dim, head_hidden_dim, 1) for _ in range(self.num_heads)]
        )

    @property
    def ensemble_size(self) -> int:
        return self.num_heads

    def q_from_readout(
        self,
        readout: torch.Tensor,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.state_dim:
            raise ValueError(f"readout must have shape [batch, {self.state_dim}]")
        head_parameter = next(self.q_heads[0].parameters())
        readout = readout.to(device=head_parameter.device, dtype=head_parameter.dtype)
        action_features = self.action_pool(action_chunks, execution_masks)
        action_features = action_features.to(device=head_parameter.device, dtype=head_parameter.dtype)
        if readout.shape[0] != action_features.shape[0]:
            raise ValueError("state and action batch sizes must match")
        fused = torch.cat([readout, action_features], dim=-1)
        return torch.stack([head(fused).squeeze(-1) for head in self.q_heads], dim=0)


class MultiHeadScalarQCritic(nn.Module):
    """One multimodal state encoder with an original-OGPO scalar Q ensemble."""

    def __init__(self, state_encoder: nn.Module, core: MultiHeadScalarQCore):
        super().__init__()
        self.state_encoder = state_encoder
        self.core = core

    @property
    def ensemble_size(self) -> int:
        return self.core.ensemble_size

    def encode_state(self, batch: Any, *, next_observation: bool = False) -> StateFeatures:
        cached = (
            batch.next_critic_features
            if next_observation
            else batch.critic_features
        )
        if cached is not None:
            return StateFeatures(readout=cached)
        readout = self.state_encoder(batch, next_observation=next_observation)
        return StateFeatures(readout=readout)

    def q_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.q_from_readout(features.readout, action_chunks, execution_masks)

    def forward(
        self,
        batch: Any,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
        *,
        next_observation: bool = False,
    ) -> torch.Tensor:
        features = self.encode_state(batch, next_observation=next_observation)
        return self.q_from_features(features, action_chunks, execution_masks)
