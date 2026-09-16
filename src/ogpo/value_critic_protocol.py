from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class StateFeatures:
    """State representation shared by all Q and categorical Value heads."""

    readout: torch.Tensor


class ValueCritic(Protocol):
    @property
    def ensemble_size(self) -> int: ...

    def encode_state(self, batch: Any, *, next_observation: bool = False) -> StateFeatures: ...

    def q_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor: ...

    def value_logits_from_features(self, features: StateFeatures) -> torch.Tensor: ...

    # Optional clean double-Q interface.  Legacy critics continue to expose
    # only q_from_features; the new main path uses these explicit axes.
    def q_pair_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor: ...

    def raw_q_ensemble_from_features(
        self,
        features: StateFeatures,
        action_chunks: torch.Tensor,
        execution_masks: torch.Tensor,
    ) -> torch.Tensor: ...
