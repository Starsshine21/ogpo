from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


class ValidationEarlyStopper:
    def __init__(self, *, mode: str, patience: int, min_delta: float = 0.0):
        if mode not in {"min", "max"}:
            raise ValueError("early-stopping mode must be 'min' or 'max'")
        if patience <= 0:
            raise ValueError("early-stopping patience must be positive")
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf if mode == "min" else -math.inf
        self.stale_evaluations = 0

    def update(self, value: float) -> bool:
        value = float(value)
        improved = (
            value <= self.best - self.min_delta
            if self.mode == "min"
            else value >= self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stale_evaluations >= self.patience


@dataclass(frozen=True)
class TrainableSnapshot:
    state: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, module: nn.Module) -> "TrainableSnapshot":
        return cls(
            {
                name: parameter.detach().cpu().clone()
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            }
        )

    @torch.no_grad()
    def restore(self, module: nn.Module) -> None:
        parameters = dict(module.named_parameters())
        missing = set(self.state) - set(parameters)
        if missing:
            raise KeyError(f"snapshot parameters are missing from module: {sorted(missing)}")
        for name, value in self.state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
