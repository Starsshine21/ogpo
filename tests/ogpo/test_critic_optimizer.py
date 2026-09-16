from types import SimpleNamespace

import pytest
import torch

from ogpo.trainer import (
    _apply_critic_lr_schedule,
    _make_critic_optimizer,
)


def test_lwd_adam_optimizer_and_cosine_decay():
    critic = torch.nn.Linear(2, 1)
    optimizer = _make_critic_optimizer(
        critic,
        None,
        {
            "optimizer": "adam",
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
        },
    )
    state = SimpleNamespace(critic_optimizer=optimizer, step=0)
    config = {
        "critic": {
            "lr_schedule": "cosine",
            "min_lr_ratio": 0.0,
        },
        "training": {"critic_steps": 5},
    }

    assert isinstance(optimizer, torch.optim.Adam)
    assert _apply_critic_lr_schedule(state, config) == pytest.approx(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)

    state.step = 2
    assert _apply_critic_lr_schedule(state, config) == pytest.approx(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)

    state.step = 4
    assert _apply_critic_lr_schedule(state, config) == pytest.approx(0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
