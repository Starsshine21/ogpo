from types import SimpleNamespace

import torch

from ogpo.critic import clone_target
from ogpo.flow_sde import GaussianFlowPolicy
from ogpo.trainer import slow_policy_update_due, sync_slow_policy


def _state():
    current = GaussianFlowPolicy(3, 4, hidden_dim=8, num_steps=3)
    slow = clone_target(current)
    with torch.no_grad():
        for parameter in current.parameters():
            parameter.fill_(1.0)
        for parameter in slow.parameters():
            parameter.zero_()
    return SimpleNamespace(policy=current, slow_policy=slow)


def test_slow_policy_tau_0005_one_and_two_updates():
    state = _state()
    sync_slow_policy(state, ema=0.9995)
    first = next(state.slow_policy.parameters())
    assert torch.allclose(first, torch.full_like(first, 0.0005), atol=1e-7)
    sync_slow_policy(state, ema=0.9995)
    second = next(state.slow_policy.parameters())
    assert torch.allclose(second, torch.full_like(second, 0.00099975), atol=1e-7)


def test_slow_policy_only_updates_for_accepted_period_boundary():
    assert slow_policy_update_due(
        accepted_actor_updates=1, update_period=1, update_accepted=True
    )
    assert not slow_policy_update_due(
        accepted_actor_updates=1, update_period=1, update_accepted=False
    )
    assert not slow_policy_update_due(
        accepted_actor_updates=1, update_period=2, update_accepted=True
    )
    assert slow_policy_update_due(
        accepted_actor_updates=2, update_period=2, update_accepted=True
    )


def test_rejected_update_leaves_slow_snapshot_unchanged():
    state = _state()
    before = [parameter.detach().clone() for parameter in state.slow_policy.parameters()]
    if slow_policy_update_due(
        accepted_actor_updates=1, update_period=1, update_accepted=False
    ):
        sync_slow_policy(state, ema=0.9995)
    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(state.slow_policy.parameters(), before, strict=True)
    )
