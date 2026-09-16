from types import SimpleNamespace

import pytest
import torch

from ogpo import trainer
from ogpo.critic import clone_target
from ogpo.flow_sde import GaussianFlowPolicy
from ogpo.full_ogpo import full_chain_chipo_ppo_loss
from ogpo.replay import make_synthetic_replay
from ogpo.trainer import (
    build_train_state,
    critic_update,
    finalize_actor_update_transaction,
    full_actor_update,
    load_checkpoint,
    save_checkpoint,
)
from train_full_ogpo import (
    periodic_actor_checkpoint_path,
    save_periodic_actor_checkpoint_if_due,
)


def _config(*, old_period=1, slow_period=1):
    return {
        "critic": {"ensemble_size": 2, "hidden_dim": 8, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {
            "hidden_dim": 8,
            "ogpo_variant": "chi2",
            "old_policy_sync_period": old_period,
            "old_policy_ema": 0.0,
            "chi2": {
                "ratio_scope": "full_chain",
                "slow_policy_tau": 0.0005,
                "slow_policy_update_period": slow_period,
            },
        },
        "flow": {
            "num_steps": 3,
            "sde_mode": "ogpo_constant_corrected",
            "constant_noise_std": 0.005,
            "learn_sde_std": False,
        },
    }


def _policy():
    return GaussianFlowPolicy(
        3,
        4,
        hidden_dim=8,
        num_steps=3,
        sde_mode="ogpo_constant_corrected",
        constant_noise_std=0.005,
        learn_sde_std=False,
    )


def _full_update_fixture():
    batch = make_synthetic_replay(
        num_samples=6,
        obs_dim=3,
        generated_horizon=2,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    config["flow"]["num_steps"] = 2
    config["actor"].update(
        {
            "group_size": 2,
            "actor_epochs_per_rollout": 1,
            "full_ratio_mode": "ais_joint",
            "gradient_microbatch_size": 2,
            "reject_update_on_kl": True,
            "post_update_kl_action": "rollback_cpu",
            "max_policy_reference_kl": float("inf"),
        }
    )
    config["regularization"] = {
        "beta_kl": 0.0,
        "lambda_fm": 0.0,
        "lambda_success": 0.0,
    }
    state = build_train_state(config, batch)
    critic_update(state, batch, config)
    return state, batch, config


def _state(current=0.0, old=0.0, slow=0.0):
    policy = _policy()
    old_policy = clone_target(policy)
    slow_policy = clone_target(policy)
    state = SimpleNamespace(
        policy=policy,
        old_policy=old_policy,
        slow_policy=slow_policy,
        actor_step=0,
        accepted_actor_updates=0,
    )
    _fill(state.policy, current)
    _fill(state.old_policy, old)
    _fill(state.slow_policy, slow)
    return state


def _fill(module, value):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _params(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _all_equal(module, value):
    return all(torch.allclose(parameter, torch.full_like(parameter, value)) for parameter in module.parameters())


def test_zero_lambda_success_skips_success_bc_forward(monkeypatch):
    state, batch, config = _full_update_fixture()

    def unexpected_success_bc(*args, **kwargs):
        raise AssertionError("success BC must not run when lambda_success is zero")

    monkeypatch.setattr(trainer, "success_buffer_loss", unexpected_success_bc)
    loss, metrics = trainer._actor_regularization_loss(
        state,
        batch,
        config,
        success_batch=batch,
    )

    assert float(loss.item()) == 0.0
    assert metrics["success_buffer_loss"] == 0.0


def test_full_actor_batch4_group4_micro1_is_one_optimizer_transaction(monkeypatch):
    state, batch, config = _full_update_fixture()
    batch = batch.index_select(torch.arange(4))
    config["actor"].update(
        {
            "group_size": 4,
            "candidate_group_size": 4,
            "gradient_microbatch_size": 1,
        }
    )

    observed_shapes = []
    original_advantages = trainer.conservative_advantages_for_candidates

    def checked_advantages(state_arg, observations, candidates, batch_arg, config_arg, **kwargs):
        assert candidates.shape[:2] == (4, 4)
        advantages, diagnostics = original_advantages(
            state_arg, observations, candidates, batch_arg, config_arg, **kwargs
        )
        observed_shapes.append(tuple(advantages.shape))
        return advantages, diagnostics

    optimizer_steps = 0
    original_step = state.actor_optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(trainer, "conservative_advantages_for_candidates", checked_advantages)
    monkeypatch.setattr(state.actor_optimizer, "step", counted_step)
    metrics = full_actor_update(state, batch, config)

    assert observed_shapes == [(4, 4)]
    assert optimizer_steps == 1
    assert metrics["state_batch_size"] == 4.0
    assert metrics["candidate_group_size"] == 4.0
    assert metrics["logical_candidate_batch_size"] == 16.0
    assert metrics["gradient_microbatch_size"] == 1.0
    assert metrics["gradient_accumulation_steps"] == 16.0


def test_old_exact_sync_after_accepted_update():
    state = _state(current=1.0, old=0.0)
    metrics = finalize_actor_update_transaction(state, accepted=True, config=_config())
    assert _all_equal(state.old_policy, 1.0)
    assert metrics["old_policy_sync_applied"] == 1.0


@pytest.mark.parametrize('microbatch,no_grad_kl', [(2, False), (4, False), (1, True), (2, True), (4, True)])
def test_speed_probe_preserves_full_update(microbatch, no_grad_kl, monkeypatch):
    from probe_actor_speed import diagnostic_kl_call
    original_kl = trainer._policy_kl_device_aware
    results = []
    for mb, optimize in [(1, False), (microbatch, no_grad_kl)]:
        torch.manual_seed(163)
        state, batch, config = _full_update_fixture()
        batch = batch.index_select(torch.arange(4))
        config['actor'].update(group_size=4, candidate_group_size=4,
                               gradient_microbatch_size=mb, kl_eval_microbatch_size=1)
        monkeypatch.setattr(trainer, '_policy_kl_device_aware',
            lambda *a, **kw: diagnostic_kl_call(lambda: original_kl(*a, **kw),
                                               beta=0.0, enabled=optimize))
        torch.manual_seed(917)
        metrics = full_actor_update(state, batch, config)
        results.append((metrics, _params(state.policy)))
    for key in ['actor_loss', 'reference_kl', 'actor_grad_norm', 'post_update_reference_kl',
                'actor_update_accepted', 'actor_update_rejected']:
        assert results[1][0][key] == pytest.approx(results[0][0][key], rel=2e-4, abs=2e-6)
    for before, after in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(before, after, rtol=2e-4, atol=2e-6)


def test_old_unchanged_after_rejected_rollback():
    state = _state(current=0.0, old=0.0)
    preupdate = _params(state.policy)
    _fill(state.policy, 1.0)  # simulated optimizer step
    with torch.no_grad():
        for parameter, before in zip(state.policy.parameters(), preupdate, strict=True):
            parameter.copy_(before)  # simulated rollback before finalization
    finalize_actor_update_transaction(state, accepted=False, config=_config())
    assert _all_equal(state.policy, 0.0)
    assert _all_equal(state.old_policy, 0.0)


def test_slow_first_ema_is_0005():
    state = _state(current=1.0, slow=0.0)
    finalize_actor_update_transaction(state, accepted=True, config=_config())
    assert _all_equal(state.slow_policy, 0.0005)


def test_slow_second_ema_is_exact():
    state = _state(current=1.0, slow=0.0)
    finalize_actor_update_transaction(state, accepted=True, config=_config())
    finalize_actor_update_transaction(state, accepted=True, config=_config())
    assert _all_equal(state.slow_policy, 0.00099975)


def test_slow_unchanged_after_rejected_transaction():
    state = _state(current=1.0, slow=0.2)
    before = _params(state.slow_policy)
    finalize_actor_update_transaction(state, accepted=False, config=_config())
    assert all(torch.equal(a, b) for a, b in zip(state.slow_policy.parameters(), before, strict=True))


def test_accepted_counter_counts_accept_reject_accept_as_two():
    state = _state(current=1.0)
    for accepted in (True, False, True):
        finalize_actor_update_transaction(state, accepted=accepted, config=_config())
    assert state.actor_step == 3
    assert state.accepted_actor_updates == 2


def test_slow_period_two_uses_accepted_count_only():
    state = _state(current=1.0, slow=0.0)
    config = _config(slow_period=2)
    accepted_flags = (True, False, True, True, False, True)
    expected = (0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.00099975)
    expected_sync = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    for accepted, value, sync in zip(accepted_flags, expected, expected_sync, strict=True):
        metrics = finalize_actor_update_transaction(state, accepted=accepted, config=config)
        assert _all_equal(state.slow_policy, value)
        assert metrics["slow_policy_sync_applied"] == sync


def test_old_period_two_uses_accepted_count_only():
    state = _state(current=1.0, old=0.0)
    config = _config(old_period=2)
    cases = (
        (True, 1.0, 0.0, 0.0),
        (False, 1.0, 0.0, 0.0),
        (True, 2.0, 2.0, 1.0),
        (True, 3.0, 2.0, 0.0),
        (False, 3.0, 2.0, 0.0),
        (True, 4.0, 4.0, 1.0),
    )
    for accepted, current, expected_old, expected_sync in cases:
        _fill(state.policy, current)
        metrics = finalize_actor_update_transaction(state, accepted=accepted, config=config)
        assert _all_equal(state.old_policy, expected_old)
        assert metrics["old_policy_sync_applied"] == expected_sync


def test_current_old_slow_are_independent_and_update_independently():
    state = _state(current=1.0, old=0.0, slow=0.2)
    assert state.old_policy is not state.policy
    assert state.slow_policy is not state.policy
    assert state.old_policy is not state.slow_policy
    slow_before = _params(state.slow_policy)
    finalize_actor_update_transaction(
        state, accepted=True, config=_config(slow_period=2)
    )
    assert _all_equal(state.old_policy, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(state.slow_policy.parameters(), slow_before, strict=True))
    old_before = _params(state.old_policy)
    finalize_actor_update_transaction(
        state, accepted=True, config=_config(old_period=0, slow_period=1)
    )
    assert all(torch.equal(a, b) for a, b in zip(state.old_policy.parameters(), old_before, strict=True))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(state.slow_policy.parameters(), slow_before, strict=True)
    )


def test_initial_references_are_exact_independent_current_copies():
    state = _state(current=0.3, old=0.3, slow=0.3)
    assert state.policy is not state.old_policy
    assert state.policy is not state.slow_policy
    assert state.old_policy is not state.slow_policy
    assert all(
        torch.equal(current, old)
        for current, old in zip(state.policy.parameters(), state.old_policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(current, slow)
        for current, slow in zip(state.policy.parameters(), state.slow_policy.parameters(), strict=True)
    )


def test_main_old_and_slow_have_different_expected_behavior():
    state = _state(current=1.0, old=0.0, slow=0.0)
    finalize_actor_update_transaction(state, accepted=True, config=_config())
    assert _all_equal(state.policy, 1.0)
    assert _all_equal(state.old_policy, 1.0)
    assert _all_equal(state.slow_policy, 0.0005)
    assert not _all_equal(state.slow_policy, 1.0)


def test_checkpoint_roundtrip_restores_all_policy_states_and_counters(tmp_path):
    batch = make_synthetic_replay(
        num_samples=4, obs_dim=3, generated_horizon=2, executed_horizon=2, action_dim=2
    )
    config = _config()
    config["flow"]["num_steps"] = 2
    state = build_train_state(config, batch)
    state.actor_optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in state.policy.parameters()).backward()
    state.actor_optimizer.step()
    _fill(state.policy, 1.0)
    _fill(state.old_policy, 0.8)
    _fill(state.slow_policy, 0.2)
    state.actor_step = 17
    state.accepted_actor_updates = 11
    path = tmp_path / "lifecycle.pt"
    save_checkpoint(state, config, path)
    restored = build_train_state(config, batch)
    load_checkpoint(path, restored)
    assert _all_equal(restored.policy, 1.0)
    assert _all_equal(restored.old_policy, 0.8)
    assert _all_equal(restored.slow_policy, 0.2)
    assert restored.actor_step == 17
    assert restored.accepted_actor_updates == 11
    assert len(restored.actor_optimizer.state) == len(state.actor_optimizer.state)
    for source, destination in zip(
        state.actor_optimizer.state.values(),
        restored.actor_optimizer.state.values(),
        strict=True,
    ):
        for key in source:
            if torch.is_tensor(source[key]):
                assert torch.equal(source[key], destination[key])


def test_periodic_actor_checkpoint_boundaries_are_exactly_every_100_steps(tmp_path):
    training = {
        "checkpoint_interval": 100,
        "keep_periodic_checkpoints": True,
        "checkpoint_path": "final.pt",
        "periodic_checkpoint_dir": "periodic",
    }
    assert periodic_actor_checkpoint_path(training, 99, root=tmp_path) is None
    assert periodic_actor_checkpoint_path(training, 100, root=tmp_path) == (
        tmp_path / "periodic/step_0100/final.pt"
    )
    assert periodic_actor_checkpoint_path(training, 101, root=tmp_path) is None
    assert periodic_actor_checkpoint_path(training, 200, root=tmp_path) == (
        tmp_path / "periodic/step_0200/final.pt"
    )


def test_periodic_actor_checkpoint_really_writes_and_restores_step_100(tmp_path):
    batch = make_synthetic_replay(
        num_samples=4,
        obs_dim=3,
        generated_horizon=2,
        executed_horizon=2,
        action_dim=2,
    )
    config = _config()
    config["flow"]["num_steps"] = 2
    config["training"] = {
        "checkpoint_interval": 100,
        "keep_periodic_checkpoints": True,
        "checkpoint_path": "actor_final.pt",
        "periodic_checkpoint_dir": "actor_periodic",
    }
    state = build_train_state(config, batch)
    _fill(state.policy, 1.0)
    _fill(state.old_policy, 0.8)
    _fill(state.slow_policy, 0.2)
    state.actor_step = 100
    state.accepted_actor_updates = 73

    checkpoint = save_periodic_actor_checkpoint_if_due(
        state, config, 100, root=tmp_path
    )
    assert checkpoint == tmp_path / "actor_periodic/step_0100/actor_final.pt"
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["actor_step"] == 100
    assert payload["accepted_actor_updates"] == 73
    assert "actor_optimizer" in payload
    assert "old_policy" in payload
    assert "slow_policy" in payload

    restored = build_train_state(config, batch)
    load_checkpoint(checkpoint, restored)
    assert restored.actor_step == 100
    assert restored.accepted_actor_updates == 73
    assert _all_equal(restored.policy, 1.0)
    assert _all_equal(restored.old_policy, 0.8)
    assert _all_equal(restored.slow_policy, 0.2)


def test_ppo_identity_after_period_one_old_sync():
    state = _state(current=1.0, old=0.0, slow=0.0)
    finalize_actor_update_transaction(state, accepted=True, config=_config())
    condition = torch.randn(2, 3)
    rollout = state.old_policy.rollout(condition)
    current_log_probs = torch.stack(
        [
            state.policy.log_prob(
                rollout.next_states[:, step], rollout.states[:, step], condition, rollout.timesteps[:, step]
            )
            for step in range(state.policy.num_steps)
        ],
        dim=1,
    )
    result = full_chain_chipo_ppo_loss(
        current_log_probs,
        rollout.log_probs,
        torch.ones(2),
        clip_eps=0.01,
        beta=0.1,
        r_max=10.0,
        logprob_normalizer=state.policy.action_dim * state.policy.num_steps,
    )
    assert torch.allclose(result.log_ratio, torch.zeros(2), atol=1e-6)
    assert torch.allclose(result.ratio, torch.ones(2), atol=1e-6)


def test_slow_does_not_equal_current_after_normal_update():
    state = _state(current=1.0, slow=0.0)
    finalize_actor_update_transaction(state, accepted=True, config=_config())
    assert not all(
        torch.equal(current, slow)
        for current, slow in zip(state.policy.parameters(), state.slow_policy.parameters(), strict=True)
    )


def test_integrated_accepted_transaction_lifecycle():
    state = _state(current=0.0, old=0.0, slow=0.0)
    optimizer = torch.optim.SGD(state.policy.parameters(), lr=0.01)

    optimizer.zero_grad()
    sum(parameter.sum() for parameter in state.policy.parameters()).backward()
    optimizer.step()
    accepted_current = _params(state.policy)
    accepted_metrics = finalize_actor_update_transaction(state, accepted=True, config=_config())
    assert accepted_metrics["old_policy_sync_applied"] == 1.0
    assert accepted_metrics["slow_policy_sync_applied"] == 1.0
    assert state.accepted_actor_updates == 1
    assert state.actor_step == 1
    assert all(
        torch.equal(current, old)
        for current, old in zip(
            state.policy.parameters(), state.old_policy.parameters(), strict=True
        )
    )
    assert any(
        not torch.equal(current, slow)
        for current, slow in zip(
            accepted_current, state.slow_policy.parameters(), strict=True
        )
    )


def test_integrated_rejected_transaction_lifecycle():
    state = _state(current=0.0, old=0.0, slow=0.0)
    optimizer = torch.optim.SGD(state.policy.parameters(), lr=0.01)

    preupdate = _params(state.policy)
    old_before = _params(state.old_policy)
    slow_before = _params(state.slow_policy)
    optimizer.zero_grad()
    sum(parameter.sum() for parameter in state.policy.parameters()).backward()
    optimizer.step()
    with torch.no_grad():
        for parameter, before in zip(state.policy.parameters(), preupdate, strict=True):
            parameter.copy_(before)  # fake KL validator rejected: rollback current
    rejected_metrics = finalize_actor_update_transaction(state, accepted=False, config=_config())
    assert rejected_metrics["old_policy_sync_applied"] == 0.0
    assert rejected_metrics["slow_policy_sync_applied"] == 0.0
    assert state.accepted_actor_updates == 0
    assert state.actor_step == 1
    assert all(torch.equal(a, b) for a, b in zip(state.policy.parameters(), preupdate, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(state.old_policy.parameters(), old_before, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(state.slow_policy.parameters(), slow_before, strict=True))


def test_full_actor_update_calls_lifecycle_after_accepted_transaction():
    state, batch, config = _full_update_fixture()
    _fill(state.slow_policy, 0.0)

    metrics = full_actor_update(state, batch, config)

    assert metrics["actor_update_accepted"] == 1.0
    assert metrics["old_policy_sync_applied"] == 1.0
    assert metrics["slow_policy_sync_applied"] == 1.0
    assert metrics["accepted_actor_updates"] == 1.0
    assert metrics["actor_step"] == 1.0
    assert all(
        torch.equal(current, old)
        for current, old in zip(
            state.policy.parameters(), state.old_policy.parameters(), strict=True
        )
    )
    assert all(
        torch.allclose(slow, current.detach() * 0.0005)
        for current, slow in zip(
            state.policy.parameters(), state.slow_policy.parameters(), strict=True
        )
    )


def test_actor_component_gradient_hooks_exactly_decompose_total_gradient():
    state, batch, config = _full_update_fixture()
    config["diagnostics"] = {"actor_component_grad_norms": True}
    config["regularization"].update({"lambda_fm": 0.0, "lambda_success": 0.1})

    metrics = full_actor_update(
        state,
        batch,
        config,
        fm_batch=batch,
        success_batch=batch,
    )

    ppo = metrics["ppo_component_grad_norm"]
    bc = metrics["bc_component_grad_norm"]
    cosine = metrics["ppo_bc_grad_cosine"]
    reconstructed = max(0.0, ppo**2 + bc**2 + 2.0 * ppo * bc * cosine) ** 0.5
    assert metrics["component_grad_norm_diagnostics"] == 1.0
    assert ppo >= 0.0
    assert bc > 0.0
    assert metrics["bc_to_ppo_grad_norm_ratio"] == pytest.approx(
        bc / (ppo + 1e-12)
    )
    assert metrics["actor_grad_norm"] == pytest.approx(reconstructed, rel=2e-4, abs=2e-6)
    assert -1.0001 <= cosine <= 1.0001


def test_actor_component_gradient_diagnostics_do_not_change_update():
    torch.manual_seed(123)
    baseline_state, baseline_batch, baseline_config = _full_update_fixture()
    torch.manual_seed(123)
    diagnostic_state, diagnostic_batch, diagnostic_config = _full_update_fixture()
    baseline_config["regularization"].update({"lambda_fm": 0.0, "lambda_success": 0.1})
    diagnostic_config["regularization"].update({"lambda_fm": 0.0, "lambda_success": 0.1})
    diagnostic_config["diagnostics"] = {"actor_component_grad_norms": True}

    torch.manual_seed(456)
    full_actor_update(
        baseline_state,
        baseline_batch,
        baseline_config,
        fm_batch=baseline_batch,
        success_batch=baseline_batch,
    )
    torch.manual_seed(456)
    full_actor_update(
        diagnostic_state,
        diagnostic_batch,
        diagnostic_config,
        fm_batch=diagnostic_batch,
        success_batch=diagnostic_batch,
    )

    assert all(
        torch.equal(baseline, diagnostic)
        for baseline, diagnostic in zip(
            baseline_state.policy.parameters(),
            diagnostic_state.policy.parameters(),
            strict=True,
        )
    )


def test_pure_component_gradient_diagnostic_never_steps_or_updates_lifecycle():
    state, batch, config = _full_update_fixture()
    config["diagnostics"] = {"actor_component_grad_norms": True}
    config["regularization"].update(
        {"lambda_fm": 0.0, "lambda_smooth": 0.0, "lambda_success": 0.1}
    )
    policy_before = _params(state.policy)
    old_before = _params(state.old_policy)
    slow_before = _params(state.slow_policy)
    optimizer_state_before = len(state.actor_optimizer.state)

    metrics = full_actor_update(
        state,
        batch,
        config,
        fm_batch=batch,
        success_batch=batch,
        diagnostic_no_optimizer_step=True,
    )

    assert state.actor_step == 0
    assert state.accepted_actor_updates == 0
    assert len(state.actor_optimizer.state) == optimizer_state_before
    assert all(
        torch.equal(before, after)
        for before, after in zip(policy_before, state.policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(old_before, state.old_policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(slow_before, state.slow_policy.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in state.policy.parameters())
    assert metrics["bc_0p1_component_grad_norm"] == pytest.approx(
        metrics["bc_component_grad_norm"]
    )
    assert metrics["combined_grad_norm"] >= 0.0
    assert metrics["would_gradient_clip"] in {0.0, 1.0}


def test_full_actor_update_calls_lifecycle_after_rejected_rollback(monkeypatch):
    state, batch, config = _full_update_fixture()
    config["actor"]["max_policy_reference_kl"] = 0.01
    current_before = _params(state.policy)
    old_before = _params(state.old_policy)
    slow_before = _params(state.slow_policy)
    monkeypatch.setattr(trainer, "_full_chain_policy_kl", lambda *args, **kwargs: 1.0)

    metrics = full_actor_update(state, batch, config)

    assert metrics["actor_update_accepted"] == 0.0
    assert metrics["actor_update_rejected"] == 1.0
    assert metrics["old_policy_sync_applied"] == 0.0
    assert metrics["slow_policy_sync_applied"] == 0.0
    assert metrics["accepted_actor_updates"] == 0.0
    assert metrics["actor_step"] == 1.0
    assert all(torch.equal(a, b) for a, b in zip(state.policy.parameters(), current_before, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(state.old_policy.parameters(), old_before, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(state.slow_policy.parameters(), slow_before, strict=True))


def test_legacy_checkpoint_missing_references_and_counters_falls_back(tmp_path):
    batch = make_synthetic_replay(
        num_samples=4, obs_dim=3, generated_horizon=2, executed_horizon=2, action_dim=2
    )
    config = _config()
    config["flow"]["num_steps"] = 2
    state = build_train_state(config, batch)
    _fill(state.policy, 0.7)
    path = tmp_path / "legacy.pt"
    save_checkpoint(state, config, path)
    payload = torch.load(path, weights_only=False)
    for key in ("old_policy", "slow_policy", "actor_step", "accepted_actor_updates"):
        payload.pop(key, None)
    torch.save(payload, path)
    restored = build_train_state(config, batch)
    with pytest.warns(RuntimeWarning, match="legacy checkpoint"):
        load_checkpoint(path, restored)
    assert all(
        torch.equal(current, old)
        for current, old in zip(restored.policy.parameters(), restored.old_policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(current, slow)
        for current, slow in zip(restored.policy.parameters(), restored.slow_policy.parameters(), strict=True)
    )
    assert restored.actor_step == 0
    assert restored.accepted_actor_updates == 0
