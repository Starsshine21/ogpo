import copy
from dataclasses import dataclass

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from ogpo.pi05_jax_flow_core import state_adaptive_kl_penalty
from ogpo.inference_policy import PI05OGPOInferencePolicy
from ogpo import inference_policy as inference_policy_module
from ogpo.pi05_pytorch_adapter import PI05FlowCondition
from ogpo.pi05_jax_adapter import PI05JaxFlowPolicy
from ogpo.pi05_jax_adapter import _ensure_orbax_jax_compatibility
from ogpo.replay import make_synthetic_replay
from ogpo import trainer
from openpi.models import model as openpi_model


@dataclass(frozen=True)
class FakeObservation:
    state: torch.Tensor
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor] | None = None
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None


class FakeJaxBackend:
    def __init__(self, action_horizon: int, action_dim: int):
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def predict_velocity(self, observation, noisy_actions, timestep, *, train=False):
        del observation, train
        return noisy_actions * 0.25 + timestep[:, None, None]


class TrainableFakeJaxBackend(nnx.Module):
    def __init__(self, action_horizon: int, action_dim: int):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.scale = nnx.Param(jnp.asarray(0.25, dtype=jnp.float32))

    def predict_velocity(self, observation, noisy_actions, timestep, *, train=False):
        del observation, train
        return noisy_actions * self.scale.value + timestep[:, None, None]


def _condition(batch_size: int) -> PI05FlowCondition:
    observation = FakeObservation(
        state=torch.randn(batch_size, 5),
        images={"base": torch.randn(batch_size, 3, 8, 8)},
        image_masks={"base": torch.ones(batch_size, dtype=torch.bool)},
    )
    return PI05FlowCondition(observation)


def _tree_allclose(left, right) -> bool:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def test_shard_jax_batch_preserves_candidate_order():
    batch = {
        "values": jnp.arange(32 * 3).reshape(32, 3),
        "mask": jnp.arange(32),
    }

    sharded = trainer._shard_jax_batch(batch, 4)

    assert sharded["values"].shape == (4, 8, 3)
    assert sharded["mask"].shape == (4, 8)
    assert np.array_equal(np.asarray(sharded["values"]).reshape(32, 3), np.asarray(batch["values"]))


def test_shard_jax_batch_rejects_uneven_device_split():
    with pytest.raises(ValueError, match="not divisible"):
        trainer._shard_jax_batch(jnp.arange(10), 4)


def test_mean_pmap_gradients_on_host_averages_replica_axis():
    gradients = {
        "weight": jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 2, 3),
        "bias": jnp.arange(4 * 2, dtype=jnp.bfloat16).reshape(4, 2),
    }

    averaged = trainer._mean_pmap_gradients_on_host(gradients)

    assert np.allclose(averaged["weight"], np.asarray(gradients["weight"]).mean(axis=0))
    assert np.allclose(
        np.asarray(averaged["bias"], dtype=np.float32),
        np.asarray(gradients["bias"], dtype=np.float32).mean(axis=0),
        atol=1e-2,
    )


def test_mean_pmap_gradients_on_device_averages_replica_axis():
    gradients = {
        "weight": jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 2, 3),
        "bias": jnp.arange(4 * 2, dtype=jnp.bfloat16).reshape(4, 2),
    }

    averaged = trainer._mean_pmap_gradients_on_device(gradients, jax.local_devices()[0])

    assert np.allclose(np.asarray(averaged["weight"]), np.asarray(gradients["weight"]).mean(axis=0))
    assert np.allclose(
        np.asarray(averaged["bias"], dtype=np.float32),
        np.asarray(gradients["bias"], dtype=np.float32).mean(axis=0),
        atol=1e-2,
    )
    assert np.isclose(float(trainer._jax_tree_l2_norm_device(averaged)), trainer._jax_tree_l2_norm(averaged))


def test_jax_tree_l2_norm_avoids_whole_leaf_square(monkeypatch):
    monkeypatch.setattr(
        np,
        "square",
        lambda *_args, **_kwargs: pytest.fail("whole-leaf square must not be used"),
    )

    norm = trainer._jax_tree_l2_norm(
        {"a": jnp.asarray([3.0, 4.0]), "b": jnp.asarray([12.0])}
    )

    assert norm == pytest.approx(13.0)


def test_jax_kl_defaults_to_fixed_beta_and_adapts_only_when_enabled():
    transition_kl = jnp.asarray([1.0, 3.0, 2.0, 4.0])
    entropy = jnp.asarray([0.0, 1.0])

    fixed_penalty, raw_kl, fixed_beta = state_adaptive_kl_penalty(
        transition_kl,
        entropy,
        group_size=2,
        beta_base=0.1,
        adapt_kl_beta=False,
        uncertainty_scale=2.0,
    )
    adaptive_penalty, _, adaptive_beta = state_adaptive_kl_penalty(
        transition_kl,
        entropy,
        group_size=2,
        beta_base=0.1,
        adapt_kl_beta=True,
        uncertainty_scale=2.0,
    )

    assert np.isclose(raw_kl, 2.5)
    assert np.isclose(fixed_beta, 0.1)
    assert np.isclose(fixed_penalty, 0.25)
    assert np.isclose(adaptive_beta, 0.2)
    assert np.isclose(adaptive_penalty, 0.55)


def test_orbax_compatibility_installs_missing_record_scalar(monkeypatch):
    monkeypatch.delattr(jax.monitoring, "record_scalar", raising=False)
    _ensure_orbax_jax_compatibility()
    assert callable(jax.monitoring.record_scalar)


def test_jax_actor_optimizer_updates_backend_parameters():
    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    policy.init_actor_optimizer(learning_rate=1e-2, optimizer="adafactor")
    before = np.asarray(policy.actor_state.to_pure_dict()["backend"]["scale"]).copy()

    def loss_fn(actor_state):
        actor = nnx.merge(policy.actor_graphdef, actor_state)
        velocity = actor.predict_velocity(
            None,
            jnp.ones((1, 3, 2), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )
        return jnp.square(velocity).mean()

    grads = jax.grad(loss_fn)(policy.actor_state)
    policy.apply_actor_gradients(grads)
    after = np.asarray(policy.actor_state.to_pure_dict()["backend"]["scale"])

    assert not np.allclose(before, after)


def test_model_dim_flow_keeps_padding_latent_but_exposes_only_environment_actions():
    class FakeBuilder:
        def action_chunks_to_flow(self, batch):
            return batch.action_chunks

        def flat_actions_to_environment(self, flat_actions, *, model_states=None):
            del model_states
            return flat_actions + 1.0

    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(
        backend,
        environment_action_dim=2,
        flow_action_dim="model",
        num_steps=2,
        residual_hidden_dim=8,
        condition_builder=FakeBuilder(),
    )
    batch = type("Batch", (), {"action_chunks": torch.ones(2, 3, 2)})()

    flow_endpoint = policy.action_chunks_to_flow(batch)
    environment_endpoint = policy.flat_actions_to_environment(
        torch.arange(24, dtype=torch.float32).reshape(2, 12)
    )

    assert policy.environment_action_dim == 2
    assert policy.flow_action_dim == 4
    assert policy.action_dim == 12
    assert tuple(policy.log_std.shape) == (12,)
    assert tuple(flow_endpoint.shape) == (2, 3, 4)
    assert torch.allclose(flow_endpoint[..., :2], torch.ones(2, 3, 2))
    assert torch.count_nonzero(flow_endpoint[..., 2:]) == 0
    expected_environment = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)[..., :2]
    assert torch.allclose(environment_endpoint.reshape(2, 3, 2), expected_environment + 1.0)


def test_model_dim_flow_padding_changes_executed_velocity_when_backend_couples_dimensions():
    class CoupledBackend(nnx.Module):
        def __init__(self):
            self.action_horizon = 1
            self.action_dim = 4

        def predict_velocity(self, observation, noisy_actions, timestep, *, train=False):
            del observation, timestep, train
            coupled = noisy_actions.sum(axis=-1, keepdims=True)
            return jnp.broadcast_to(coupled, noisy_actions.shape)

    policy = PI05JaxFlowPolicy(
        CoupledBackend(),
        environment_action_dim=2,
        flow_action_dim="model",
        num_steps=2,
        residual_hidden_dim=8,
    )
    actor = nnx.merge(policy.actor_graphdef, policy.actor_state)
    without_padding_noise = actor.predict_velocity(
        None,
        jnp.asarray([[[1.0, 2.0, 0.0, 0.0]]]),
        jnp.asarray([0.5]),
    ).reshape(1, 1, 4)
    with_padding_noise = actor.predict_velocity(
        None,
        jnp.asarray([[[1.0, 2.0, 3.0, 4.0]]]),
        jnp.asarray([0.5]),
    ).reshape(1, 1, 4)

    assert np.allclose(without_padding_noise[..., :2], 3.0)
    assert np.allclose(with_padding_noise[..., :2], 10.0)


@pytest.mark.parametrize("sde_mode", ["gaussian_adapter", "ogpo_corrected"])
def test_joint_jax_inference_rollout_matches_stepwise_euler(sde_mode):
    policy = PI05JaxFlowPolicy(
        TrainableFakeJaxBackend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        num_steps=3,
        stochastic_variance=0.01,
        sde_mode=sde_mode,
        residual_hidden_dim=8,
    )
    observation = openpi_model.Observation(
        state=jnp.zeros((1, 5), dtype=jnp.float32),
        images={"base": jnp.zeros((1, 8, 8, 3), dtype=jnp.float32)},
        image_masks={"base": jnp.ones((1,), dtype=jnp.bool_)},
    )
    noise = jnp.linspace(-1.0, 1.0, 6, dtype=jnp.float32).reshape(1, 3, 2)
    actor = nnx.merge(policy.actor_graphdef, policy.actor_state)

    expected = noise
    dt = -1.0 / policy.num_steps
    for step_index in range(policy.num_steps):
        time = jnp.asarray(1.0 + dt * step_index, dtype=jnp.float32)
        time_batch = jnp.broadcast_to(time, (1,))
        velocity = actor.predict_velocity(
            observation,
            expected,
            time_batch,
        ).reshape(expected.shape)
        if sde_mode == "ogpo_corrected":
            sigma_squared = jnp.exp(2.0 * actor.log_std.value).reshape(expected.shape)
            velocity = velocity + 0.5 * sigma_squared * (
                (1.0 - time) * velocity + expected
            )
        expected = expected + dt * velocity

    policy.prepare_inference()
    actual = policy.sample_actions_jax(observation, noise=noise)

    assert np.allclose(actual, expected, atol=1e-6)


def test_native_ode_inference_does_not_apply_training_sde_drift_correction():
    policy = PI05JaxFlowPolicy(
        TrainableFakeJaxBackend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        num_steps=3,
        stochastic_variance=0.01,
        sde_mode="ogpo_corrected",
        inference_dynamics="native_ode",
        residual_hidden_dim=8,
    )
    observation = openpi_model.Observation(
        state=jnp.zeros((1, 5), dtype=jnp.float32),
        images={"base": jnp.zeros((1, 8, 8, 3), dtype=jnp.float32)},
        image_masks={"base": jnp.ones((1,), dtype=jnp.bool_)},
    )
    noise = jnp.linspace(-1.0, 1.0, 6, dtype=jnp.float32).reshape(1, 3, 2)
    actor = nnx.merge(policy.actor_graphdef, policy.actor_state)

    expected = noise
    dt = -1.0 / policy.num_steps
    for step_index in range(policy.num_steps):
        time = jnp.asarray(1.0 + dt * step_index, dtype=jnp.float32)
        velocity = actor.predict_velocity(
            observation,
            expected,
            jnp.broadcast_to(time, (1,)),
        ).reshape(expected.shape)
        expected = expected + dt * velocity

    policy.prepare_inference()
    actual = policy.sample_actions_jax(observation, noise=noise)

    assert np.allclose(actual, expected, atol=1e-6)


def test_native_ode_torch_compatibility_path_uses_the_same_plain_euler_step():
    policy = PI05JaxFlowPolicy(
        FakeJaxBackend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        num_steps=3,
        stochastic_variance=0.01,
        sde_mode="ogpo_corrected",
        inference_dynamics="native_ode",
        residual_hidden_dim=8,
    )
    condition = _condition(1)
    x_t = torch.linspace(-1.0, 1.0, 6).reshape(1, 6)
    timestep = torch.ones(1, 1)

    velocity = policy.predict_velocity(x_t, condition, timestep)
    expected = policy.flow_spec.euler_step(x_t, velocity)
    actual = policy.transition_mean(x_t, condition, timestep)

    assert torch.allclose(actual, expected)


def test_legacy_22d_checkpoint_adapter_can_use_32d_native_ode_at_inference():
    policy = PI05JaxFlowPolicy(
        TrainableFakeJaxBackend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        flow_action_dim=2,
        inference_flow_action_dim="model",
        num_steps=3,
        sde_mode="ogpo_corrected",
        inference_dynamics="native_ode",
        residual_hidden_dim=8,
    )
    observation = openpi_model.Observation(
        state=jnp.zeros((1, 5), dtype=jnp.float32),
        images={"base": jnp.zeros((1, 8, 8, 3), dtype=jnp.float32)},
        image_masks={"base": jnp.ones((1,), dtype=jnp.bool_)},
    )
    noise = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)

    policy.prepare_inference()
    actions = policy.sample_actions_jax(observation, noise=noise)

    assert policy.flow_action_dim == 2
    assert policy.inference_flow_action_dim == 4
    assert tuple(policy.log_std.shape) == (6,)
    assert actions.shape == (1, 3, 4)
    assert np.isfinite(actions).all()


def test_ogpo_inference_uses_joint_jax_path_without_torch_bridge(monkeypatch):
    flow_policy = PI05JaxFlowPolicy(
        TrainableFakeJaxBackend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        num_steps=3,
        stochastic_variance=0.01,
        sde_mode="ogpo_corrected",
        residual_hidden_dim=8,
    )
    flow_policy.prepare_inference()
    monkeypatch.setattr(
        inference_policy_module,
        "_to_batched_torch",
        lambda *_args, **_kwargs: pytest.fail("JAX inference entered the Torch bridge"),
    )
    policy = PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=lambda raw: {
            "state": raw["state"],
            "image": {"base": raw["base"]},
            "image_mask": {"base": np.asarray(True)},
        },
        output_transform=lambda data: {"actions": data["actions"][:, :2]},
        observation_type=openpi_model.Observation,
    )

    result = policy.infer(
        {
            "state": np.zeros(5, dtype=np.float32),
            "base": np.zeros((8, 8, 3), dtype=np.float32),
        },
        noise=np.arange(12, dtype=np.float32).reshape(3, 4),
    )

    assert result["actions"].shape == (3, 2)
    assert np.isfinite(result["actions"]).all()
    assert result["policy_timing"]["infer_ms"] >= 0.0


def test_bf16_value_anchor_changes_value_without_changing_current_gradient():
    old = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
    current = jnp.asarray([3.0, 4.0], dtype=jnp.float32)

    anchored = trainer._conditionally_anchor_current_to_old_value(
        current,
        old,
        1.0,
    )
    unanchored = trainer._conditionally_anchor_current_to_old_value(
        current,
        old,
        0.0,
    )
    anchored_grad = jax.grad(
        lambda value: trainer._conditionally_anchor_current_to_old_value(
            value,
            old,
            1.0,
        ).sum()
    )(current)

    assert np.allclose(anchored, old)
    assert np.allclose(unanchored, current)
    assert np.allclose(anchored_grad, jnp.ones_like(current))


def test_jax_residual_is_zero_initialized():
    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    actor = nnx.merge(policy.actor_graphdef, policy.actor_state)

    residual = actor.residual(
        jnp.ones((1, 3, 2), dtype=jnp.float32),
        jnp.ones((1, 3, 2), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )

    assert np.allclose(residual, 0.0)


def test_jax_adapter_sync_old_policy_ema_updates_actor_state():
    backend = FakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    old_policy = policy.clone_adapter(trainable=True)
    policy.init_actor_optimizer(learning_rate=1e-3)
    old_policy.init_actor_optimizer(learning_rate=1e-3)

    class DummyState:
        pass

    state = DummyState()
    state.policy = policy
    state.old_policy = old_policy

    old_before = old_policy.adapter_state_dict()["log_std"].clone()
    actor = trainer.nnx.merge(policy.actor_graphdef, policy.actor_state)
    actor.log_std.value = actor.log_std.value + 1.0
    policy.actor_graphdef, policy.actor_state = trainer.nnx.split(actor)
    policy._sync_torch_adapter_from_jax()
    policy_now = policy.adapter_state_dict()["log_std"].clone()

    trainer.sync_old_policy(state, ema=0.5)

    old_after = old_policy.adapter_state_dict()["log_std"]
    assert torch.allclose(old_after, 0.5 * old_before + 0.5 * policy_now)


def test_jax_adapter_hard_sync_updates_full_backend_state():
    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    old_policy = policy.clone_adapter()
    actor = nnx.merge(policy.actor_graphdef, policy.actor_state)
    actor.backend.scale.value = actor.backend.scale.value + 0.75
    policy.actor_graphdef, policy.actor_state = nnx.split(actor)

    class DummyState:
        pass

    state = DummyState()
    state.policy = policy
    state.old_policy = old_policy
    trainer.sync_old_policy(state, ema=0.0)

    current_scale = policy.actor_state.to_pure_dict()["backend"]["scale"]
    old_scale = old_policy.actor_state.to_pure_dict()["backend"]["scale"]
    assert np.allclose(current_scale, old_scale)


def test_jax_adapter_checkpoint_roundtrip_preserves_adapter_state(tmp_path):
    backend = FakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    state_dict = policy.adapter_state_dict()
    actor = trainer.nnx.merge(policy.actor_graphdef, policy.actor_state)
    actor.log_std.value = actor.log_std.value + 0.3
    policy.actor_graphdef, policy.actor_state = trainer.nnx.split(actor)
    policy._sync_torch_adapter_from_jax()

    checkpoint = tmp_path / "jax_adapter.pt"
    torch.save({"policy": {"format": "pi05_residual_adapter", "state": policy.adapter_state_dict()}}, checkpoint)

    restored = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    payload = torch.load(checkpoint, weights_only=False)
    trainer._load_policy_checkpoint_state(restored, payload["policy"])

    assert not torch.allclose(state_dict["log_std"], restored.adapter_state_dict()["log_std"])
    assert torch.allclose(restored.adapter_state_dict()["log_std"], policy.adapter_state_dict()["log_std"])


def test_jax_full_training_checkpoint_restores_actor_old_policy_and_optimizer(tmp_path):
    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    old_policy = policy.clone_adapter()
    policy.init_actor_optimizer(learning_rate=1e-3, optimizer="adafactor")

    actor = nnx.merge(policy.actor_graphdef, policy.actor_state)
    actor.backend.scale.value = actor.backend.scale.value + 0.5
    policy.actor_graphdef, policy.actor_state = nnx.split(actor)
    old_actor = nnx.merge(old_policy.actor_graphdef, old_policy.actor_state)
    old_actor.backend.scale.value = old_actor.backend.scale.value + 0.2
    old_policy.actor_graphdef, old_policy.actor_state = nnx.split(old_actor)
    expected_policy = policy.actor_state.to_pure_dict()
    expected_old = old_policy.actor_state.to_pure_dict()
    expected_opt = policy.actor_opt_state

    checkpoint_dir = tmp_path / "jax_state"
    policy.save_training_checkpoint(checkpoint_dir, old_policy=old_policy)

    restored_backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    restored = PI05JaxFlowPolicy(
        restored_backend,
        environment_action_dim=2,
        num_steps=2,
        residual_hidden_dim=8,
    )
    restored_old = restored.clone_adapter()
    restored.init_actor_optimizer(learning_rate=1e-3, optimizer="adafactor")
    restored.restore_training_checkpoint(
        checkpoint_dir,
        old_policy=restored_old,
        restore_optimizer=True,
    )

    assert _tree_allclose(restored.actor_state.to_pure_dict(), expected_policy)
    assert _tree_allclose(restored_old.actor_state.to_pure_dict(), expected_old)
    assert _tree_allclose(restored.actor_opt_state, expected_opt)


def test_jax_checkpoint_can_restore_weights_without_adafactor_state(tmp_path):
    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(
        backend,
        environment_action_dim=2,
        num_steps=2,
        residual_hidden_dim=8,
    )
    policy.init_actor_optimizer(learning_rate=1e-2, optimizer="adafactor")

    def loss_fn(actor_state):
        actor = nnx.merge(policy.actor_graphdef, actor_state)
        return jnp.square(
            actor.predict_velocity(
                None,
                jnp.ones((1, 3, 2), dtype=jnp.float32),
                jnp.zeros((1,), dtype=jnp.float32),
            )
        ).mean()

    policy.apply_actor_gradients(jax.grad(loss_fn)(policy.actor_state))
    trained_optimizer_state = policy.actor_opt_state
    checkpoint_dir = tmp_path / "jax_state_reset_optimizer"
    policy.save_training_checkpoint(checkpoint_dir)

    restored = PI05JaxFlowPolicy(
        TrainableFakeJaxBackend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        num_steps=2,
        residual_hidden_dim=8,
    )
    restored.init_actor_optimizer(learning_rate=1e-2, optimizer="adafactor")
    fresh_optimizer_state = restored.actor_opt_state
    restored.restore_training_checkpoint(
        checkpoint_dir,
        restore_optimizer=False,
    )

    assert _tree_allclose(restored.actor_state, policy.actor_state)
    assert _tree_allclose(restored.actor_opt_state, fresh_optimizer_state)
    assert not _tree_allclose(restored.actor_opt_state, trained_optimizer_state)


def test_trainer_checkpoint_uses_full_jax_sidecar(monkeypatch, tmp_path):
    batch = make_synthetic_replay(num_samples=4, generated_horizon=3, executed_horizon=2, action_dim=2)

    def fake_loader(**kwargs):
        def builder(replay_batch, *, next_observation=False, device="cpu"):
            state = replay_batch.next_observations if next_observation else replay_batch.observations
            return PI05FlowCondition(
                FakeObservation(
                    state=state.to(device),
                    images={"base": torch.zeros(state.shape[0], 3, 8, 8, device=device)},
                    image_masks={"base": torch.ones(state.shape[0], dtype=torch.bool, device=device)},
                )
            )

        return PI05JaxFlowPolicy(
            TrainableFakeJaxBackend(action_horizon=3, action_dim=4),
            environment_action_dim=kwargs["environment_action_dim"],
            num_steps=kwargs["num_steps"],
            sde_mode=kwargs["sde_mode"],
            residual_hidden_dim=8,
            condition_builder=builder,
        )

    monkeypatch.setattr(trainer, "load_pi05_jax_flow_policy", fake_loader)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {"group_size": 1, "hidden_dim": 8, "optimizer": "adafactor"},
        "flow": {
            "adapter": "pi05_jax",
            "checkpoint_dir": "/tmp/fake-pi05-jax",
            "train_config": "fake",
            "image_mapping": {"base": "front"},
            "num_steps": 2,
            "sde_mode": "gaussian_adapter",
        },
    }
    state = trainer.build_train_state(cfg, batch)
    actor = nnx.merge(state.policy.actor_graphdef, state.policy.actor_state)
    actor.backend.scale.value = actor.backend.scale.value + 0.4
    state.policy.actor_graphdef, state.policy.actor_state = nnx.split(actor)
    old_actor = nnx.merge(state.old_policy.actor_graphdef, state.old_policy.actor_state)
    old_actor.backend.scale.value = old_actor.backend.scale.value + 0.2
    state.old_policy.actor_graphdef, state.old_policy.actor_state = nnx.split(old_actor)
    expected_policy = state.policy.actor_state.to_pure_dict()
    expected_old = state.old_policy.actor_state.to_pure_dict()
    checkpoint = tmp_path / "ogpo.pt"

    trainer.save_checkpoint(state, cfg, checkpoint)

    payload = torch.load(checkpoint, weights_only=False)
    assert payload["policy"]["format"] == "pi05_jax_full_finetune"
    assert (tmp_path / payload["policy"]["jax_sidecar"]).is_dir()
    restored = trainer.build_train_state(cfg, batch)
    trainer.load_checkpoint(checkpoint, restored)
    assert _tree_allclose(restored.policy.actor_state.to_pure_dict(), expected_policy)
    assert _tree_allclose(restored.old_policy.actor_state.to_pure_dict(), expected_old)

    state.old_policy.actor_state = jax.tree.map(
        lambda value: value,
        state.policy.actor_state,
    )
    cfg["actor"]["checkpoint_old_policy"] = False
    compact_checkpoint = tmp_path / "ogpo_compact.pt"
    trainer.save_checkpoint(state, cfg, compact_checkpoint)
    compact_payload = torch.load(compact_checkpoint, weights_only=False)
    assert compact_payload["policy"]["jax_sidecar_has_old_policy"] is False

    compact_restored = trainer.build_train_state(cfg, batch)
    trainer.load_checkpoint(compact_checkpoint, compact_restored)
    assert _tree_allclose(
        compact_restored.policy.actor_state.to_pure_dict(),
        expected_policy,
    )
    assert _tree_allclose(
        compact_restored.old_policy.actor_state.to_pure_dict(),
        expected_policy,
    )


def test_torch_facing_jax_policy_uses_current_actor_state():
    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)
    policy = PI05JaxFlowPolicy(backend, environment_action_dim=2, num_steps=2, residual_hidden_dim=8)
    condition = _condition(1)
    x_t = torch.ones(1, 6)
    timestep = torch.zeros(1, 1)
    before = policy.predict_velocity(x_t, condition, timestep)

    actor = nnx.merge(policy.actor_graphdef, policy.actor_state)
    actor.backend.scale.value = actor.backend.scale.value + 0.5
    policy.actor_graphdef, policy.actor_state = nnx.split(actor)
    after = policy.predict_velocity(x_t, condition, timestep)

    assert not torch.allclose(before, after)


def test_jax_flash_and_full_updates_return_finite_metrics(monkeypatch):
    batch = make_synthetic_replay(num_samples=6, generated_horizon=3, executed_horizon=2, action_dim=2)
    backend = FakeJaxBackend(action_horizon=3, action_dim=4)

    def fake_loader(**kwargs):
        class Builder:
            def __call__(self, replay_batch, *, next_observation=False, device="cpu"):
                state = replay_batch.next_observations if next_observation else replay_batch.observations
                obs = FakeObservation(
                    state=state.to(device),
                    images={"base": torch.zeros(state.shape[0], 3, 8, 8, device=device)},
                    image_masks={"base": torch.ones(state.shape[0], dtype=torch.bool, device=device)},
                )
                return PI05FlowCondition(obs)

            def action_chunks_to_flow(self, replay_batch):
                return replay_batch.action_chunks

            def flat_actions_to_environment(self, flat_actions, *, model_states=None):
                del model_states
                return flat_actions

        policy = PI05JaxFlowPolicy(
            backend,
            environment_action_dim=kwargs["environment_action_dim"],
            flow_action_dim=kwargs.get("flow_action_dim"),
            num_steps=kwargs["num_steps"],
            sde_mode=kwargs["sde_mode"],
            residual_hidden_dim=8,
            condition_builder=Builder(),
        )
        policy.init_actor_optimizer(learning_rate=1e-4)
        return policy

    monkeypatch.setattr(trainer, "load_pi05_jax_flow_policy", fake_loader)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {
            "group_size": 2,
            "hidden_dim": 8,
            "actor_epochs_per_rollout": 2,
            "normalize_logprob_by_action_dim": True,
            "ppo_action_horizon": 2,
            "max_grad_norm": 200.0,
        },
        "flow": {
            "adapter": "pi05_jax",
            "checkpoint_dir": "/tmp/fake-pi05-jax",
            "train_config": "fake",
            "image_mapping": {"base": "front"},
            "latent_action_dim": "model",
            "num_steps": 2,
            "sde_mode": "gaussian_adapter",
            "selected_timestep": 1,
        },
    }

    state = trainer.build_train_state(cfg, batch)
    trainer.critic_update(state, batch, cfg)
    flash_metrics = trainer.flash_actor_update(state, batch, cfg)
    assert flash_metrics["ppo_epoch_1_importance_ratio_mean"] == pytest.approx(
        1.0,
        abs=1e-5,
    )
    assert flash_metrics["attempted_actor_epochs"] == 2.0
    assert flash_metrics["ppo_event_dim"] == 4.0
    assert np.isfinite(flash_metrics["reference_kl"])
    assert flash_metrics["normalize_logprob_by_action_dim"] == 1.0
    assert flash_metrics["logprob_normalizer"] == 4.0
    cfg["actor"]["actor_epochs_per_rollout"] = 1
    full_metrics = trainer.full_actor_update(state, batch, cfg)
    streaming_cfg = copy.deepcopy(cfg)
    streaming_cfg["actor"].update(
        {
            "full_chain_streaming_backward": True,
            "full_ratio_mode": "ais_joint",
            "gradient_microbatch_size": 1,
            "normalize_logprob_by_denoising_steps": True,
        }
    )
    streaming_cfg["regularization"] = {
        "beta_kl": 0.0,
        "lambda_fm": 0.0,
        "lambda_success": 0.0,
    }
    streaming_metrics = trainer.full_actor_update(state, batch, streaming_cfg)

    assert torch.isfinite(torch.tensor(flash_metrics["actor_loss"]))
    assert torch.isfinite(torch.tensor(full_metrics["actor_loss"]))
    assert torch.isfinite(torch.tensor(streaming_metrics["actor_loss"]))
    assert streaming_metrics["full_chain_streaming_backward"] == 1.0
    assert np.isfinite(flash_metrics["ppo_epoch_2_importance_ratio_mean"])
    assert flash_metrics["reference_kl_beta"] == pytest.approx(0.01)
    assert full_metrics["reference_kl_beta"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    (
        "lambda_fm",
        "lambda_success",
        "regularization_batch",
        "reject_update",
        "actor_step",
        "expected_update",
    ),
    [
        (1.0, 0.0, "fm", False, 1, True),
        (0.0, 1.0, "success", False, 0, True),
        (0.0, 1.0, "success", True, 0, False),
        (0.0, 1.0, "success", False, 1, False),
    ],
)
def test_jax_behavior_regularization_updates_backend_when_ppo_advantage_is_zero(
    monkeypatch,
    lambda_fm,
    lambda_success,
    regularization_batch,
    reject_update,
    actor_step,
    expected_update,
):
    batch = make_synthetic_replay(num_samples=4, generated_horizon=3, executed_horizon=2, action_dim=2)
    backend = TrainableFakeJaxBackend(action_horizon=3, action_dim=4)

    def fake_loader(**kwargs):
        def builder(replay_batch, *, next_observation=False, device="cpu"):
            state = replay_batch.next_observations if next_observation else replay_batch.observations
            return PI05FlowCondition(
                FakeObservation(
                    state=state.to(device),
                    images={"base": torch.zeros(state.shape[0], 3, 8, 8, device=device)},
                    image_masks={"base": torch.ones(state.shape[0], dtype=torch.bool, device=device)},
                )
            )

        policy = PI05JaxFlowPolicy(
            backend,
            environment_action_dim=kwargs["environment_action_dim"],
            num_steps=kwargs["num_steps"],
            sde_mode=kwargs["sde_mode"],
            residual_hidden_dim=8,
            condition_builder=builder,
        )
        policy.init_actor_optimizer(learning_rate=1e-2, optimizer="adafactor")
        return policy

    monkeypatch.setattr(trainer, "load_pi05_jax_flow_policy", fake_loader)

    def zero_advantages(state, observations, candidates, replay_batch, config):
        del state, observations, replay_batch, config
        return torch.zeros(candidates.shape[:2]), {}

    monkeypatch.setattr(trainer, "conservative_advantages_for_candidates", zero_advantages)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {
            "group_size": 1,
            "candidate_group_size": 2,
            "rollout_state_microbatch_size": 2,
            "gradient_microbatch_size": 1,
            "kl_eval_microbatch_size": 2,
            "hidden_dim": 8,
            "actor_epochs_per_rollout": 1,
            "learning_rate": 1e-2,
            "optimizer": "adafactor",
            "reject_update_on_kl": reject_update,
            "max_policy_reference_kl": -1.0 if reject_update else float("inf"),
        },
        "flow": {
            "adapter": "pi05_jax",
            "checkpoint_dir": "/tmp/fake-pi05-jax",
            "train_config": "fake",
            "image_mapping": {"base": "front"},
            "num_steps": 2,
            "sde_mode": "gaussian_adapter",
            "selected_timestep": 1,
        },
        "regularization": {
            "beta_kl": 0.0,
            "lambda_fm": lambda_fm,
            "lambda_success": lambda_success,
            "success_update_period": 4,
            "lambda_smooth": 0.0,
        },
    }
    state = trainer.build_train_state(cfg, batch)
    actor_state_before = state.policy.actor_state
    optimizer_state_before = state.policy.actor_opt_state
    before = np.asarray(actor_state_before.to_pure_dict()["backend"]["scale"]).copy()

    kwargs = {f"{regularization_batch}_batch": batch}
    metrics = trainer.flash_actor_update(
        state,
        batch,
        cfg,
        actor_step=actor_step,
        **kwargs,
    )

    after = np.asarray(state.policy.actor_state.to_pure_dict()["backend"]["scale"])
    assert metrics["candidate_group_size"] == 2
    assert metrics["rollout_state_microbatch_size"] == 2
    assert metrics["gradient_microbatch_size"] == 1
    assert metrics["kl_eval_microbatch_size"] == 2
    assert metrics["success_update_period"] == 4
    if reject_update:
        assert np.allclose(before, after)
        assert _tree_allclose(state.policy.actor_state, actor_state_before)
        assert _tree_allclose(state.policy.actor_opt_state, optimizer_state_before)
        assert metrics["actor_update_rejected"] == 1.0
        assert metrics["jax_rejection_cleanup_applied"] == 1.0
    elif expected_update:
        assert not np.allclose(before, after)
        assert metrics["actor_update_rejected"] == 0.0
        assert metrics["jax_rejection_cleanup_applied"] == 0.0
    else:
        assert np.allclose(before, after)
        assert metrics["actor_update_rejected"] == 0.0
        assert metrics["jax_rejection_cleanup_applied"] == 0.0
    assert metrics["success_update_applied"] == float(actor_step % 4 == 0)


def test_numpy_reference_kl_can_limit_event_to_executed_prefix():
    mean_reference = np.zeros((1, 6), dtype=np.float32)
    mean_current = mean_reference.copy()
    mean_current[:, 2:] = 1.0
    log_std = np.zeros_like(mean_reference)

    full_kl = trainer._numpy_gaussian_kl_diag(
        mean_current,
        log_std,
        mean_reference,
        log_std,
    )
    prefix_kl = trainer._numpy_gaussian_kl_diag(
        mean_current,
        log_std,
        mean_reference,
        log_std,
        event_dim=2,
    )

    assert full_kl.item() == pytest.approx(2.0)
    assert prefix_kl.item() == pytest.approx(0.0)
