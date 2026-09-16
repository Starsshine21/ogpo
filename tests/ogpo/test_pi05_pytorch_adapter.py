from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from ogpo.inference_policy import PI05OGPOInferencePolicy
from ogpo.losses import flow_matching_anchor_loss
from ogpo.multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic
from ogpo.pi05_pytorch_adapter import (
    PI05FlowCondition,
    PI05PytorchFlowPolicy,
    PI05ReplayConditionBuilder,
)
from ogpo.replay import make_synthetic_replay
from ogpo import trainer


@dataclass(frozen=True)
class FakeObservation:
    state: torch.Tensor
    images: dict[str, torch.Tensor]

    @classmethod
    def from_dict(cls, data):
        return cls(state=data["state"], images=data["image"])


class FakePI05Backend(nn.Module):
    def __init__(self, action_horizon: int, action_dim: int):
        super().__init__()
        self.config = type("Config", (), {"action_horizon": action_horizon, "action_dim": action_dim})()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.seen_shape = None

    def predict_velocity(self, observation, noisy_actions, timestep, *, train=False):
        self.seen_shape = noisy_actions.shape
        return noisy_actions * self.scale + timestep[:, None, None]


def _condition(batch_size: int) -> PI05FlowCondition:
    observation = FakeObservation(
        state=torch.randn(batch_size, 5),
        images={"base": torch.randn(batch_size, 3, 8, 8)},
    )
    return PI05FlowCondition(observation)


def test_pi05_adapter_uses_real_backend_shape_and_repeats_condition_groups():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    policy = PI05PytorchFlowPolicy(
        backend,
        environment_action_dim=2,
        num_steps=2,
        residual_hidden_dim=16,
    )

    rollout = policy.rollout(_condition(2), group_size=3)

    assert rollout.endpoint.shape == (6, 3 * 2)
    assert backend.seen_shape == (6, 3, 4)
    assert not backend.scale.requires_grad


def test_pi05_adapter_clone_starts_with_unit_ratio_and_only_trains_residual():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    policy = PI05PytorchFlowPolicy(
        backend,
        environment_action_dim=2,
        num_steps=2,
        residual_hidden_dim=16,
    )
    old_policy = policy.clone_adapter()
    condition = _condition(2)
    rollout = old_policy.rollout(condition)

    new_log_prob = policy.log_prob(
        rollout.next_states[:, 0], rollout.states[:, 0], condition, rollout.timesteps[:, 0]
    )
    assert torch.allclose(torch.exp(new_log_prob - rollout.log_probs[:, 0]), torch.ones(2), atol=1e-5)

    endpoint = torch.randn(2, 6)
    loss = flow_matching_anchor_loss(policy, condition, endpoint).loss
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.residual.parameters())
    assert backend.scale.grad is None


def test_pi05_adapter_clone_preserves_corrected_sde_mode():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    policy = PI05PytorchFlowPolicy(
        backend,
        environment_action_dim=2,
        num_steps=2,
        sde_mode="ogpo_corrected",
    )

    clone = policy.clone_adapter()

    assert clone.sde_mode == "ogpo_corrected"


def test_pi05_clean_reference_can_disable_residual_controller():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    policy = PI05PytorchFlowPolicy(
        backend,
        environment_action_dim=2,
        num_steps=2,
        residual_enabled=False,
    )
    condition = _condition(2)
    x_t = torch.randn(2, 6)
    timestep = torch.full((2, 1), 0.5)
    # The adapter's model-space padding carries x_t's environment prefix.
    model_x = x_t.reshape(2, 3, 2).new_zeros(2, 3, 4)
    model_x[..., :2] = x_t.reshape(2, 3, 2)
    expected = backend.predict_velocity(condition.observation, model_x, timestep[:, 0])[
        ..., :2
    ].reshape(2, -1)
    actual = policy.predict_velocity(x_t, condition, timestep)
    assert torch.allclose(actual, expected)
    assert all(parameter.grad is None for parameter in policy.residual.parameters())
    assert not any(parameter.requires_grad for parameter in policy.residual.parameters())


def test_inference_native_ode_bypasses_training_sde_transition():
    backend = FakePI05Backend(action_horizon=2, action_dim=2)
    flow_policy = PI05PytorchFlowPolicy(
        backend,
        environment_action_dim=2,
        num_steps=2,
        sde_mode="ogpo_constant_corrected",
        constant_noise_std=0.005,
        learn_sde_std=False,
        residual_enabled=False,
    )

    def transform(raw):
        return {
            "state": np.asarray(raw["state"], dtype=np.float32),
            "image": {"base": np.asarray(raw["base"], dtype=np.float32)},
        }

    def output_transform(data):
        return {"actions": np.asarray(data["actions"])[..., :2]}

    inference = PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=transform,
        output_transform=output_transform,
        observation_type=FakeObservation,
        dynamics="native_ode",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("training SDE transition was used during native inference")

    flow_policy.transition_mean = forbidden
    result = inference.infer(
        {
            "state": np.zeros(5, dtype=np.float32),
            "base": np.zeros((3, 4, 4), dtype=np.float32),
        },
        noise=np.zeros((2, 2), dtype=np.float32),
    )
    assert result["actions"].shape == (2, 2)


def test_pi05_full_backend_training_uses_frozen_functional_reference():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    policy = PI05PytorchFlowPolicy(
        backend,
        environment_action_dim=2,
        num_steps=2,
        backend_train_mode="full",
    )
    reference = policy.clone_adapter()
    condition = _condition(2)
    x_t = torch.randn(2, 6)
    timestep = torch.full((2, 1), 0.5)
    reference_before = reference.predict_velocity(x_t, condition, timestep).detach().clone()

    policy.predict_velocity(x_t, condition, timestep).sum().backward()
    assert backend.scale.grad is not None
    with torch.no_grad():
        backend.scale.add_(1.0)

    reference_after = reference.predict_velocity(x_t, condition, timestep)
    current_after = policy.predict_velocity(x_t, condition, timestep)
    assert torch.allclose(reference_before, reference_after)
    assert not torch.allclose(reference_after, current_after)


def test_pi05_full_backend_state_roundtrip():
    policy = PI05PytorchFlowPolicy(
        FakePI05Backend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        backend_train_mode="full",
    )
    with torch.no_grad():
        policy.backend.scale.fill_(1.75)
    state = policy.adapter_state_dict()

    restored = PI05PytorchFlowPolicy(
        FakePI05Backend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        backend_train_mode="full",
    )
    restored.load_adapter_state_dict(state)

    assert torch.equal(restored.backend.scale, policy.backend.scale)


def test_pi05_functional_slow_snapshot_ema_and_roundtrip():
    policy = PI05PytorchFlowPolicy(
        FakePI05Backend(action_horizon=3, action_dim=4),
        environment_action_dim=2,
        backend_train_mode="full",
    )
    slow = policy.clone_adapter()
    assert slow._backend_parameter_snapshot is not None
    before = slow._backend_parameter_snapshot["scale"].detach().clone()
    with torch.no_grad():
        policy.backend.scale.fill_(1.75)
        policy.residual[-1].bias.fill_(0.5)

    slow.sync_adapter_snapshot_from(policy, ema=0.5)

    assert torch.allclose(
        slow._backend_parameter_snapshot["scale"],
        0.5 * before + 0.5 * policy.backend.scale,
    )
    snapshot_state = slow.adapter_state_dict()
    assert "backend.scale" in snapshot_state
    restored = policy.clone_adapter()
    restored.load_adapter_state_dict(snapshot_state)
    assert torch.allclose(
        restored._backend_parameter_snapshot["scale"],
        slow._backend_parameter_snapshot["scale"],
    )
    assert torch.allclose(restored.residual[-1].bias, slow.residual[-1].bias)


def test_replay_condition_builder_applies_transform_and_selects_next_images():
    batch = make_synthetic_replay(num_samples=3, generated_horizon=3, action_dim=2)
    images = {"front": torch.arange(3 * 2 * 2 * 3, dtype=torch.uint8).reshape(3, 2, 2, 3)}
    next_images = {"front": images["front"] + 1}
    batch = dataclass_replace(batch, images=images, next_images=next_images)

    class FakeModelObservation:
        @classmethod
        def from_dict(cls, data):
            return FakeObservation(state=data["state"], images=data["image"])

    prompts = []

    def transform(raw):
        prompts.append(str(raw["prompt"]))
        return {"state": raw["state"], "image": {"base": raw["base"]}}

    builder = PI05ReplayConditionBuilder(
        input_transform=transform,
        observation_type=FakeObservation,
        image_mapping={"base": "front"},
    )
    condition = builder(batch, next_observation=True, device="cpu")

    assert condition.batch_size == 3
    assert torch.equal(condition.observation.images["base"], next_images["front"])
    assert prompts == batch.languages


def test_replay_condition_builder_can_nest_mapped_images():
    batch = make_synthetic_replay(num_samples=1, generated_horizon=3, action_dim=2)
    images = {"front": torch.zeros(1, 2, 2, 3, dtype=torch.uint8)}
    batch = dataclass_replace(batch, images=images, next_images=images)

    def transform(raw):
        assert set(raw["images"]) == {"cam_high"}
        return {"state": raw["state"], "image": {"base": raw["images"]["cam_high"]}}

    builder = PI05ReplayConditionBuilder(
        input_transform=transform,
        observation_type=FakeObservation,
        image_mapping={"cam_high": "front"},
        image_container_key="images",
        transpose_images_to_chw=True,
    )

    condition = builder(batch)
    assert condition.batch_size == 1
    assert condition.observation.images["base"].shape == (1, 3, 2, 2)


def test_replay_action_normalization_roundtrips_through_checkpoint_transforms():
    batch = make_synthetic_replay(num_samples=3, generated_horizon=3, action_dim=2)
    images = {"front": torch.zeros(3, 2, 2, 3, dtype=torch.uint8)}
    batch = dataclass_replace(batch, images=images, next_images=images)

    def input_transform(raw):
        result = {"state": raw["state"], "image": {"base": raw["base"]}}
        if "actions" in raw:
            padded = np.zeros((3, 4), dtype=np.float32)
            padded[:, :2] = raw["actions"] * 2.0
            result["actions"] = padded
        return result

    def output_transform(data):
        return {"actions": data["actions"][:, :2] / 2.0}

    builder = PI05ReplayConditionBuilder(
        input_transform=input_transform,
        output_transform=output_transform,
        observation_type=FakeObservation,
        image_mapping={"base": "front"},
        model_action_dim=4,
        environment_action_dim=2,
    )
    normalized = builder.action_chunks_to_flow(batch)
    restored = builder.flat_actions_to_environment(normalized.reshape(3, -1))

    assert torch.allclose(normalized, batch.action_chunks * 2.0)
    assert torch.allclose(restored.reshape_as(batch.action_chunks), batch.action_chunks)


def dataclass_replace(instance, **changes):
    from dataclasses import replace

    return replace(instance, **changes)


def test_train_state_wires_pi05_adapter_without_copying_backend(monkeypatch, tmp_path):
    batch = make_synthetic_replay(
        num_samples=6,
        generated_horizon=3,
        action_dim=2,
        executed_horizon=2,
    )
    backend = FakePI05Backend(action_horizon=3, action_dim=4)

    def fake_loader(**kwargs):
        def builder(replay_batch, *, next_observation=False, device="cpu"):
            state = replay_batch.next_observations if next_observation else replay_batch.observations
            return PI05FlowCondition(FakeObservation(state=state.to(device), images={}))

        return PI05PytorchFlowPolicy(
            backend,
            environment_action_dim=kwargs["environment_action_dim"],
            num_steps=kwargs["num_steps"],
            sde_mode=kwargs["sde_mode"],
            residual_hidden_dim=8,
            condition_builder=builder,
        )

    monkeypatch.setattr(trainer, "load_pi05_pytorch_flow_policy", fake_loader)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 16, "num_layers": 1},
        "divl": {"num_atoms": 11, "v_min": -2.0, "v_max": 2.0},
        "actor": {"group_size": 2, "hidden_dim": 8},
        "flow": {
            "adapter": "pi05_pytorch",
            "checkpoint_dir": "/tmp/fake-pi05",
            "train_config": "fake",
            "image_mapping": {"base": "front"},
            "num_steps": 2,
            "sde_mode": "ogpo_corrected",
        },
    }

    state = trainer.build_train_state(cfg, batch)
    conversion_calls = {"replay_to_flow": 0, "flow_to_environment": 0}
    original_replay_to_flow = state.policy.action_chunks_to_flow
    original_flow_to_environment = state.old_policy.flat_actions_to_environment

    def replay_to_flow(replay_batch):
        conversion_calls["replay_to_flow"] += 1
        return original_replay_to_flow(replay_batch)

    def flow_to_environment(flat_actions, condition=None):
        conversion_calls["flow_to_environment"] += 1
        return original_flow_to_environment(flat_actions, condition)

    state.policy.action_chunks_to_flow = replay_to_flow
    state.old_policy.flat_actions_to_environment = flow_to_environment
    trainer.critic_update(state, batch, cfg)
    full_metrics = trainer.full_actor_update(state, batch, cfg)
    flash_metrics = trainer.flash_actor_update(state, batch, cfg)

    assert isinstance(state.policy, PI05PytorchFlowPolicy)
    assert state.policy.sde_mode == "ogpo_corrected"
    assert state.policy.backend is state.old_policy.backend is state.reference_policy.backend
    assert conversion_calls == {"replay_to_flow": 2, "flow_to_environment": 2}
    assert torch.isfinite(torch.tensor(full_metrics["actor_loss"]))
    assert torch.isfinite(torch.tensor(flash_metrics["actor_loss"]))

    checkpoint = tmp_path / "pi05_adapter.pt"
    trainer.save_checkpoint(state, cfg, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["policy"]["format"] == "pi05_residual_adapter"
    assert "backend.scale" not in payload["policy"]["state"]


def test_pi05_ogpo_inference_policy_returns_environment_action_chunk():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    flow_policy = PI05PytorchFlowPolicy(backend, environment_action_dim=2, num_steps=2)

    def input_transform(raw):
        return {"state": raw["state"], "image": {"base": raw["base"]}}

    def output_transform(data):
        return {"actions": data["actions"][:, :2]}

    policy = PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=input_transform,
        output_transform=output_transform,
        observation_type=FakeObservation,
    )
    result = policy.infer(
        {
            "state": np.zeros(5, dtype=np.float32),
            "base": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        noise=np.zeros((3, 2), dtype=np.float32),
    )

    assert result["actions"].shape == (3, 2)
    assert np.isfinite(result["actions"]).all()


def test_pi05_ogpo_inference_policy_uses_request_noise_seed():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    flow_policy = PI05PytorchFlowPolicy(backend, environment_action_dim=2, num_steps=2)
    policy = PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=lambda raw: {
            "state": raw["state"],
            "image": {"base": raw["base"]},
        },
        output_transform=lambda data: {"actions": data["actions"][:, :2]},
        observation_type=FakeObservation,
    )
    observation = {
        "state": np.zeros(5, dtype=np.float32),
        "base": np.zeros((8, 8, 3), dtype=np.uint8),
        "_policy_noise_seed": 1234,
    }

    first = policy.infer(observation)["actions"]
    second = policy.infer(observation)["actions"]

    assert np.array_equal(first, second)


def test_pi05_ogpo_inference_policy_uses_portable_request_noise():
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    flow_policy = PI05PytorchFlowPolicy(backend, environment_action_dim=2, num_steps=2)
    policy = PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=lambda raw: {
            "state": raw["state"],
            "image": {"base": raw["base"]},
        },
        output_transform=lambda data: {"actions": data["actions"][:, :2]},
        observation_type=FakeObservation,
    )
    observation = {
        "state": np.zeros(5, dtype=np.float32),
        "base": np.zeros((8, 8, 3), dtype=np.uint8),
        "_policy_noise_seed": 999,
        "_policy_noise": np.arange(12, dtype=np.float32).reshape(3, 4),
    }

    first = policy.infer(observation)["actions"]
    second = policy.infer(observation)["actions"]

    assert np.array_equal(first, second)


def test_pi05_ogpo_inference_reports_frozen_q_and_reference_action_divergence():
    class FixedCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, observations, action_chunks, execution_masks):
            assert execution_masks.tolist() == [[True, True, False]]
            score = action_chunks[:, :2].sum(dim=(1, 2)) + observations.sum(dim=1) * 0.0
            return torch.stack([score, score + 2.0]) + self.anchor

    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    flow_policy = PI05PytorchFlowPolicy(backend, environment_action_dim=2, num_steps=2)
    reference_policy = flow_policy.clone_adapter()
    with torch.no_grad():
        flow_policy.residual[-1].bias.fill_(0.5)

    def input_transform(raw):
        return {"state": raw["state"], "image": {"base": raw["base"]}}

    def output_transform(data):
        return {"actions": data["actions"][:, :2]}

    policy = PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=input_transform,
        output_transform=output_transform,
        observation_type=FakeObservation,
        reference_flow_policy=reference_policy,
        critic=FixedCritic(),
        executed_horizon=2,
    )

    result = policy.infer(
        {
            "state": np.zeros(5, dtype=np.float32),
            "base": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        noise=np.zeros((3, 2), dtype=np.float32),
    )

    assert np.isfinite(result["predicted_q"])
    assert result["policy_reference_action_divergence"] > 0.0


def test_pi05_ogpo_inference_scores_multimodal_critic_from_raw_observation():
    class OnlineEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(5, 8)

        def forward(self, batch, *, next_observation=False):
            assert not next_observation
            assert set(batch.images) == {"image_base"}
            assert batch.languages == ["click the mouse"]
            return self.projection(batch.proprioceptions)

    critic = MultiHeadUdivlCritic(
        OnlineEncoder(),
        MultiHeadUdivlCore(
            state_dim=8,
            action_dim=2,
            max_horizon=3,
            action_hidden_dim=8,
            head_hidden_dim=8,
            num_attention_heads=2,
            num_value_atoms=11,
            num_pairs=3,
        ),
    )
    backend = FakePI05Backend(action_horizon=3, action_dim=4)
    flow_policy = PI05PytorchFlowPolicy(backend, environment_action_dim=2, num_steps=2)

    policy = PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=lambda raw: {"state": raw["state"], "image": {"base": raw["base"]}},
        output_transform=lambda data: {"actions": data["actions"][:, :2]},
        observation_type=FakeObservation,
        critic=critic,
        critic_camera_keys=("image_base",),
        critic_online_camera_mapping={"image_base": "base"},
        default_language="click the mouse",
        executed_horizon=2,
    )
    result = policy.infer(
        {
            "state": np.zeros(5, dtype=np.float32),
            "base": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        noise=np.zeros((3, 2), dtype=np.float32),
    )

    assert np.isfinite(result["predicted_q"])
