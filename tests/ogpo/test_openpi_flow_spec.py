import pytest
import torch

from ogpo.losses import flow_matching_anchor_loss, weighted_flow_matching_loss
from ogpo.openpi_flow_spec import OpenPIFlowSpec, OpenPIStochasticFlowPolicy


class ConstantVelocityPolicy(OpenPIStochasticFlowPolicy):
    def __init__(self, action_dim: int, velocity: torch.Tensor, **kwargs):
        super().__init__(action_dim=action_dim, **kwargs)
        self.register_buffer("velocity", velocity.reshape(1, -1))

    def predict_velocity(self, x_t: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.velocity.expand_as(x_t)


def test_stochastic_flow_policy_is_an_abstract_interface():
    with pytest.raises(TypeError, match="abstract"):
        OpenPIStochasticFlowPolicy(action_dim=2)


def test_openpi_training_interpolation_matches_source_equations():
    spec = OpenPIFlowSpec(num_steps=4)
    action = torch.tensor([[2.0, -1.0]])
    noise = torch.tensor([[10.0, 3.0]])
    time = torch.tensor([[0.25]])

    x_t, target_velocity = spec.training_pair(action, noise, time)

    assert torch.allclose(x_t, torch.tensor([[4.0, 0.0]]))
    assert torch.allclose(target_velocity, torch.tensor([[8.0, 4.0]]))


def test_openpi_euler_step_integrates_backwards_from_noise_to_action():
    spec = OpenPIFlowSpec(num_steps=4)
    x_t = torch.tensor([[1.0, 3.0]])
    velocity = torch.tensor([[8.0, -4.0]])

    x_prev = spec.euler_step(x_t, velocity)

    assert torch.allclose(x_prev, torch.tensor([[-1.0, 4.0]]))


def test_ogpo_corrected_noise_tapers_from_pi_noise_time_to_clean_endpoint():
    policy = ConstantVelocityPolicy(
        action_dim=2,
        velocity=torch.tensor([0.5, -0.25]),
        num_steps=4,
        stochastic_variance=0.04,
        sde_mode="ogpo_corrected",
    )
    x_t = torch.zeros(3, 2)

    at_noise = policy.transition_std(x_t, torch.ones(3, 1))
    at_midpoint = policy.transition_std(x_t, torch.full((3, 1), 0.25))
    at_clean = policy.transition_std(x_t, torch.zeros(3, 1))

    assert torch.allclose(at_noise, torch.full_like(x_t, 0.2))
    assert torch.allclose(at_midpoint, torch.full_like(x_t, 0.1))
    assert torch.equal(at_clean, torch.zeros_like(x_t))


def test_ogpo_corrected_pi_mean_matches_cleanward_time_derivation():
    policy = ConstantVelocityPolicy(
        action_dim=2,
        velocity=torch.tensor([0.5, -0.25]),
        num_steps=4,
        stochastic_variance=0.04,
        sde_mode="ogpo_corrected",
    )
    x_t = torch.tensor([[1.0, -2.0]])
    timestep = torch.tensor([[0.75]])

    mean = policy.transition_mean(x_t, torch.zeros(1, 3), timestep)

    velocity = torch.tensor([[0.5, -0.25]])
    sigma_squared = 0.04
    corrected_pi_velocity = velocity + 0.5 * sigma_squared * (
        (1.0 - timestep) * velocity + x_t
    )
    expected = x_t - 0.25 * corrected_pi_velocity
    assert torch.allclose(mean, expected)


def test_ogpo_corrected_transition_log_prob_is_finite_at_rollout_times():
    policy = ConstantVelocityPolicy(
        action_dim=2,
        velocity=torch.tensor([0.5, -0.25]),
        num_steps=4,
        stochastic_variance=0.04,
        sde_mode="ogpo_corrected",
    )

    rollout = policy.rollout(torch.zeros(3, 2))

    assert torch.isfinite(rollout.states).all()
    assert torch.isfinite(rollout.log_probs).all()


def test_openpi_policy_equal_old_ratio_is_one_for_selected_transition():
    velocity = torch.tensor([0.5, -0.25])
    policy = ConstantVelocityPolicy(action_dim=2, velocity=velocity, num_steps=3, stochastic_variance=0.04)
    old = ConstantVelocityPolicy(action_dim=2, velocity=velocity, num_steps=3, stochastic_variance=0.04)
    old.load_state_dict(policy.state_dict())

    condition = torch.randn(4, 3)
    rollout = old.rollout(condition, group_size=2)
    selected = 1

    new_lp = policy.log_prob(
        rollout.next_states[:, selected],
        rollout.states[:, selected],
        condition.repeat_interleave(2, dim=0),
        rollout.timesteps[:, selected],
    )
    ratio = torch.exp(new_lp - rollout.log_probs[:, selected])

    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


def test_flow_matching_anchor_uses_openpi_velocity_target():
    action = torch.tensor([[2.0, -1.0]])
    noise = torch.tensor([[10.0, 3.0]])
    time = torch.tensor([[0.25]])
    target_velocity = noise - action
    policy = ConstantVelocityPolicy(action_dim=2, velocity=target_velocity, num_steps=4)

    result = flow_matching_anchor_loss(policy, torch.zeros(1, 3), action, noise=noise, timestep=time)

    assert torch.allclose(result.loss, torch.tensor(0.0))


def test_weighted_flow_matching_applies_per_replay_sample_weights():
    action = torch.tensor([[0.0], [2.0]])
    noise = torch.zeros_like(action)
    time = torch.full((2, 1), 0.5)
    policy = ConstantVelocityPolicy(action_dim=1, velocity=torch.tensor([0.0]), num_steps=2)

    result = weighted_flow_matching_loss(
        policy,
        torch.zeros(2, 3),
        action,
        torch.tensor([1.0, 0.0]),
        noise=noise,
        timestep=time,
    )

    assert torch.allclose(result.loss, torch.tensor(0.0))
    assert result.diagnostics["awr_weight_max"] == 1.0
