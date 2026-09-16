import torch

from ogpo.flash_ogpo import sample_flash_rollout
from ogpo.openpi_flow_spec import OpenPIStochasticFlowPolicy
from ogpo.trainer import _select_flash_steps


class ConstantVelocityPolicy(OpenPIStochasticFlowPolicy):
    def __init__(self, action_dim: int, velocity: torch.Tensor, **kwargs):
        super().__init__(action_dim=action_dim, **kwargs)
        self.register_buffer("velocity", velocity.reshape(1, -1))

    def predict_velocity(self, x_t: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.velocity.expand_as(x_t)


def test_flash_rollout_uses_one_selected_timestep_per_state_group():
    policy = ConstantVelocityPolicy(action_dim=2, velocity=torch.tensor([0.1, -0.2]), num_steps=4)
    condition = torch.randn(2, 3)
    selected_steps = torch.tensor([0, 2])

    rollout = sample_flash_rollout(
        policy,
        condition,
        group_size=3,
        selected_step=selected_steps,
        generator=torch.Generator().manual_seed(5),
    )

    expected_times = policy.flow_spec.timestep_values(
        device=condition.device,
        dtype=condition.dtype,
    )[selected_steps]
    grouped_times = rollout.timestep.reshape(2, 3, 1).squeeze(-1)

    assert torch.allclose(grouped_times, expected_times[:, None].expand(2, 3))
    assert torch.equal(rollout.selected_step, selected_steps)


def test_stratified_flash_steps_cover_the_schedule():
    selected = _select_flash_steps(
        {"selected_timestep_distribution": "stratified_uniform"},
        batch_size=8,
        num_steps=8,
        device=torch.device("cpu"),
        seed=17,
    )

    assert torch.equal(torch.sort(selected).values, torch.arange(8))
