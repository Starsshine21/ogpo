import torch

from ogpo.flow_sde import GaussianFlowPolicy


def _policy():
    return GaussianFlowPolicy(
        3,
        4,
        hidden_dim=8,
        num_steps=4,
        sde_mode="ogpo_constant_corrected",
        constant_noise_std=0.005,
        learn_sde_std=False,
    )


def test_constant_sde_schedule_final_step_and_frozen_std():
    policy = _policy()
    condition = torch.randn(2, 3)
    rollout = policy.rollout(condition)
    assert rollout.transition_stds is not None
    assert rollout.stochastic_masks is not None
    assert torch.allclose(
        rollout.transition_stds[:, :-1],
        torch.full_like(rollout.transition_stds[:, :-1], 0.005),
        atol=1e-6,
    )
    assert torch.equal(
        rollout.transition_stds[:, -1], torch.zeros_like(rollout.transition_stds[:, -1])
    )
    assert rollout.stochastic_masks[:, :-1].all()
    assert not rollout.stochastic_masks[:, -1].any()
    assert torch.equal(rollout.log_probs[:, -1], torch.zeros(2))
    assert "log_std" not in dict(policy.named_parameters())


def test_sampler_and_logprob_reevaluation_share_transition_spec():
    policy = _policy()
    condition = torch.randn(2, 3)
    rollout = policy.rollout(condition)
    assert rollout.transition_means is not None
    assert rollout.transition_stds is not None
    assert rollout.stochastic_masks is not None
    for step in range(policy.num_steps):
        mean, std, mask = policy.transition_parameters(
            rollout.states[:, step], condition, rollout.timesteps[:, step]
        )
        assert torch.allclose(mean, rollout.transition_means[:, step])
        assert torch.equal(std, rollout.transition_stds[:, step])
        assert torch.equal(mask, rollout.stochastic_masks[:, step])
        reevaluated = policy.log_prob(
            rollout.next_states[:, step],
            rollout.states[:, step],
            condition,
            rollout.timesteps[:, step],
        )
        assert torch.allclose(reevaluated, rollout.log_probs[:, step])


def test_local_to_ogpo_time_mapping_and_native_final_step():
    policy = _policy()
    assert torch.allclose(
        policy.ogpo_time(torch.tensor([[1.0], [0.25]])),
        torch.tensor([[0.0], [0.75]]),
    )
    condition = torch.randn(2, 3)
    x_t = torch.randn(2, 4)
    final = torch.full((2, 1), 0.25)
    assert torch.allclose(
        policy.transition_mean(x_t, condition, final),
        policy.native_transition_mean(x_t, condition, final),
    )
