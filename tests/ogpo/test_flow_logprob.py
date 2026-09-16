import torch

from ogpo.flow_logprob import gaussian_log_prob
from ogpo.flow_sde import GaussianFlowPolicy


def test_gaussian_logprob_matches_torch_distribution():
    value = torch.tensor([[0.5, -0.25]])
    mean = torch.zeros_like(value)
    log_std = torch.zeros_like(value)
    ours = gaussian_log_prob(value, mean, log_std)
    dist = torch.distributions.Normal(mean, torch.ones_like(value))
    expected = dist.log_prob(value).sum(dim=-1)
    assert torch.allclose(ours, expected)


def test_policy_equal_old_ratio_is_one():
    policy = GaussianFlowPolicy(condition_dim=3, action_dim=2, num_steps=3)
    old = GaussianFlowPolicy(condition_dim=3, action_dim=2, num_steps=3)
    old.load_state_dict(policy.state_dict())
    condition = torch.randn(4, 3)
    rollout = old.rollout(condition)
    new_lp = policy.log_prob(
        rollout.next_states[:, 1],
        rollout.states[:, 1],
        condition,
        rollout.timesteps[:, 1],
    )
    ratio = torch.exp(new_lp - rollout.log_probs[:, 1])
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)
