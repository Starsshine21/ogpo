import torch

from ogpo.chi2_regularization import full_chain_chipo_ratio
from ogpo.critic import clone_target
from ogpo.flow_sde import GaussianFlowPolicy
from ogpo.full_ogpo import full_chain_chipo_ppo_loss


def test_detached_advantage_keeps_current_actor_gradient_only():
    current = GaussianFlowPolicy(
        3,
        4,
        hidden_dim=8,
        num_steps=3,
        sde_mode="ogpo_constant_corrected",
        constant_noise_std=0.005,
        learn_sde_std=False,
    )
    old = clone_target(current)
    slow = clone_target(current)
    condition = torch.randn(2, 3)
    rollout = old.rollout(condition)
    current_lp = torch.stack(
        [
            current.log_prob(
                rollout.next_states[:, step],
                rollout.states[:, step],
                condition,
                rollout.timesteps[:, step],
            )
            for step in range(current.num_steps)
        ],
        dim=1,
    )
    with torch.no_grad():
        slow_lp = torch.stack(
            [
                slow.log_prob(
                    rollout.next_states[:, step],
                    rollout.states[:, step],
                    condition,
                    rollout.timesteps[:, step],
                )
                for step in range(slow.num_steps)
            ],
            dim=1,
        )
        chi_ratio, _ = full_chain_chipo_ratio(
            current_lp.detach(),
            slow_lp,
            action_dim=current.action_dim,
            normalize_logprob_by_action_dim=True,
            normalize_logprob_by_denoising_steps=True,
        )
    assert not chi_ratio.requires_grad
    advantage = torch.tensor([1.0, -1.0]).detach()
    loss = full_chain_chipo_ppo_loss(
        current_lp,
        rollout.log_probs,
        advantage,
        clip_eps=0.01,
        beta=0.1,
        r_max=10.0,
        logprob_normalizer=current.action_dim * current.num_steps,
    ).loss
    loss.backward()
    current_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in current.parameters()
        if parameter.grad is not None
    )
    assert current_grad > 0.0
    assert all(parameter.grad is None for parameter in old.parameters())
    assert all(parameter.grad is None for parameter in slow.parameters())
