from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, output):
    from ogpo.actor_role_dp import average_gradients, global_validation, global_chipo
    from ogpo.chi2_regularization import apply_chipo_to_ca_advantage
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([2., 4., 6.]) if rank == 0 else torch.tensor([4., 6., 8.])
    average_gradients([p], bucket_bytes=8)
    torch.testing.assert_close(p.grad, torch.tensor([3., 5., 7.]))
    kl, bad = global_validation(0.1 if rank == 0 else 0.3, False, torch.device('cpu'))
    assert abs(kl - 0.2) < 1e-7 and not bad
    _, bad = global_validation(0.1, rank == 1, torch.device('cpu'))
    assert bad
    q = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 20
    q[:, 2:] *= 3
    ca = q.mean(0) - q.mean(0).mean(-1, keepdim=True)
    ratio = torch.linspace(0.9, 1.1, 16).reshape(4, 4)
    kwargs = dict(beta_base=.1, q_std_target=1., ensemble_alpha=5., r_max=10., normalize_group=False)
    expected, stats = apply_chipo_to_ca_advantage(ca, q, ratio, **kwargs)
    sl = slice(rank*2, rank*2+2)
    actual, actual_stats = global_chipo(ca[sl], q[:, sl], ratio[sl], **kwargs)
    torch.testing.assert_close(actual, expected[sl])
    assert actual_stats.beta == stats.beta
    # Compare the real full-chain update on four states against two replicas.
    from dataclasses import fields, replace
    from test_actor_policy_lifecycle import _full_update_fixture
    from ogpo.trainer import full_actor_update
    outcomes = []
    for distributed in (False, True):
        torch.manual_seed(193)
        state, batch, config = _full_update_fixture()
        batch = batch.index_select(torch.arange(4))
        config['actor'].update(ogpo_variant='ca_chi2', advantage_mode='conservative',
                               group_size=4, candidate_group_size=4, gradient_microbatch_size=1,
                               role_data_parallel=distributed)
        config['actor']['chi2']['apply_project_safety_gates'] = False
        torch.manual_seed(351)
        with torch.no_grad():
            rollout = state.old_policy.rollout(batch.observations, group_size=4)
        if distributed:
            updates = {f.name: (getattr(rollout, f.name)[rank*8:(rank+1)*8]
                       if torch.is_tensor(getattr(rollout, f.name))
                       and getattr(rollout, f.name).ndim > 0
                       and getattr(rollout, f.name).shape[0] == 16
                       else getattr(rollout, f.name)) for f in fields(rollout)}
            rollout = replace(rollout, **updates)
            batch = batch.index_select(torch.arange(rank*2, (rank+1)*2))
        state.old_policy.rollout = lambda *a, **kw: rollout
        metrics = full_actor_update(state, batch, config)
        outcomes.append((metrics, [p.detach().clone() for p in state.policy.parameters()]))
    for a, b in zip(outcomes[0][1], outcomes[1][1], strict=True):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
    for key in ['actor_grad_norm', 'post_update_reference_kl', 'actor_update_accepted']:
        assert abs(outcomes[0][0][key]-outcomes[1][0][key]) < 2e-5
    # One rank's failed post-update validation must roll back both policies.
    from ogpo import trainer
    torch.manual_seed(193)
    state, batch, config = _full_update_fixture()
    batch = batch.index_select(torch.arange(rank*2, (rank+1)*2))
    config['actor'].update(ogpo_variant='ca_chi2', advantage_mode='conservative',
                           group_size=4, candidate_group_size=4, gradient_microbatch_size=1,
                           role_data_parallel=True)
    config['actor']['chi2']['apply_project_safety_gates'] = False
    before = [p.detach().clone() for p in state.policy.parameters()]
    original_kl = trainer._full_chain_policy_kl
    calls = 0
    def failed_validation(*a, **kw):
        nonlocal calls
        calls += 1
        value = original_kl(*a, **kw)
        return float('nan') if rank == 1 and calls == 1 else value
    trainer._full_chain_policy_kl = failed_validation
    try:
        metrics = full_actor_update(state, batch, config)
    finally:
        trainer._full_chain_policy_kl = original_kl
    assert metrics['actor_update_rejected'] == 1
    assert metrics['actor_update_accepted'] == 0
    for a, b in zip(before, state.policy.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    Path(output, str(rank)).write_text('passed')
    dist.destroy_process_group()


def test_collectives_preserve_global_objective(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path/'rdzv'), str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path/'0').read_text() == (tmp_path/'1').read_text() == 'passed'
