from dataclasses import replace
import torch
import pytest
from ogpo.replay import make_synthetic_replay
from ogpo.episode_bootstrap import MemberEpisodeBootstrapReplay, member_owner_loss_mask


def fixture():
    batch = replace(make_synthetic_replay(num_samples=40),
                    task_ids=['a'] * 20 + ['b'] * 20,
                    episode_ids=torch.arange(40), successes=torch.tensor([0, 1] * 20).float())
    masks = {'members': {str(m): {'a': list(range(m * 2, 20)), 'b': list(range(20 + m * 2, 40))} for m in range(5)}}
    return batch, masks


def test_persistent_balanced_pools_and_determinism():
    batch, masks = fixture()
    replay = MemberEpisodeBootstrapReplay(batch, masks=masks, task_names=['a', 'b'])
    sample = replay.sample(4000, generator=torch.Generator().manual_seed(12))
    again = replay.sample(4000, generator=torch.Generator().manual_seed(12))
    assert torch.equal(sample.episode_ids, again.episode_ids)
    assert sample.batch_size == 20000
    for m in range(5):
        part = sample.index_select(torch.arange(m * 4000, (m + 1) * 4000))
        assert abs(part.successes.float().mean().item() - .5) < .03
        assert abs(part.task_ids.count('a') / 4000 - .5) < .03
        for task, eid in zip(part.task_ids, part.episode_ids.tolist()):
            assert eid in masks['members'][str(m)][task]
        mask = member_owner_loss_mask(part, 5, device='cpu')
        assert mask[m].all() and mask.sum() == 4000


def test_both_q_heads_and_v_only_receive_own_loss():
    batch, masks = fixture()
    sample = MemberEpisodeBootstrapReplay(batch, masks=masks, task_names=['a', 'b']).sample(1)
    sample = sample.index_select(torch.tensor([2]))
    mask = member_owner_loss_mask(sample, 5, device='cpu')
    q = torch.ones(5, 2, 1, requires_grad=True)
    v = torch.ones(5, 1, requires_grad=True)
    ((q.square() * mask[:, None]).sum() / 2 + (v.square() * mask).sum()).backward()
    assert (q.grad[2] != 0).all() and (v.grad[2] != 0).all()
    assert q.grad[[0, 1, 3, 4]].count_nonzero() == 0
    assert v.grad[[0, 1, 3, 4]].count_nonzero() == 0


def test_empty_pool_fails_closed():
    batch, masks = fixture()
    masks['members']['0']['a'] = [0]
    with pytest.raises(ValueError, match='empty bootstrap pool'):
        MemberEpisodeBootstrapReplay(batch, masks=masks, task_names=['a', 'b'])


@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('shared', [False, True])
def test_real_double_q_update_and_independent_initialization(categorical, shared):
    from test_categorical_q_ranking import _config, _factory
    from ogpo.trainer import build_train_state, accumulated_critic_update
    from ogpo.evaluator import validation_metrics_for_training
    batch, masks = fixture()
    config = _config(categorical=categorical, categorical_q_loss='one_hot_ce')
    config['critic'].update(ensemble_size=5, q_heads_per_member=2,
                           double_q_divl=True, rank_consensus_enabled=False,
                           bootstrap_probability=1.0, member_initialization_seed=20260915)
    config['training']['critic_sampling'] = {'mode': 'shared_episode_bootstrap' if shared else 'member_episode_bootstrap'}
    state = build_train_state(config, batch, multimodal_critic_factory=_factory)
    weights = [head[0].weight for head in state.critic.core.q_heads]
    assert all(not torch.equal(weights[i], weights[j]) for i in range(10) for j in range(i))
    from ogpo.episode_bootstrap import SharedEpisodeBootstrapReplay
    replay = (SharedEpisodeBootstrapReplay if shared else MemberEpisodeBootstrapReplay)(batch, masks=masks, task_names=['a', 'b'])
    result = accumulated_critic_update(state, replay.sample(8 if shared else 1), config, microbatch_size=1)
    assert state.step == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    diagnostics = validation_metrics_for_training(state, batch, config)
    assert 'validation_raw10_q_head_correlation' in diagnostics


def test_shared_normalization_equals_per_member_mean_under_microbatching():
    from ogpo.episode_bootstrap import SharedEpisodeBootstrapReplay, normalize_shared_member_weights
    batch,masks = fixture()
    sample=SharedEpisodeBootstrapReplay(batch,masks=masks,task_names=['a','b']).sample(32,generator=torch.Generator().manual_seed(7))
    assert sample.batch_size==32
    sample=normalize_shared_member_weights(sample,device='cpu')
    include=torch.tensor([m['bootstrap_inclusion'] for m in sample.behavior_metadata]).T
    weights=torch.tensor([m['bootstrap_loss_weights'] for m in sample.behavior_metadata]).T
    loss=torch.arange(160).reshape(5,32).float()
    expected=((loss*include).sum(1)/include.sum(1).clamp_min(1)).mean()
    assert torch.allclose((loss*weights).sum()/32,expected)
    micro=sum((loss[:,i]*weights[:,i]).sum()/32 for i in range(32))
    assert torch.allclose(micro,expected)
    assert (weights[~include]==0).all()
