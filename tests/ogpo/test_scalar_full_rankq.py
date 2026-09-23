from dataclasses import replace
import copy
import math

import pytest
import torch
import torch.nn.functional as F

from test_rankq import _factory, _config
from ogpo import trainer
from ogpo.rankq import make_rankq_actions, compute_rankq_loss, full_rankq_settings, ddp_global_valid_mean_scale, rank_pair_loss
from ogpo.critic_raw10_evaluator import raw10_fidelity_metrics, same_state_action_ranking_metrics, actor_signal_metrics
from ogpo.evaluator import validation_metrics_for_training
from ogpo.replay import make_synthetic_replay


def config():
    cfg = _config(enable_rankq=False, q_representation="scalar")
    cfg["critic"].update(ensemble_size=5, q_heads_per_member=2, double_q_divl=True,
                         bootstrap_probability=1., max_grad_norm=1e6)
    cfg["critic"]["rankq"] = dict(mode="full", enabled=True, noise_sigma=.04,
                                     lambda_rank=1., alpha_success=1., alpha_failure=1.)
    cfg["evaluation"] = {"raw10_validation": True}
    return cfg


def batch():
    data = make_synthetic_replay(num_samples=8, generated_horizon=4, executed_horizon=2, action_dim=2)
    return replace(data, action_chunks=data.action_chunks.clamp(-.8, .8),
                   successes=torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.]),
                   task_ids=["a", "a", "b", "b", "a", "a", "b", "b"],
                   episode_ids=torch.arange(8), mc_returns=torch.linspace(0, 1, 8))


def actions(data):
    return make_rankq_actions(data.action_chunks, data.execution_masks, action_mean=torch.tensor([.2, -.1]),
                             action_std=torch.tensor([.5, 2.]), action_min=torch.tensor([-.8, -.8]),
                             action_max=torch.tensor([.8, .8]), noise_sigma=.04,
                             task_ids=data.task_ids, episode_ids=data.episode_ids, timesteps=data.timesteps,
                             generator=torch.Generator().manual_seed(18))


def test_scalar_double_shapes_and_v_distribution():
    data = batch()
    state = trainer.build_train_state(config(), data, multimodal_critic_factory=_factory)
    features = state.critic.encode_state(data)
    pairs = state.critic.q_pair_from_features(features, data.action_chunks, data.execution_masks)
    raw = state.critic.raw_q_ensemble_from_features(features, data.action_chunks, data.execution_masks)
    assert pairs.shape == (5, 2, 8)
    assert raw.shape == (10, 8)
    assert torch.equal(raw, pairs.flatten(0, 1))
    assert state.critic.core.q_support is None
    assert state.critic.value_logits_from_features(features).shape == (5, 8, 201)


def test_noise_2x_iid_bounds_and_execution():
    data = batch()
    made = actions(data)
    assert torch.allclose(made.very_noisy-data.action_chunks, 2*(made.noisy-data.action_chunks), atol=2e-7)
    for candidate in (made.noisy, made.very_noisy, made.random, made.permuted):
        assert torch.equal(candidate[~data.execution_masks], data.action_chunks[~data.execution_masks])
        assert candidate.min() >= -.800001 and candidate.max() <= .800001
    assert (made.noisy-data.action_chunks)[data.execution_masks].std() > 0


def test_permutation_same_task_episode_priority_singleton_and_different_state():
    data = replace(batch(), task_ids=["a", "a", "a", "a", "b", "b", "c", "d"],
                   episode_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 4]))
    made = actions(data)
    for i in range(6):
        j = int(made.permutation[i])
        assert j != i and data.task_ids[j] == data.task_ids[i]
        if i < 4:
            assert data.episode_ids[i] != data.episode_ids[j]
    assert made.permuted_valid_mask.tolist() == [True]*6+[False]*2
    assert made.same_episode_permuted_mask.tolist() == [False]*4+[True]*2+[False]*2
    # Duplicate draws of exactly the same transition cannot act as a donor.
    data = replace(data, timesteps=torch.zeros(8, dtype=torch.long))
    assert not bool(actions(data).permuted_valid_mask[4:6].any())


def values(n=8):
    gen = torch.Generator().manual_seed(29)
    return {name: torch.randn(10, n, generator=gen, requires_grad=True)
            for name in ("positive", "noisy", "very_noisy", "random", "permuted")}


def test_full_six_relations_and_failure_raw10_independence():
    q = values()
    success = batch().successes.bool()
    valid = torch.tensor([True, False, True, False, True, True, True, True])
    result = compute_rankq_loss(q, success, permuted_valid_mask=valid, alpha_success=2., alpha_failure=3.)
    pairs = [("positive", "noisy"), ("positive", "very_noisy"), ("positive", "random"),
             ("positive", "permuted"), ("noisy", "very_noisy"), ("very_noisy", "random")]
    expected = sum(F.softplus(q[b][:, success & valid if b == "permuted" else success]-q[a][:, success & valid if b == "permuted" else success]).mean() for a, b in pairs)
    failure = F.softplus(q["random"][:, ~success]-q["positive"][:, ~success]).mean()
    assert torch.allclose(result.loss, 2*expected+3*failure)
    result.loss.backward()
    for name in ("noisy", "very_noisy", "permuted"):
        assert torch.equal(q[name].grad[:, ~success], torch.zeros(10, 4))
    assert bool((q["positive"].grad.abs().sum(1) > 0).all())
    assert len(result.relation_losses) == 7
    # Nonlinearity must precede head averaging.
    averaged = {name: value.detach().mean(0, keepdim=True) for name, value in q.items()}
    assert not torch.allclose(result.loss.detach(), compute_rankq_loss(averaged, success, permuted_valid_mask=valid, alpha_success=2., alpha_failure=3.).loss)


@pytest.mark.parametrize("success", [True, False])
def test_empty_outcome_and_permutation_graph_safe(success):
    q = values(3)
    result = compute_rankq_loss(q, torch.full((3,), success), permuted_valid_mask=torch.zeros(3, dtype=torch.bool))
    assert torch.isfinite(result.loss)
    assert result.relation_losses["pos_permuted"] == 0
    result.loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in q.values())


@pytest.mark.parametrize(
    "loss_kwargs",
    [{}, {"pair_loss": "temperature_softplus_guarded", "temperature": .1, "max_gap": 1.}],
    ids=["legacy", "guarded"],
)
def test_microbatch_loss_gradient_and_ddp_scales(loss_kwargs):
    q = values()
    success = batch().successes.bool()
    valid = torch.tensor([True, False, True, True, True, True, True, True])
    full = compute_rankq_loss(q, success, permuted_valid_mask=valid, **loss_kwargs).loss
    full_grads = torch.autograd.grad(full, tuple(q.values()), retain_graph=True)
    total = 0
    # A all success, B all failure; permutation has its own denominator.
    for sl in (slice(0, 4), slice(4, 8)):
        scales = {name: ddp_global_valid_mean_scale(int(mask[sl].sum()), int(mask.sum()), 1)
                  for name, mask in (("success", success), ("failure", ~success), ("permuted", success & valid))}
        total += compute_rankq_loss({name: v[:, sl] for name, v in q.items()}, success[sl],
                                   permuted_valid_mask=valid[sl], mean_scales=scales, **loss_kwargs).loss
    assert torch.allclose(full, total)
    grads = torch.autograd.grad(total, tuple(q.values()))
    assert all(torch.allclose(a, b) for a, b in zip(full_grads, grads))
    # DDP ranks with no success / no failure still average to the full mean.
    assert ddp_global_valid_mean_scale(0, 4, 2) == 0
    assert ddp_global_valid_mean_scale(4, 4, 2)/2 == 1


@pytest.mark.parametrize("mode", [None, "stochastic_bootstrap", "member_episode_bootstrap", "shared_episode_bootstrap"])
def test_real_full_vs_microbatch_training(mode):
    cfg = config()
    data = batch()
    if mode == "stochastic_bootstrap":
        cfg["critic"]["bootstrap_probability"] = .6
    elif mode:
        cfg["training"]["critic_sampling"] = {"mode": mode}
        data = replace(data, behavior_metadata=[dict(bootstrap_owner=i%5,
                       bootstrap_inclusion=[(i+m)%3 != 0 for m in range(5)]) for i in range(8)])
    torch.manual_seed(41)
    full = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
    micro = copy.deepcopy(full)
    torch.manual_seed(51)
    a = trainer.accumulated_critic_update(full, data, cfg, microbatch_size=8)
    torch.manual_seed(51)
    b = trainer.accumulated_critic_update(micro, data, cfg, microbatch_size=4)
    for p, r in zip(full.critic.parameters(), micro.critic.parameters()):
        if p.grad is None or r.grad is None:
            assert p.grad is None and r.grad is None
        else:
            assert torch.allclose(p.grad, r.grad, atol=5e-7, rtol=3e-5)
        assert torch.allclose(p, r, atol=2e-6, rtol=3e-5)
    for key in ("rankq_success_loss", "rankq_failure_loss", "rankq_chain_loss", "rankq_raw_loss", "q_loss", "divl_loss"):
        assert a[key] == pytest.approx(b[key], rel=3e-5, abs=5e-7)
    assert "critic/q_entropy_mean" not in a
    assert a["batch_success_fraction"] == .5


def test_threecam_main_config_has_scalar_raw10_and_full_rankq():
    from pathlib import Path
    from train_udivl_critic import load_config
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root/"configs/ogpo/threecam_scalar_guarded.yaml")
    critic = cfg["critic"]
    assert critic["architecture"] == "gemma_siglip_multihead"
    assert critic["q_representation"] == "scalar" and critic["double_q_divl"]
    assert critic["ensemble_size"]*critic["q_heads_per_member"] == 10
    assert full_rankq_settings(critic)["full"]
    assert cfg["divl"]["enabled"] and cfg["divl"]["num_atoms"] == 201
    assert cfg["training"]["critic_sampling"]["mode"] == "task_outcome_balanced"
    assert cfg["training"]["critic_sampling"]["success_probability"] == .5
    assert cfg["actor"]["q_ensemble_source"] == "raw_heads"
    assert not cfg["evaluation"]["checkpoint_selection"]["enabled"]
    assert cfg["critic"]["backbone"]["camera_keys"] == [
        "image_base", "image_left_wrist", "image_right_wrist"
    ]
    assert len(cfg["data"]["distributed_dataset_paths"]) == 2


def test_threecam_ablation_configs_preserve_scalar_divl_geometry():
    from pathlib import Path
    from train_udivl_critic import load_config
    root = Path(__file__).resolve().parents[2]
    new = load_config(root/"configs/ogpo/threecam_scalar_guarded.yaml")
    no_rank = load_config(root/"configs/ogpo/threecam_scalar_norank.yaml")
    tau1 = load_config(root/"configs/ogpo/threecam_scalar_tau1.yaml")
    settings = full_rankq_settings(new["critic"])
    assert settings["pair_loss"] == "temperature_softplus_guarded"
    assert settings["temperature"] == .1 and settings["max_gap"] == 1.
    for key in ("architecture", "ensemble_size", "q_heads_per_member", "double_q_divl", "q_representation"):
        assert new["critic"][key] == no_rank["critic"][key] == tau1["critic"][key]
    for key in ("noise_sigma", "alpha_success", "alpha_failure", "permutation_mode"):
        assert new["critic"]["rankq"][key] == tau1["critic"]["rankq"][key]
    assert not no_rank["critic"]["rankq"]["enabled"]
    assert tau1["critic"]["rankq"]["temperature"] == 1.0
    assert tau1["critic"]["rankq"]["max_gap"] is None


def test_guarded_pair_loss_gradient_scale_saturation_continuity_and_no_detach():
    positive = torch.tensor([0.0], requires_grad=True)
    negative = torch.tensor([0.0], requires_grad=True)
    loss = rank_pair_loss(positive, negative, pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.)
    loss.backward()
    assert positive.grad.item() == pytest.approx(-.5, abs=1e-6)
    assert negative.grad.item() == pytest.approx(.5, abs=1e-6)

    gaps = torch.tensor([.1, .5, 1.])
    slopes = torch.sigmoid(-gaps/.1)
    assert slopes.tolist() == pytest.approx([.2689414, .00669285, 4.539787e-5], rel=1e-5)

    for gap in (.5,):
        guarded_gap = torch.tensor(gap, requires_grad=True)
        natural_gap = guarded_gap.detach().clone().requires_grad_()
        guarded = rank_pair_loss(guarded_gap, torch.zeros_like(guarded_gap), pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.)
        natural = rank_pair_loss(natural_gap, torch.zeros_like(natural_gap), pair_loss="temperature_softplus", temperature=.1)
        assert torch.autograd.grad(guarded, guarded_gap)[0] == pytest.approx(torch.autograd.grad(natural, natural_gap)[0])

    epsilon = 1e-5
    below = rank_pair_loss(torch.tensor(1.-epsilon), torch.tensor(0.), pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.)
    assert 0 < below.item() < 1e-8
    for gap in (1., 1.5):
        pos = torch.tensor(gap, requires_grad=True); neg = torch.tensor(0., requires_grad=True)
        guarded = rank_pair_loss(pos, neg, pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.)
        guarded.backward()
        assert guarded.item() == 0. and pos.grad.item() == 0. and neg.grad.item() == 0.

    pos = torch.tensor(.05, requires_grad=True); neg = torch.tensor(0., requires_grad=True)
    rank_pair_loss(pos, neg, pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.).backward()
    assert pos.grad.item() < 0 and neg.grad.item() > 0


def test_guard_metrics_and_all_seven_relations_share_canonical_loss(monkeypatch):
    gaps = torch.tensor([.2, .7, 1., 1.5])
    q = {name: torch.zeros(2, 4) for name in ("positive", "noisy", "very_noisy", "random", "permuted")}
    q["positive"] = gaps.repeat(2, 1)
    calls = []
    from ogpo import rankq as rankq_module
    original = rankq_module.rank_pair_loss
    def wrapped(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(rankq_module, "rank_pair_loss", wrapped)
    result = compute_rankq_loss(
        q, torch.tensor([True, True, False, False]),
        pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.,
    )
    assert len(calls) == 7
    assert all(c == {"pair_loss": "temperature_softplus_guarded", "temperature": .1, "max_gap": 1.} for c in calls)
    # Failure positive-vs-random sees gaps [1.0, 1.5] on both heads.
    assert result.metrics["rankq/guard/failure_pos_random_fraction"] == 1.
    # Success positive-vs-noisy sees [.2, .7], neither guarded.
    assert result.metrics["rankq/guard/pos_noisy_fraction"] == 0.

    q["positive"] = gaps.repeat(2, 1)
    result = compute_rankq_loss(
        q, torch.tensor([True, True, True, True]),
        pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.,
    )
    assert result.metrics["rankq/guard/pos_noisy_fraction"] == .5
    assert result.metrics["rankq/guard/pos_noisy_any_head_fraction"] == .5
    assert result.metrics["rankq/guard/pos_noisy_all_heads_fraction"] == .5
    assert result.metrics["rankq/guard_trigger_but_natural_slope_gt_1e2_fraction"] == 0.


def test_scalar_td_half_mse_member_mask():
    data = batch()
    cfg = config()
    cfg["critic"]["target_mode"] = "mc_return"
    cfg["critic"]["rankq"]["enabled"] = False
    state = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
    features = state.critic.encode_state(data)
    q = state.critic.q_pair_from_features(features, data.action_chunks, data.execution_masks)
    weights = torch.zeros(5, 8)
    weights[0, :4] = 1/4
    result = trainer._multimodal_double_q_divl_update(state, data, cfg, optimizer_step=False, base_member_weights=weights)
    expected = .5*(q[0, :, :4]-data.mc_returns[None, :4]).square().mean()
    assert result["q_loss"] == pytest.approx(float(expected), rel=1e-6)


def test_artificial_scalar_td_target_and_mask(monkeypatch):
    data = batch()
    cfg = config()
    cfg["critic"]["rankq"]["enabled"] = False
    cfg["critic"]["target_mode"] = "mc_return"
    state = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
    q = torch.arange(80, dtype=torch.float32).reshape(5, 2, 8).requires_grad_()
    monkeypatch.setattr(state.critic, "q_pair_from_features", lambda *args: q)
    weights = torch.zeros(5, 8)
    weights[1, 1:3] = .5
    result = trainer._multimodal_double_q_divl_update(state, data, cfg, optimizer_step=False, base_member_weights=weights)
    expected = .5*(q.detach()[1, :, 1:3]-data.mc_returns[None, 1:3]).square().mean()
    assert result["q_loss"] == pytest.approx(float(expected), rel=1e-6)
    assert torch.equal(q.grad[0], torch.zeros(2, 8))
    assert torch.allclose(q.grad[1, :, 1:3], (q.detach()[1, :, 1:3]-data.mc_returns[None, 1:3])/4)


def test_scalar_outputs_are_not_support_clamped():
    data = batch()
    state = trainer.build_train_state(config(), data, multimodal_critic_factory=_factory)
    with torch.no_grad():
        for head in state.critic.core.q_heads:
            head[-1].weight.zero_()
            head[-1].bias.fill_(10.)
    raw = state.critic.raw_q_ensemble_from_features(state.critic.encode_state(data), data.action_chunks, data.execution_masks)
    assert torch.equal(raw, torch.full((10, 8), 10.))


def test_scalar_saturation_no_support_cap():
    margins = torch.tensor([0., 1., 5., 10.])
    slope = torch.sigmoid(-margins)
    assert bool((slope[:-1] > slope[1:]).all())
    assert slope[0] == .5 and slope[-1] < 1e-2
    q = {name: torch.zeros(10, 4) for name in ("positive", "noisy", "very_noisy", "random", "permuted")}
    q["positive"] = margins.repeat(10, 1)
    result = compute_rankq_loss(q, torch.ones(4, dtype=torch.bool))
    assert result.metrics["rankq_pos_vs_random_softplus_slope_below_1e2"] == .5


def test_categorical_to_scalar_initialization_no_partial_q(tmp_path, capsys):
    data = batch()
    source_cfg = config()
    source_cfg["critic"].update(q_representation="categorical", categorical_q_loss="one_hot_ce")
    source = trainer.build_train_state(source_cfg, data, multimodal_critic_factory=_factory)
    with torch.no_grad():
        for head in source.critic.core.q_heads:
            head[0].weight.fill_(123.)
        source.critic.core.value_heads[0][0].weight.fill_(.42)
        source.critic.core.action_pool.query.fill_(.24)
    path = tmp_path/"cat.pt"
    trainer.save_checkpoint(source, source_cfg, path)
    destination = trainer.build_train_state(config(), data, multimodal_critic_factory=_factory)
    trainer.initialize_critic_from_checkpoint(path, destination)
    assert torch.equal(destination.critic.state_encoder.projection.weight, source.critic.state_encoder.projection.weight)
    assert torch.equal(destination.critic.core.value_heads[0][0].weight, source.critic.core.value_heads[0][0].weight)
    assert torch.equal(destination.critic.core.action_pool.query, source.critic.core.action_pool.query)
    assert all(not bool((head[0].weight == 123).any()) for head in destination.critic.core.q_heads)
    assert all(torch.equal(p, r) for p, r in zip(destination.critic.core.q_heads.parameters(), destination.target_critic.core.q_heads.parameters()))
    assert not destination.critic_optimizer.state
    logs = capsys.readouterr().out
    assert "source q_representation = categorical" in logs
    assert "destination q_representation = scalar" in logs


def test_scalar_resume_and_raw10_evaluator(tmp_path):
    data = batch()
    cfg = config()
    state = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
    trainer.critic_update(state, data, cfg)
    path = tmp_path/"scalar.pt"
    trainer.save_checkpoint(state, cfg, path)
    restored = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
    trainer.load_critic_checkpoint(path, restored)
    assert restored.step == 1 and restored.critic_optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(state.critic.parameters(), restored.critic.parameters()))
    raw = torch.randn(10, 8)*4
    fidelity = raw10_fidelity_metrics(raw, data.mc_returns)
    ranking = same_state_action_ranking_metrics(raw, torch.randn(4, 10, 8))
    signal = actor_signal_metrics(torch.randn(10, 8, 4), chi2_config={})
    assert "raw10_member_rank_min" in fidelity
    assert "raw10_action_rank_unanimous" in ranking and "ca_nonzero_ratio" in signal
    validation = validation_metrics_for_training(state, data, cfg)
    assert "validation_raw10_mc_spearman" in validation
    assert all(math.isfinite(v) for v in validation.values())


def test_full_settings_reject_legacy_sigmas_and_default_no_warmup():
    cfg = config()["critic"]
    resolved = full_rankq_settings(cfg)
    assert resolved["lambda_rank"] == 1 and resolved["noise_sigma"] == .04
    assert not resolved["lambda_rank_schedule_enabled"]
    cfg["rankq"]["mild_sigma"] = .02
    cfg["rankq"]["strong_sigma"] = .05
    with pytest.raises(ValueError, match="noise_sigma"):
        full_rankq_settings(cfg)


def test_nonfinite_rankq_fails_loudly():
    q = values()
    q["random"] = torch.full((10, 8), float("nan"))
    with pytest.raises(FloatingPointError):
        compute_rankq_loss(q, batch().successes)
    with pytest.raises(FloatingPointError):
        raw10_fidelity_metrics(torch.full((10, 8), float("nan")), batch().mc_returns)


def _ddp_worker(rank, rendezvous, output_dir):
    import torch.distributed as dist
    from pathlib import Path
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        data = batch()
        cfg = config()
        cfg["critic"]["rankq"].update(
            pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.
        )
        torch.manual_seed(123)
        state = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
        local = data.index_select(torch.arange(rank*4, rank*4+4))
        torch.manual_seed(321)
        metrics = trainer.accumulated_critic_update(state, local, cfg, microbatch_size=2)
        gradients = {name: p.grad.cpu() for name, p in state.critic.named_parameters() if p.grad is not None}
        torch.save({"gradients": gradients, "metrics": metrics}, Path(output_dir)/f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_real_two_rank_ddp_success_only_and_failure_only(tmp_path):
    import torch.multiprocessing as mp
    mp.start_processes(_ddp_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path)), nprocs=2, start_method="spawn", join=True)
    cfg = config()
    cfg["critic"]["rankq"].update(
        pair_loss="temperature_softplus_guarded", temperature=.1, max_gap=1.
    )
    data = batch()
    torch.manual_seed(123)
    state = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
    torch.manual_seed(321)
    metrics = trainer.accumulated_critic_update(state, data, cfg, microbatch_size=8)
    for rank in range(2):
        result = torch.load(tmp_path/f"rank{rank}.pt", weights_only=False)
        for name, p in state.critic.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, result["gradients"][name], atol=8e-7, rtol=5e-5)
        for key in ("rankq_raw_loss", "rankq_success_loss", "rankq_failure_loss", "rankq_permuted_valid_fraction",
                    "rankq/guard/overall_fraction", "rankq/guard/overall_count",
                    "rankq/guard/overall_valid_count", "rankq/natural_slope_mean"):
            assert metrics[key] == pytest.approx(result["metrics"][key], abs=5e-7, rel=3e-5)


def test_full_gradient_probe_keeps_parameters_and_step():
    data = batch()
    cfg = config()
    state = trainer.build_train_state(cfg, data, multimodal_critic_factory=_factory)
    before = copy.deepcopy(state.critic.state_dict())
    metrics = trainer._multimodal_double_q_divl_update(state, data, cfg, diagnostic_component_gradients=True)
    assert state.step == 0
    assert all(torch.equal(before[k], value) for k, value in state.critic.state_dict().items())
    assert all(p.grad is None for p in state.critic.parameters())
    assert metrics["rankq_q_head_gradient_count"] == 10
    assert metrics["rankq_value_head_grad_norm"] == 0
    assert math.isfinite(metrics["divl_rankq_gradient_cosine"])
