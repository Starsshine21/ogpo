from ogpo.evaluator import offline_calibration_metrics
from ogpo.replay import make_synthetic_replay
from ogpo.trainer import (
    build_train_state,
    critic_update,
    flash_actor_update,
    full_actor_update,
    load_critic_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_critic_full_flash_checkpoint_smoke(tmp_path):
    batch = make_synthetic_replay(num_samples=16, generated_horizon=4, executed_horizon=2, action_dim=2)
    cfg = {
        "critic": {"ensemble_size": 2, "hidden_dim": 32, "num_layers": 1},
        "divl": {"num_atoms": 21, "v_min": -5.0, "v_max": 5.0},
        "actor": {"group_size": 2, "hidden_dim": 32},
        "flow": {"num_steps": 3, "selected_timestep": 1},
        "regularization": {"lambda_fm": 0.01, "beta_kl": 0.01},
    }
    state = build_train_state(cfg, batch)
    critic_metrics = critic_update(state, batch, cfg)
    assert critic_metrics["critic_loss"] > 0.0
    full_metrics = full_actor_update(state, batch, cfg)
    flash_metrics = flash_actor_update(state, batch, cfg)
    assert "actor_loss" in full_metrics
    assert "actor_loss" in flash_metrics
    ckpt = tmp_path / "ogpo.pt"
    expected_support = state.support.clone()
    save_checkpoint(state, cfg, ckpt)
    state.support = state.support.new_zeros(state.support.shape)
    load_checkpoint(ckpt, state)
    payload = load_checkpoint(ckpt, state)
    assert "schedulers" in payload
    assert "support" in payload
    assert state.support.equal(expected_support)
    critic_only = build_train_state(cfg, batch)
    critic_only.support = critic_only.support.new_zeros(critic_only.support.shape)
    load_critic_checkpoint(ckpt, critic_only)
    assert critic_only.support.equal(expected_support)
    for expected, actual in zip(state.critic.parameters(), critic_only.critic.parameters(), strict=True):
        assert expected.equal(actual)
    critic_without_optimizer = build_train_state(cfg, batch)
    load_critic_checkpoint(ckpt, critic_without_optimizer, load_optimizer=False)
    assert not critic_without_optimizer.critic_optimizer.state
    metrics = offline_calibration_metrics(state.critic, batch)
    assert "q_rmse" in metrics
    assert "success_buffer_loss" in full_metrics
    assert "success_buffer_loss" in flash_metrics
