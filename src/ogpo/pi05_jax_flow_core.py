from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class JaxFlowRollout:
    states: jax.Array
    next_states: jax.Array
    timesteps: jax.Array
    log_probs: jax.Array
    endpoint: jax.Array


@dataclass(frozen=True)
class JaxFlashRollout:
    x_t: jax.Array
    x_prev: jax.Array
    timestep: jax.Array
    old_log_prob: jax.Array
    endpoint: jax.Array
    selected_step: jax.Array


def sample_flash_rollout(*, actor: Any, flow_spec: OpenPIJaxFlowSpec, observation: Any, group_size: int, selected_step: jax.Array, rng: jax.Array, sde_mode: str) -> JaxFlashRollout:
    batch = selected_step.shape[0]
    total = batch * group_size
    selected_g = jnp.repeat(selected_step, group_size)
    endpoint_dim = actor.model_horizon * actor.flow_action_dim
    timestep_values = flow_spec.timestep_values()
    x_t = jax.random.normal(rng, (total, endpoint_dim), dtype=jnp.float32)

    def body(carry, step_index):
        x_curr, key = carry
        key, subkey = jax.random.split(key)
        t_scalar = timestep_values[step_index]
        timestep = jnp.full((total, 1), t_scalar, dtype=x_curr.dtype)
        mean = transition_mean(actor=actor, flow_spec=flow_spec, x_t=x_curr, observation=observation, timestep=timestep, sde_mode=sde_mode)
        log_std = transition_log_std(actor=actor, flow_spec=flow_spec, x_t=x_curr, timestep=timestep, sde_mode=sde_mode)
        noise = jax.random.normal(subkey, mean.shape, dtype=mean.dtype)
        stochastic = mean + noise * jnp.exp(log_std)
        deterministic_mask = step_index != selected_g
        x_prev = jnp.where(deterministic_mask[:, None], mean, stochastic)
        log_prob = gaussian_log_prob(x_prev, mean, log_std)
        out = (x_curr, x_prev, timestep, log_prob)
        return (x_prev, key), out

    (_, _), (states, next_states, timesteps, log_probs) = jax.lax.scan(body, (x_t, rng), jnp.arange(flow_spec.num_steps))
    states = jnp.swapaxes(states, 0, 1)
    next_states = jnp.swapaxes(next_states, 0, 1)
    timesteps = jnp.swapaxes(timesteps, 0, 1)
    log_probs = jnp.swapaxes(log_probs, 0, 1)
    gather_idx = selected_g[:, None]
    batch_idx = jnp.arange(total)[:, None]
    return JaxFlashRollout(
        x_t=states[batch_idx, gather_idx].squeeze(1),
        x_prev=next_states[batch_idx, gather_idx].squeeze(1),
        timestep=timesteps[batch_idx, gather_idx].squeeze(1),
        old_log_prob=log_probs[batch_idx, gather_idx].squeeze(1),
        endpoint=next_states[:, -1],
        selected_step=selected_step,
    )


def rollout(*, actor: Any, flow_spec: OpenPIJaxFlowSpec, observation: Any, group_size: int, rng: jax.Array, sde_mode: str) -> JaxFlowRollout:
    batch = observation.state.shape[0]
    total = batch * group_size
    endpoint_dim = actor.model_horizon * actor.flow_action_dim
    timestep_values = flow_spec.timestep_values()
    x_t = jax.random.normal(rng, (total, endpoint_dim), dtype=jnp.float32)

    def _repeat_tree(value):
        if isinstance(value, dict):
            return {key: _repeat_tree(item) for key, item in value.items()}
        if value is None:
            return None
        return jnp.repeat(value, group_size, axis=0)

    observation_g = jax.tree_util.tree_map(lambda x: x, observation)
    observation_g = jax.tree_util.tree_map(lambda x: _repeat_tree(x) if not isinstance(x, type(None)) else x, observation_g)

    def body(carry, step_index):
        x_curr, key = carry
        key, subkey = jax.random.split(key)
        t_scalar = timestep_values[step_index]
        timestep = jnp.full((total, 1), t_scalar, dtype=x_curr.dtype)
        mean = transition_mean(actor=actor, flow_spec=flow_spec, x_t=x_curr, observation=observation_g, timestep=timestep, sde_mode=sde_mode)
        log_std = transition_log_std(actor=actor, flow_spec=flow_spec, x_t=x_curr, timestep=timestep, sde_mode=sde_mode)
        noise = jax.random.normal(subkey, mean.shape, dtype=mean.dtype)
        x_prev = mean + noise * jnp.exp(log_std)
        log_prob = gaussian_log_prob(x_prev, mean, log_std)
        out = (x_curr, x_prev, timestep, log_prob)
        return (x_prev, key), out

    (_, _), (states, next_states, timesteps, log_probs) = jax.lax.scan(body, (x_t, rng), jnp.arange(flow_spec.num_steps))
    return JaxFlowRollout(
        states=jnp.swapaxes(states, 0, 1),
        next_states=jnp.swapaxes(next_states, 0, 1),
        timesteps=jnp.swapaxes(timesteps, 0, 1),
        log_probs=jnp.swapaxes(log_probs, 0, 1),
        endpoint=jnp.swapaxes(next_states, 0, 1)[:, -1],
    )


@dataclass(frozen=True)
class JaxPPOStats:
    loss: jax.Array
    ratio: jax.Array
    clipped_ratio: jax.Array
    per_sample_loss: jax.Array


@dataclass(frozen=True)
class JaxFullPPOStats:
    loss: jax.Array
    ratio: jax.Array
    clipped_ratio: jax.Array


class OpenPIJaxFlowSpec:
    def __init__(self, num_steps: int):
        self.num_steps = int(num_steps)

    @property
    def dt(self) -> float:
        return -1.0 / max(1, self.num_steps)

    def timestep_values(self, *, dtype=jnp.float32) -> jax.Array:
        return jnp.linspace(1.0, 1.0 / max(1, self.num_steps), self.num_steps, dtype=dtype)

    def expand_timestep(self, timestep: jax.Array, target: jax.Array) -> jax.Array:
        t = timestep.astype(target.dtype)
        while t.ndim < target.ndim:
            t = jnp.expand_dims(t, axis=-1)
        return t

    def euler_step(self, x_t: jax.Array, velocity: jax.Array) -> jax.Array:
        return x_t + self.dt * velocity


def gaussian_log_prob(value: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    var = jnp.exp(2.0 * log_std)
    log_prob = -0.5 * (((value - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))
    return log_prob.reshape(log_prob.shape[0], -1).sum(axis=-1)


def gaussian_kl_diag(mean_p: jax.Array, log_std_p: jax.Array, mean_q: jax.Array, log_std_q: jax.Array) -> jax.Array:
    var_p = jnp.exp(2.0 * log_std_p)
    var_q = jnp.exp(2.0 * log_std_q)
    kl = log_std_q - log_std_p + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5
    return kl.reshape(kl.shape[0], -1).sum(axis=-1)


def state_adaptive_kl_penalty(
    transition_kl: jax.Array,
    entropy_norm: jax.Array,
    *,
    group_size: int,
    beta_base: float,
    adapt_kl_beta: bool,
    uncertainty_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Match the torch KL semantics while keeping U-state adaptation optional."""
    per_state_kl = transition_kl.reshape(entropy_norm.size, int(group_size)).mean(axis=1)
    if adapt_kl_beta:
        beta = float(beta_base) * (
            1.0 + float(uncertainty_scale) * jnp.clip(entropy_norm, 0.0, 1.0)
        )
    else:
        beta = jnp.full_like(entropy_norm, float(beta_base))
    return jnp.mean(beta * per_state_kl), jnp.mean(per_state_kl), jnp.mean(beta)


def flow_matching_loss(
    *,
    actor: Any,
    observation: Any,
    action_endpoint: jax.Array,
    noise: jax.Array,
    timestep: jax.Array,
) -> jax.Array:
    """PI0.5 linear-interpolation flow matching with gradients through actor."""
    actions = action_endpoint.reshape(
        action_endpoint.shape[0],
        actor.model_horizon,
        actor.flow_action_dim,
    )
    noise = noise.reshape(actions.shape)
    time = timestep.reshape(-1)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    target_velocity = noise - actions
    predicted = actor.predict_velocity(observation, x_t, time).reshape(actions.shape)
    return jnp.mean(jnp.square(predicted - target_velocity))


def transition_mean(*, actor: Any, flow_spec: OpenPIJaxFlowSpec, x_t: jax.Array, observation: Any, timestep: jax.Array, sde_mode: str) -> jax.Array:
    velocity = actor.predict_velocity(
        observation,
        x_t.reshape(x_t.shape[0], actor.model_horizon, actor.flow_action_dim),
        timestep.reshape(-1),
    )
    velocity = velocity.reshape(x_t.shape)
    if sde_mode == "ogpo_corrected":
        t = flow_spec.expand_timestep(timestep, x_t)
        sigma_squared = jnp.exp(2.0 * actor.log_std.value).reshape((1, -1)).astype(x_t.dtype)
        sigma_squared = jnp.broadcast_to(sigma_squared, x_t.shape)
        velocity = velocity + 0.5 * sigma_squared * ((1.0 - t) * velocity + x_t)
    return flow_spec.euler_step(x_t, velocity)


def transition_log_std(*, actor: Any, flow_spec: OpenPIJaxFlowSpec, x_t: jax.Array, timestep: jax.Array, sde_mode: str) -> jax.Array:
    base_std = jnp.exp(actor.log_std.value).reshape((1, -1)).astype(x_t.dtype)
    base_std = jnp.broadcast_to(base_std, x_t.shape)
    if sde_mode == "gaussian_adapter":
        return jnp.log(jnp.clip(base_std, a_min=jnp.finfo(x_t.dtype).tiny))
    t = jnp.clip(flow_spec.expand_timestep(timestep, x_t), a_min=0.0, a_max=1.0)
    return jnp.log(jnp.clip(base_std * jnp.sqrt(t), a_min=jnp.finfo(x_t.dtype).tiny))


def transition_log_prob(*, actor: Any, flow_spec: OpenPIJaxFlowSpec, x_prev: jax.Array, x_t: jax.Array, observation: Any, timestep: jax.Array, sde_mode: str) -> jax.Array:
    mean = transition_mean(actor=actor, flow_spec=flow_spec, x_t=x_t, observation=observation, timestep=timestep, sde_mode=sde_mode)
    log_std = transition_log_std(actor=actor, flow_spec=flow_spec, x_t=x_t, timestep=timestep, sde_mode=sde_mode)
    return gaussian_log_prob(x_prev, mean, log_std)


def transition_kl(*, actor: Any, other_actor: Any, flow_spec: OpenPIJaxFlowSpec, x_t: jax.Array, observation: Any, timestep: jax.Array, sde_mode: str) -> jax.Array:
    return gaussian_kl_diag(
        transition_mean(actor=actor, flow_spec=flow_spec, x_t=x_t, observation=observation, timestep=timestep, sde_mode=sde_mode),
        transition_log_std(actor=actor, flow_spec=flow_spec, x_t=x_t, timestep=timestep, sde_mode=sde_mode),
        transition_mean(actor=other_actor, flow_spec=flow_spec, x_t=x_t, observation=observation, timestep=timestep, sde_mode=sde_mode),
        transition_log_std(actor=other_actor, flow_spec=flow_spec, x_t=x_t, timestep=timestep, sde_mode=sde_mode),
    )


def flash_ppo_loss(new_log_prob: jax.Array, old_log_prob: jax.Array, advantages: jax.Array, *, clip_eps: jax.Array, rectification_weight: jax.Array, log_ratio_clip: float = 20.0) -> JaxPPOStats:
    log_ratio = jnp.clip(new_log_prob - old_log_prob, -log_ratio_clip, log_ratio_clip)
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    objective = jnp.minimum(ratio * advantages, clipped_ratio * advantages) * rectification_weight
    return JaxPPOStats(loss=-jnp.mean(objective), ratio=ratio, clipped_ratio=clipped_ratio, per_sample_loss=-objective)


def full_chain_ppo_loss(new_log_probs: jax.Array, old_log_probs: jax.Array, advantages: jax.Array, *, clip_eps: jax.Array, timestep_weights: jax.Array | None = None, log_ratio_clip: float = 20.0) -> JaxFullPPOStats:
    log_ratio = jnp.clip(new_log_probs - old_log_probs, -log_ratio_clip, log_ratio_clip)
    ratio = jnp.exp(log_ratio)
    adv = advantages[:, None]
    if clip_eps.ndim == 1:
        clip_eps = clip_eps[:, None]
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    objective = jnp.minimum(ratio * adv, clipped_ratio * adv)
    if timestep_weights is not None:
        objective = objective * timestep_weights
    return JaxFullPPOStats(loss=-jnp.mean(objective), ratio=ratio, clipped_ratio=clipped_ratio)


def full_chain_ais_ppo_loss(new_log_probs: jax.Array, old_log_probs: jax.Array, advantages: jax.Array, *, clip_eps: jax.Array, log_ratio_clip: float = 20.0) -> JaxFullPPOStats:
    log_ratio = jnp.sum(new_log_probs - old_log_probs, axis=1)
    ratio = jnp.exp(jnp.clip(log_ratio, -log_ratio_clip, log_ratio_clip))
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    objective = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
    return JaxFullPPOStats(loss=-jnp.mean(objective), ratio=ratio, clipped_ratio=clipped_ratio)
