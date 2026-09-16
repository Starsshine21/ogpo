from __future__ import annotations

import abc
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn

from .flow_logprob import gaussian_kl_diag, gaussian_log_prob


@dataclass(frozen=True)
class FlowRollout:
    states: torch.Tensor
    next_states: torch.Tensor
    timesteps: torch.Tensor
    log_probs: torch.Tensor
    endpoint: torch.Tensor
    raw_velocity_norms: torch.Tensor | None = None
    corrected_drift_norms: torch.Tensor | None = None
    sde_correction_norms: torch.Tensor | None = None
    transition_means: torch.Tensor | None = None
    transition_stds: torch.Tensor | None = None
    stochastic_masks: torch.Tensor | None = None


@dataclass(frozen=True)
class OpenPIFlowSpec:
    """OpenPI PI0/PI0.5 flow-matching convention.

    OpenPI uses t=1 for noise and t=0 for the clean action endpoint. The
    learned velocity points from the clean action toward the sampled noise, and
    inference integrates backward with a negative Euler step.
    """

    num_steps: int = 10

    @property
    def dt(self) -> float:
        return -1.0 / max(1, int(self.num_steps))

    def timestep_values(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(
            1.0,
            1.0 / max(1, int(self.num_steps)),
            int(self.num_steps),
            device=device,
            dtype=dtype,
        )

    def expand_timestep(self, timestep: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=target.dtype, device=target.device)
        while t.ndim < target.ndim:
            t = t.unsqueeze(-1)
        return t

    def sample_training_time(
        self,
        batch_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        uniform = torch.rand(batch_shape, device=device, dtype=dtype, generator=generator)
        return uniform.pow(1.0 / 1.5) * 0.999 + 0.001

    def training_pair(
        self,
        action_endpoint: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.expand_timestep(timestep, action_endpoint)
        x_t = t * noise + (1.0 - t) * action_endpoint
        target_velocity = noise - action_endpoint
        return x_t, target_velocity

    def euler_step(self, x_t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return x_t + self.dt * velocity


class OpenPIStochasticFlowPolicy(nn.Module, abc.ABC):
    """Stochastic transition wrapper around OpenPI's deterministic Euler mean.

    Subclasses provide `predict_velocity(x_t, condition, timestep)`. The
    wrapper turns each deterministic Euler update into a diagonal Gaussian
    transition so PPO-style objectives can evaluate transition log-probability
    ratios without learning an inner flow-state value function.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        num_steps: int = 10,
        stochastic_variance: float = 0.04,
        sde_mode: str = "gaussian_adapter",
        constant_noise_std: float = 0.005,
        learn_sde_std: bool = True,
        randn_clip_value: float = 3.0,
    ):
        super().__init__()
        if sde_mode not in {
            "gaussian_adapter",
            "ogpo_corrected",
            "ogpo_constant_corrected",
        }:
            raise ValueError(f"unsupported SDE mode: {sde_mode}")
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        if not math.isfinite(float(constant_noise_std)) or float(constant_noise_std) <= 0.0:
            raise ValueError("constant_noise_std must be finite and positive")
        if not math.isfinite(float(randn_clip_value)) or float(randn_clip_value) <= 0.0:
            raise ValueError("randn_clip_value must be finite and positive")
        self.action_dim = int(action_dim)
        self.num_steps = int(num_steps)
        self.sde_mode = sde_mode
        self.constant_noise_std = float(constant_noise_std)
        self.learn_sde_std = bool(learn_sde_std)
        self.randn_clip_value = float(randn_clip_value)
        self._capture_transition_diagnostics = False
        self._last_transition_diagnostics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._last_transition_parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.flow_spec = OpenPIFlowSpec(num_steps=self.num_steps)
        init_std = (
            self.constant_noise_std
            if self.sde_mode == "ogpo_constant_corrected"
            else math.sqrt(float(stochastic_variance))
        )
        if not math.isfinite(float(init_std)) or float(init_std) <= 0.0:
            raise ValueError("stochastic_variance must imply a finite positive std")
        log_std = torch.full((self.action_dim,), math.log(init_std))
        if self.sde_mode == "ogpo_constant_corrected" and not self.learn_sde_std:
            # A buffer makes the clean reference's noise scale impossible to
            # optimize and keeps it out of the actor optimizer parameter list.
            self.register_buffer("log_std", log_std)
        else:
            self.log_std = nn.Parameter(log_std)

    @abc.abstractmethod
    def predict_velocity(self, x_t: torch.Tensor, condition: Any, timestep: torch.Tensor) -> torch.Tensor:
        """Predict the PI0.5 flow velocity for one batched latent state."""

    def condition_batch_size(self, condition: Any) -> int:
        return int(condition.shape[0])

    def condition_device_dtype(self, condition: Any) -> tuple[torch.device, torch.dtype]:
        return condition.device, condition.dtype

    def repeat_condition(self, condition: Any, repeats: int) -> Any:
        return condition.repeat_interleave(repeats, dim=0)

    def action_chunks_to_flow(self, batch: Any) -> torch.Tensor:
        """Map replay action chunks into the flow model's action space."""
        return batch.action_chunks

    def flat_actions_to_environment(
        self,
        flat_actions: torch.Tensor,
        condition: Any | None = None,
    ) -> torch.Tensor:
        """Map flat flow endpoints into the critic/environment action space."""
        del condition
        return flat_actions

    def _final_transition_mask(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return [batch] mask for the deterministic final reverse-time step."""
        if timestep.ndim == 0:
            t_local = timestep.reshape(1)
        else:
            t_local = timestep.reshape(timestep.shape[0], -1)[:, 0]
        # OpenPI's reverse-time schedule is t_local=1 (noise) down to 1/K
        # (clean endpoint).  The official OGPO schedule is the opposite
        # direction, t_ogpo=1-t_local; its final t->1 step is this last local
        # transition.  Keep the mapping in one helper for sampling/log-prob.
        return t_local <= (1.0 / float(self.num_steps) + 1e-6)

    def _initial_transition_mask(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            t_local = timestep.reshape(1)
        else:
            t_local = timestep.reshape(timestep.shape[0], -1)[:, 0]
        return t_local >= 1.0 - 1e-6

    def _constant_sde_initial_log_prob(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Official full-chain initial N(0,I) prior contribution."""
        if self.sde_mode != "ogpo_constant_corrected":
            return x_t.new_zeros(x_t.shape[0])
        prior = gaussian_log_prob(
            x_t,
            torch.zeros_like(x_t),
            torch.zeros_like(x_t),
        )
        return prior * self._initial_transition_mask(timestep).to(prior.dtype)

    def ogpo_time(self, timestep: torch.Tensor) -> torch.Tensor:
        """Map local OpenPI reverse time to OGPO's noise->data time."""
        t_local = self.flow_spec.expand_timestep(timestep, timestep)
        return (1.0 - t_local).clamp(0.0, 1.0)

    def _base_noise_std(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.sde_mode == "ogpo_constant_corrected" and not self.learn_sde_std:
            return x_t.new_full((), self.constant_noise_std).expand_as(x_t)
        return self.log_std.to(x_t.device, x_t.dtype).exp().expand_as(x_t)

    def _transition_velocity(
        self,
        x_t: torch.Tensor,
        condition: Any,
        timestep: torch.Tensor,
        *,
        apply_sde_correction: bool = True,
    ) -> torch.Tensor:
        velocity = self.predict_velocity(x_t, condition, timestep)
        raw_velocity = velocity
        if not apply_sde_correction:
            return velocity
        if self.sde_mode == "ogpo_corrected":
            # OGPO is written in cleanward time tau=1-t. Mapping its tapered
            # CondOT correction back to PI's reverse-time derivative gives
            # v_pi + sigma_base^2 / 2 * ((1-t) v_pi + x_t).
            t = self.flow_spec.expand_timestep(timestep, x_t)
            sigma_squared = self._base_noise_std(x_t).square()
            velocity = velocity + 0.5 * sigma_squared * ((1.0 - t) * velocity + x_t)
        elif self.sde_mode == "ogpo_constant_corrected":
            # PyTorch/OpenPI uses reverse time (noise at t_local=1), while
            # OGPO defines t_ogpo=0 at noise and t_ogpo->1 at data.  Reversing
            # the OGPO CondOT score correction gives this local-time form:
            #   v_local + sigma^2/2 * ((1-t_local)v_local+x_t)/t_local.
            # This is a direct PyTorch port of OGPO_public
            # pg_helper.py::sde_drift_correction at commit
            # 0b3be413cde766a41257c6b19c0c2b06393a557f.  The last transition
            # is deterministic and must not evaluate the singular correction.
            final_mask = self._final_transition_mask(timestep)
            # The official constant-noise branch does not evaluate the score
            # correction at all on its deterministic final transition.
            if bool(final_mask.all()):
                if self._capture_transition_diagnostics:
                    raw_flat = raw_velocity.detach().float().flatten(start_dim=1)
                    raw_norm = raw_flat.norm(dim=1)
                    self._last_transition_diagnostics = (
                        raw_norm,
                        raw_norm,
                        torch.zeros_like(raw_norm),
                    )
                return self.flow_spec.euler_step(x_t, velocity)
            t_local = self.flow_spec.expand_timestep(timestep, x_t).clamp_min(1e-6)
            sigma_squared = self._base_noise_std(x_t).square()
            correction = 0.5 * sigma_squared * (
                ((1.0 - t_local) * velocity + x_t) / t_local
            )
            while final_mask.ndim < correction.ndim:
                final_mask = final_mask.unsqueeze(-1)
            velocity = velocity + torch.where(
                final_mask, torch.zeros_like(correction), correction
            )
        if self._capture_transition_diagnostics:
            raw_flat = raw_velocity.detach().float().flatten(start_dim=1)
            corrected_flat = velocity.detach().float().flatten(start_dim=1)
            correction_flat = (velocity - raw_velocity).detach().float().flatten(start_dim=1)
            self._last_transition_diagnostics = (
                raw_flat.norm(dim=1),
                corrected_flat.norm(dim=1),
                correction_flat.norm(dim=1),
            )
        return self.flow_spec.euler_step(x_t, velocity)

    def transition_mean(
        self,
        x_t: torch.Tensor,
        condition: Any,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Mean of the training SDE transition (legacy public API)."""
        return self._transition_velocity(x_t, condition, timestep)

    def native_transition_mean(
        self,
        x_t: torch.Tensor,
        condition: Any,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Native π0.5 ODE step, deliberately excluding all SDE correction."""
        return self.flow_spec.euler_step(
            x_t, self._transition_velocity(x_t, condition, timestep, apply_sde_correction=False)
        )

    def transition_parameters(
        self,
        x_t: torch.Tensor,
        condition: Any,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared transition definition used by sampler and every log-prob path.

        Returns ``(mean, std, stochastic_mask)``.  The mask is [batch] and is
        false only for the constant-corrected deterministic final step.
        """
        mean = self._transition_velocity(x_t, condition, timestep)
        if self.sde_mode == "ogpo_constant_corrected":
            std = self._base_noise_std(x_t)
            final_mask = self._final_transition_mask(timestep)
            while final_mask.ndim < std.ndim:
                final_mask = final_mask.unsqueeze(-1)
            std = torch.where(final_mask, torch.zeros_like(std), std)
            stochastic_mask = ~self._final_transition_mask(timestep)
        else:
            std = self.transition_std(x_t, timestep)
            stochastic_mask = torch.ones(
                x_t.shape[0], dtype=torch.bool, device=x_t.device
            )
        return mean, std, stochastic_mask

    def transition_std(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        base_std = self._base_noise_std(x_t)
        if self.sde_mode == "gaussian_adapter":
            return base_std
        if self.sde_mode == "ogpo_constant_corrected":
            final_mask = self._final_transition_mask(timestep)
            while final_mask.ndim < base_std.ndim:
                final_mask = final_mask.unsqueeze(-1)
            return torch.where(final_mask, torch.zeros_like(base_std), base_std)
        t = self.flow_spec.expand_timestep(timestep, x_t).clamp(min=0.0, max=1.0)
        return base_std * t.sqrt()

    def transition_log_std(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.transition_std(x_t, timestep).clamp_min(torch.finfo(x_t.dtype).tiny).log()

    def sample_transition(
        self,
        x_t: torch.Tensor,
        condition: Any,
        timestep: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool | torch.Tensor = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std, stochastic_mask = self.transition_parameters(
            x_t, condition, timestep
        )
        if self._capture_transition_diagnostics:
            self._last_transition_parameters = (
                mean.detach(),
                std.detach(),
                stochastic_mask.detach(),
            )
        # Avoid ever passing std=0 to the Gaussian helper.  The corresponding
        # final transition is a Dirac/Euler step and is masked from log-prob.
        # Use a finite dummy unit variance for deterministic entries.  A
        # tiny epsilon can underflow ``exp(2*log_std)`` to zero and produce
        # ``0/0`` NaNs before the final-step log-prob mask is applied.
        safe_std = torch.where(std > 0.0, std, torch.ones_like(std))
        log_std = safe_std.log()
        deterministic_mask = torch.zeros(
            x_t.shape[0], dtype=torch.bool, device=x_t.device
        )
        if isinstance(deterministic, bool) and deterministic:
            x_prev = mean
            deterministic_mask.fill_(True)
        elif isinstance(deterministic, torch.Tensor):
            mask = deterministic.to(dtype=torch.bool, device=mean.device)
            while mask.ndim < mean.ndim:
                mask = mask.unsqueeze(-1)
            deterministic_mask = deterministic.to(dtype=torch.bool, device=mean.device).reshape(-1)
            if bool((~deterministic_mask & stochastic_mask).any()):
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device=mean.device,
                    generator=generator,
                )
                stochastic = mean + noise * log_std.exp()
                if self.sde_mode == "ogpo_constant_corrected":
                    stochastic = torch.maximum(
                        torch.minimum(
                            stochastic,
                            mean + self.randn_clip_value * std,
                        ),
                        mean - self.randn_clip_value * std,
                    )
                x_prev = torch.where(mask, mean, stochastic)
            else:
                x_prev = mean
        else:
            if self.sde_mode == "ogpo_constant_corrected" and not bool(stochastic_mask.any()):
                x_prev = mean
            else:
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device=mean.device,
                    generator=generator,
                )
                x_prev = mean + noise * log_std.exp()
                if self.sde_mode == "ogpo_constant_corrected":
                    x_prev = torch.maximum(
                        torch.minimum(
                            x_prev,
                            mean + self.randn_clip_value * std,
                        ),
                        mean - self.randn_clip_value * std,
                    )
        log_prob = gaussian_log_prob(x_prev, mean, log_std)
        if self.sde_mode == "ogpo_constant_corrected":
            # Official constant SDE skips the final deterministic transition;
            # explicit deterministic_except_selected masks are still scored
            # normally for compatibility with legacy callers.
            log_prob = log_prob * stochastic_mask.to(log_prob.dtype)
            log_prob = log_prob + self._constant_sde_initial_log_prob(x_t, timestep)
        del deterministic_mask
        return x_prev, log_prob

    def log_prob(
        self,
        x_prev: torch.Tensor,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        mean, std, stochastic_mask = self.transition_parameters(
            x_t, condition, timestep
        )
        log_prob = gaussian_log_prob(
            x_prev,
            mean,
            torch.where(std > 0.0, std, torch.ones_like(std)).log(),
        )
        if self.sde_mode == "ogpo_constant_corrected":
            log_prob = log_prob * stochastic_mask.to(log_prob.dtype)
            log_prob = log_prob + self._constant_sde_initial_log_prob(x_t, timestep)
        return log_prob

    def kl_to(
        self,
        other: "OpenPIStochasticFlowPolicy",
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        mean_p, std_p, mask_p = self.transition_parameters(x_t, condition, timestep)
        mean_q, std_q, mask_q = other.transition_parameters(x_t, condition, timestep)
        safe_p = torch.where(std_p > 0.0, std_p, torch.ones_like(std_p))
        safe_q = torch.where(std_q > 0.0, std_q, torch.ones_like(std_q))
        value = gaussian_kl_diag(
            mean_p,
            safe_p.log(),
            mean_q,
            safe_q.log(),
        )
        if self.sde_mode == "ogpo_constant_corrected" or other.sde_mode == "ogpo_constant_corrected":
            value = value * (mask_p & mask_q).to(value.dtype)
        return value

    def rollout(
        self,
        condition: torch.Tensor,
        *,
        group_size: int = 1,
        selected_timestep: int | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        deterministic_except_selected: bool = False,
    ) -> FlowRollout:
        batch = self.condition_batch_size(condition)
        device, dtype = self.condition_device_dtype(condition)
        condition_g = self.repeat_condition(condition, group_size)
        selected_g: int | torch.Tensor | None
        if isinstance(selected_timestep, torch.Tensor):
            selected = selected_timestep.to(device=device, dtype=torch.long)
            if selected.shape == (batch,):
                selected_g = selected.repeat_interleave(group_size)
            elif selected.shape == (batch * group_size,):
                selected_g = selected
            else:
                raise ValueError("selected_timestep tensor must have shape [batch] or [batch * group_size]")
        else:
            selected_g = selected_timestep
        x_t = torch.randn(
            batch * group_size,
            self.action_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        states = []
        next_states = []
        timesteps = []
        log_probs = []
        raw_velocity_norms = []
        corrected_drift_norms = []
        sde_correction_norms = []
        transition_means = []
        transition_stds = []
        stochastic_masks = []
        timestep_values = self.flow_spec.timestep_values(device=device, dtype=dtype)
        self._capture_transition_diagnostics = True
        try:
            for step_index, t_scalar in enumerate(timestep_values):
                t_value = t_scalar.expand(batch * group_size, 1)
                deterministic = (
                    deterministic_except_selected
                    and selected_g is not None
                    and (
                        (step_index != selected_g)
                        if isinstance(selected_g, torch.Tensor)
                        else step_index != selected_g
                    )
                )
                x_prev, log_prob = self.sample_transition(
                    x_t,
                    condition_g,
                    t_value,
                    generator=generator,
                    deterministic=deterministic,
                )
                states.append(x_t)
                next_states.append(x_prev)
                timesteps.append(t_value)
                log_probs.append(log_prob)
                if self._last_transition_diagnostics is None:
                    raise AssertionError("transition diagnostics were not captured")
                if self._last_transition_parameters is None:
                    raise AssertionError("transition parameters were not captured")
                raw_norm, drift_norm, correction_norm = self._last_transition_diagnostics
                captured_mean, captured_std, captured_mask = self._last_transition_parameters
                raw_velocity_norms.append(raw_norm)
                corrected_drift_norms.append(drift_norm)
                sde_correction_norms.append(correction_norm)
                transition_means.append(captured_mean)
                transition_stds.append(captured_std)
                stochastic_masks.append(captured_mask)
                x_t = x_prev
        finally:
            self._capture_transition_diagnostics = False
            self._last_transition_diagnostics = None
            self._last_transition_parameters = None
        return FlowRollout(
            states=torch.stack(states, dim=1),
            next_states=torch.stack(next_states, dim=1),
            timesteps=torch.stack(timesteps, dim=1),
            log_probs=torch.stack(log_probs, dim=1),
            endpoint=x_t,
            raw_velocity_norms=torch.stack(raw_velocity_norms, dim=1),
            corrected_drift_norms=torch.stack(corrected_drift_norms, dim=1),
            sde_correction_norms=torch.stack(sde_correction_norms, dim=1),
            transition_means=torch.stack(transition_means, dim=1),
            transition_stds=torch.stack(transition_stds, dim=1),
            stochastic_masks=torch.stack(stochastic_masks, dim=1),
        )
