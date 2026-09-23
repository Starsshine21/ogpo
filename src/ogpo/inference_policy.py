from __future__ import annotations

import copy
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from .critic import ScalarQEnsemble
from .gemma_siglip_backbone import GemmaSiglipStateBackbone, install_final_gemma_lora
from .multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic
from .openpi_flow_spec import OpenPIStochasticFlowPolicy
from .pi05_jax_adapter import PI05JaxFlowPolicy, _ensure_orbax_jax_compatibility
from .pi05_pytorch_adapter import PI05FlowCondition, PI05PytorchFlowPolicy


def _to_batched_torch(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _to_batched_torch(item, device=device) for key, item in value.items()}
    array = np.asarray(value)
    return torch.as_tensor(array, device=device).unsqueeze(0)


def _to_unbatched_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_unbatched_numpy(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value[0].detach().cpu().numpy()
    return np.asarray(value)[0]


class PI05OGPOInferencePolicy:
    """OpenPI server-compatible policy that applies an OGPO residual adapter."""

    def __init__(
        self,
        flow_policy: OpenPIStochasticFlowPolicy,
        *,
        input_transform,
        output_transform,
        observation_type,
        metadata: dict[str, Any] | None = None,
        reference_flow_policy: OpenPIStochasticFlowPolicy | None = None,
        critic: torch.nn.Module | None = None,
        executed_horizon: int | None = None,
        critic_camera_keys: tuple[str, ...] = (),
        critic_online_camera_mapping: dict[str, str] | None = None,
        default_language: str = "",
        dynamics: str = "native_ode",
    ):
        self.flow_policy = flow_policy.eval()
        self.reference_flow_policy = None if reference_flow_policy is None else reference_flow_policy.eval()
        self.critic = None if critic is None else critic.eval()
        self.executed_horizon = executed_horizon
        self.critic_camera_keys = tuple(critic_camera_keys)
        self.critic_online_camera_mapping = dict(critic_online_camera_mapping or {})
        self.default_language = str(default_language)
        self.dynamics = str(dynamics).lower()
        if self.dynamics == "training_sde_drift":
            self.dynamics = "training_sde"
        if self.dynamics not in {"native_ode", "training_sde"}:
            raise ValueError("inference dynamics must be 'native_ode' or 'training_sde'")
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.observation_type = observation_type
        self._metadata = metadata or {}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @torch.no_grad()
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        started = time.monotonic()
        device = self.flow_policy.log_std.device
        raw_obs = copy.deepcopy(obs)
        policy_noise_seed = raw_obs.pop("_policy_noise_seed", None)
        request_noise = raw_obs.pop("_policy_noise", None)
        if noise is not None and request_noise is not None:
            raise ValueError("noise was provided both as an argument and in the observation")
        if request_noise is not None:
            noise = np.asarray(request_noise, dtype=np.float32)
        if (
            isinstance(self.flow_policy, PI05JaxFlowPolicy)
            and self.reference_flow_policy is None
            and self.critic is None
        ):
            transformed = self.input_transform(raw_obs)
            jax_inputs = jax.tree.map(
                lambda value: jnp.asarray(value)[None, ...],
                transformed,
            )
            observation = self.observation_type.from_dict(jax_inputs)
            flow_actions = self.flow_policy.sample_actions_jax(
                observation,
                noise=noise,
                noise_seed=(
                    int(policy_noise_seed)
                    if policy_noise_seed is not None
                    else None
                ),
            )
            model_actions = jnp.zeros(
                (
                    flow_actions.shape[0],
                    self.flow_policy.model_horizon,
                    self.flow_policy.model_action_dim,
                ),
                dtype=flow_actions.dtype,
            )
            model_actions = model_actions.at[
                ..., : self.flow_policy.inference_flow_action_dim
            ].set(flow_actions)
            outputs = {
                "state": np.asarray(jax_inputs["state"][0]),
                "actions": np.asarray(model_actions[0]),
            }
            outputs = self.output_transform(outputs)
            outputs["policy_timing"] = {
                "infer_ms": (time.monotonic() - started) * 1000.0
            }
            return outputs
        transformed = self.input_transform(raw_obs)
        inputs = _to_batched_torch(transformed, device=device)
        observation = self.observation_type.from_dict(inputs)
        condition = PI05FlowCondition(observation)
        batch_size = condition.batch_size
        horizon = self.flow_policy.model_horizon
        env_dim = self.flow_policy.environment_action_dim
        flow_dim = getattr(
            self.flow_policy,
            "inference_flow_action_dim",
            getattr(self.flow_policy, "flow_action_dim", env_dim),
        )

        if noise is None:
            generator = None
            if policy_noise_seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(policy_noise_seed))
            x_t = torch.randn(
                batch_size,
                horizon * flow_dim,
                device=device,
                generator=generator,
            )
        else:
            noise_tensor = torch.as_tensor(noise, dtype=torch.float32, device=device)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor.unsqueeze(0)
            if (
                noise_tensor.shape[-1] == self.flow_policy.model_action_dim
                and flow_dim < self.flow_policy.model_action_dim
            ):
                noise_tensor = noise_tensor[..., :flow_dim]
            expected = (batch_size, horizon, flow_dim)
            if tuple(noise_tensor.shape) != expected:
                raise ValueError(f"expected inference noise shape {expected}, got {tuple(noise_tensor.shape)}")
            x_t = noise_tensor.reshape(batch_size, -1)

        reference_x_t = x_t.clone() if self.reference_flow_policy is not None else None
        for timestep in self.flow_policy.flow_spec.timestep_values(device=device, dtype=torch.float32):
            time_batch = timestep.expand(batch_size, 1)
            transition = (
                self.flow_policy.native_transition_mean
                if self.dynamics == "native_ode"
                and isinstance(self.flow_policy, PI05PytorchFlowPolicy)
                else self.flow_policy.transition_mean
            )
            x_t = transition(x_t, condition, time_batch)
            if reference_x_t is not None:
                reference_transition = (
                    self.reference_flow_policy.native_transition_mean
                    if self.dynamics == "native_ode"
                    and isinstance(self.reference_flow_policy, PI05PytorchFlowPolicy)
                    else self.reference_flow_policy.transition_mean
                )
                reference_x_t = reference_transition(
                    reference_x_t,
                    condition,
                    time_batch,
                )

        flow_actions = x_t.reshape(batch_size, horizon, flow_dim)
        model_actions = flow_actions.new_zeros(batch_size, horizon, self.flow_policy.model_action_dim)
        model_actions[..., :flow_dim] = flow_actions
        outputs = _to_unbatched_numpy({"state": inputs["state"], "actions": model_actions})
        outputs = self.output_transform(outputs)
        if reference_x_t is not None:
            reference_flow_actions = reference_x_t.reshape(batch_size, horizon, flow_dim)
            reference_model_actions = reference_flow_actions.new_zeros(
                batch_size,
                horizon,
                self.flow_policy.model_action_dim,
            )
            reference_model_actions[..., :flow_dim] = reference_flow_actions
            reference_outputs = _to_unbatched_numpy(
                {"state": inputs["state"], "actions": reference_model_actions}
            )
            reference_outputs = self.output_transform(reference_outputs)
            outputs["policy_reference_action_divergence"] = float(
                np.mean(
                    np.square(
                        np.asarray(outputs["actions"], dtype=np.float64)
                        - np.asarray(reference_outputs["actions"], dtype=np.float64)
                    )
                )
            )
        if self.critic is not None:
            raw_actions = torch.as_tensor(
                np.asarray(outputs["actions"], dtype=np.float32),
                device=device,
            ).unsqueeze(0)
            raw_state = torch.as_tensor(
                np.asarray(obs["state"], dtype=np.float32),
                device=device,
            ).reshape(1, -1)
            executed_horizon = min(
                horizon,
                int(self.executed_horizon if self.executed_horizon is not None else horizon),
            )
            execution_mask = torch.arange(horizon, device=device).unsqueeze(0) < executed_horizon
            if isinstance(self.critic, MultiHeadUdivlCritic):
                online_camera_keys = {
                    key: (
                        key
                        if key in obs
                        else self.critic_online_camera_mapping.get(key, key)
                    )
                    for key in self.critic_camera_keys
                }
                missing = [
                    key
                    for key, online_key in online_camera_keys.items()
                    if online_key not in obs
                ]
                if missing:
                    raise KeyError(f"online observation is missing critic cameras: {missing}")
                language = obs.get("prompt", obs.get("language", self.default_language))
                if isinstance(language, np.ndarray) and language.ndim == 0:
                    language = language.item()
                if isinstance(language, bytes):
                    language = language.decode("utf-8")
                online_batch = SimpleNamespace(
                    images={
                        key: torch.as_tensor(
                            np.asarray(obs[online_camera_keys[key]]),
                            device=device,
                        ).unsqueeze(0)
                        for key in self.critic_camera_keys
                    },
                    next_images=None,
                    proprioceptions=raw_state,
                    next_proprioceptions=raw_state,
                    languages=[str(language)],
                )
                features = self.critic.encode_state(online_batch)
                q_values = self.critic.q_from_features(features, raw_actions, execution_mask)
            else:
                q_values = self.critic(raw_state, raw_actions, execution_mask)
            outputs["predicted_q"] = float(q_values.mean().item())
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
        return outputs


def create_pi05_ogpo_inference_policy(
    *,
    pi05_checkpoint_dir: str | Path,
    train_config_name: str,
    ogpo_checkpoint: str | Path,
    device: str = "cuda",
    include_reference_policy: bool = True,
    include_critic: bool = True,
    inference_dynamics: str | None = None,
    inference_num_steps: int | None = None,
    require_model_flow_dim: bool = False,
) -> PI05OGPOInferencePolicy:
    """Load a PI0.5 base (JAX or PyTorch) and a compact OGPO residual checkpoint.

    The backend is auto-detected: if ``model.safetensors`` is present the
    converted PyTorch model is used, otherwise the native JAX (Orbax) checkpoint
    is loaded directly without any JAX->PyTorch conversion.
    """
    from openpi.models import model as openpi_model  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    pi05_checkpoint_dir = Path(pi05_checkpoint_dir).expanduser().resolve()
    is_pytorch = (pi05_checkpoint_dir / "model.safetensors").exists()
    if is_pytorch:
        trained_policy = policy_config.create_trained_policy(
            training_config.get_config(train_config_name),
            pi05_checkpoint_dir,
            pytorch_device=device,
        )
        if not getattr(trained_policy, "_is_pytorch_model", False):
            raise TypeError("model.safetensors present but policy is not PyTorch")
        model_horizon = int(trained_policy._model.config.action_horizon)
        flow_policy_cls = PI05PytorchFlowPolicy
    else:
        _ensure_orbax_jax_compatibility()
        trained_policy = policy_config.create_trained_policy(
            training_config.get_config(train_config_name),
            pi05_checkpoint_dir,
        )
        if getattr(trained_policy, "_is_pytorch_model", False):
            raise TypeError("pi05_jax inference requires a native JAX checkpoint (no model.safetensors)")
        model_horizon = int(trained_policy._model.action_horizon)
        flow_policy_cls = PI05JaxFlowPolicy

    payload = torch.load(Path(ogpo_checkpoint), map_location="cpu", weights_only=False)
    policy_payload = payload["policy"]
    policy_format = policy_payload.get("format")
    if policy_format not in {
        "pi05_residual_adapter",
        "pi05_jax_full_finetune",
        "pi05_pytorch_full_finetune",
    }:
        raise ValueError("checkpoint does not contain a PI0.5 OGPO actor")
    if policy_format == "pi05_jax_full_finetune" and is_pytorch:
        raise TypeError("full-finetune JAX actor cannot be loaded into a PyTorch PI0.5 backend")
    if policy_format == "pi05_pytorch_full_finetune" and not is_pytorch:
        raise TypeError("full-finetune PyTorch actor requires a PyTorch PI0.5 backend")
    adapter_state = policy_payload["state"]
    backend_tensor_count = sum(key.startswith("backend.") for key in adapter_state)
    if policy_format == "pi05_pytorch_full_finetune" and backend_tensor_count == 0:
        raise ValueError("full-finetune PyTorch actor checkpoint contains no backend tensors")
    print(
        "[ogpo-inference] "
        f"format={policy_format} checkpoint_step={payload.get('step', 'unknown')} "
        f"actor_tensors={len(adapter_state)} backend_tensors={backend_tensor_count} "
        f"checkpoint={Path(ogpo_checkpoint).expanduser().resolve()}",
        flush=True,
    )
    environment_action_dim = int(
        policy_payload.get(
            "environment_action_dim",
            adapter_state["residual.2.bias"].numel(),
        )
    )
    flow_action_dim = int(
        policy_payload.get(
            "flow_action_dim",
            adapter_state["log_std"].numel() // model_horizon,
        )
    )
    model_action_dim = int(
        trained_policy._model.config.action_dim
        if is_pytorch
        else trained_policy._model.action_dim
    )
    inference_flow_action_dim = model_action_dim if require_model_flow_dim else flow_action_dim
    residual_hidden_dim = int(adapter_state["residual.0.weight"].shape[0])
    flow_cfg = payload.get("config", {}).get("flow", {})
    resolved_num_steps = int(
        inference_num_steps
        if inference_num_steps is not None
        else flow_cfg.get("num_steps", 10)
    )
    if resolved_num_steps <= 0:
        raise ValueError("inference_num_steps must be positive")
    resolved_inference_dynamics = str(
        inference_dynamics
        if inference_dynamics is not None
        else flow_cfg.get("inference_dynamics", "training_sde_drift")
    )
    flow_policy_kwargs = dict(
        environment_action_dim=environment_action_dim,
        num_steps=resolved_num_steps,
        stochastic_variance=float(flow_cfg.get("stochastic_variance", 0.01)),
        sde_mode=str(
            flow_cfg.get("sde_mode", policy_payload.get("sde_mode", "gaussian_adapter"))
        ),
        residual_hidden_dim=residual_hidden_dim,
        checkpoint_dir=str(pi05_checkpoint_dir),
        train_config_name=train_config_name,
    )
    if is_pytorch:
        flow_policy_kwargs.update(
            flow_action_dim=flow_action_dim,
            constant_noise_std=float(
                flow_cfg.get(
                    "constant_noise_std", policy_payload.get("constant_noise_std", 0.005)
                )
            ),
            learn_sde_std=bool(
                flow_cfg.get("learn_sde_std", policy_payload.get("learn_sde_std", True))
            ),
            randn_clip_value=float(
                flow_cfg.get("randn_clip_value", policy_payload.get("randn_clip_value", 3.0))
            ),
            residual_enabled=bool(
                payload.get("config", {})
                .get("actor", {})
                .get("residual_enabled", policy_payload.get("residual_enabled", True))
            ),
        )
    if flow_policy_cls is PI05JaxFlowPolicy:
        flow_policy_kwargs["flow_action_dim"] = flow_action_dim
        flow_policy_kwargs["inference_flow_action_dim"] = inference_flow_action_dim
        flow_policy_kwargs["inference_dynamics"] = resolved_inference_dynamics
    flow_policy = flow_policy_cls(trained_policy._model, **flow_policy_kwargs).to(device)
    reference_flow_policy = flow_policy.clone_adapter() if include_reference_policy else None
    flow_policy.load_adapter_state_dict(adapter_state)
    if policy_format == "pi05_jax_full_finetune":
        if not isinstance(flow_policy, PI05JaxFlowPolicy):
            raise TypeError("full-finetune JAX checkpoint requires PI05JaxFlowPolicy")
        sidecar = Path(ogpo_checkpoint).expanduser().resolve().parent / policy_payload["jax_sidecar"]
        flow_policy.restore_training_checkpoint(sidecar, restore_optimizer=False)
        flow_policy.prepare_inference()
        if isinstance(reference_flow_policy, PI05JaxFlowPolicy):
            reference_flow_policy.prepare_inference()
    config = payload.get("config", {})
    critic_cfg = config.get("critic", {})
    critic_camera_keys: tuple[str, ...] = ()
    critic = None
    if include_critic and payload.get("critic_format") == "gemma_siglip_multihead":
        metadata = payload.get("critic_metadata", {})
        backbone_cfg = critic_cfg.get("backbone", {})
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[str(backbone_cfg.get("dtype", "bfloat16"))]
        critic_state = payload["multimodal_critic"]
        proprio_dim = int(critic_state["state_encoder.proprio_projection.weight"].shape[1])
        critic_camera_keys = tuple(metadata.get("camera_keys", backbone_cfg.get("camera_keys", ())))
        backbone = GemmaSiglipStateBackbone.from_local_checkpoints(
            gemma_path=metadata.get("gemma_path", backbone_cfg["gemma_path"]),
            siglip_path=metadata.get("siglip_path", backbone_cfg["siglip_path"]),
            camera_keys=critic_camera_keys,
            proprio_dim=proprio_dim,
            max_language_tokens=int(backbone_cfg.get("max_language_tokens", 128)),
            dtype=dtype,
        )
        core = MultiHeadUdivlCore(
            state_dim=backbone.hidden_size,
            action_dim=environment_action_dim,
            max_horizon=model_horizon,
            action_hidden_dim=int(critic_cfg.get("action_hidden_dim", 256)),
            head_hidden_dim=int(critic_cfg.get("head_hidden_dim", 512)),
            num_attention_heads=int(critic_cfg.get("action_attention_heads", 8)),
            num_value_atoms=int(config.get("divl", {}).get("num_atoms", 51)),
            num_pairs=int(critic_cfg.get("ensemble_size", 3)),
            q_representation=str(
                metadata.get(
                    "q_representation",
                    critic_cfg.get("q_representation", "scalar"),
                )
            ),
            q_num_bins=int(critic_cfg.get("q_num_bins", 201)),
            q_vmin=float(critic_cfg.get("q_vmin", -0.1)),
            q_vmax=float(critic_cfg.get("q_vmax", 1.1)),
            q_heads_per_member=int(
                metadata.get(
                    "q_heads_per_member",
                    critic_cfg.get("q_heads_per_member", 1),
                )
            ),
        )
        lora_cfg = critic_cfg.get("gemma_lora", {})
        if int(lora_cfg.get("final_n_layers", 0)) > 0:
            install_final_gemma_lora(
                backbone,
                final_n_layers=int(lora_cfg["final_n_layers"]),
                rank=int(lora_cfg.get("rank", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                target_suffixes=tuple(lora_cfg.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
            )
        critic = MultiHeadUdivlCritic(backbone, core).to(device)
        critic.load_state_dict(critic_state)
    elif include_critic:
        first_weight = payload["critic_ensemble"]["members.0.net.0.weight"]
        critic_obs_dim = int(first_weight.shape[1]) - model_horizon * environment_action_dim
        critic = ScalarQEnsemble(
            ensemble_size=int(critic_cfg.get("ensemble_size", 3)),
            obs_dim=critic_obs_dim,
            generated_horizon=model_horizon,
            action_dim=environment_action_dim,
            hidden_dim=int(critic_cfg.get("hidden_dim", 256)),
            num_layers=int(critic_cfg.get("num_layers", 2)),
            randomized_prior_scale=float(critic_cfg.get("randomized_prior_scale", 0.0)),
        ).to(device)
        incompatible = critic.load_state_dict(payload["critic_ensemble"], strict=False)
        if incompatible.unexpected_keys or any(".prior." not in key for key in incompatible.missing_keys):
            raise RuntimeError(f"incompatible inference critic checkpoint: {incompatible}")
    if critic is not None:
        critic.requires_grad_(False)
    return PI05OGPOInferencePolicy(
        flow_policy,
        input_transform=trained_policy._input_transform,
        output_transform=trained_policy._output_transform,
        observation_type=openpi_model.Observation,
        metadata=trained_policy.metadata,
        reference_flow_policy=reference_flow_policy,
        critic=critic,
        executed_horizon=int(config.get("data", {}).get("executed_horizon", model_horizon)),
        critic_camera_keys=critic_camera_keys,
        critic_online_camera_mapping={
            replay_key: online_key
            for online_key, replay_key in flow_cfg.get("image_mapping", {}).items()
        },
        default_language=str(config.get("data", {}).get("language", "")),
        dynamics=str(
            inference_dynamics
            if inference_dynamics is not None
            else config.get("inference", {}).get("dynamics", "native_ode")
        ),
    )
