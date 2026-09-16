from __future__ import annotations

import copy
import dataclasses
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .openpi_flow_spec import OpenPIStochasticFlowPolicy


def _predict_pi05_backend_velocity(
    backend: nn.Module,
    observation: Any,
    noisy_actions: torch.Tensor,
    time: torch.Tensor,
    *,
    train: bool = False,
) -> torch.Tensor:
    """Call the explicit PI0.5 velocity path across OpenPI revisions.

    The training checkout exposes ``PI0Pytorch.predict_velocity`` while the
    RoboTwin vendored OpenPI revision only exposes the same operations through
    its sampling helpers.  Keep the compatibility implementation here so an
    OGPO checkpoint is evaluated by the exact one-step flow model, rather than
    silently falling back to base ``sample_actions`` or a synthetic velocity.
    """
    predict_velocity = getattr(backend, "predict_velocity", None)
    if predict_velocity is not None:
        return predict_velocity(observation, noisy_actions, time, train=train)

    images, image_masks, language_tokens, language_masks, state = (
        backend._preprocess_observation(observation, train=train)
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = backend.embed_prefix(
        images, image_masks, language_tokens, language_masks
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = backend.embed_suffix(
        state, noisy_actions, time
    )
    language_model = backend.paligemma_with_expert.paligemma.language_model
    if language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    attention_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    backend_module = importlib.import_module(backend.__class__.__module__)
    attention_masks_2d = backend_module.make_att_2d_masks(pad_masks, attention_masks)
    attention_masks_4d = backend._prepare_attention_masks_4d(attention_masks_2d)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    def forward_func(prefix, suffix, masks, positions, conditioning):
        (_, suffix_output), _ = backend.paligemma_with_expert.forward(
            attention_mask=masks,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, conditioning],
        )
        return suffix_output

    suffix_output = backend._apply_checkpoint(
        forward_func,
        prefix_embs,
        suffix_embs,
        attention_masks_4d,
        position_ids,
        adarms_cond,
    )
    suffix_output = suffix_output[:, -backend.config.action_horizon :].to(
        dtype=torch.float32
    )
    return backend._apply_checkpoint(backend.action_out_proj, suffix_output)


def _tree_map_tensor(value: Any, fn) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: _tree_map_tensor(item, fn) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        updates = {
            field.name: _tree_map_tensor(getattr(value, field.name), fn)
            for field in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **updates)
    if value is None:
        return None
    return value


def _tree_index_select(value: Any, indices: torch.Tensor) -> Any:
    return _tree_map_tensor(value, lambda tensor: tensor.index_select(0, indices))


@dataclass(frozen=True)
class PI05FlowCondition:
    """Batched, model-ready PI0.5 observation used by flow transitions."""

    observation: Any

    @property
    def state(self) -> torch.Tensor:
        return self.observation.state

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])

    def repeat_interleave(self, repeats: int) -> "PI05FlowCondition":
        return PI05FlowCondition(
            _tree_map_tensor(self.observation, lambda tensor: tensor.repeat_interleave(repeats, dim=0))
        )

    def index_select(self, indices: torch.Tensor) -> "PI05FlowCondition":
        return PI05FlowCondition(_tree_index_select(self.observation, indices))

    def to(self, device: torch.device | str) -> "PI05FlowCondition":
        return PI05FlowCondition(
            _tree_map_tensor(self.observation, lambda tensor: tensor.to(device))
        )


def _stack_tree(values: list[Any], *, device: torch.device | str) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack_tree([value[key] for value in values], device=device) for key in first}
    arrays = [np.asarray(value) for value in values]
    return torch.as_tensor(np.stack(arrays, axis=0), device=device)


@dataclass(frozen=True)
class PI05ReplayConditionBuilder:
    """Convert raw replay images/state/language into model-ready observations."""

    input_transform: Any
    observation_type: Any
    image_mapping: dict[str, str]
    image_container_key: str | None = None
    transpose_images_to_chw: bool = False
    output_transform: Any | None = None
    model_action_dim: int | None = None
    environment_action_dim: int | None = None

    def _raw_sample(self, batch, index: int, *, next_observation: bool = False) -> dict[str, Any]:
        images = batch.next_images if next_observation else batch.images
        if images is None:
            raise ValueError(
                "PI0.5 flow adapter requires replay RGB observations; rebuild the dataset from replay.zarr "
                "after collecting synchronized image arrays"
            )
        missing = set(self.image_mapping.values()) - set(images)
        if missing:
            raise KeyError(f"replay is missing PI0.5 camera arrays: {sorted(missing)}")
        states = batch.next_proprioceptions if next_observation else batch.proprioceptions
        mapped_images = {}
        for role, key in self.image_mapping.items():
            image = images[key][index].detach().cpu().numpy()
            if self.transpose_images_to_chw and image.ndim == 3 and image.shape[-1] in {1, 3, 4}:
                image = np.moveaxis(image, -1, 0)
            mapped_images[role] = image
        raw = (
            {self.image_container_key: mapped_images}
            if self.image_container_key is not None
            else mapped_images
        )
        raw["state"] = states[index].detach().cpu().numpy()
        raw["prompt"] = np.asarray(batch.languages[index])
        return raw

    def __call__(
        self,
        batch,
        *,
        next_observation: bool = False,
        device: torch.device | str = "cpu",
    ) -> PI05FlowCondition:
        transformed = []
        for index in range(batch.batch_size):
            transformed.append(self.input_transform(self._raw_sample(batch, index, next_observation=next_observation)))
        model_inputs = _stack_tree(transformed, device=device)
        return PI05FlowCondition(self.observation_type.from_dict(model_inputs))

    def action_chunks_to_flow(self, batch) -> torch.Tensor:
        if self.environment_action_dim is None:
            raise RuntimeError("PI0.5 environment action dimension is not configured")
        transformed_actions = []
        for index in range(batch.batch_size):
            raw = self._raw_sample(batch, index)
            raw["actions"] = batch.action_chunks[index].detach().cpu().numpy()
            transformed = self.input_transform(raw)
            if "actions" not in transformed:
                raise KeyError("PI0.5 input transform did not return normalized actions")
            transformed_actions.append(
                np.asarray(transformed["actions"], dtype=np.float32)[..., : self.environment_action_dim]
            )
        return torch.as_tensor(
            np.stack(transformed_actions),
            device=batch.action_chunks.device,
            dtype=batch.action_chunks.dtype,
        )

    def flat_actions_to_environment(
        self,
        flat_actions: torch.Tensor,
        *,
        model_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.output_transform is None or self.model_action_dim is None or self.environment_action_dim is None:
            raise RuntimeError("PI0.5 output action transform is not configured")
        batch_size = flat_actions.shape[0]
        if flat_actions.shape[1] % self.environment_action_dim:
            raise ValueError("flat PI0.5 actions are not divisible by the environment action dimension")
        horizon = flat_actions.shape[1] // self.environment_action_dim
        flow_actions = flat_actions.detach().reshape(batch_size, horizon, self.environment_action_dim).cpu().numpy()
        states = model_states.detach().cpu().numpy() if model_states is not None else None
        environment_actions = []
        for index in range(batch_size):
            padded = np.zeros((horizon, self.model_action_dim), dtype=flow_actions.dtype)
            padded[..., : self.environment_action_dim] = flow_actions[index]
            output = {"actions": padded}
            if states is not None:
                output["state"] = states[index]
            transformed = self.output_transform(output)
            environment_actions.append(
                np.asarray(transformed["actions"], dtype=np.float32)[..., : self.environment_action_dim]
            )
        return torch.as_tensor(
            np.stack(environment_actions),
            device=flat_actions.device,
            dtype=flat_actions.dtype,
        ).reshape(batch_size, -1)


class PI05PytorchFlowPolicy(OpenPIStochasticFlowPolicy):
    """Stochastic OGPO adapter backed by the real PyTorch PI0.5 action expert.

    The backend can either remain frozen (the historical residual-adapter
    mode), train only its action expert, or train the complete PyTorch model.
    ``residual_enabled`` controls the optional legacy residual controller;
    disabling it leaves the native backend velocity untouched. Frozen
    old/reference policies use functional parameter snapshots so a trainable
    4B backend does not need to be duplicated as a module.
    """

    def __init__(
        self,
        backend: nn.Module,
        *,
        environment_action_dim: int,
        num_steps: int = 10,
        stochastic_variance: float = 0.04,
        sde_mode: str = "gaussian_adapter",
        constant_noise_std: float = 0.005,
        learn_sde_std: bool = True,
        randn_clip_value: float = 3.0,
        residual_hidden_dim: int = 128,
        residual_enabled: bool = True,
        condition_builder: PI05ReplayConditionBuilder | None = None,
        checkpoint_dir: str | None = None,
        train_config_name: str | None = None,
        backend_train_mode: str = "none",
        register_backend: bool = True,
        backend_trainable_names: tuple[str, ...] = (),
        backend_parameter_snapshot: dict[str, torch.Tensor] | None = None,
    ):
        model_horizon = int(backend.config.action_horizon)
        model_action_dim = int(backend.config.action_dim)
        environment_action_dim = int(environment_action_dim)
        if environment_action_dim > model_action_dim:
            raise ValueError("environment action dimension cannot exceed PI0.5 model action dimension")
        super().__init__(
            action_dim=model_horizon * environment_action_dim,
            num_steps=num_steps,
            stochastic_variance=stochastic_variance,
            sde_mode=sde_mode,
            constant_noise_std=constant_noise_std,
            learn_sde_std=learn_sde_std,
            randn_clip_value=randn_clip_value,
        )
        backend_train_mode = str(backend_train_mode)
        if backend_train_mode not in {"none", "action_expert", "full"}:
            raise ValueError(
                "backend_train_mode must be 'none', 'action_expert', or 'full'"
            )
        if register_backend:
            self.backend = backend
        else:
            object.__setattr__(self, "backend", backend)
        self.model_horizon = model_horizon
        self.model_action_dim = model_action_dim
        self.environment_action_dim = environment_action_dim
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.residual_enabled = bool(residual_enabled)
        self.condition_builder = condition_builder
        self.checkpoint_dir = checkpoint_dir
        self.train_config_name = train_config_name
        self.backend_train_mode = backend_train_mode
        object.__setattr__(
            self,
            "_backend_parameter_snapshot",
            None if backend_parameter_snapshot is None else dict(backend_parameter_snapshot),
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * environment_action_dim + 1, self.residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.residual_hidden_dim, environment_action_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        if not self.residual_enabled:
            # Keep the module/state-dict for legacy checkpoint compatibility,
            # but make the clean reference a pure PI0.5 backend velocity.
            self.residual.requires_grad_(False)
        if register_backend:
            self.backend.requires_grad_(False)
            if backend_train_mode == "full":
                self.backend.requires_grad_(True)
            elif backend_train_mode == "action_expert":
                trainable_prefixes = (
                    "paligemma_with_expert.gemma_expert.",
                    "action_in_proj.",
                    "action_out_proj.",
                    "time_mlp_in.",
                    "time_mlp_out.",
                    "state_proj.",
                    "action_time_mlp_in.",
                    "action_time_mlp_out.",
                )
                for name, parameter in self.backend.named_parameters():
                    parameter.requires_grad_(name.startswith(trainable_prefixes))
            backend_trainable_names = tuple(
                name for name, parameter in self.backend.named_parameters() if parameter.requires_grad
            )
            if backend_train_mode != "none" and hasattr(
                self.backend, "gradient_checkpointing_enable"
            ):
                self.backend.gradient_checkpointing_enable()
        self._backend_trainable_names = tuple(backend_trainable_names)
        if backend_train_mode == "none" and register_backend:
            self.backend.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backend_train_mode == "none":
            self.backend.eval()
        elif self._backend_parameter_snapshot is None:
            self.backend.train(mode)
        return self

    def condition_batch_size(self, condition: PI05FlowCondition) -> int:
        return condition.batch_size

    def condition_device_dtype(self, condition: PI05FlowCondition) -> tuple[torch.device, torch.dtype]:
        return condition.state.device, torch.float32

    def repeat_condition(self, condition: PI05FlowCondition, repeats: int) -> PI05FlowCondition:
        return condition.repeat_interleave(repeats)

    def condition_from_batch(self, batch, *, next_observation: bool = False) -> PI05FlowCondition:
        if self.condition_builder is None:
            raise RuntimeError("PI0.5 replay condition builder is not configured")
        return self.condition_builder(
            batch,
            next_observation=next_observation,
            device=self.log_std.device,
        )

    def action_chunks_to_flow(self, batch) -> torch.Tensor:
        if self.condition_builder is None or not hasattr(self.condition_builder, "action_chunks_to_flow"):
            return super().action_chunks_to_flow(batch)
        return self.condition_builder.action_chunks_to_flow(batch)

    def flat_actions_to_environment(
        self,
        flat_actions: torch.Tensor,
        condition: PI05FlowCondition | None = None,
    ) -> torch.Tensor:
        if self.condition_builder is None or not hasattr(self.condition_builder, "flat_actions_to_environment"):
            return super().flat_actions_to_environment(flat_actions, condition)
        model_states = None if condition is None else condition.state
        converted = self.condition_builder.flat_actions_to_environment(
            flat_actions,
            model_states=model_states,
        )
        if flat_actions.requires_grad:
            converted = converted + flat_actions - flat_actions.detach()
        return converted

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        condition: PI05FlowCondition,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch = x_t.shape[0]
        x_env = x_t.reshape(batch, self.model_horizon, self.environment_action_dim)
        x_model = x_env.new_zeros(batch, self.model_horizon, self.model_action_dim)
        x_model[..., : self.environment_action_dim] = x_env
        time = timestep.reshape(batch, -1)[:, 0].to(dtype=torch.float32)
        if self._backend_parameter_snapshot is not None:
            class _VelocityCall(nn.Module):
                def __init__(self, backend):
                    super().__init__()
                    self.backend = backend

                def forward(self, observation, noisy_actions, time):
                    return _predict_pi05_backend_velocity(
                        self.backend,
                        observation,
                        noisy_actions,
                        time,
                        train=False,
                    )

            replacement = {
                f"backend.{name}": value
                for name, value in self._backend_parameter_snapshot.items()
            }
            base_model = torch.func.functional_call(
                _VelocityCall(self.backend),
                replacement,
                (condition.observation, x_model, time),
                strict=False,
            )
        elif self.backend_train_mode != "none":
            base_model = _predict_pi05_backend_velocity(
                self.backend,
                condition.observation,
                x_model,
                time,
                train=False,
            )
        else:
            with torch.no_grad():
                base_model = _predict_pi05_backend_velocity(
                    self.backend,
                    condition.observation,
                    x_model,
                    time,
                    train=False,
                )
        base = base_model[..., : self.environment_action_dim].to(dtype=x_env.dtype)
        time_features = time.to(dtype=x_env.dtype)[:, None, None].expand(batch, self.model_horizon, 1)
        if self.residual_enabled:
            residual = self.residual(torch.cat([x_env, base, time_features], dim=-1))
            base = base + residual
        return base.reshape(batch, -1)

    def clone_adapter(self, *, trainable: bool = False) -> "PI05PytorchFlowPolicy":
        backend_snapshot = {
            name: parameter.detach().clone()
            for name, parameter in self.backend.named_parameters()
            if name in self._backend_trainable_names
        }
        clone = PI05PytorchFlowPolicy(
            self.backend,
            environment_action_dim=self.environment_action_dim,
            num_steps=self.num_steps,
            stochastic_variance=float(self.log_std.detach().exp().square().mean().item()),
            sde_mode=self.sde_mode,
            constant_noise_std=self.constant_noise_std,
            learn_sde_std=self.learn_sde_std,
            randn_clip_value=self.randn_clip_value,
            residual_hidden_dim=self.residual_hidden_dim,
            residual_enabled=self.residual_enabled,
            condition_builder=self.condition_builder,
            checkpoint_dir=self.checkpoint_dir,
            train_config_name=self.train_config_name,
            backend_train_mode="none",
            register_backend=False,
            backend_trainable_names=self._backend_trainable_names,
            backend_parameter_snapshot=backend_snapshot,
        ).to(self.log_std.device)
        clone.log_std.data.copy_(self.log_std.data)
        clone.residual.load_state_dict(copy.deepcopy(self.residual.state_dict()))
        if not trainable:
            clone.requires_grad_(False)
        return clone

    def adapter_state_dict(
        self,
        *,
        include_backend: bool = True,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        state = {"log_std": self.log_std.detach().to(device)}
        state.update(
            {
                f"residual.{key}": value.detach().to(device)
                for key, value in self.residual.state_dict().items()
            }
        )
        if include_backend:
            if self._backend_parameter_snapshot is not None:
                state.update(
                    {
                        f"backend.{name}": parameter.detach().to(device)
                        for name, parameter in self._backend_parameter_snapshot.items()
                    }
                )
            else:
                state.update(
                    {
                        f"backend.{name}": parameter.detach().to(device)
                        for name, parameter in self.backend.named_parameters()
                        if name in self._backend_trainable_names
                    }
                )
        return state

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.log_std.data.copy_(state["log_std"].to(self.log_std.device))
        residual_state = {
            key.removeprefix("residual."): value.to(self.log_std.device)
            for key, value in state.items()
            if key.startswith("residual.")
        }
        self.residual.load_state_dict(residual_state)
        backend_state = {
            key.removeprefix("backend."): value
            for key, value in state.items()
            if key.startswith("backend.")
        }
        if backend_state:
            if self._backend_parameter_snapshot is not None:
                self._backend_parameter_snapshot.update(
                    {
                        name: value.to(self.log_std.device)
                        for name, value in backend_state.items()
                    }
                )
            else:
                named_parameters = dict(self.backend.named_parameters())
                missing = set(backend_state) - set(named_parameters)
                if missing:
                    raise KeyError(f"unknown PI0.5 backend parameters: {sorted(missing)[:5]}")
                for name, value in backend_state.items():
                    named_parameters[name].data.copy_(
                        value.to(named_parameters[name].device, dtype=named_parameters[name].dtype)
                    )

    @torch.no_grad()
    def sync_backend_snapshot_from(self, source: "PI05PytorchFlowPolicy", *, ema: float = 0.0) -> None:
        source_parameters = dict(source.backend.named_parameters())
        if self._backend_parameter_snapshot is not None:
            destination_parameters = self._backend_parameter_snapshot
        elif self.backend is not source.backend:
            destination_parameters = {
                name: parameter
                for name, parameter in self.backend.named_parameters()
                if name in self._backend_trainable_names
            }
        else:
            return
        for name, old_value in destination_parameters.items():
            new_value = source_parameters[name].detach().to(
                device=old_value.device, dtype=old_value.dtype
            )
            if ema == 0.0:
                old_value.copy_(new_value)
            else:
                old_value.mul_(ema).add_(new_value, alpha=1.0 - ema)

    @torch.no_grad()
    def sync_adapter_snapshot_from(
        self,
        source: "PI05PytorchFlowPolicy",
        *,
        ema: float = 0.0,
    ) -> None:
        """EMA-copy a trainable PI0.5 actor into a functional frozen snapshot."""
        if not 0.0 <= float(ema) < 1.0:
            raise ValueError("policy EMA must be in [0, 1)")
        if float(ema) == 0.0:
            self.log_std.copy_(
                source.log_std.detach().to(
                    device=self.log_std.device,
                    dtype=self.log_std.dtype,
                )
            )
            self.residual.load_state_dict(source.residual.state_dict())
        else:
            self.log_std.mul_(float(ema)).add_(
                source.log_std.detach().to(
                    device=self.log_std.device,
                    dtype=self.log_std.dtype,
                ),
                alpha=1.0 - float(ema),
            )
            source_residual = dict(source.residual.named_parameters())
            for name, parameter in self.residual.named_parameters():
                parameter.mul_(float(ema)).add_(
                    source_residual[name].detach().to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                    alpha=1.0 - float(ema),
                )
        self.sync_backend_snapshot_from(source, ema=float(ema))


def load_pi05_pytorch_flow_policy(
    *,
    checkpoint_dir: str | Path,
    train_config_name: str,
    image_mapping: dict[str, str],
    image_container_key: str | None = None,
    transpose_images_to_chw: bool = False,
    environment_action_dim: int,
    num_steps: int,
    stochastic_variance: float,
    sde_mode: str,
    constant_noise_std: float = 0.005,
    learn_sde_std: bool = True,
    randn_clip_value: float = 3.0,
    residual_hidden_dim: int = 128,
    residual_enabled: bool = True,
    device: torch.device | str,
    backend_train_mode: str = "none",
) -> PI05PytorchFlowPolicy:
    """Load a converted PI0.5 checkpoint and construct the OGPO adapter."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    weights = checkpoint_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"{weights} is missing. Convert the JAX checkpoint with "
            "openpi/examples/convert_jax_model_to_pytorch.py before OGPO actor training."
        )

    from openpi.models import model as openpi_model  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    trained_policy = policy_config.create_trained_policy(
        training_config.get_config(train_config_name),
        checkpoint_dir,
        pytorch_device=str(device),
    )
    if not getattr(trained_policy, "_is_pytorch_model", False):
        raise TypeError("OGPO requires a PyTorch PI0.5 checkpoint")
    builder = PI05ReplayConditionBuilder(
        input_transform=trained_policy._input_transform,
        output_transform=trained_policy._output_transform,
        observation_type=openpi_model.Observation,
        image_mapping=dict(image_mapping),
        image_container_key=image_container_key,
        transpose_images_to_chw=transpose_images_to_chw,
        model_action_dim=int(trained_policy._model.config.action_dim),
        environment_action_dim=int(environment_action_dim),
    )
    return PI05PytorchFlowPolicy(
        trained_policy._model,
        environment_action_dim=environment_action_dim,
        num_steps=num_steps,
        stochastic_variance=stochastic_variance,
        sde_mode=sde_mode,
        constant_noise_std=constant_noise_std,
        learn_sde_std=learn_sde_std,
        randn_clip_value=randn_clip_value,
        residual_hidden_dim=residual_hidden_dim,
        residual_enabled=residual_enabled,
        condition_builder=builder,
        checkpoint_dir=str(checkpoint_dir),
        train_config_name=train_config_name,
        backend_train_mode=backend_train_mode,
    ).to(device)
