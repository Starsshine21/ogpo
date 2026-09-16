from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-rank residual around a frozen linear projection."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.lora_a = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + residual * self.scaling


def _resolve_parent(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def install_final_gemma_lora(
    state_encoder: nn.Module,
    *,
    final_n_layers: int,
    rank: int,
    alpha: float,
    target_suffixes: Sequence[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
) -> tuple[int, ...]:
    """Install adapters only in the final Gemma decoder layers."""
    if final_n_layers <= 0:
        return ()
    candidates: list[tuple[str, int, nn.Linear]] = []
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    for name, child in state_encoder.gemma_model.named_modules():
        match = pattern.search(name)
        if match and isinstance(child, nn.Linear) and name.endswith(tuple(target_suffixes)):
            candidates.append((name, int(match.group(1)), child))
    if not candidates:
        raise ValueError("no matching Gemma linear projections were found for LoRA")
    layer_indices = sorted({index for _, index, _ in candidates})
    selected = tuple(layer_indices[-int(final_n_layers) :])
    for name, index, child in candidates:
        if index not in selected:
            continue
        parent, attribute = _resolve_parent(state_encoder.gemma_model, name)
        setattr(parent, attribute, LoRALinear(child, rank=rank, alpha=alpha))
    return selected


def configure_critic_stage(critic: nn.Module, stage: str) -> None:
    """Apply the production trainability policy for a multimodal critic stage."""
    if stage not in {"head_mc", "head_td", "gemma_lora_td", "full_td"}:
        raise ValueError(f"unsupported critic stage: {stage}")
    encoder = critic.state_encoder
    if stage == "full_td":
        critic.requires_grad_(True)
        if hasattr(encoder, "unfreeze_pretrained"):
            encoder.unfreeze_pretrained()
        return
    critic.requires_grad_(False)
    if hasattr(encoder, "freeze_pretrained"):
        encoder.freeze_pretrained()
    for name in ("visual_projection", "proprio_projection"):
        module = getattr(encoder, name, None)
        if module is not None:
            module.requires_grad_(True)
    if hasattr(encoder, "readout_token"):
        encoder.readout_token.requires_grad_(True)
    critic.core.requires_grad_(True)
    if stage == "gemma_lora_td" and hasattr(encoder, "gemma_model"):
        for module in encoder.gemma_model.modules():
            if isinstance(module, LoRALinear):
                module.lora_a.requires_grad_(True)
                module.lora_b.requires_grad_(True)


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
    return parameter.device, dtype


def _config_hidden_size(config: Any) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise AttributeError("model config does not expose hidden_size")


class GemmaSiglipStateBackbone(nn.Module):
    """Encode replay images, language, and proprioception into one readout token."""

    def __init__(
        self,
        *,
        vision_model: nn.Module,
        gemma_model: nn.Module,
        image_processor: Any,
        tokenizer: Any,
        camera_keys: Sequence[str],
        proprio_dim: int,
        vision_hidden_size: int,
        gemma_hidden_size: int,
        max_language_tokens: int,
        train_pretrained: bool = False,
    ):
        super().__init__()
        if not camera_keys:
            raise ValueError("camera_keys cannot be empty")
        self.vision_model = vision_model
        self.gemma_model = gemma_model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.camera_keys = tuple(camera_keys)
        self.proprio_dim = int(proprio_dim)
        self.hidden_size = int(gemma_hidden_size)
        self.max_language_tokens = int(max_language_tokens)
        self.train_pretrained = bool(train_pretrained)
        self.visual_projection = nn.Linear(int(vision_hidden_size), self.hidden_size)
        self.proprio_projection = nn.Linear(self.proprio_dim, self.hidden_size)
        self.readout_token = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        nn.init.normal_(self.readout_token, std=0.02)
        if self.train_pretrained:
            self.unfreeze_pretrained()
        else:
            self.freeze_pretrained()

    @classmethod
    def from_local_checkpoints(
        cls,
        *,
        gemma_path: str | Path,
        siglip_path: str | Path,
        camera_keys: Sequence[str],
        proprio_dim: int,
        max_language_tokens: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        train_pretrained: bool = False,
    ) -> "GemmaSiglipStateBackbone":
        from transformers import (  # noqa: PLC0415
            AutoImageProcessor,
            AutoTokenizer,
            Gemma3ForCausalLM,
            SiglipVisionModel,
        )

        gemma_dir = Path(gemma_path)
        siglip_dir = Path(siglip_path)
        if not gemma_dir.is_dir():
            raise FileNotFoundError(f"Gemma checkpoint directory does not exist: {gemma_dir}")
        if not siglip_dir.is_dir():
            raise FileNotFoundError(f"SigLIP checkpoint directory does not exist: {siglip_dir}")
        gemma = Gemma3ForCausalLM.from_pretrained(
            gemma_dir,
            torch_dtype=dtype,
            local_files_only=True,
            attn_implementation="eager",
        )
        vision = SiglipVisionModel.from_pretrained(
            siglip_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(gemma_dir, local_files_only=True)
        image_processor = AutoImageProcessor.from_pretrained(siglip_dir, local_files_only=True)
        return cls(
            vision_model=vision,
            gemma_model=gemma,
            image_processor=image_processor,
            tokenizer=tokenizer,
            camera_keys=camera_keys,
            proprio_dim=proprio_dim,
            vision_hidden_size=int(vision.config.hidden_size),
            gemma_hidden_size=_config_hidden_size(gemma.config),
            max_language_tokens=max_language_tokens,
            train_pretrained=train_pretrained,
        )

    def freeze_pretrained(self) -> None:
        self.train_pretrained = False
        self.vision_model.requires_grad_(False)
        self.gemma_model.requires_grad_(False)
        self.vision_model.eval()
        self.gemma_model.eval()

    def unfreeze_pretrained(self) -> None:
        self.train_pretrained = True
        self.vision_model.requires_grad_(True)
        self.gemma_model.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_pretrained:
            self.vision_model.eval()
            self.gemma_model.eval()
        return self

    def _encode_camera(self, images: torch.Tensor) -> torch.Tensor:
        image_list = [np.asarray(image.detach().cpu()) for image in images]
        processed = self.image_processor(images=image_list, return_tensors="pt")
        vision_device, vision_dtype = _module_device_dtype(self.vision_model)
        pixel_values = processed["pixel_values"].to(device=vision_device, dtype=vision_dtype)
        grad_context = nullcontext() if self.train_pretrained else torch.no_grad()
        with grad_context:
            output = self.vision_model(pixel_values=pixel_values)
        projection_device, projection_dtype = _module_device_dtype(self.visual_projection)
        return self.visual_projection(
            output.last_hidden_state.to(device=projection_device, dtype=projection_dtype)
        )

    def _language_tokens(self, languages: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            languages,
            padding=True,
            truncation=True,
            max_length=self.max_language_tokens,
            return_tensors="pt",
        )
        embedding = self.gemma_model.get_input_embeddings()
        gemma_device, _ = _module_device_dtype(embedding)
        input_ids = encoded["input_ids"].to(gemma_device)
        attention_mask = encoded["attention_mask"].to(device=gemma_device, dtype=torch.bool)
        grad_context = nullcontext() if self.train_pretrained else torch.no_grad()
        with grad_context:
            tokens = embedding(input_ids)
        return tokens, attention_mask

    def forward(self, batch: Any, *, next_observation: bool = False) -> torch.Tensor:
        images = batch.next_images if next_observation else batch.images
        if images is None:
            raise ValueError("multimodal critic requires replay RGB images")
        missing = [key for key in self.camera_keys if key not in images]
        if missing:
            raise KeyError(f"replay is missing critic camera arrays: {missing}")
        proprio = batch.next_proprioceptions if next_observation else batch.proprioceptions
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprioception must have shape [batch, {self.proprio_dim}]")

        visual_tokens = [self._encode_camera(images[key]) for key in self.camera_keys]
        language_tokens, language_mask = self._language_tokens(list(batch.languages))
        device = language_tokens.device
        dtype = language_tokens.dtype
        visual_tokens = [tokens.to(device=device, dtype=dtype) for tokens in visual_tokens]
        proprio_token = self.proprio_projection(
            proprio.to(
                device=self.proprio_projection.weight.device,
                dtype=self.proprio_projection.weight.dtype,
            )
        ).to(device=device, dtype=dtype).unsqueeze(1)
        readout = self.readout_token.to(device=device, dtype=dtype).expand(proprio.shape[0], -1, -1)

        tokens = torch.cat([*visual_tokens, language_tokens, proprio_token, readout], dim=1)
        visual_length = sum(token.shape[1] for token in visual_tokens)
        prefix_mask = torch.ones(
            proprio.shape[0],
            visual_length,
            dtype=torch.bool,
            device=device,
        )
        suffix_mask = torch.ones(proprio.shape[0], 2, dtype=torch.bool, device=device)
        attention_mask = torch.cat([prefix_mask, language_mask, suffix_mask], dim=1)

        transformer = getattr(self.gemma_model, "model", self.gemma_model)
        output = transformer(
            inputs_embeds=tokens,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return output.last_hidden_state[:, -1]


def build_gemma_siglip_critic(batch: Any, config: dict[str, Any]):
    """Build one shared-backbone critic with a configurable number of Q/V pairs."""
    from .multimodal_critic import MultiHeadUdivlCore, MultiHeadUdivlCritic  # noqa: PLC0415

    if batch.images is None:
        raise ValueError("gemma_siglip_multihead requires replay RGB images")
    critic_cfg = config.get("critic", {})
    backbone_cfg = critic_cfg.get("backbone", {})
    dtype_name = str(backbone_cfg.get("dtype", "bfloat16"))
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported critic backbone dtype: {dtype_name}")
    camera_keys = tuple(backbone_cfg.get("camera_keys", tuple(batch.images)))
    backbone = GemmaSiglipStateBackbone.from_local_checkpoints(
        gemma_path=backbone_cfg["gemma_path"],
        siglip_path=backbone_cfg["siglip_path"],
        camera_keys=camera_keys,
        proprio_dim=int(batch.proprioceptions.shape[-1]),
        max_language_tokens=int(backbone_cfg.get("max_language_tokens", 128)),
        dtype=dtype_by_name[dtype_name],
        train_pretrained=bool(backbone_cfg.get("train_vlm_full", False)),
    )
    if bool(backbone_cfg.get("gradient_checkpointing", False)):
        for module in (backbone.vision_model, backbone.gemma_model):
            enable = getattr(module, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()
    core = MultiHeadUdivlCore(
        state_dim=backbone.hidden_size,
        action_dim=batch.action_dim,
        max_horizon=batch.generated_horizon,
        action_hidden_dim=int(critic_cfg.get("action_hidden_dim", 256)),
        head_hidden_dim=int(critic_cfg.get("head_hidden_dim", 512)),
        num_attention_heads=int(critic_cfg.get("action_attention_heads", 8)),
        num_value_atoms=int(config.get("divl", {}).get("num_atoms", 51)),
        num_pairs=int(critic_cfg.get("ensemble_size", 3)),
        q_representation=str(critic_cfg.get("q_representation", "scalar")),
        q_num_bins=int(critic_cfg.get("q_num_bins", 201)),
        q_vmin=float(critic_cfg.get("q_vmin", -0.1)),
        q_vmax=float(critic_cfg.get("q_vmax", 1.1)),
        q_heads_per_member=int(critic_cfg.get("q_heads_per_member", 1)),
    )
    executed_actions = batch.action_chunks[batch.execution_masks.bool()]
    with torch.no_grad():
        core.action_pool.action_mean.copy_(executed_actions.mean(dim=0))
        core.action_pool.action_std.copy_(executed_actions.std(dim=0, unbiased=False).clamp_min(1e-6))
        core.action_pool.action_min.copy_(executed_actions.min(dim=0).values)
        core.action_pool.action_max.copy_(executed_actions.max(dim=0).values)
    critic = MultiHeadUdivlCritic(backbone, core)
    critic.model_metadata = {
        "gemma_path": str(backbone_cfg["gemma_path"]),
        "siglip_path": str(backbone_cfg["siglip_path"]),
        "camera_keys": camera_keys,
        "dtype": dtype_name,
        "q_representation": core.q_representation,
        "q_num_bins": int(core.q_support.numel()) if core.q_support is not None else 1,
        "q_heads_per_member": core.q_heads_per_member,
        "num_pairs": core.num_pairs,
        "num_raw_q_heads": core.num_q_heads,
    }
    return critic


def build_gemma_siglip_scalar_q_critic(batch: Any, config: dict[str, Any]):
    """Build the scalar Q ensemble used by the OGPO-origin path."""
    from .multimodal_critic import MultiHeadScalarQCore, MultiHeadScalarQCritic  # noqa: PLC0415

    if batch.images is None:
        raise ValueError("gemma_siglip_scalar_q requires replay RGB images")
    critic_cfg = config.get("critic", {})
    backbone_cfg = critic_cfg.get("backbone", {})
    dtype_name = str(backbone_cfg.get("dtype", "bfloat16"))
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported critic backbone dtype: {dtype_name}")
    train_vlm_full = bool(backbone_cfg.get("train_vlm_full", False))
    camera_keys = tuple(backbone_cfg.get("camera_keys", tuple(batch.images)))
    backbone = GemmaSiglipStateBackbone.from_local_checkpoints(
        gemma_path=backbone_cfg["gemma_path"],
        siglip_path=backbone_cfg["siglip_path"],
        camera_keys=camera_keys,
        proprio_dim=int(batch.proprioceptions.shape[-1]),
        max_language_tokens=int(backbone_cfg.get("max_language_tokens", 128)),
        dtype=dtype_by_name[dtype_name],
        train_pretrained=train_vlm_full,
    )
    if bool(backbone_cfg.get("gradient_checkpointing", False)):
        for module in (backbone.vision_model, backbone.gemma_model):
            enable = getattr(module, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()
    core = MultiHeadScalarQCore(
        state_dim=backbone.hidden_size,
        action_dim=batch.action_dim,
        max_horizon=batch.generated_horizon,
        action_hidden_dim=int(critic_cfg.get("action_hidden_dim", 256)),
        head_hidden_dim=int(critic_cfg.get("head_hidden_dim", 512)),
        num_attention_heads=int(critic_cfg.get("action_attention_heads", 8)),
        num_heads=int(critic_cfg.get("ensemble_size", 10)),
    )
    executed_actions = batch.action_chunks[batch.execution_masks.bool()]
    with torch.no_grad():
        core.action_pool.action_mean.copy_(executed_actions.mean(dim=0))
        core.action_pool.action_std.copy_(
            executed_actions.std(dim=0, unbiased=False).clamp_min(1e-6)
        )
        core.action_pool.action_min.copy_(executed_actions.min(dim=0).values)
        core.action_pool.action_max.copy_(executed_actions.max(dim=0).values)
    critic = MultiHeadScalarQCritic(backbone, core)
    critic_stage = str(critic_cfg.get("stage", "head_td"))
    if critic_stage == "full_td" and not train_vlm_full:
        raise ValueError("critic.stage=full_td requires critic.backbone.train_vlm_full=true")
    configure_critic_stage(critic, critic_stage)
    if critic_stage != "full_td":
        # Frozen image OGPO may cache replay embeddings. Full finetuning must
        # always encode raw observations so gradients reach Gemma and SigLIP.
        critic.state_encoder.requires_grad_(False)
        critic.core.requires_grad_(True)
    critic.model_metadata = {
        "gemma_path": str(backbone_cfg["gemma_path"]),
        "siglip_path": str(backbone_cfg["siglip_path"]),
        "camera_keys": camera_keys,
        "dtype": dtype_name,
        "vlm_frozen": critic_stage != "full_td",
    }
    return critic
