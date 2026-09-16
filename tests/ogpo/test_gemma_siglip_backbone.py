from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ogpo.gemma_siglip_backbone import GemmaSiglipStateBackbone


class FakeImageProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        values = torch.stack([torch.as_tensor(image).permute(2, 0, 1) for image in images]).float()
        return {"pixel_values": values / 255.0}


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["padding"] is True
        ids = torch.tensor([[len(text) % 13 + 1, 2, 3] for text in texts], dtype=torch.long)
        mask = torch.tensor([[True, True, False] for _ in texts])
        return {"input_ids": ids, "attention_mask": mask}


class FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 6, bias=False)

    def forward(self, pixel_values):
        rgb = pixel_values.mean(dim=(-2, -1))
        token = self.projection(rgb)
        return SimpleNamespace(last_hidden_state=torch.stack([token, token * 2.0], dim=1))


class FakeGemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
        assert use_cache is False
        assert return_dict is True
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=masked.cumsum(dim=1))


class TinyImageBatch:
    def __init__(self):
        self.images = {
            "image_base": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
            "image_wrist": torch.full((2, 4, 4, 3), 32, dtype=torch.uint8),
        }
        self.next_images = {key: value + 1 for key, value in self.images.items()}
        self.proprioceptions = torch.zeros(2, 5)
        self.next_proprioceptions = torch.ones(2, 5)
        self.languages = ["click mouse", "press button"]


def _backbone() -> GemmaSiglipStateBackbone:
    return GemmaSiglipStateBackbone(
        vision_model=FakeVisionModel(),
        gemma_model=FakeGemmaModel(),
        image_processor=FakeImageProcessor(),
        tokenizer=FakeTokenizer(),
        camera_keys=("image_base", "image_wrist"),
        proprio_dim=5,
        vision_hidden_size=6,
        gemma_hidden_size=8,
        max_language_tokens=8,
    )


def test_multimodal_backbone_uses_both_cameras_language_and_proprioception():
    torch.manual_seed(0)
    backbone = _backbone()
    batch = TinyImageBatch()

    original = backbone(batch)
    changed_base = TinyImageBatch()
    changed_base.images["image_base"].fill_(255)
    changed_wrist = TinyImageBatch()
    changed_wrist.images["image_wrist"].fill_(255)
    changed_proprio = TinyImageBatch()
    changed_proprio.proprioceptions.fill_(2.0)
    changed_language = TinyImageBatch()
    changed_language.languages = ["a", "bb"]

    assert original.shape == (2, 8)
    assert not torch.allclose(original, backbone(changed_base))
    assert not torch.allclose(original, backbone(changed_wrist))
    assert not torch.allclose(original, backbone(changed_proprio))
    assert not torch.allclose(original, backbone(changed_language))


def test_next_observation_switches_images_and_proprioception():
    backbone = _backbone()
    batch = TinyImageBatch()

    assert not torch.allclose(backbone(batch), backbone(batch, next_observation=True))


def test_pretrained_modules_are_frozen_but_fusion_parameters_are_trainable():
    backbone = _backbone()

    assert all(not parameter.requires_grad for parameter in backbone.vision_model.parameters())
    assert all(not parameter.requires_grad for parameter in backbone.gemma_model.parameters())
    assert all(parameter.requires_grad for parameter in backbone.visual_projection.parameters())
    assert all(parameter.requires_grad for parameter in backbone.proprio_projection.parameters())
    assert backbone.readout_token.requires_grad


def test_missing_camera_is_rejected():
    backbone = _backbone()
    batch = TinyImageBatch()
    del batch.images["image_wrist"]

    with pytest.raises(KeyError, match="image_wrist"):
        backbone(batch)
