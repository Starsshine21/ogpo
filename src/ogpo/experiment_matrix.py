from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    config_path: str
    algorithm: str
    requires_ogpo_checkpoint: bool = True


METHOD_SPECS: dict[str, MethodSpec] = {
    "pi05_sft": MethodSpec(
        config_path="configs/ogpo/methods/pi05_sft.yaml",
        algorithm="sft",
        requires_ogpo_checkpoint=False,
    ),
    "scalar_single_q_awr": MethodSpec(
        config_path="configs/ogpo/pi05_awr.yaml",
        algorithm="awr",
    ),
    "scalar_q_ensemble_full": MethodSpec(
        config_path="configs/ogpo/methods/scalar_q_ensemble_full.yaml",
        algorithm="full_ogpo",
    ),
    "divl_full": MethodSpec(
        config_path="configs/ogpo/methods/divl_full.yaml",
        algorithm="full_ogpo",
    ),
    "divl_flash": MethodSpec(
        config_path="configs/ogpo/methods/divl_flash.yaml",
        algorithm="flash_ogpo",
    ),
    "udivl_flash": MethodSpec(
        config_path="configs/ogpo/pi05_flash_ogpo.yaml",
        algorithm="flash_ogpo",
    ),
}


def get_method_spec(name: str) -> MethodSpec:
    try:
        return METHOD_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown OGPO evaluation method: {name}") from exc
