from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class JSONLMetricsWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                clean[key] = float(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().tolist()
            else:
                clean[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, sort_keys=True) + "\n")


class TensorBoardMetricsWriter:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            self.writer = None
        else:
            self.writer = SummaryWriter(str(self.log_dir))

    def write(self, payload: dict[str, Any]) -> None:
        if self.writer is None:
            return
        step = int(payload.get("step", 0))
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(key, float(value.detach().cpu().item()), step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(key, float(value), step)
        self.writer.flush()


class CompositeMetricsWriter:
    def __init__(self, writers):
        self.writers = list(writers)

    def write(self, payload: dict[str, Any]) -> None:
        for writer in self.writers:
            writer.write(payload)


def create_metrics_writer(jsonl_path: str | Path, tensorboard_dir: str | Path | None = None):
    writers = [JSONLMetricsWriter(jsonl_path)]
    if tensorboard_dir:
        writers.append(TensorBoardMetricsWriter(tensorboard_dir))
    return CompositeMetricsWriter(writers)


def add_run_metadata(metrics: dict[str, Any], *, config: dict[str, Any], step: int) -> dict[str, Any]:
    evaluation_cfg = config.get("evaluation", {})
    training_cfg = config.get("training", {})
    tasks = evaluation_cfg.get("tasks", ["unknown_task"])
    seeds = evaluation_cfg.get("seeds", [training_cfg.get("seed", 0)])
    metrics["step"] = step
    metrics["task_id"] = ",".join(str(task) for task in tasks)
    metrics["seed"] = int(seeds[0]) if seeds else int(training_cfg.get("seed", 0))
    return metrics


def grad_norm(parameters) -> float:
    norms = []
    for param in parameters:
        if param.grad is not None:
            norms.append(param.grad.detach().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())
