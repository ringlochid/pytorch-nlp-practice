from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_run_dir(output_root: str, experiment_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | os.PathLike, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_device(preferred: str = "cpu") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
