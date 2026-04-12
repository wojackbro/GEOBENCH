from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(max(mse, 0.0)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0:
        r2 = 0.0
    else:
        r2 = float(1.0 - (np.sum(err**2) / denom))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def target_encode(y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return y
    if mode in {"log1p", "auto"}:
        return torch.sign(y) * torch.log1p(torch.abs(y))
    raise ValueError(f"Unknown target transform: {mode}")


def target_decode(y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return y
    if mode in {"log1p", "auto"}:
        return torch.sign(y) * (torch.expm1(torch.abs(y)))
    raise ValueError(f"Unknown target transform: {mode}")
