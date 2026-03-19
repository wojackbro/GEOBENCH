import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


GLOBAL_TASKS = ["gdp", "pop", "acc2health", "carbon", "build_height"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace("__", "_")
        .strip("_")
    )


def results_root(data_root: Path) -> Path:
    return data_root / "Results" / "global_learned"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ExperimentConfig:
    task_name: str
    branch: str
    satellite_model: str
    street_model: str
    fusion_type: str
    pooling: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    backbone_lr: float
    head_lr: float
    val_frac: float
    seed: int
    lora_r: int
    target_transform: str
    num_workers: int

    def experiment_name(self) -> str:
        bits = [
            self.task_name,
            self.branch,
            self.satellite_model,
            self.street_model,
            self.fusion_type,
            self.pooling,
            f"bs{self.batch_size}",
            f"ep{self.epochs}",
            f"lr{self.lr}",
            f"blr{self.backbone_lr}",
            f"hlr{self.head_lr}",
            f"tt{self.target_transform}",
            f"seed{self.seed}",
        ]
        return slugify("-".join([b for b in bits if b and b != "none"]))


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=np.float32)
    y_pred = np.asarray(list(y_pred), dtype=np.float32)
    if len(y_true) == 0:
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan"), "rmse": float("nan")}
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2), "rmse": float(rmse)}


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_csv(path: Path, row: Dict, fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_rows_csv(path: Path, rows: Sequence[Dict], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_experiment_log(
    data_root: Path,
    config: ExperimentConfig,
    split_path: Path,
    checkpoint_dir: Path,
    metrics: Dict[str, float],
    status: str,
) -> None:
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "task_name": config.task_name,
        "branch": config.branch,
        "satellite_model": config.satellite_model,
        "street_model": config.street_model,
        "fusion_type": config.fusion_type,
        "pooling": config.pooling,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "val_frac": config.val_frac,
        "seed": config.seed,
        "lora_r": config.lora_r,
        "split_path": str(split_path),
        "checkpoint_dir": str(checkpoint_dir),
        "status": status,
        **metrics,
    }
    append_csv(
        results_root(data_root) / "experiment_log.csv",
        row,
        fieldnames=list(row.keys()),
    )


def dump_config(path: Path, config: ExperimentConfig) -> None:
    save_json(path, asdict(config))


def compute_group_metrics(rows: Sequence[Dict], group_key: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"target": [], "prediction": []})
    for row in rows:
        group = row.get(group_key)
        if group in (None, ""):
            continue
        grouped[str(group)]["target"].append(float(row["target"]))
        grouped[str(group)]["prediction"].append(float(row["prediction"]))
    return {group: compute_metrics(vals["target"], vals["prediction"]) for group, vals in grouped.items()}
