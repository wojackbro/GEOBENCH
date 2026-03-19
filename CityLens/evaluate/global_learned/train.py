from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    GlobalFusionDataset,
    GlobalSatelliteDataset,
    GlobalStreetViewDataset,
    filter_items_for_branch,
    load_global_items,
    make_or_load_split,
    resolve_data_root,
)
from .models import (
    FusionRegressor,
    SatelliteRegressor,
    StreetViewRegressor,
    make_satellite_encoder,
    make_street_encoder,
    materialize_model,
)
from .utils import (
    ExperimentConfig,
    GLOBAL_TASKS,
    append_experiment_log,
    compute_group_metrics,
    compute_metrics,
    dump_config,
    ensure_dir,
    results_root,
    save_json,
    set_seed,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CityLens learned global-task baselines")
    parser.add_argument("--task_name", default="all", choices=["all"] + GLOBAL_TASKS)
    parser.add_argument("--branch", default="satellite", choices=["satellite", "street", "fusion"])
    parser.add_argument("--satellite_model", default="prithvi_rgb_lora")
    parser.add_argument("--street_model", default="clip_vitb16")
    parser.add_argument("--fusion_type", default="late", choices=["late", "gated"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "attention"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--target_transform", default="auto", choices=["auto", "raw", "log1p"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--skip_if_done", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class TargetTransform:
    def __init__(self, mode: str):
        self.mode = mode

    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "log1p":
            return torch.log1p(torch.clamp(x, min=0.0))
        return x

    def decode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "log1p":
            return torch.expm1(x)
        return x

    def decode_float(self, value: float) -> float:
        if self.mode == "log1p":
            return max(0.0, math.expm1(value))
        return value


def resolve_target_transform(task_name: str, requested: str) -> str:
    if requested == "auto":
        return "log1p" if task_name in GLOBAL_TASKS else "raw"
    return requested


def build_dataloaders(
    branch: str,
    train_items: List[Dict],
    val_items: List[Dict],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    if branch == "satellite":
        train_ds = GlobalSatelliteDataset(train_items, image_size=image_size, train=True)
        val_ds = GlobalSatelliteDataset(val_items, image_size=image_size, train=False)
    elif branch == "street":
        train_ds = GlobalStreetViewDataset(train_items, image_size=image_size, train=True)
        val_ds = GlobalStreetViewDataset(val_items, image_size=image_size, train=False)
    else:
        train_ds = GlobalFusionDataset(train_items, image_size=image_size, train=True)
        val_ds = GlobalFusionDataset(val_items, image_size=image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg: ExperimentConfig) -> nn.Module:
    if cfg.branch == "satellite":
        return SatelliteRegressor(make_satellite_encoder(cfg.satellite_model, lora_r=cfg.lora_r))
    if cfg.branch == "street":
        return StreetViewRegressor(make_street_encoder(cfg.street_model), pooling=cfg.pooling)
    return FusionRegressor(
        make_satellite_encoder(cfg.satellite_model, lora_r=cfg.lora_r),
        make_street_encoder(cfg.street_model),
        pooling=cfg.pooling,
        fusion_type=cfg.fusion_type,
    )


def forward_batch(
    model: nn.Module,
    batch: Dict,
    branch: str,
    device: torch.device,
    target_transform: TargetTransform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_target = batch["target"].to(device)
    target = target_transform.encode_tensor(raw_target)
    if branch == "satellite":
        pred = model(batch["image"].to(device))
    elif branch == "street":
        pred = model(batch["street_views"].to(device), batch["street_mask"].to(device))
    else:
        pred = model(
            batch["image"].to(device),
            batch["street_views"].to(device),
            batch["street_mask"].to(device),
        )
    return pred, target, raw_target


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    branch: str,
    device: torch.device,
    target_transform: TargetTransform,
) -> Tuple[Dict[str, float], List[Dict]]:
    model.eval()
    y_true: List[float] = []
    y_pred: List[float] = []
    predictions: List[Dict] = []
    with torch.no_grad():
        for batch in loader:
            pred, _, raw_target = forward_batch(model, batch, branch, device, target_transform)
            pred_raw = target_transform.decode_tensor(pred)
            pred_cpu = pred_raw.detach().cpu().tolist()
            target_cpu = raw_target.detach().cpu().tolist()
            ids = batch["id"]
            cities = batch["city"]
            y_true.extend(target_cpu)
            y_pred.extend(pred_cpu)
            predictions.extend(
                {"id": sample_id, "city": city, "target": float(t), "prediction": float(p)}
                for sample_id, city, t, p in zip(ids, cities, target_cpu, pred_cpu)
            )
    return compute_metrics(y_true, y_pred), predictions


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in ["head", "input_adapter", "gate", "pool"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": cfg.backbone_lr,
                "weight_decay": cfg.weight_decay,
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": cfg.head_lr,
                "weight_decay": cfg.weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)


def save_predictions(path: Path, rows: List[Dict]) -> None:
    write_rows_csv(path, rows, fieldnames=["id", "city", "target", "prediction"])


def save_per_city_metrics(path: Path, rows: List[Dict]) -> None:
    metrics = compute_group_metrics(rows, "city")
    json_path = path.with_suffix(".json")
    csv_path = path.with_suffix(".csv")
    save_json(json_path, metrics)
    csv_rows = [{"city": city, **vals} for city, vals in metrics.items()]
    write_rows_csv(csv_path, csv_rows, fieldnames=["city", "mse", "mae", "r2", "rmse"])


def run_task(task_name: str, args: argparse.Namespace, data_root: Path) -> None:
    cfg = ExperimentConfig(
        task_name=task_name,
        branch=args.branch,
        satellite_model=args.satellite_model,
        street_model=args.street_model,
        fusion_type=args.fusion_type if args.branch == "fusion" else "none",
        pooling=args.pooling if args.branch in {"street", "fusion"} else "none",
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr=args.backbone_lr if args.backbone_lr is not None else args.lr,
        head_lr=args.head_lr,
        val_frac=args.val_frac,
        seed=args.seed,
        lora_r=args.lora_r,
        target_transform=resolve_target_transform(task_name, args.target_transform),
        num_workers=args.num_workers,
    )

    task_root = results_root(data_root) / task_name / cfg.experiment_name()
    ckpt_dir = ensure_dir(task_root / "checkpoints")
    split_path = results_root(data_root) / "splits" / f"{task_name}_{cfg.branch}_seed{args.seed}_val{args.val_frac}.json"
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    metrics_path = task_root / "metrics.json"
    preds_path = task_root / "val_predictions.csv"
    per_city_path = task_root / "per_city_metrics"
    history_path = task_root / "history.csv"

    if args.skip_if_done and best_ckpt.exists() and metrics_path.exists():
        metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
        print(f"[skip] {task_name}: found existing checkpoint at {best_ckpt}")
        append_experiment_log(data_root, cfg, split_path, ckpt_dir, metrics, status="skipped")
        return

    items, load_report = load_global_items(task_name, data_root, return_report=True)
    filtered_items, filter_report = filter_items_for_branch(items, cfg.branch)
    if not filtered_items:
        raise RuntimeError(f"No records available for branch={cfg.branch} task={task_name}")

    train_items, val_items = make_or_load_split(filtered_items, split_path, seed=args.seed, val_frac=args.val_frac)
    train_loader, val_loader = build_dataloaders(
        cfg.branch, train_items, val_items, args.image_size, args.batch_size, args.num_workers
    )
    dataset_report = {
        **load_report,
        **filter_report,
        "train_records": len(train_items),
        "val_records": len(val_items),
    }
    save_json(task_root / "dataset_report.json", dataset_report)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    sample_batch = next(iter(train_loader if len(train_loader) > 0 else val_loader))
    materialize_model(model, cfg.branch, sample_batch, device)
    optimizer = build_optimizer(model, cfg)
    loss_fn = nn.MSELoss()
    target_transform = TargetTransform(cfg.target_transform)
    start_epoch = 1
    best_rmse = float("inf")
    history_rows: List[Dict] = []

    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError:
            print(f"[resume] {task_name}: optimizer state mismatch, continuing with fresh optimizer")
        start_epoch = int(state["epoch"]) + 1
        best_rmse = float(state.get("best_rmse", best_rmse))
        print(f"[resume] {task_name}: resuming from epoch {start_epoch}")

    dump_config(task_root / "config.json", cfg)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        prog = tqdm(train_loader, desc=f"{task_name} epoch {epoch}/{args.epochs}", leave=False)
        for batch in prog:
            pred, target, _ = forward_batch(model, batch, cfg.branch, device, target_transform)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = target.numel()
            running += loss.item() * batch_n
            seen += batch_n
            prog.set_postfix(train_loss=running / max(1, seen))

        val_metrics, predictions = evaluate(model, val_loader, cfg.branch, device, target_transform)
        pred_values = [row["prediction"] for row in predictions]
        target_values = [row["target"] for row in predictions]
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "pred_mean": float(sum(pred_values) / max(1, len(pred_values))),
            "pred_std": float(torch.tensor(pred_values).std(unbiased=False).item()) if pred_values else float("nan"),
            "target_mean": float(sum(target_values) / max(1, len(target_values))),
            "target_std": float(torch.tensor(target_values).std(unbiased=False).item()) if target_values else float("nan"),
            **val_metrics,
        }
        history_rows.append(row)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_rmse": best_rmse,
                    "config": cfg.__dict__,
                },
                best_ckpt,
            )
            save_predictions(preds_path, predictions)
            save_per_city_metrics(per_city_path, predictions)
            save_json(metrics_path, {"best_epoch": epoch, "val_records": len(predictions), **val_metrics})

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_rmse": best_rmse,
                "config": cfg.__dict__,
            },
            last_ckpt,
        )

        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "pred_mean",
                    "pred_std",
                    "target_mean",
                    "target_std",
                    "mse",
                    "mae",
                    "r2",
                    "rmse",
                ],
            )
            writer.writeheader()
            writer.writerows(history_rows)

        print(
            f"[{task_name}] epoch={epoch} val_rmse={val_metrics['rmse']:.4f} "
            f"val_r2={val_metrics['r2']:.4f} pred_std={row['pred_std']:.4f} "
            f"target_std={row['target_std']:.4f} transform={cfg.target_transform}"
        )

    final_metrics = json.load(open(metrics_path, "r", encoding="utf-8")) if metrics_path.exists() else {}
    append_experiment_log(data_root, cfg, split_path, ckpt_dir, final_metrics, status="finished")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = resolve_data_root(args.data_root)
    tasks = GLOBAL_TASKS if args.task_name == "all" else [args.task_name]
    for task_name in tasks:
        run_task(task_name, args, data_root)


if __name__ == "__main__":
    main()
