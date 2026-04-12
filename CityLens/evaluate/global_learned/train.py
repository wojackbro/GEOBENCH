from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import TASKS, GlobalLearnedDataset, collate_batch, load_global_items
from .models import FusionRegressor, SatelliteRegressor, StreetViewRegressor, make_satellite_encoder, make_street_encoder
from .utils import append_csv_row, ensure_dir, load_json, metric_dict, save_json, set_seed, target_decode, target_encode


def _experiment_name(args: argparse.Namespace, task_name: str) -> str:
    bb = f"-blr{args.backbone_lr}" if args.backbone_lr is not None else ""
    hl = f"-hlr{args.head_lr}" if args.head_lr is not None else ""
    tt = f"-tt{args.target_transform}" if args.target_transform else ""
    if args.branch == "satellite":
        return (
            f"{task_name}-satellite-{args.satellite_model}-{args.street_model}"
            f"-bs{args.batch_size}-ep{args.epochs}-lr{args.lr}{bb}{hl}{tt}-seed{args.seed}"
        )
    if args.branch == "street":
        return (
            f"{task_name}-street-{args.satellite_model}-{args.street_model}-{args.pooling}"
            f"-bs{args.batch_size}-ep{args.epochs}-lr{args.lr}{bb}{hl}{tt}-seed{args.seed}"
        )
    return (
        f"{task_name}-fusion-{args.satellite_model}-{args.street_model}-{args.fusion_type}-{args.pooling}"
        f"-bs{args.batch_size}-ep{args.epochs}-lr{args.lr}{bb}{hl}{tt}-seed{args.seed}"
    )


def _split_path(data_root: Path, task_name: str, args: argparse.Namespace) -> Path:
    return data_root / "Results" / "global_learned" / "splits" / f"{task_name}_{args.split_key}_seed{args.seed}_val{args.val_frac}.json"


def _select_by_split(items: list[Any], split_json: Path) -> tuple[list[Any], list[Any]]:
    split = load_json(split_json)
    train_ids = set(split["train_ids"])
    val_ids = set(split["val_ids"])
    tr = [x for x in items if x.uid in train_ids]
    va = [x for x in items if x.uid in val_ids]
    return tr, va


def _build_model(args: argparse.Namespace) -> nn.Module:
    sat = make_satellite_encoder(args.satellite_model, lora_r=args.lora_r, img_size=args.image_size)
    if args.branch == "satellite":
        return SatelliteRegressor(sat)
    stv = make_street_encoder(args.street_model, img_size=args.image_size)
    if args.branch == "street":
        return StreetViewRegressor(stv, pooling=args.pooling)
    return FusionRegressor(sat, stv, pooling=args.pooling, fusion_type=args.fusion_type)


def _forward(model: nn.Module, batch: dict[str, torch.Tensor], branch: str, device: torch.device) -> torch.Tensor:
    if branch == "satellite":
        return model(batch["image"].to(device))
    if branch == "street":
        return model(batch["street_views"].to(device), batch["street_mask"].to(device))
    return model(batch["image"].to(device), batch["street_views"].to(device), batch["street_mask"].to(device))


def evaluate(model: nn.Module, dl: DataLoader, branch: str, device: torch.device, target_transform: str) -> dict[str, Any]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for b in dl:
            pred_t = _forward(model, b, branch, device)
            pred = target_decode(pred_t, target_transform).cpu().numpy()
            true = b["target"].cpu().numpy()
            ys.append(true)
            ps.append(pred)
            for i in range(len(pred)):
                rows.append({"id": b["id"][i], "city": b["city"][i], "y_true": float(true[i]), "y_pred": float(pred[i])})
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
    p = np.concatenate(ps, axis=0) if ps else np.zeros((0,), dtype=np.float32)
    m = metric_dict(y, p) if len(y) > 0 else {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}
    m["val_records"] = int(len(y))
    m["rows"] = rows
    return m


def run_task(task_name: str, args: argparse.Namespace, data_root: Path) -> None:
    split_path = _split_path(data_root, task_name, args)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file missing: {split_path}. Run make_shared_split first.")

    branch_for_load = "satellite" if args.branch == "satellite" else "street"
    items, report = load_global_items(data_root, task_name, branch=branch_for_load, return_report=True)
    tr_items, va_items = _select_by_split(items, split_path)
    if len(tr_items) == 0 or len(va_items) == 0:
        raise RuntimeError(f"No records available for branch={args.branch} task={task_name}")

    exp_name = _experiment_name(args, task_name)
    exp_dir = ensure_dir(data_root / "Results" / "global_learned" / task_name / exp_name)
    ckpt_dir = ensure_dir(exp_dir / "checkpoints")
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    metrics_path = exp_dir / "metrics.json"
    history_path = exp_dir / "history.csv"

    if args.skip_if_done and best_ckpt.exists() and metrics_path.exists():
        print(f"[skip] {task_name}: found existing checkpoint at {best_ckpt}")
        append_csv_row(
            data_root / "Results" / "global_learned" / "experiment_log.csv",
            {
                "task_name": task_name,
                "branch": args.branch,
                "satellite_model": args.satellite_model,
                "street_model": args.street_model,
                "fusion_type": args.fusion_type,
                "pooling": args.pooling,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "val_frac": args.val_frac,
                "seed": args.seed,
                "split_path": str(split_path),
                "checkpoint_dir": str(ckpt_dir),
                "status": "skipped",
            },
        )
        return

    save_json(
        exp_dir / "config.json",
        {
            "task_name": task_name,
            "branch": args.branch,
            "satellite_model": args.satellite_model,
            "street_model": args.street_model,
            "fusion_type": args.fusion_type,
            "pooling": args.pooling,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "lora_r": args.lora_r,
            "target_transform": args.target_transform,
            "split_path": str(split_path),
            "checkpoint_dir": str(ckpt_dir),
        },
    )
    save_json(exp_dir / "dataset_report.json", {"loader_report": report, "n_train": len(tr_items), "n_val": len(va_items)})

    tr_ds = GlobalLearnedDataset(tr_items, image_size=args.image_size)
    va_ds = GlobalLearnedDataset(va_items, image_size=args.image_size)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_batch)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    start_epoch = 1
    best_r2 = -1e9
    best_epoch = 0

    if args.resume and last_ckpt.exists():
        obj = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(obj["model"])
        opt.load_state_dict(obj["optimizer"])
        start_epoch = int(obj.get("epoch", 0)) + 1
        best_r2 = float(obj.get("best_r2", best_r2))
        best_epoch = int(obj.get("best_epoch", best_epoch))
        print(f"[resume] {task_name}: epoch={start_epoch} from {last_ckpt}")

    hist_exists = history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as f_hist:
        writer = csv.DictWriter(
            f_hist,
            fieldnames=["epoch", "train_loss", "mse", "mae", "r2", "rmse", "val_records", "pred_mean", "pred_std"],
        )
        if not hist_exists:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            losses = []
            for b in tr_dl:
                opt.zero_grad(set_to_none=True)
                pred_raw = _forward(model, b, args.branch, device)
                y = b["target"].to(device)
                pred = target_encode(pred_raw, args.target_transform)
                tgt = target_encode(y, args.target_transform)
                loss = criterion(pred, tgt)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            val = evaluate(model, va_dl, args.branch, device, args.target_transform)
            pred_vals = np.array([r["y_pred"] for r in val["rows"]], dtype=np.float64) if val["rows"] else np.zeros((0,))
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses) if losses else 0.0),
                    "mse": val["mse"],
                    "mae": val["mae"],
                    "r2": val["r2"],
                    "rmse": val["rmse"],
                    "val_records": val["val_records"],
                    "pred_mean": float(pred_vals.mean()) if len(pred_vals) else 0.0,
                    "pred_std": float(pred_vals.std()) if len(pred_vals) else 0.0,
                }
            )
            f_hist.flush()

            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "best_r2": best_r2, "best_epoch": best_epoch},
                last_ckpt,
            )
            if val["r2"] >= best_r2:
                best_r2 = float(val["r2"])
                best_epoch = int(epoch)
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "best_r2": best_r2,
                        "best_epoch": best_epoch,
                    },
                    best_ckpt,
                )

    # final eval from best checkpoint
    best_obj = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_obj["model"])
    final = evaluate(model, va_dl, args.branch, device, args.target_transform)
    final["best_epoch"] = int(best_obj.get("best_epoch", best_epoch))
    save_json(metrics_path, {k: v for k, v in final.items() if k != "rows"})

    pred_csv = exp_dir / "val_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "city", "y_true", "y_pred"])
        w.writeheader()
        w.writerows(final["rows"])

    # simple per-city metrics
    city_groups: dict[str, list[dict[str, Any]]] = {}
    for row in final["rows"]:
        city_groups.setdefault(str(row["city"]), []).append(row)
    per_city_json: dict[str, dict[str, float]] = {}
    for c, rows in city_groups.items():
        yt = np.array([r["y_true"] for r in rows], dtype=np.float64)
        yp = np.array([r["y_pred"] for r in rows], dtype=np.float64)
        per_city_json[c] = metric_dict(yt, yp)
    save_json(exp_dir / "per_city_metrics.json", per_city_json)
    with (exp_dir / "per_city_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city", "mse", "mae", "r2", "rmse"])
        w.writeheader()
        for c, m in per_city_json.items():
            w.writerow({"city": c, **m})

    append_csv_row(
        data_root / "Results" / "global_learned" / "experiment_log.csv",
        {
            "task_name": task_name,
            "branch": args.branch,
            "satellite_model": args.satellite_model,
            "street_model": args.street_model,
            "fusion_type": args.fusion_type,
            "pooling": args.pooling,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "lora_r": args.lora_r,
            "split_path": str(split_path),
            "checkpoint_dir": str(ckpt_dir),
            "status": "finished",
            "best_epoch": final["best_epoch"],
            "val_records": final["val_records"],
            "mse": final["mse"],
            "mae": final["mae"],
            "r2": final["r2"],
            "rmse": final["rmse"],
        },
    )
    print(f"[done] {task_name}: r2={final['r2']:.4f} rmse={final['rmse']:.4f} best_epoch={final['best_epoch']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", type=str, default="all")
    p.add_argument("--branch", type=str, choices=["satellite", "street", "fusion"], required=True)
    p.add_argument("--satellite_model", type=str, default="prithvi_rgb_lora")
    p.add_argument("--street_model", type=str, default="clip_vitb16")
    p.add_argument("--fusion_type", type=str, default="late")
    p.add_argument("--pooling", type=str, default="mean")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backbone_lr", type=float, default=2e-4)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--split_key", type=str, default="fusion")
    p.add_argument("--target_transform", type=str, default="log1p")
    p.add_argument("--skip_if_done", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data_root", type=str, default=None)
    args = p.parse_args()

    data_root = Path(args.data_root or os.environ.get("CITYLENS_DATA_ROOT", "")).expanduser()
    if not str(data_root):
        raise RuntimeError("Missing data root: pass --data_root or set CITYLENS_DATA_ROOT")
    set_seed(args.seed)

    tasks = TASKS if args.task_name == "all" else [args.task_name]
    for t in tasks:
        run_task(t, args, data_root)


if __name__ == "__main__":
    main()
