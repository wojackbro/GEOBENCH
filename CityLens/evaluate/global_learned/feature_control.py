from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import GlobalStreetViewDataset, filter_items_for_branch, load_global_items, make_or_load_split, resolve_data_root
from .models import StreetViewRegressor, make_street_encoder
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
    parser = argparse.ArgumentParser(description="CityLens low-cost street-view control")
    parser.add_argument("--task_name", default="all", choices=["all"] + GLOBAL_TASKS)
    parser.add_argument("--street_model", default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--skip_if_done", action="store_true")
    return parser.parse_args()


def make_loader(items: List[Dict], image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    ds = GlobalStreetViewDataset(items, image_size=image_size, train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def extract_embeddings(
    loader: DataLoader, model: StreetViewRegressor, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feats = []
    targets = []
    ids: List[str] = []
    cities: List[str] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="extract"):
            x = batch["street_views"].to(device)
            mask = batch["street_mask"].to(device)
            emb = model.forward_features(x, mask).detach().cpu().numpy()
            feats.append(emb)
            targets.extend(batch["target"].cpu().numpy().tolist())
            ids.extend(list(batch["id"]))
            cities.extend(list(batch["city"]))
    return np.concatenate(feats, axis=0), np.asarray(targets, dtype=np.float32), ids, cities


def save_predictions(path: Path, rows: List[Dict]) -> None:
    write_rows_csv(path, rows, fieldnames=["id", "city", "target", "prediction"])


def run_task(task_name: str, args: argparse.Namespace, data_root: Path) -> None:
    cfg = ExperimentConfig(
        task_name=task_name,
        branch="street_feature_control",
        satellite_model="none",
        street_model=args.street_model,
        fusion_type="none",
        pooling="mean",
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=0,
        lr=0.0,
        weight_decay=0.0,
        val_frac=args.val_frac,
        seed=args.seed,
        lora_r=0,
        num_workers=args.num_workers,
    )

    task_root = results_root(data_root) / task_name / cfg.experiment_name()
    ckpt_dir = ensure_dir(task_root / "checkpoints")
    metrics_path = task_root / "metrics.json"
    preds_path = task_root / "val_predictions.csv"
    per_city_path = task_root / "per_city_metrics"
    split_path = results_root(data_root) / "splits" / f"{task_name}_{cfg.branch}_seed{args.seed}_val{args.val_frac}.json"
    artifact_path = ckpt_dir / "lasso.pkl"

    if args.skip_if_done and metrics_path.exists() and artifact_path.exists():
        metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
        append_experiment_log(data_root, cfg, split_path, ckpt_dir, metrics, status="skipped")
        print(f"[skip] {task_name}: feature control already exists")
        return

    items, load_report = load_global_items(task_name, data_root, return_report=True)
    filtered_items, filter_report = filter_items_for_branch(items, cfg.branch)
    train_items, val_items = make_or_load_split(filtered_items, split_path, seed=args.seed, val_frac=args.val_frac)
    save_json(
        task_root / "dataset_report.json",
        {**load_report, **filter_report, "train_records": len(train_items), "val_records": len(val_items)},
    )
    train_loader = make_loader(train_items, args.image_size, args.batch_size, args.num_workers)
    val_loader = make_loader(val_items, args.image_size, args.batch_size, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = make_street_encoder(args.street_model)
    model = StreetViewRegressor(encoder, pooling="mean").to(device)

    x_train, y_train, _, _ = extract_embeddings(train_loader, model, device)
    x_val, y_val, val_ids, val_cities = extract_embeddings(val_loader, model, device)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    reg = LassoCV(cv=5, random_state=args.seed, max_iter=10000)
    reg.fit(x_train_scaled, y_train)
    y_pred = reg.predict(x_val_scaled)
    metrics = compute_metrics(y_val, y_pred)
    prediction_rows = [
        {"id": sample_id, "city": city, "target": float(target), "prediction": float(pred)}
        for sample_id, city, target, pred in zip(val_ids, val_cities, y_val.tolist(), y_pred.tolist())
    ]

    dump_config(task_root / "config.json", cfg)
    save_json(metrics_path, {"val_records": len(prediction_rows), **metrics})
    save_predictions(preds_path, prediction_rows)
    save_json(per_city_path.with_suffix(".json"), compute_group_metrics(prediction_rows, "city"))
    write_rows_csv(
        per_city_path.with_suffix(".csv"),
        [{"city": city, **vals} for city, vals in compute_group_metrics(prediction_rows, "city").items()],
        fieldnames=["city", "mse", "mae", "r2", "rmse"],
    )
    with open(artifact_path, "wb") as f:
        pickle.dump({"scaler": scaler, "regressor": reg}, f)

    append_experiment_log(data_root, cfg, split_path, ckpt_dir, metrics, status="finished")
    print(f"[{task_name}] feature-control rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = resolve_data_root(args.data_root)
    tasks = GLOBAL_TASKS if args.task_name == "all" else [args.task_name]
    for task_name in tasks:
        run_task(task_name, args, data_root)


if __name__ == "__main__":
    main()
