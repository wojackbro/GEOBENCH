"""
Post-hoc ensemble of satellite-only and street-only checkpoints.

You cannot "pick the model with the best prediction" at inference time without labels.
This script does what *is* valid:

1. **Convex blend** on the validation set: find w in [0, 1] that minimizes MSE of
      y_hat = w * y_sat + (1 - w) * y_street
   (in raw target space, after log1p decode). At test time you use the same fixed w.

2. **Oracle bound** (analysis only): per sample, use whichever of the two preds is closer
   to y. This is an upper bound — not deployable — useful to see if modalities are
   complementary.

Usage (from CityLens package root, same as train.py):

  python -m evaluate.global_learned.ensemble_blend \\
    --task_name gdp \\
    --data_root /path/to/CityLens-data \\
    --sat_exp_dir .../Results/global_learned/gdp/<satellite-run-folder> \\
    --street_exp_dir .../Results/global_learned/gdp/<street-run-folder> \\
    --split_branch fusion

`--split_branch fusion` uses the fusion train/val split and only samples with street views,
so satellite and street models are evaluated on the **same** validation IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    GlobalSatelliteDataset,
    GlobalStreetViewDataset,
    filter_items_for_branch,
    load_global_items,
    make_or_load_split,
    resolve_data_root,
)
from .models import (
    StreetViewRegressor,
    SatelliteRegressor,
    make_satellite_encoder,
    make_street_encoder,
    materialize_model,
)
from .train import TargetTransform, forward_batch
from .utils import ExperimentConfig, compute_metrics, results_root


def load_cfg(exp_dir: Path) -> ExperimentConfig:
    path = exp_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    data = json.load(open(path, "r", encoding="utf-8"))
    return ExperimentConfig(**data)


def build_val_loaders(
    val_items: List[Dict],
    cfg_sat: ExperimentConfig,
    cfg_st: ExperimentConfig,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    sat_ds = GlobalSatelliteDataset(val_items, image_size=cfg_sat.image_size, train=False)
    st_ds = GlobalStreetViewDataset(val_items, image_size=cfg_st.image_size, train=False)
    common = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return DataLoader(sat_ds, **common), DataLoader(st_ds, **common)


@torch.no_grad()
def collect_preds(
    model: torch.nn.Module,
    loader: DataLoader,
    branch: str,
    device: torch.device,
    tt: TargetTransform,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    ys: List[float] = []
    ps: List[float] = []
    ids: List[str] = []
    for batch in loader:
        pred, _, raw = forward_batch(model, batch, branch, device, tt)
        pred_raw = tt.decode_tensor(pred)
        ys.extend(raw.detach().cpu().tolist())
        ps.extend(pred_raw.detach().cpu().tolist())
        ids.extend(batch["id"])
    return np.asarray(ys, dtype=np.float64), np.asarray(ps, dtype=np.float64), ids


def best_convex_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Minimize MSE(y, w*a + (1-w)*b) for w in [0,1]."""
    best_w = 0.0
    best_mse = float("inf")
    for w in np.linspace(0.0, 1.0, 201):
        p = w * a + (1.0 - w) * b
        mse = float(np.mean((y - p) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_w = float(w)
    p = best_w * a + (1.0 - best_w) * b
    metrics = compute_metrics(y.tolist(), p.tolist())
    return best_w, {"weight_satellite": best_w, **metrics}


def oracle_metrics(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Per-sample pick closer prediction (not usable at deployment)."""
    err_a = np.abs(y - a)
    err_b = np.abs(y - b)
    p = np.where(err_a <= err_b, a, b)
    return compute_metrics(y.tolist(), p.tolist())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend satellite + street checkpoints (val-tuned weight)")
    p.add_argument("--task_name", required=True)
    p.add_argument("--data_root", default=None)
    p.add_argument("--sat_exp_dir", required=True, help="Folder with config.json + checkpoints/best.pt")
    p.add_argument("--street_exp_dir", required=True)
    p.add_argument(
        "--split_branch",
        default="fusion",
        choices=["fusion", "street", "satellite"],
        help="Which saved split JSON to use (fusion = same cohort as fusion training)",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    sat_dir = Path(args.sat_exp_dir)
    st_dir = Path(args.street_exp_dir)

    cfg_sat = load_cfg(sat_dir)
    cfg_st = load_cfg(st_dir)
    if cfg_sat.task_name != args.task_name or cfg_st.task_name != args.task_name:
        raise ValueError("task_name mismatch vs config.json in experiment dirs")
    if cfg_sat.target_transform != cfg_st.target_transform:
        raise ValueError(
            f"target_transform must match: sat={cfg_sat.target_transform} st={cfg_st.target_transform}"
        )

    split_path = (
        results_root(data_root)
        / "splits"
        / f"{args.task_name}_{args.split_branch}_seed{cfg_sat.seed}_val{cfg_sat.val_frac}.json"
    )
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split not found: {split_path}. Train {args.split_branch} once with same seed/val_frac."
        )

    items, _ = load_global_items(args.task_name, data_root, return_report=True)
    filtered, _ = filter_items_for_branch(items, args.split_branch)
    _, val_items = make_or_load_split(filtered, split_path, seed=cfg_sat.seed, val_frac=cfg_sat.val_frac)
    if not val_items:
        raise RuntimeError("No validation items")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tt = TargetTransform(cfg_sat.target_transform)

    sat_loader, st_loader = build_val_loaders(val_items, cfg_sat, cfg_st, args.batch_size, args.num_workers)

    sat_model = SatelliteRegressor(make_satellite_encoder(cfg_sat.satellite_model, lora_r=cfg_sat.lora_r)).to(
        device
    )
    st_model = StreetViewRegressor(
        make_street_encoder(cfg_st.street_model), pooling=cfg_st.pooling
    ).to(device)

    sb = next(iter(sat_loader))
    tb = next(iter(st_loader))
    materialize_model(sat_model, "satellite", sb, device)
    materialize_model(st_model, "street", tb, device)

    sat_state = torch.load(sat_dir / "checkpoints" / "best.pt", map_location=device)
    st_state = torch.load(st_dir / "checkpoints" / "best.pt", map_location=device)
    sat_model.load_state_dict(sat_state["model"], strict=True)
    st_model.load_state_dict(st_state["model"], strict=True)

    y_sat, p_sat, ids_sat = collect_preds(sat_model, sat_loader, "satellite", device, tt)
    y_st, p_st, ids_st = collect_preds(st_model, st_loader, "street", device, tt)

    if ids_sat != ids_st or not np.allclose(y_sat, y_st):
        raise RuntimeError("Batch alignment failed: check batch_size and datasets")

    y = y_sat
    solo_sat = compute_metrics(y.tolist(), p_sat.tolist())
    solo_st = compute_metrics(y.tolist(), p_st.tolist())
    w, blend = best_convex_weight(y, p_sat, p_st)
    ora = oracle_metrics(y, p_sat, p_st)

    print(json.dumps({"val_n": len(y), "satellite_only": solo_sat, "street_only": solo_st}, indent=2))
    print(json.dumps({"convex_blend_val_tuned": blend}, indent=2))
    print(json.dumps({"oracle_pick_closer_per_sample_NOT_deployable": ora}, indent=2))
    print(
        f"\nUse at inference (raw space): pred = {w:.3f} * sat_pred + {1 - w:.3f} * street_pred\n"
        f"(Tune w on val only once; same w for test.)"
    )


if __name__ == "__main__":
    main()
