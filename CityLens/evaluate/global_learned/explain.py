from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image

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
from .utils import GLOBAL_TASKS, ensure_dir, results_root, save_json, set_seed, write_rows_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain learned CityLens global-task models")
    parser.add_argument("--task_name", required=True, choices=GLOBAL_TASKS)
    parser.add_argument("--branch", required=True, choices=["satellite", "street", "fusion"])
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--satellite_model", default="prithvi_rgb_lora")
    parser.add_argument("--street_model", default="clip_vitb16")
    parser.add_argument("--fusion_type", default="late", choices=["late", "gated"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "attention"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=8)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.branch == "satellite":
        return SatelliteRegressor(make_satellite_encoder(args.satellite_model))
    if args.branch == "street":
        return StreetViewRegressor(make_street_encoder(args.street_model), pooling=args.pooling)
    return FusionRegressor(
        make_satellite_encoder(args.satellite_model),
        make_street_encoder(args.street_model),
        pooling=args.pooling,
        fusion_type=args.fusion_type,
    )


def apply_checkpoint_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = Path(args.checkpoint_dir) / "config.json"
    if config_path.exists():
        cfg = json_load(config_path)
        for key in ["branch", "satellite_model", "street_model", "fusion_type", "pooling", "image_size", "seed", "val_frac"]:
            if key in cfg:
                setattr(args, key, cfg[key])
    return args


def json_load(path: Path) -> Dict:
    import json

    return json.load(open(path, "r", encoding="utf-8"))


def load_val_samples(args: argparse.Namespace, data_root: Path) -> List[Dict]:
    items, _ = load_global_items(args.task_name, data_root, return_report=True)
    filtered_items, _ = filter_items_for_branch(items, args.branch)
    split_path = results_root(data_root) / "splits" / f"{args.task_name}_{args.branch}_seed{args.seed}_val{args.val_frac}.json"
    _, val_items = make_or_load_split(filtered_items, split_path, seed=args.seed, val_frac=args.val_frac)
    if args.branch == "satellite":
        ds = GlobalSatelliteDataset(val_items, image_size=args.image_size, train=False)
    elif args.branch == "street":
        ds = GlobalStreetViewDataset(val_items, image_size=args.image_size, train=False)
    else:
        ds = GlobalFusionDataset(val_items, image_size=args.image_size, train=False)
    return [ds[i] for i in range(min(len(ds), args.max_samples))]


def save_heatmap_png(attr: np.ndarray, path: Path) -> None:
    arr = np.abs(attr).mean(axis=0)
    arr = arr - arr.min()
    arr = arr / max(arr.max(), 1e-8)
    image = Image.fromarray((arr * 255).astype(np.uint8))
    image.save(path)


def save_input_preview(image_tensor: torch.Tensor, path: Path) -> None:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.detach().cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def satellite_integrated_gradients(model: torch.nn.Module, sample: Dict, device: torch.device) -> np.ndarray:
    model.eval()
    x = sample["image"].unsqueeze(0).to(device)
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, target=None)
    return attr.detach().cpu().numpy()[0]


def street_leave_one_view_out(model: torch.nn.Module, sample: Dict, device: torch.device) -> List[Dict]:
    model.eval()
    x = sample["street_views"].unsqueeze(0).to(device)
    mask = sample["street_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        base = model(x, mask).item()
    drops = []
    for i in range(x.size(1)):
        if mask[0, i].item() <= 0:
            continue
        x_mod = x.clone()
        mask_mod = mask.clone()
        mask_mod[0, i] = 0
        x_mod[0, i] = 0
        with torch.no_grad():
            pred = model(x_mod, mask_mod).item()
        drops.append({"view_index": i, "base_prediction": base, "ablated_prediction": pred, "delta": base - pred})
    return drops


def fusion_modality_ablation(model: torch.nn.Module, sample: Dict, device: torch.device) -> Dict:
    model.eval()
    image = sample["image"].unsqueeze(0).to(device)
    views = sample["street_views"].unsqueeze(0).to(device)
    mask = sample["street_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        full = model(image, views, mask).item()
        sat_zero = model(torch.zeros_like(image), views, mask).item()
        stv_zero = model(image, torch.zeros_like(views), torch.zeros_like(mask)).item()
    return {
        "id": sample["id"],
        "city": sample["city"],
        "full_prediction": full,
        "no_satellite_prediction": sat_zero,
        "no_street_prediction": stv_zero,
        "satellite_contribution_delta": full - sat_zero,
        "street_contribution_delta": full - stv_zero,
    }


def main() -> None:
    args = apply_checkpoint_config(parse_args())
    set_seed(args.seed)
    data_root = resolve_data_root(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    samples = load_val_samples(args, data_root)
    if not samples:
        raise RuntimeError("No validation samples available for explainability")
    materialize_model(model, args.branch, samples[0], device)
    ckpt = torch.load(Path(args.checkpoint_dir) / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    out_dir = ensure_dir(Path(args.checkpoint_dir) / "explain")
    save_json(
        out_dir / "explainability_manifest.json",
        {
            "branch": args.branch,
            "task_name": args.task_name,
            "max_samples": len(samples),
            "feasibility_note": (
                "Integrated gradients, leave-one-view-out, and modality ablation are feasible here. "
                "True multispectral spectral ablation is not supported because CityLens is RGB-only."
            ),
        },
    )

    if args.branch == "satellite":
        stats = []
        for idx, sample in enumerate(samples):
            attr = satellite_integrated_gradients(model, sample, device)
            sample_dir = ensure_dir(out_dir / f"sample_{idx:02d}_{sample['id']}")
            np.save(sample_dir / "integrated_gradients.npy", attr)
            save_input_preview(sample["image"], sample_dir / "input.png")
            save_heatmap_png(attr, sample_dir / "heatmap.png")
            stats.append(
                {
                    "id": sample["id"],
                    "city": sample["city"],
                    "shape": list(attr.shape),
                    "mean_abs": float(np.abs(attr).mean()),
                    "max_abs": float(np.abs(attr).max()),
                }
            )
        save_json(out_dir / "integrated_gradients_summary.json", {"rows": stats})
    elif args.branch == "street":
        rows = []
        for sample in samples:
            for row in street_leave_one_view_out(model, sample, device):
                rows.append({"id": sample["id"], "city": sample["city"], **row})
        save_json(out_dir / "leave_one_view_out.json", {"rows": rows})
        write_rows_csv(
            out_dir / "leave_one_view_out.csv",
            rows,
            fieldnames=["id", "city", "view_index", "base_prediction", "ablated_prediction", "delta"],
        )
    else:
        rows = [fusion_modality_ablation(model, sample, device) for sample in samples]
        save_json(out_dir / "modality_ablation.json", {"rows": rows})
        write_rows_csv(
            out_dir / "modality_ablation.csv",
            rows,
            fieldnames=[
                "id",
                "city",
                "full_prediction",
                "no_satellite_prediction",
                "no_street_prediction",
                "satellite_contribution_delta",
                "street_contribution_delta",
            ],
        )

    print(f"Saved explainability outputs to {out_dir}")


if __name__ == "__main__":
    main()
