from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import GLOBAL_TASKS, ensure_dir, save_json


def resolve_data_root(root: Optional[str] = None) -> Path:
    if root:
        return Path(root)
    env_root = os.environ.get("CITYLENS_DATA_ROOT")
    if env_root:
        return Path(env_root)
    try:
        from path_config import DATA_ROOT  # type: ignore

        return Path(DATA_ROOT)
    except Exception as exc:
        raise RuntimeError("CITYLENS_DATA_ROOT is not set and path_config is unavailable") from exc


def global_task_candidates(task_name: str) -> List[str]:
    if task_name not in GLOBAL_TASKS:
        raise ValueError(f"Unsupported global task: {task_name}")
    return [
        f"Benchmark/{task_name}_all.json",
        f"Benchmark/all_global_{task_name}_task.json",
        f"Dataset/all_global_{task_name}_task_all.json",
        f"Dataset/all_global_{task_name}_task-all.json",
    ]


def resolve_task_json(data_root: Path, task_name: str) -> Path:
    for rel in global_task_candidates(task_name):
        p = data_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find task JSON for {task_name} under {data_root}")


def build_image_index(root: Path, folder_name: str) -> Dict[str, Path]:
    folder = root / folder_name
    index: Dict[str, Path] = {}
    if not folder.exists():
        return index
    for dirpath, _, files in os.walk(folder):
        for name in files:
            if name not in index:
                index[name] = Path(dirpath) / name
    return index


def resolve_image_path(
    image_path: str,
    data_root: Path,
    image_index: Optional[Dict[str, Path]] = None,
    default_subdir: Optional[str] = None,
) -> Path:
    p = Path(image_path)
    if p.is_file():
        return p

    candidates = [
        data_root / image_path.lstrip("/"),
        data_root / p.name,
    ]
    if default_subdir:
        candidates.append(data_root / default_subdir / p.name)

    for cand in candidates:
        if cand.is_file():
            return cand

    if image_index and p.name in image_index:
        return image_index[p.name]

    raise FileNotFoundError(f"Could not resolve image path: {image_path}")


def normalize_global_item(item: Dict, data_root: Path, sat_index: Dict[str, Path], stv_index: Dict[str, Path]) -> Dict:
    images = item.get("images", [])
    if not images:
        raise ValueError("Benchmark item has no images")
    sat = resolve_image_path(images[0], data_root, sat_index, "satellite_image")
    stv = []
    for path in images[1:]:
        try:
            stv.append(resolve_image_path(path, data_root, stv_index, "street_view_image"))
        except FileNotFoundError:
            continue
    return {
        "id": item.get("area") or item.get("img_name") or item.get("ct") or sat.stem,
        "city": item.get("city") or item.get("city_name") or sat.parent.name,
        "satellite": sat,
        "street_views": stv,
        "reference": float(item["reference"]),
        "reference_normalized": item.get("reference_normalized"),
        "prompt": item.get("prompt", ""),
        "raw": item,
    }


def load_global_items(task_name: str, data_root: Path, return_report: bool = False):
    task_json = resolve_task_json(data_root, task_name)
    items = json.load(open(task_json, "r", encoding="utf-8"))
    sat_index = build_image_index(data_root, "satellite_image")
    stv_index = build_image_index(data_root, "street_view_image")
    normalized = []
    skipped = Counter()
    for item in items:
        try:
            normalized.append(normalize_global_item(item, data_root, sat_index, stv_index))
        except Exception as exc:
            skipped[type(exc).__name__] += 1
    if not normalized:
        raise RuntimeError(f"No usable items found for {task_name} from {task_json}")
    report = {
        "task_name": task_name,
        "task_json": str(task_json),
        "total_records": len(items),
        "usable_records": len(normalized),
        "skipped_records": int(sum(skipped.values())),
        "skip_reasons": dict(skipped),
        "records_with_street_views": sum(1 for item in normalized if item["street_views"]),
        "records_without_street_views": sum(1 for item in normalized if not item["street_views"]),
    }
    if return_report:
        return normalized, report
    return normalized


def filter_items_for_branch(items: Sequence[Dict], branch: str) -> Tuple[List[Dict], Dict]:
    if branch in {"street", "fusion", "street_feature_control"}:
        filtered = [item for item in items if item["street_views"]]
    else:
        filtered = list(items)
    report = {
        "branch": branch,
        "branch_input_records": len(items),
        "branch_retained_records": len(filtered),
        "branch_dropped_records": len(items) - len(filtered),
    }
    return filtered, report


def make_or_load_split(
    items: Sequence[Dict],
    split_path: Path,
    seed: int = 42,
    val_frac: float = 0.1,
) -> Tuple[List[Dict], List[Dict]]:
    ensure_dir(split_path.parent)
    if split_path.exists():
        split = json.load(open(split_path, "r", encoding="utf-8"))
        id_map = {item["id"]: item for item in items}
        train_items = [id_map[i] for i in split["train_ids"] if i in id_map]
        val_items = [id_map[i] for i in split["val_ids"] if i in id_map]
        return train_items, val_items

    ids = [item["id"] for item in items]
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    save_json(split_path, {"seed": seed, "val_frac": val_frac, "train_ids": train_ids, "val_ids": val_ids})
    id_map = {item["id"]: item for item in items}
    return [id_map[i] for i in train_ids], [id_map[i] for i in val_ids]


def make_rgb_transform(image_size: int, train: bool) -> transforms.Compose:
    ops: List[object] = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip()]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


class GlobalSatelliteDataset(Dataset):
    def __init__(self, items: Sequence[Dict], image_size: int = 224, train: bool = True):
        self.items = list(items)
        self.transform = make_rgb_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = Image.open(item["satellite"]).convert("RGB")
        return {
            "image": self.transform(image),
            "target": torch.tensor(item["reference"], dtype=torch.float32),
            "id": item["id"],
            "city": item["city"],
        }


class GlobalStreetViewDataset(Dataset):
    def __init__(self, items: Sequence[Dict], image_size: int = 224, train: bool = True, max_views: int = 10):
        self.items = [x for x in items if x["street_views"]]
        self.transform = make_rgb_transform(image_size, train)
        self.max_views = max_views
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def _load_views(self, paths: Sequence[Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = []
        mask = []
        for path in list(paths)[: self.max_views]:
            image = Image.open(path).convert("RGB")
            images.append(self.transform(image))
            mask.append(1.0)
        while len(images) < self.max_views:
            images.append(
                torch.zeros_like(images[0]) if images else torch.zeros(3, self.image_size, self.image_size)
            )
            mask.append(0.0)
        return torch.stack(images, dim=0), torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        views, mask = self._load_views(item["street_views"])
        return {
            "street_views": views,
            "street_mask": mask,
            "target": torch.tensor(item["reference"], dtype=torch.float32),
            "id": item["id"],
            "city": item["city"],
        }


class GlobalFusionDataset(Dataset):
    def __init__(self, items: Sequence[Dict], image_size: int = 224, train: bool = True, max_views: int = 10):
        self.items = [x for x in items if x["street_views"]]
        self.sat_transform = make_rgb_transform(image_size, train)
        self.stv_transform = make_rgb_transform(image_size, train)
        self.max_views = max_views
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        sat = self.sat_transform(Image.open(item["satellite"]).convert("RGB"))

        street = []
        mask = []
        for path in list(item["street_views"])[: self.max_views]:
            street.append(self.stv_transform(Image.open(path).convert("RGB")))
            mask.append(1.0)
        while len(street) < self.max_views:
            street.append(torch.zeros(3, self.image_size, self.image_size))
            mask.append(0.0)

        return {
            "image": sat,
            "street_views": torch.stack(street, dim=0),
            "street_mask": torch.tensor(mask, dtype=torch.float32),
            "target": torch.tensor(item["reference"], dtype=torch.float32),
            "id": item["id"],
            "city": item["city"],
        }
