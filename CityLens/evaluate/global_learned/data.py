from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TASKS = ["gdp", "acc2health", "build_height", "pop", "carbon"]


@dataclass
class Item:
    uid: str
    city: str
    target: float
    image: Path
    street_views: list[Path]


def _candidate_task_paths(data_root: Path, task: str) -> list[Path]:
    return [
        data_root / "Benchmark" / f"{task}_all.json",
        data_root / "Benchmark" / f"{task}.json",
        data_root / "Dataset" / f"{task}_all.json",
        data_root / "Dataset" / f"{task}.json",
    ]


def _to_path(data_root: Path, value: str | None) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return data_root / p


def _target_from_record(rec: dict[str, Any], task: str) -> Optional[float]:
    for k in [task, "label", "target", "value", "y"]:
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except Exception:
                pass
    return None


def _satellite_from_record(data_root: Path, rec: dict[str, Any]) -> Optional[Path]:
    for k in ["satellite_image", "satellite_path", "image", "img"]:
        if k in rec:
            p = _to_path(data_root, rec.get(k))
            if p is not None:
                return p
    imgs = rec.get("images")
    if isinstance(imgs, list) and imgs:
        p = _to_path(data_root, imgs[0])
        if p is not None:
            return p
    return None


def _street_from_record(data_root: Path, rec: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    for k in ["street_view_image", "street_views", "street_paths"]:
        v = rec.get(k)
        if isinstance(v, list):
            for x in v:
                p = _to_path(data_root, x)
                if p is not None:
                    out.append(p)
    imgs = rec.get("images")
    if isinstance(imgs, list) and len(imgs) > 1:
        for x in imgs[1:]:
            p = _to_path(data_root, x)
            if p is not None:
                out.append(p)
    # dedupe in order
    seen = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def load_global_items(
    data_root: Path,
    task_name: str,
    branch: str = "satellite",
    return_report: bool = False,
) -> list[Item] | tuple[list[Item], dict[str, Any]]:
    task_path = None
    for p in _candidate_task_paths(data_root, task_name):
        if p.exists():
            task_path = p
            break
    if task_path is None:
        raise FileNotFoundError(f"Could not find task json for {task_name} in Benchmark/ or Dataset/")

    payload = __import__("json").loads(task_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for k in ["data", "items", "records", "samples"]:
            if k in payload and isinstance(payload[k], list):
                records = payload[k]
                break
        else:
            raise ValueError(f"Task json has unsupported format: {task_path}")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Task json has unsupported format: {task_path}")

    items: list[Item] = []
    usable = 0
    with_street = 0
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        y = _target_from_record(rec, task_name)
        sat = _satellite_from_record(data_root, rec)
        stv = _street_from_record(data_root, rec)
        if sat is None or y is None:
            continue
        if not sat.exists():
            continue
        usable += 1
        stv = [p for p in stv if p.exists()]
        if stv:
            with_street += 1
        uid = str(rec.get("id") or rec.get("uid") or rec.get("region_id") or rec.get("sample_id") or i)
        city = str(rec.get("city") or rec.get("city_name") or "unknown")
        item = Item(uid=uid, city=city, target=float(y), image=sat, street_views=stv)
        if branch == "satellite":
            items.append(item)
        else:
            if item.street_views:
                items.append(item)

    report = {
        "task_json": str(task_path),
        "usable_records": usable,
        "records_with_street_views": with_street,
        "selected_for_branch": len(items),
        "branch": branch,
    }
    if return_report:
        return items, report
    return items


class GlobalLearnedDataset(Dataset):
    def __init__(self, items: list[Item], image_size: int = 224, max_views: int = 10):
        self.items = items
        self.image_size = image_size
        self.max_views = max_views
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def _read(self, path: Path) -> torch.Tensor:
        im = Image.open(path).convert("RGB")
        return self.tf(im)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        image = self._read(it.image)
        views = it.street_views[: self.max_views]
        if views:
            st = [self._read(p) for p in views]
            street_views = torch.stack(st, dim=0)
            street_mask = torch.ones(street_views.size(0), dtype=torch.float32)
        else:
            street_views = torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.float32)
            street_mask = torch.zeros(1, dtype=torch.float32)
        return {
            "id": it.uid,
            "city": it.city,
            "image": image,
            "street_views": street_views,
            "street_mask": street_mask,
            "target": torch.tensor(it.target, dtype=torch.float32),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    b = len(batch)
    image = torch.stack([x["image"] for x in batch], dim=0)
    target = torch.stack([x["target"] for x in batch], dim=0)
    ids = [x["id"] for x in batch]
    cities = [x["city"] for x in batch]
    max_v = max(x["street_views"].size(0) for x in batch)
    c, h, w = batch[0]["street_views"].size(1), batch[0]["street_views"].size(2), batch[0]["street_views"].size(3)
    street = torch.zeros((b, max_v, c, h, w), dtype=torch.float32)
    mask = torch.zeros((b, max_v), dtype=torch.float32)
    for i, x in enumerate(batch):
        v = x["street_views"].size(0)
        street[i, :v] = x["street_views"]
        mask[i, :v] = x["street_mask"]
    return {"id": ids, "city": cities, "image": image, "street_views": street, "street_mask": mask, "target": target}
