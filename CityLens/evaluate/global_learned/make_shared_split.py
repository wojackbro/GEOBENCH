from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import TASKS, load_global_items
from .utils import ensure_dir, save_json


def make_split(task_name: str, seed: int, val_frac: float, split_key: str, data_root: Path) -> Path:
    # Shared split for fair sat/street/fusion comparison: always built on street-available cohort.
    items, report = load_global_items(data_root, task_name, branch="street", return_report=True)
    if len(items) < 2:
        raise RuntimeError(f"Not enough records for split task={task_name}; report={report}")
    ids = [x.uid for x in items]
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    payload = {
        "task_name": task_name,
        "seed": seed,
        "val_frac": val_frac,
        "split_key": split_key,
        "n_total": len(ids),
        "n_val": len(val_ids),
        "n_train": len(train_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "report": report,
    }
    out = ensure_dir(data_root / "Results" / "global_learned" / "splits") / f"{task_name}_{split_key}_seed{seed}_val{val_frac}.json"
    save_json(out, payload)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", type=str, default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--split_key", type=str, default="fusion")
    p.add_argument("--data_root", type=str, default=None)
    args = p.parse_args()

    data_root = Path(args.data_root or "").expanduser()
    if not str(data_root):
        import os

        data_root = Path(os.environ.get("CITYLENS_DATA_ROOT", "")).expanduser()
    if not str(data_root):
        raise RuntimeError("Missing data root: pass --data_root or set CITYLENS_DATA_ROOT")

    tasks = TASKS if args.task_name == "all" else [args.task_name]
    for t in tasks:
        out = make_split(t, args.seed, args.val_frac, args.split_key, data_root)
        print(f"[ok] split {t}: {out}")


if __name__ == "__main__":
    main()
