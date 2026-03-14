#!/usr/bin/env python3
"""List CityLens data directory structure. Run after extracting CityLens-Data.zip."""
import os
from pathlib import Path

DATA_ROOT = os.environ.get("CITYLENS_DATA_ROOT", "")
if not DATA_ROOT:
    DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "CityLens-Data"
DATA_ROOT = Path(DATA_ROOT)

if not DATA_ROOT.exists():
    print(f"Directory not found: {DATA_ROOT}")
    print("Run scripts/download_citylens.py first, or set CITYLENS_DATA_ROOT.")
    exit(1)

print(f"Data root: {DATA_ROOT}\n")
for d in sorted(DATA_ROOT.iterdir()):
    if d.is_dir():
        n = len(list(d.iterdir()))
        print(f"  {d.name}/  ({n} items)")
        if d.name in ("Benchmark", "Dataset") and n <= 50:
            for f in sorted(d.iterdir())[:20]:
                print(f"    - {f.name}")
            if n > 20:
                print(f"    ... and {n - 20} more")
    else:
        print(f"  {d.name}")
