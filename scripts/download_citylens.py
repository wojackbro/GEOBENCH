#!/usr/bin/env python3
"""
Download CityLens dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/Tianhui-Liu/CityLens-Data
Correct repo: Tianhui-Liu/CityLens-Data (not abidhossain123/CityLens-Data).
"""
import os
import zipfile
from pathlib import Path

REPO_ID = "Tianhui-Liu/CityLens-Data"
FILENAME = "CityLens-Data.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRACT_DIR = DATA_DIR / "CityLens-Data"
HF_RESOLVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FILENAME}"


def _progress_hook(block_num, block_size, total_size):
    if total_size <= 0:
        print(".", end="", flush=True)
        return
    downloaded = block_num * block_size
    pct = min(100, 100 * downloaded / total_size)
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(f"\r  {pct:.1f}% ({mb:.1f} / {total_mb:.1f} MB)", end="", flush=True)


def download_via_url():
    """Direct download from Hugging Face resolve URL (with progress)."""
    import urllib.request
    zip_path = DATA_DIR / FILENAME
    print(f"Downloading from Hugging Face (this may take several minutes) ...")
    urllib.request.urlretrieve(HF_RESOLVE_URL, zip_path, reporthook=_progress_hook)
    print()
    return zip_path


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / FILENAME

    if not zip_path.is_file():
        # Try direct URL first (shows progress; often more reliable for large files)
        try:
            zip_path = download_via_url()
        except Exception as e:
            print(f"Direct URL failed: {e}. Trying huggingface_hub ...")
            try:
                from huggingface_hub import hf_hub_download
                print(f"Downloading {REPO_ID} (no progress bar; may take 10+ min) ...")
                path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=FILENAME,
                    repo_type="dataset",
                    local_dir=DATA_DIR,
                )
                zip_path = Path(path) if os.path.isfile(path) else (DATA_DIR / FILENAME)
            except Exception as e2:
                print(f"huggingface_hub failed: {e2}")
                raise SystemExit(1)
    else:
        print(f"Using existing {zip_path}")

    if zip_path.is_file():
        print(f"Extracting to {EXTRACT_DIR}")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(EXTRACT_DIR)
        for p in sorted(EXTRACT_DIR.iterdir()):
            print(f"  {p.name}")
        print("Done. Set CITYLENS_DATA_ROOT to:", os.path.abspath(EXTRACT_DIR))
    else:
        print("No zip file found at", zip_path)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
