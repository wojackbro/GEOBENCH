#!/usr/bin/env python3
"""
Patch CityLens evaluate/global/global_indicator.py for Colab:
1. Insert path fallback from CITYLENS_DATA_ROOT before output_dir = ...
2. Wrap os.makedirs(output_dir, exist_ok=True) in "if output_dir and not os.path.exists(output_dir):"

Run from repo root (e.g. /content/CityLens): python scripts/patch_global_indicator_colab.py
Or from project root: python scripts/patch_global_indicator_colab.py /path/to/CityLens
"""
import os
import re
import sys

def main():
    if len(sys.argv) > 1:
        repo_root = sys.argv[1]
    else:
        repo_root = os.getcwd()
    path = os.path.join(repo_root, "evaluate", "global", "global_indicator.py")
    if not os.path.isfile(path):
        print("Not found:", path)
        sys.exit(1)
    with open(path) as f:
        t = f.read()

    # 1. Path fallback: before "output_dir = os.path.dirname(response_path)"
    if "if not task_path or not response_path" not in t and "output_dir = os.path.dirname(response_path)" in t:
        needle = "output_dir = os.path.dirname(response_path)"
        insert = (
            "\n    if not task_path or not response_path:\n"
            "        data_root = os.environ.get(\"CITYLENS_DATA_ROOT\", \"/content/CityLens-Data\")\n"
            "        task_path = os.path.join(data_root, \"Benchmark\", f\"{task_name}_{city}.json\")\n"
            "        response_path = os.path.join(data_root, \"Results\", f\"{task_name}_{city}_{model_name_full}_{prompt_type}.json\")\n"
            "    "
        )
        before, _, after = t.partition(needle)
        t = before + insert + needle + after
        print("Inserted path fallback.")

    # 2. Guard makedirs: any indentation, flexible spacing in (output_dir, exist_ok=True)
    pattern = r"(\n)(\s+)os\.makedirs\s*\(\s*output_dir\s*,\s*exist_ok\s*=\s*True\s*\)"
    replacement = r"\1\2if output_dir and not os.path.exists(output_dir):\n\2    os.makedirs(output_dir, exist_ok=True)"
    t_new = re.sub(pattern, replacement, t)
    if t_new != t:
        t = t_new
        print("Applied makedirs guard.")
    else:
        if "os.makedirs(output_dir" in t and "if output_dir and not os.path.exists(output_dir)" not in t:
            print("WARNING: makedirs guard pattern did not match. Check file content.")

    with open(path, "w") as f:
        f.write(t)
    print("Wrote", path)

if __name__ == "__main__":
    main()
