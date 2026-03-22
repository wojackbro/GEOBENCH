# GeoBench: CityLens Geospatial FM Workflow

This repository tracks an end-to-end workflow for CityLens regression experiments, centered on:

- `prithvi_rgb_lora` (satellite geospatial FM adaptation)
- controlled satellite baselines (`dinov2_sat`, `resnet50_sat`)
- street-only and fusion phases (`clip_vitb16`, `late`, `gated`) — see official report

## Current status

**Single source of truth:** `docs/OFFICIAL_REPORT_SATELLITE_PHASE.md` (satellite + street + fusion tables, methods, limitations, ethics, future work).

- Satellite, street, and fusion **seed-42 snapshots** for `gdp`, `acc2health`, `build_height`, `pop` are documented there.
- `prithvi_rgb_lora` is the strongest **satellite** backbone on those tasks; `gdp + resnet50_sat` remains a documented failure (negative R²).
- Fusion improves R² only on **`build_height`** in this snapshot; satellite Prithvi leads **`gdp`** and **`acc2health`**.
- Per-city, multi-seed, and shared-ID validation are still open.

See also:

- `docs/PRITHVI_SATELLITE_REFERENCE.md`
- `docs/GLOBAL_LEARNED_PIPELINE.md`

*(Journal/conference draft files under `docs/` are stubs that point to the official report—do not maintain duplicate tables there.)*

## Recommended execution path

### 1) Use Colab notebooks

- `colab_citylens_full.ipynb`: full setup + pipeline notebook
- `colab_citylens_baseline_compare.ipynb`: compact baseline comparison notebook

Both are designed to run with checkpoint-safe behavior (`--resume --skip_if_done`) for restart resilience.

### 2) Data root

Set `CITYLENS_DATA_ROOT` to the folder containing:

- `Benchmark/`
- `Dataset/`
- `satellite_image/`
- `street_view_image/`

### 3) Core global learned commands

Run from `CityLens/` root:

```bash
# Satellite-only
python -m evaluate.global_learned.train \
  --task_name all \
  --branch satellite \
  --satellite_model prithvi_rgb_lora \
  --epochs 5 \
  --batch_size 8 \
  --image_size 224 \
  --seed 42 \
  --skip_if_done \
  --resume

# Street-only
python -m evaluate.global_learned.train \
  --task_name all \
  --branch street \
  --street_model clip_vitb16 \
  --pooling mean \
  --epochs 5 \
  --batch_size 8 \
  --image_size 224 \
  --seed 42 \
  --skip_if_done \
  --resume

# Fusion
python -m evaluate.global_learned.train \
  --task_name all \
  --branch fusion \
  --satellite_model prithvi_rgb_lora \
  --street_model clip_vitb16 \
  --fusion_type late \
  --pooling mean \
  --epochs 5 \
  --batch_size 8 \
  --image_size 224 \
  --seed 42 \
  --skip_if_done \
  --resume
```

## Checkpoint and resume rules (critical)

- Artifacts are saved under `Results/global_learned/<task>/<experiment_name>/`.
- Checkpoints: `checkpoints/best.pt` and `checkpoints/last.pt`.
- Resume only works within the same experiment folder.
- Experiment names include `ep{epochs}`:
  - changing `--epochs` creates a new folder
  - new folder means no automatic resume from old folder unless checkpoint is copied intentionally

## Outputs you should expect

Per experiment:

- `config.json`
- `dataset_report.json`
- `history.csv`
- `metrics.json`
- `val_predictions.csv`
- `per_city_metrics.json`
- `per_city_metrics.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`

Global logs:

- `Results/global_learned/experiment_log.csv`
- `Results/global_learned/splits/*.json`

## Next milestones

- complete street-only matrix (`clip_vitb16`, `resnet50`, `dinov2_vitb14`)
- complete fusion matrix (`late`, `gated`)
- rerun satellite anchors on the same street-enabled subset for strict fairness
- run per-city error analysis
- run XAI package (`explain.py`) and integrate qualitative findings into paper drafts
- add multi-seed confidence intervals for publication-grade claims
