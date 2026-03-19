# Global Learned Pipeline

This document describes the learned-model experimental stack for the 5 global CityLens tasks:

- `gdp`
- `pop`
- `acc2health`
- `carbon`
- `build_height`

The implementation lives under `CityLens/evaluate/global_learned/` and is designed to be used from `colab_citylens_full.ipynb`.

## Architectural stance

CityLens global tasks provide:
- `1` RGB satellite PNG
- `10` RGB street-view images
- a scalar target in `Benchmark/*.json`

Official `Prithvi-EO-2.0-300M` was pretrained on `6` HLS bands (`BLUE`, `GREEN`, `RED`, `NIR_NARROW`, `SWIR_1`, `SWIR_2`), often multi-temporal. Therefore this repo treats Prithvi on CityLens as an **adaptation experiment**, not a native Prithvi setup.

## Implemented branches

### Low-cost control
- frozen street-view embeddings + `LassoCV`
- implemented in `CityLens/evaluate/global_learned/feature_control.py`

### Satellite-only
- `prithvi_rgb_lora`
- `prithvi_rgb_lora_tl`
- `dinov2_sat`
- `resnet50_sat`

Current mitigation for unstable raw-target regression:
- the learned training path now supports target transforms and defaults to `log1p` under `--target_transform auto`
- validation metrics are still computed in raw target space
- history logs include prediction spread (`pred_mean`, `pred_std`) so collapsed predictions are easier to detect

### Street-view-only
- `resnet50`
- `clip_vitb16`
- `dinov2_vitb14`
- optional `swin_t`

### Fusion
- `late`
- `gated`

## Checkpointing and resume

All learned runs write under:

`CITYLENS_DATA_ROOT/Results/global_learned/`

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
- `Results/global_learned/splits/<task>_seed<seed>_val<frac>.json`

Use both flags when running in Colab:

- `--skip_if_done`
- `--resume`

Useful tuning flags:

- `--target_transform auto|raw|log1p`
- `--backbone_lr <float>`
- `--head_lr <float>`

## Core commands

From `CityLens/` repo root:

```bash
# Low-cost street-view control
python -m evaluate.global_learned.feature_control \
  --task_name all \
  --street_model resnet50 \
  --image_size 224 \
  --batch_size 8 \
  --seed 42 \
  --skip_if_done

# Satellite-only across all 5 global tasks
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

# Street-view-only baseline
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

# Fusion baseline
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

## Explainability hooks

Implemented in `CityLens/evaluate/global_learned/explain.py`:

- satellite: integrated gradients over multiple validation samples, with saved `npy` arrays and heatmap previews
- street: leave-one-view-out importance over multiple validation samples
- fusion: leave-one-modality-out ablation over multiple validation samples

Example:

```bash
python -m evaluate.global_learned.explain \
  --task_name gdp \
  --branch satellite \
  --satellite_model prithvi_rgb_lora \
  --checkpoint_dir /path/to/Results/global_learned/gdp/<experiment_name> \
  --max_samples 8
```

Explainability outputs are written under `<experiment_dir>/explain/` and now include:

- `explainability_manifest.json`
- satellite: per-sample `integrated_gradients.npy`, `input.png`, `heatmap.png`, plus `integrated_gradients_summary.json`
- street: `leave_one_view_out.json` and `leave_one_view_out.csv`
- fusion: `modality_ablation.json` and `modality_ablation.csv`

## Honest ablation scope

Scientifically defensible ablations in this repo:

- low-cost control vs learned street-view model comparison
- LoRA rank changes via `--lora_r`
- satellite backbone comparison (`prithvi_rgb_lora` vs `dinov2_sat` vs `resnet50_sat`)
- street backbone comparison (`resnet50` vs `clip_vitb16` vs `dinov2_vitb14`)
- pooling comparison (`mean` vs `attention`)
- fusion comparison (`late` vs `gated`)

What this repo does **not** claim:

- true multispectral spectral ablations on native HLS bands
- native Prithvi evaluation on the same data distribution as official pretraining
- UK transfer learning or Prithvi+LVLM feature fusion, which are still outside the implemented pipeline

CityLens does not provide the six native HLS bands required for a real spectral study, so any channel-level study should be framed as an **RGB-to-6 adapter approximation**, not a genuine spectral ablation.

## Recommended experiment order

1. low-cost control: `feature_control.py` with `resnet50`
2. `prithvi_rgb_lora` satellite-only
3. `dinov2_sat` satellite fallback
4. `resnet50` street-only
5. `clip_vitb16` street-only
6. `dinov2_vitb14` street-only
7. `late` fusion
8. `gated` fusion

Keep `swin_t` optional unless Colab runtime is stable.
