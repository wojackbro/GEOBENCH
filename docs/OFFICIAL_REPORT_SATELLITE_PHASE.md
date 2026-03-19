# Official Technical Report: Satellite-Only Baseline Phase

## Project

**Title:** Geospatial Foundation Model Adaptation for CityLens Global-Task Regression  
**Repo:** `wojackbro/GEOBENCH`  
**Phase:** Satellite-only model comparison (completed)  
**Date:** 2026-03-19

## Scope of this report

This report documents the completed satellite-only comparison stage for CityLens global tasks:

- `gdp`
- `acc2health`
- `build_height`
- `pop`

Note: the full learned pipeline defines five global tasks (`gdp`, `pop`, `acc2health`, `carbon`, `build_height`). This completed phase reports the four tasks with finalized comparable outputs; `carbon` remains queued for the extension phase.

Models compared:

- `prithvi_rgb_lora` (adapted geospatial foundation model)
- `dinov2_sat` (generic vision transformer baseline)
- `resnet50_sat` (generic CNN baseline)

Branch used: `satellite` only (no street-view input in this phase).

## Experimental protocol

### Shared setup

- branch: `satellite`
- image size: `224`
- batch size: `8`
- seed: `42`
- target transform: `log1p`
- lr: `2e-4`
- backbone lr: `2e-4`
- head lr: `1e-3`
- weight decay: `1e-2`
- val fraction: `0.1`
- checkpointing: `best.pt` and `last.pt`
- resume flags: `--resume --skip_if_done`

### Epoch budgets

- `gdp`: 20
- `acc2health`: 30
- `build_height`: 30
- `pop`: 5

### Reproducibility conditions

- Task splits are reused by branch/seed/validation fraction (`Results/global_learned/splits/...`).
- Runs are logged under `CITYLENS_DATA_ROOT/Results/global_learned/<task>/<experiment_name>/`.
- Experiment folder names include `ep{epochs}`. Changing `--epochs` creates a different folder, so resume is folder-local unless checkpoints are copied intentionally.

## Main results

| Task | Model | Best Epoch | R2 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| acc2health | dinov2_sat | 20 | 0.0985 | 11.6113 | 8.7746 |
| acc2health | prithvi_rgb_lora | 9 | 0.3901 | 9.5502 | 7.1910 |
| acc2health | resnet50_sat | 22 | 0.2124 | 10.8530 | 7.2042 |
| build_height | dinov2_sat | 18 | 0.6791 | 3.9542 | 2.8124 |
| build_height | prithvi_rgb_lora | 11 | 0.8599 | 2.6130 | 1.9100 |
| build_height | resnet50_sat | 26 | 0.8004 | 3.1182 | 2.2806 |
| gdp | dinov2_sat | 19 | 0.4535 | 3.7845e8 | 2.3281e8 |
| gdp | prithvi_rgb_lora | 14 | 0.5808 | 3.3146e8 | 1.9811e8 |
| gdp | resnet50_sat | 5 | -4.0046 | 1.1452e9 | 3.9336e8 |
| pop | dinov2_sat | 5 | -0.1840 | 23175.54 | 11363.06 |
| pop | prithvi_rgb_lora | 2 | -0.0324 | 21641.25 | 10020.26 |
| pop | resnet50_sat | 2 | -0.2661 | 23965.72 | 11871.83 |

## Primary findings

1. `prithvi_rgb_lora` outperforms both generic satellite baselines on all reported tasks.
2. Gains are strongest on:
   - `build_height` (0.8599 vs 0.8004 vs 0.6791)
   - `gdp` (0.5808 vs 0.4535 vs -4.0046)
3. All models underperform on `pop` (negative R2), but Prithvi is the least poor.

## Important caveat: GDP + ResNet instability

`gdp + resnet50_sat` produced highly negative R2 (`-4.0046`), indicating a pathological failure mode under the current training setup.

Possible causes:

- optimization instability with this backbone under shared LR schedule
- weak target alignment for heavy-tailed GDP with current head/backbone parameterization
- poor feature-target geometry from generic ImageNet features in this remote-sensing-like regression setting
- target-transform sensitivity (`log1p` decode in raw-space evaluation) amplifying prediction collapse

Reporting guidance:

- keep the value in the table for transparency
- explicitly mark this as a failed baseline behavior
- do not over-claim from this single unstable point

## Architecture-based interpretation (working hypothesis)

### Why Prithvi likely wins

- Prithvi carries EO-oriented inductive biases from geospatial pretraining.
- Even with RGB-to-6 adaptation, it appears to capture built-form and spatial context more effectively than generic backbones.
- This aligns with strongest gains on morphology-heavy tasks (`build_height`) and macro-structure-sensitive tasks (`gdp`).

### Why DINOv2 is mid-tier

- ViT representations are strong and general, but not specialized for remote-sensing socioeconomic regression.
- DINOv2 performs reasonably on `gdp` and `build_height`, but trails Prithvi consistently.

### Why ResNet can fail

- CNN local texture bias may be less suitable for long-range global context in this benchmark.
- Under a shared optimizer schedule, it appears less stable in GDP regression.

## Why `clip_vitb16` is the current street default

Planned street branch starts with `clip_vitb16` because:

- strong generic image-text visual features
- practical memory/throughput profile in Colab
- established baseline in existing project pipeline

This does **not** imply it is universally best for street tasks. Planned comparison (`resnet50`, `clip_vitb16`, `dinov2_vitb14`) is still required.

## Explainability status and value

The pipeline already supports:

- satellite integrated gradients
- street leave-one-view-out
- fusion modality ablation

Planned use:

- validate whether the model attends to plausible built-form and infrastructure regions
- test whether failure cases (`pop`, unstable GDP baselines) correspond to diffuse or collapsed saliency
- support, not replace, quantitative claims

## Limitations of current phase

- no street-only results yet in this report
- no fusion (`late`, `gated`) results yet in this report
- no completed per-city cross-model analysis yet
- no multi-seed variance analysis yet

## Actionable next phase (already planned)

1. Street-only baseline with `clip_vitb16` (then other street encoders).
2. Fusion baseline with `late`, then `gated`.
3. Per-city comparisons and failure-mode analysis.
4. Explainability section for qualitative validation.

## Current claim boundary

Supported now:

> In satellite-only global-task regression on CityLens, adapted Prithvi (`prithvi_rgb_lora`) consistently outperforms generic satellite backbones (`dinov2_sat`, `resnet50_sat`) under matched settings.

Not yet supported:

> Full multimodal superiority claims over street-only or fusion alternatives.
