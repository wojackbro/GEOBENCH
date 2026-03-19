# Official Technical Report: CityLens Satellite-Only Phase

## 1) Executive Summary

This report documents the completed satellite-only comparison stage of our CityLens learned-model pipeline.  
Under matched training settings, `prithvi_rgb_lora` consistently outperforms `dinov2_sat` and `resnet50_sat` on all finalized tasks in this phase (`gdp`, `acc2health`, `build_height`, `pop`).

Key caveat: `gdp + resnet50_sat` is a failed baseline behavior (R2 = `-4.0046`) and is reported transparently as instability/mismatch, not as competitive performance.

## 2) Benchmark Context and Positioning

CityLens is a multimodal urban socioeconomic benchmark introduced as a large-scale evaluation framework for visual models across tasks and cities.  
Our current work is a controlled **satellite-only** stage designed to answer a narrower question:

> Does a geospatially adapted satellite model (`prithvi_rgb_lora`) outperform generic satellite backbones under matched protocol in CityLens global-task regression?

This report does **not** claim final multimodal leadership yet. Street-only, fusion, per-city, and full XAI synthesis are scheduled as next-stage deliverables.

## 3) Scope of This Phase

### Completed tasks

- `gdp`
- `acc2health`
- `build_height`
- `pop`

### Models compared

- `prithvi_rgb_lora` (geospatial FM adaptation with LoRA)
- `dinov2_sat` (generic ViT baseline)
- `resnet50_sat` (generic CNN baseline)

### Out-of-scope for this report

- street-only results
- fusion (`late`, `gated`) results
- per-city ranking and fairness audit integration
- multi-seed confidence intervals

Note: the full learned pipeline covers `gdp`, `pop`, `acc2health`, `carbon`, `build_height`; `carbon` is reserved for the extension round.

## 4) Experimental Protocol (Matched Setup)

- branch: `satellite`
- image size: `224`
- batch size: `8`
- seed: `42`
- target transform: `log1p`
- lr/backbone lr: `2e-4`
- head lr: `1e-3`
- weight decay: `1e-2`
- validation fraction: `0.1`
- checkpointing: `best.pt`, `last.pt`
- restart safety: `--resume --skip_if_done`

### Epoch budgets

- `gdp`: 20
- `acc2health`: 30
- `build_height`: 30
- `pop`: 5

### Reproducibility note

Experiment folders encode `ep{epochs}`.  
Changing `--epochs` creates a new directory, so resume is folder-local unless checkpoint migration is performed intentionally.

## 5) Results

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

## 6) Findings

1. `prithvi_rgb_lora` is best on all four finalized tasks.
2. Largest practical gap appears on `build_height` and `gdp`.
3. `pop` remains difficult for all backbones (negative R2), indicating unresolved signal/label complexity under this phase design.

## 7) Mandatory Caveat: Failed Baseline Behavior

`gdp + resnet50_sat` gives R2 = `-4.0046`.

Interpretation:

- this is a failed run behavior under current setup
- it may indicate instability, optimization mismatch, or poor representation-task fit
- it should be reported as a negative control outcome, not used to inflate Prithvi gains

## 8) Architecture-Level Interpretation (Hypotheses)

### Why Prithvi is strong in this phase

- EO-oriented priors likely improve morphology/context encoding from satellite views.
- LoRA adaptation preserves useful pretrained structure while allowing task fitting.
- Gains align with structurally visible tasks (`build_height`) and macro-pattern tasks (`gdp`).

### Why DINOv2 is competitive but below Prithvi

- strong generic representation quality
- weaker domain alignment for socioeconomic satellite regression compared with EO-adapted encoder

### Why ResNet may break on GDP

- local texture bias may be insufficient for broad geospatial semantics
- shared LR schedule may be suboptimal for this backbone/task pair
- GDP heavy-tail sensitivity can amplify regression instability

## 9) Explainability and Verification Plan

Implemented hooks in pipeline:

- satellite integrated gradients
- street leave-one-view-out
- fusion modality ablation

Planned use in next phase:

- validate whether strong/weak predictions align with plausible regions
- compare saliency concentration between successful and failed baselines
- avoid causal over-claims; treat XAI as consistency evidence

## 10) Publication-Ready Claim Boundary

### Supported claim now

In CityLens satellite-only global-task regression under matched settings, adapted Prithvi consistently outperforms generic satellite baselines.

### Not yet supported

- multimodal superiority claims
- final per-city robustness/fairness conclusions
- statistically strong cross-seed confidence claims

## 11) Next Deliverables

1. Street-only matrix (`clip_vitb16`, `resnet50`, `dinov2_vitb14`)
2. Fusion matrix (`late`, `gated`)
3. Per-city and error taxonomy analysis
4. Multi-seed uncertainty and significance pass
5. Final integrated manuscript version (conference and journal)
