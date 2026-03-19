# Conference Paper Draft (Prestige Format)

## Title

**Adapting a Geospatial Foundation Model to CityLens: A Satellite-Only Benchmark with Strong Baseline Controls**

## Abstract

Urban socioeconomic sensing benchmarks are increasingly used to evaluate vision-language and vision-only models, yet robust satellite-only controlled baselines remain limited. We present a satellite-only study on CityLens global-task regression using an adapted geospatial foundation model (`prithvi_rgb_lora`) and two generic controls (`dinov2_sat`, `resnet50_sat`) under matched protocols. Across `gdp`, `acc2health`, `build_height`, and `pop`, Prithvi achieves the best R2 in all tasks. We also identify a critical failed-baseline behavior (`gdp + resnet50_sat`, R2 = -4.00), which we report explicitly for transparency. Our findings suggest that EO-oriented pretraining provides practical advantages even when adapted to RGB-only CityLens inputs. We release a reproducible pipeline with checkpointed training, result logging, and explainability hooks, and outline pending multimodal extensions (street-only and fusion) as future work.

## 1. Introduction

City-scale prediction of socioeconomic indicators from imagery is an important but difficult regression problem due to domain shift, label noise, and non-stationarity across cities. Recent CityLens evaluations focus heavily on LVLM paradigms, but the role of specialized geospatial pretraining in controlled learned baselines is underexplored.

This work asks: **does a geospatial foundation model adapted to RGB satellite imagery outperform generic computer vision backbones under matched training settings?**

### Contributions

- A controlled satellite-only benchmark comparison on CityLens global tasks.
- Evidence that `prithvi_rgb_lora` outperforms `dinov2_sat` and `resnet50_sat` across all tested tasks.
- Explicit failure analysis of unstable baseline behavior (`gdp + resnet50_sat`).
- A reproducible training/reporting pipeline with resume-safe checkpoints and explainability hooks.

## 2. Related Work

### 2.1 Urban socioeconomic sensing with CityLens

CityLens introduces broad cross-domain city indicators and multimodal (satellite + street-view) inputs for benchmarked evaluation.

### 2.2 Foundation models in geospatial vision

Geospatial foundation models are trained with remote-sensing priors that may better encode built environment structure, land patterns, and scale-sensitive context.

### 2.3 Generic visual backbones for regression transfer

ImageNet/self-supervised backbones (ResNet, DINOv2) are strong general encoders but are not explicitly designed for socioeconomic inference from aerial context.

## 3. Methodology

### 3.1 Tasks

- `gdp`
- `acc2health`
- `build_height`
- `pop`

The broader pipeline includes `carbon`; this draft reports the four tasks with finalized satellite-only comparison outputs and leaves `carbon` for extension reporting.

### 3.2 Models

- `prithvi_rgb_lora` (satellite-only adapted geospatial foundation model)
- `dinov2_sat` (ViT baseline)
- `resnet50_sat` (CNN baseline)

### 3.3 Training protocol

- image size: 224
- batch size: 8
- seed: 42
- target transform: `log1p`
- lr/backbone lr: 2e-4
- head lr: 1e-3
- weight decay: 1e-2
- validation fraction: 0.1
- resume and skip-done enabled
- run naming includes `ep{epochs}`, so resume is specific to that experiment directory

Task budgets: GDP (20), Acc2Health (30), Build Height (30), Pop (5).

### 3.4 Metrics

Primary: R2.  
Secondary: RMSE, MAE, MSE.

## 4. Results

### 4.1 Main comparison (R2)

| Task | dinov2_sat | prithvi_rgb_lora | resnet50_sat |
| --- | ---: | ---: | ---: |
| acc2health | 0.0985 | **0.3901** | 0.2124 |
| build_height | 0.6791 | **0.8599** | 0.8004 |
| gdp | 0.4535 | **0.5808** | -4.0046 |
| pop | -0.1840 | **-0.0324** | -0.2661 |

### 4.2 Key observations

- Prithvi is best on all four tasks.
- The strongest gain is on `build_height`.
- `pop` remains difficult for all models (negative R2), indicating unresolved signal limitations.
- `gdp + resnet50_sat` is a failed baseline behavior and should be interpreted as instability/mismatch, not a competitive result.

## 5. Analysis and Discussion

### 5.1 Why Prithvi may outperform generic backbones

- EO-oriented pretraining likely improves geospatial structure encoding.
- Better alignment with morphology-sensitive targets (`build_height`) and macro patterns (`gdp`).
- LoRA adaptation offers efficient transfer without full-model overfitting.

### 5.2 Why ResNet can fail on GDP in this setup

Potential factors:

- local texture bias vs long-range urban context requirements
- optimizer/learning-rate mismatch under shared schedule
- sensitivity to heavy-tailed socioeconomic targets
- representation collapse under current regression head settings

### 5.3 On CLIP ViT-B/16 for planned street branch

`clip_vitb16` is selected first for practicality and strong transferable features.  
It is a starting baseline, not an assumed optimum. Planned comparisons with `resnet50` and `dinov2_vitb14` remain necessary.

## 6. Limitations

- Street-only experiments pending.
- Fusion (`late`, `gated`) pending.
- Per-city analysis pending.
- Multi-seed variance and statistical significance pending.

## 7. Future Work

- Complete street-only baselines under matched budgets.
- Complete late and gated fusion runs.
- Add per-city and stratified error analysis.
- Add explainability-driven failure analysis and sanity checks.
- Perform multi-seed robustness with confidence intervals.

## 8. Conclusion

Under matched satellite-only settings on CityLens global tasks, adapted Prithvi consistently outperforms generic backbone controls. The current evidence supports a strong satellite-phase claim while motivating a full multimodal extension for complete benchmark positioning.

## Ethics, Risks, and Responsible Use

Socioeconomic prediction from imagery can amplify biases, overlook local context, and be misused for high-stakes decisions. Models should be used for aggregate analysis, not individual-level inference, with explicit fairness and uncertainty reporting.

## Reproducibility Checklist

- [x] Defined tasks and metrics
- [x] Fixed protocol hyperparameters
- [x] Checkpointed training (`best.pt`, `last.pt`)
- [x] Structured logs (`config.json`, `history.csv`, `metrics.json`)
- [ ] Multi-seed confidence intervals
- [ ] External validation beyond current split protocol

## Appendix (Placeholders)

- A. Full command table: [TODO]
- B. Per-epoch learning curves: [TODO]
- C. Street-only and fusion extension tables: [TODO]
- D. Explainability figures and case studies: [TODO]
