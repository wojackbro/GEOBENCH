# Conference Paper Draft (ICLR/NeurIPS-Style)

## Title

**Geospatial Foundation Model Adaptation on CityLens: A Controlled Satellite-Only Study with Transparent Baseline Failure Analysis**

## Abstract

Urban socioeconomic sensing from visual data is promising but still unstable across tasks and model families. We present a controlled satellite-only study on CityLens global-task regression to isolate the value of geospatial pretraining. We compare an adapted geospatial model (`prithvi_rgb_lora`) against generic satellite backbones (`dinov2_sat`, `resnet50_sat`) under matched optimization and split settings. Across `gdp`, `acc2health`, `build_height`, and `pop`, `prithvi_rgb_lora` achieves the highest R2 on all tasks. We explicitly report a failed baseline behavior (`gdp + resnet50_sat`, R2 = `-4.0046`) and treat it as instability/mismatch rather than competitive evidence. Results indicate that EO-aligned priors remain beneficial even under RGB adaptation on CityLens. We release a reproducible pipeline with restart-safe checkpoints, structured experiment artifacts, and explainability hooks; multimodal extensions (street and fusion) are defined as the next stage.

## 1. Introduction

Urban indicators such as GDP, health accessibility, and built-form characteristics are central to policy and planning, but high-quality local socioeconomic measurements are expensive and unevenly available. Benchmarks such as CityLens establish a broad multimodal evaluation setting, yet controlled satellite-only model-comparison evidence remains underdeveloped in many practical pipelines.

This work asks:

> Under matched training settings, does an adapted geospatial foundation model outperform generic satellite backbones for CityLens-style global-task regression?

We intentionally constrain the scope to satellite-only inputs to prevent modality confounding and to establish a strong unimodal baseline before running street/fusion stages.

### Contributions

- Controlled satellite-only benchmark in a unified pipeline.
- Consistent superiority of `prithvi_rgb_lora` over `dinov2_sat` and `resnet50_sat` on finalized tasks.
- Mandatory reporting of failed baseline behavior (`gdp + resnet50_sat`).
- Reproducibility-oriented engineering: split reuse, checkpoint resume, structured logs, explainability hooks.

## 2. Related Work

### 2.1 Urban socioeconomic sensing benchmarks

CityLens frames multimodal urban sensing across indicators and regions and highlights that socioeconomic prediction remains difficult even for modern LVLM pipelines.

### 2.2 Geospatial foundation models

EO-pretrained models encode domain-specific priors (spatial texture, land-use structure, morphology) that may transfer better than generic visual encoders in remote-sensing-like regression settings.

### 2.3 Generic vision backbones

ResNet and DINOv2 are strong and widely used transfer baselines; however, their socioeconomic regression performance can vary substantially by task and optimization regime.

## 3. Methods

### 3.1 Task scope

Completed in this draft:

- `gdp`
- `acc2health`
- `build_height`
- `pop`

Pipeline-wide pending extension task:

- `carbon`

### 3.2 Model set

- `prithvi_rgb_lora` (geospatial FM adaptation with LoRA)
- `dinov2_sat` (generic ViT baseline)
- `resnet50_sat` (generic CNN baseline)

### 3.3 Protocol

- image size `224`, batch size `8`, seed `42`
- target transform `log1p`
- lr/backbone lr `2e-4`, head lr `1e-3`
- weight decay `1e-2`, val fraction `0.1`
- `--resume --skip_if_done`

Epoch budgets:

- `gdp`: 20
- `acc2health`: 30
- `build_height`: 30
- `pop`: 5

Implementation detail: experiment naming includes `ep{epochs}`, so resume is experiment-folder-specific.

### 3.4 Metrics

- primary: R2
- secondary: RMSE, MAE, MSE

## 4. Results

### 4.1 Main R2 comparison

| Task | dinov2_sat | prithvi_rgb_lora | resnet50_sat |
| --- | ---: | ---: | ---: |
| acc2health | 0.0985 | **0.3901** | 0.2124 |
| build_height | 0.6791 | **0.8599** | 0.8004 |
| gdp | 0.4535 | **0.5808** | -4.0046 |
| pop | -0.1840 | **-0.0324** | -0.2661 |

### 4.2 Summary

- `prithvi_rgb_lora` ranks first on all reported tasks.
- `build_height` shows strongest absolute performance.
- `pop` remains unresolved for all backbones (all negative R2).
- `gdp + resnet50_sat` is a failed baseline behavior and must be reported explicitly.

## 5. Discussion

### 5.1 Why geospatial adaptation likely helps

The gains are consistent with a domain-prior hypothesis: EO-oriented pretraining supports extraction of morphology and macro-structure cues relevant to urban socioeconomic proxies.

### 5.2 Interpreting the GDP-ResNet failure

R2 = `-4.0046` suggests severe mismatch under this setup. Plausible causes include:

- optimization schedule mismatch for CNN backbone
- weak long-range context capture for GDP proxy patterns
- heavy-tail target sensitivity and unstable regression dynamics

### 5.3 Why `clip_vitb16` is next for street phase

`clip_vitb16` is a practical first street baseline (strong transferable semantics, manageable Colab profile), but not assumed to be globally optimal; dedicated street-backbone comparison remains required.

## 6. Limitations

- no street-only and no fusion results yet
- no multi-seed confidence interval in this stage
- no integrated per-city fairness/robustness analysis yet

## 7. Future Work

- run street-only matrix (`clip_vitb16`, `resnet50`, `dinov2_vitb14`)
- run fusion matrix (`late`, `gated`)
- perform per-city and failure-taxonomy analysis
- add seed-level robustness and statistical uncertainty
- integrate explainability evidence into final narrative

## 8. Ethics and Responsible Use

Socioeconomic inference from imagery can be misused if treated as individual-level truth. This work targets aggregate research benchmarking; operational policy use requires fairness audits, uncertainty reporting, and domain governance.

## 9. Reproducibility Checklist

- [x] explicit protocol and hyperparameters
- [x] saved metrics/config/history/predictions/checkpoints
- [x] restart-safe training (`best.pt`, `last.pt`, resume flags)
- [ ] multi-seed confidence intervals
- [ ] external transfer validation

## 10. Conclusion

In a controlled satellite-only CityLens setting, adapted Prithvi is consistently stronger than generic satellite backbones on finalized tasks. The result is substantial but phase-bounded; full multimodal ranking requires the planned street and fusion completion.

## Appendix: Fill Slots for Camera-Ready

- A. Command appendix and environment hashes: [TODO]
- B. Full learning curves and early-stop traces: [TODO]
- C. Street/fusion completion table: [TODO]
- D. Per-city and XAI figures: [TODO]
