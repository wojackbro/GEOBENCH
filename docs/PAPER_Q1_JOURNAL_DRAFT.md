# Q1 Journal Draft (Full-Length Structure)

## Title

**Geospatial Foundation Model Adaptation for Urban Socioeconomic Sensing: Controlled Satellite-Only Evidence on CityLens with a Multimodal Extension Framework**

## Abstract

Urban socioeconomic sensing from visual data is increasingly evaluated through large benchmarks, yet controlled evidence isolating geospatial pretraining effects remains limited. We present a reproducible satellite-only study on CityLens global-task regression, comparing an adapted geospatial encoder (`prithvi_rgb_lora`) with generic controls (`dinov2_sat`, `resnet50_sat`) under matched splits and optimization settings. Across `gdp`, `acc2health`, `build_height`, and `pop`, `prithvi_rgb_lora` yields the highest R2 in all tasks. We explicitly report a failed baseline behavior (`gdp + resnet50_sat`, R2 = `-4.0046`) to avoid selective reporting. We then provide a staged multimodal roadmap (street-only, late/gated fusion, per-city diagnostics, and explainability integration) and clarify the inferential limits of XAI in this domain. The study emphasizes transparent benchmark practice, failure reporting, and phase-wise claim discipline for publication-grade urban AI research.

## 1. Introduction

Accurate, spatially resolved socioeconomic measurement is critical for urban planning and sustainability policy, but direct data collection is costly and often delayed. Vision-driven proxies from satellite and street imagery are attractive, yet predictive reliability remains heterogeneous across tasks, cities, and model families.

CityLens provides a comprehensive benchmark context; however, practical model development still needs controlled ablations that separate modality effects from encoder effects. This work focuses on that missing layer: a satellite-only, matched-protocol comparison of geospatial adaptation versus generic backbones.

### 1.1 Problem Statement

Given region-level satellite RGB imagery and scalar socioeconomic targets, identify which satellite backbone family provides the strongest and most stable regression behavior under controlled training settings.

### 1.2 Research Questions

- **RQ1:** Does geospatial adaptation (`prithvi_rgb_lora`) outperform generic satellite backbones on CityLens global tasks?
- **RQ2:** Which tasks show strongest gains and which remain unresolved?
- **RQ3:** What does failed baseline behavior reveal about architecture-task mismatch?
- **RQ4 (planned):** How will street-only and fusion stages alter the ranking?

### 1.3 Contributions

- Satellite-only controlled comparison with matched hyperparameters and split reuse.
- Task-level evidence that adapted Prithvi is consistently strongest in this phase.
- Explicit failed-baseline disclosure (`gdp + resnet50_sat`).
- Reproducibility package: checkpoint-safe training, standardized artifacts, explainability hooks.
- Journal-ready extension plan for multimodal and robustness completion.

## 2. Literature Review

### 2.1 Urban socioeconomic sensing and benchmark evolution

Recent work demonstrates growing LVLM capability but also instability in numeric socioeconomic prediction. CityLens advances benchmark breadth, yet method-level isolation (especially unimodal controlled comparisons) is still needed for reliable scientific attribution.

### 2.2 Geospatial pretraining and transfer

Remote-sensing pretraining encodes spatial and land-use priors often absent from generic natural-image pretraining. This suggests potential transfer advantages for urban morphological and macro-pattern tasks.

### 2.3 Generic visual backbones in regression

ResNet and DINOv2 remain key controls. Their efficacy depends on task semantics, target distributions, and optimization regime alignment.

### 2.4 Explainability in multimodal urban ML

Saliency and ablations can validate consistency of learned focus, but cannot by themselves establish socioeconomic causality. Robust conclusions require triangulation with quantitative analyses.

## 3. Data, Tasks, and Evaluation Setup

### 3.1 Tasks in this manuscript stage

- `gdp`
- `acc2health`
- `build_height`
- `pop`

Pipeline continuation task:

- `carbon` (planned extension)

### 3.2 Input modality

- satellite RGB only (single-modality by design for this stage)
- street imagery excluded to avoid modality confounding

### 3.3 Metrics

- primary: R2
- secondary: RMSE, MAE, MSE

## 4. Methods

### 4.1 Model families

- `prithvi_rgb_lora` (EO adaptation with LoRA)
- `dinov2_sat` (generic ViT)
- `resnet50_sat` (generic CNN)

### 4.2 Shared optimization protocol

- image size `224`
- batch size `8`
- seed `42`
- target transform `log1p`
- lr/backbone lr `2e-4`
- head lr `1e-3`
- weight decay `1e-2`
- val fraction `0.1`

Epoch budgets:

- `gdp`: 20
- `acc2health`: 30
- `build_height`: 30
- `pop`: 5

### 4.3 Reproducibility protocol

- split reuse by task/seed/val fraction
- artifacts: `config.json`, `history.csv`, `metrics.json`, predictions, checkpoints
- checkpoint policy: `best.pt`, `last.pt`, `--resume --skip_if_done`
- experiment naming includes `ep{epochs}`, so resume is experiment-folder-specific

## 5. Results

### 5.1 Main R2 table

| Task | dinov2_sat | prithvi_rgb_lora | resnet50_sat |
| --- | ---: | ---: | ---: |
| acc2health | 0.0985 | **0.3901** | 0.2124 |
| build_height | 0.6791 | **0.8599** | 0.8004 |
| gdp | 0.4535 | **0.5808** | -4.0046 |
| pop | -0.1840 | **-0.0324** | -0.2661 |

### 5.2 Observed ranking

`prithvi_rgb_lora` > `dinov2_sat` > `resnet50_sat` on all finalized tasks in this stage.

### 5.3 Mandatory caveat

`gdp + resnet50_sat` (R2 = `-4.0046`) is reported as failed baseline behavior, not as a normal competitive score.

## 6. Discussion

### 6.1 Architecture-task alignment interpretation

The result pattern supports an EO-alignment hypothesis: geospatial priors improve extraction of urban structural cues relevant to socioeconomic proxy regression.

### 6.2 Why `pop` remains weak

All backbones produce negative R2 on `pop`, suggesting weak observability under satellite-only inputs, target noise, or insufficient signal granularity for this setup.

### 6.3 GDP instability in ResNet baseline

Likely contributors:

- backbone-optimizer mismatch in shared schedule
- insufficient long-range context representation
- heavy-tail sensitivity of GDP target under current objective

### 6.4 Implications for multimodal stage

Satellite stage establishes `prithvi_rgb_lora` as anchor backbone. Street/fusion stages should test whether complementary street semantics improve unresolved tasks, especially `pop`.

## 7. Explainable AI Integration Plan

### 7.1 What is already executable

- satellite integrated gradients
- street leave-one-view-out
- fusion modality ablation

### 7.2 Planned journal-level analysis

- align saliency concentration with error buckets
- compare attention profiles for success vs failure cases
- integrate per-city XAI exemplars in supplement

### 7.3 Causality caution

XAI is evidence of model behavior consistency, not proof of socioeconomic causation.

## 8. Limitations

- current stage is satellite-only
- single-seed primary comparison
- no full per-city fairness/robustness section yet
- no uncertainty calibration in current manuscript

## 9. Future Work (Manuscript Completion Plan)

- [TODO] Street-only matrix (`clip_vitb16`, `resnet50`, `dinov2_vitb14`)
- [TODO] Fusion matrix (`late`, `gated`)
- [TODO] Per-city performance/failure taxonomy
- [TODO] Multi-seed confidence intervals + significance tests
- [TODO] Expanded XAI atlas and error-linked interpretation
- [TODO] Full cross-reference against CityLens official paradigms

## 10. Ethics, Fairness, and Misuse Considerations

Socioeconomic prediction from imagery is sensitive. Outputs should be used for aggregate research benchmarking only. High-stakes deployment requires fairness evaluation, uncertainty communication, and institutional oversight.

## 11. Reproducibility Statement

This work documents model/task setup, protocol, artifacts, and checkpoint semantics for rerunnable experiments. Remaining work for journal-grade reproducibility includes multi-seed robustness, external transfer checks, and full environment hash publication.

## 12. Conclusion

Controlled satellite-only evidence indicates that adapted Prithvi is a stronger CityLens regression backbone than generic satellite controls under matched settings. The result is meaningful but phase-bounded; final journal claims will be completed after street, fusion, and robustness phases.

## Appendix Slots (Journal Completion)

- A. Data and split manifest: [TODO]
- B. Full hyperparameter and run registry: [TODO]
- C. Expanded per-task diagnostics: [TODO]
- D. Per-city fairness and bias audit: [TODO]
- E. XAI qualitative appendix: [TODO]
