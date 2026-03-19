# Q1 Journal Draft (Comprehensive Version)

## Provisional Title

**From Generic Vision to Geospatial Foundation Models: A Reproducible CityLens Study with Satellite-Only Controls and Multimodal Roadmap**

## Abstract

Urban socioeconomic sensing from Earth observation and street-view imagery has become a central benchmark problem for modern vision systems. However, rigorous controlled analyses that isolate the value of geospatial pretraining remain limited. We present a reproducible CityLens study centered on satellite-only global-task regression using an adapted geospatial foundation model (`prithvi_rgb_lora`) against two strong generic controls (`dinov2_sat`, `resnet50_sat`). Under matched protocols, Prithvi achieves the best R2 across `gdp`, `acc2health`, `build_height`, and `pop`. We explicitly document an unstable baseline outcome (`gdp + resnet50_sat`, R2 = -4.00) as failed behavior rather than a competitive result. We further provide a full multimodal roadmap (street-only, late/gated fusion, per-city diagnostics, and explainability analyses) and discuss how XAI can validate mechanistic hypotheses without over-claiming causality. This paper emphasizes transparent failure reporting, reproducibility, and phase-wise evidence building for high-integrity urban AI research.

## 1. Introduction

Estimating urban socioeconomic indicators from imagery is high-impact and high-risk. Progress is constrained by domain heterogeneity, target noise, and weakly observable causal pathways. CityLens offers a broad and practical benchmark, but many evaluations mix modality and model effects, making it difficult to isolate the contribution of geospatially specialized pretraining.

This study takes a phase-wise approach:

1. Fix protocol and compare satellite-only backbones.
2. Quantify gains from geospatial adaptation vs generic controls.
3. Expand to street-only and fusion only after satellite evidence is stabilized.

### 1.1 Research Questions

- **RQ1:** Does adapted geospatial pretraining improve CityLens satellite-only regression over generic backbones?
- **RQ2:** Which tasks benefit most from this specialization?
- **RQ3:** What failure signatures emerge, and what do they suggest about architecture-task mismatch?
- **RQ4 (planned):** How do street and fusion modalities alter ranking and robustness?

### 1.2 Contributions

- Controlled satellite-only benchmark with transparent caveat reporting.
- Architecture-informed interpretation of success and failure patterns.
- Reproducible pipeline with resume-safe checkpointing and structured outputs.
- A publication-ready roadmap for multimodal and explainability extensions.

## 2. Background and Literature Review

### 2.1 Urban sensing benchmarks and CityLens context

CityLens evaluates socioeconomic/urban indicators across multiple domains and supports satellite and street-view modalities. Prior results motivate multimodal learning but also expose model-specific sensitivity to prompt strategy, data quality, and transfer assumptions.

### 2.2 Remote sensing foundation models

Geospatial FMs encode priors from Earth-observation domains (spatial texture, land-use patterns, scale context). Such priors can transfer to urban proxy tasks even when direct modality alignment is imperfect.

### 2.3 Generic computer vision backbones

ResNet and DINOv2 remain strong transfer baselines. Their relative weakness on some socioeconomic tasks may reflect mismatch between generic pretraining objectives and geospatial regression targets.

### 2.4 Explainability in geospatial ML

Saliency and ablation methods can detect brittle behavior and guide error analysis, but they do not establish causal truth. Best practice is triangulation: combine XAI with quantitative validation and stratified performance breakdowns.

## 3. Data, Tasks, and Evaluation

### 3.1 Tasks in this phase

- `gdp`
- `acc2health`
- `build_height`
- `pop`

The full benchmark pipeline includes `carbon`. This manuscript stage reports four tasks with completed and validated satellite-only outputs; `carbon` is reserved for the next reporting increment.

### 3.2 Inputs

- Satellite RGB imagery (single-modality for this phase).
- Street imagery excluded in this phase by design.

### 3.3 Metrics

- Primary: R2
- Secondary: RMSE, MAE, MSE

R2 is prioritized for cross-task comparative interpretation, with raw-space RMSE/MAE used to retain scale-aware error context.

## 4. Methods

### 4.1 Models

- `prithvi_rgb_lora`: geospatial FM adaptation via LoRA.
- `dinov2_sat`: generic transformer baseline.
- `resnet50_sat`: generic CNN baseline.

### 4.2 Optimization protocol

- image size 224, batch size 8, seed 42
- `log1p` target transform
- learning rates: 2e-4 (global/backbone), 1e-3 (head)
- weight decay 1e-2
- validation fraction 0.1
- checkpointing and resume enabled

Task-specific epoch budgets: GDP 20, Acc2Health 30, Build Height 30, Pop 5.

### 4.3 Reproducibility design

- deterministic split reuse per task/seed/val fraction
- complete artifacts (`config`, `history`, `metrics`, predictions, checkpoints)
- explicit run naming by task/model/hyperparameters
- resume semantics are experiment-folder-specific because folder names encode epoch budget (`ep{epochs}`)

## 5. Results

### 5.1 Aggregate results (R2)

| Task | dinov2_sat | prithvi_rgb_lora | resnet50_sat |
| --- | ---: | ---: | ---: |
| acc2health | 0.0985 | **0.3901** | 0.2124 |
| build_height | 0.6791 | **0.8599** | 0.8004 |
| gdp | 0.4535 | **0.5808** | -4.0046 |
| pop | -0.1840 | **-0.0324** | -0.2661 |

### 5.2 Observed ranking

`prithvi_rgb_lora` > `dinov2_sat` > `resnet50_sat` on all tasks by R2, with the strongest separation on GDP and Build Height.

### 5.3 Mandatory caveat

`gdp + resnet50_sat` is a failed baseline behavior (R2 = -4.00) and should be explicitly reported as instability/mismatch in this setting.

## 6. Discussion

### 6.1 Architecture-task alignment hypothesis

Prithvi may gain from EO-oriented priors that better match urban morphology and macro-structure cues. DINOv2 captures broad semantics but may lack geospatially tuned biases. ResNet may underperform where long-range contextual encoding is critical.

### 6.2 Why does population remain hard?

Negative R2 across all backbones suggests weak satellite-only observability for `pop` at this setup level, label granularity mismatch, or noisy proxy relationships.

### 6.3 Implications for multimodal design

The satellite phase suggests a strong satellite anchor model (Prithvi). Planned street/fusion experiments should test whether complementary street signals improve difficult tasks (especially `pop` and potentially `acc2health`).

## 7. Explainable AI Plan and Scientific Value

### 7.1 What can be tested now

- Integrated gradients on satellite predictions.
- Failure-case saliency for `pop` and unstable GDP baselines.

### 7.2 What to test after street/fusion runs

- Street leave-one-view-out to quantify view contribution.
- Fusion modality ablation (drop satellite/street branch).
- Correlate saliency concentration with error and uncertainty.

### 7.3 What XAI cannot prove

XAI cannot independently prove causal mechanisms. It should be framed as consistency evidence, not causal confirmation.

## 8. Limitations

- single-seed main comparison
- no multimodal results yet in this manuscript version
- no per-city significance testing yet
- no uncertainty calibration analysis yet

## 9. Future Work

- [TODO] Street-only matrix (`clip_vitb16`, `resnet50`, `dinov2_vitb14`)
- [TODO] Fusion matrix (`late`, `gated`)
- [TODO] Per-city and cross-city robustness analysis
- [TODO] Multi-seed confidence intervals and significance tests
- [TODO] Extended XAI qualitative atlas + quantitative alignment checks

## 10. Conclusion

This study provides reproducible evidence that an adapted geospatial foundation model is a stronger satellite-only baseline than generic vision backbones on CityLens global tasks under matched settings. The result is promising but phase-bounded. A complete claim requires the planned street-only, fusion, per-city, and multi-seed analyses.

## 11. Practical Positioning for Publication

### Conference version (short-form)

Focus on controlled satellite-only evidence, transparent failed baseline reporting, and concise methodology.

### Q1 journal version (full-form)

Extend with multimodal experiments, robustness statistics, per-city breakdowns, and a substantial explainability section.

## Acknowledged Open Sections (to be filled later)

- [TODO] Full street-only results table
- [TODO] Full fusion results table
- [TODO] Per-city leaderboard and failure taxonomy
- [TODO] XAI figures and case narratives
- [TODO] Cross-reference against CityLens official multimodal baselines
