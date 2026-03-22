# Report for writing

**Manuscript draft (structure aligned with top-tier ML / CV conference expectations)**  
**Project:** CityLens geospatial regression — satellite imagery, street-level imagery, and multimodal fusion for socioeconomic and built-environment indicators.

**PDF:** run **`.venv-pdf/bin/python docs/render_report_pdf.py`** → **`docs/Report_for_writing.pdf`** (single file: full paper shell **and** seed-42 quantitative snapshot below). Requires **`pandoc`** on your PATH and **WeasyPrint** in `.venv-pdf`.

**Multi-seed + shared-validation aggregates** remain placeholders where noted — paste after you finish paired reruns.

---

## Abstract

**Problem:** Predicting neighborhood-scale socioeconomic and urban indicators from remote sensing is often studied with **inconsistent evaluation protocols**—different train/validation splits across modalities, single-seed reporting, and weak baselines—making it hard to tell whether gains come from the model or from evaluation leakage and variance.

**Core idea:** We study **paired, modality-aligned evaluation** of **geospatial foundation models** and strong CNN/ViT baselines on CityLens-style global benchmarks, combining **RGB-to-multispectral adaptation** for Prithvi-EO with **parameter-efficient fine-tuning (LoRA)**.

**Methodology:** For each task and random seed, we fix a **shared validation cohort** on the **street-available subset** so that **satellite-only, street-only, and fusion** models are scored on **identical held-out region IDs**; we report **multi-seed** metrics to quantify instability.

**Results (seed 42, single run — see §4.3):** On satellite, **Prithvi + LoRA** reaches **R² = 0.58** on `gdp`, **0.39** on `acc2health`, **0.87** on `build_height`, and **negative R²** on `pop` under the locked protocol. Street-only **ResNet-50** on the street cohort reaches **R² ≈ 0.36 / 0.33 / 0.34** on `gdp` / `acc2health` / `build_height`; **fusion** in the documented snapshot helps **only** on **`build_height`** vs. satellite Prithvi, while **`pop`** stays difficult. *[Replace with mean ± std after multi-seed shared-split runs.]*

**Impact:** We provide a **transparent protocol** (shared splits, seeds, and baselines) for comparing Earth-observation FMs and generic vision encoders on spatial regression. *[Adjust if code is private: e.g., “Implementation details and reproducibility artifacts are available under collaboration / upon request.”]*

---

## 1. Introduction

### The hook
Cities generate vast **geospatial and street-level imagery**; policymakers and researchers want **scalable measurement** of wealth, health proxies, building height, and related indicators. Remote sensing and street-view ML promise **wall-to-wall coverage** where traditional surveys are sparse, but **generalization and fair comparison** across methods remain open problems.

### The gap
**Earth observation foundation models** (e.g., IBM–NASA **Prithvi-EO 2.x**) leverage **large-scale multispectral pretraining** on harmonized Landsat/Sentinel-style data and achieve strong transfer on segmentation and classification. In parallel, **general vision transformers** (**DINOv2**, **CLIP**) provide strong features for natural images. **However**, regression on **joint satellite + street** benchmarks is often reported with **modality-specific splits** or **single seeds**, inflating apparent differences. **Despite this progress**, a **key challenge remains:** *paired* comparison under **identical held-out geographies** and **variance-aware** reporting.

### Our approach
**In this paper, we** frame CityLens-style tasks as **scalar regression** from single-patch satellite RGB (adapted to six-band Prithvi inputs), variable-length street-view tiles, or **late / gated fusion** of both towers, with a shared MLP regression head.

### Contributions
We summarize contributions as follows (edit numbering to match venue style):

1. **We propose an evaluation protocol** that fixes **shared validation IDs** on the **street cohort** across **satellite, street, and fusion** branches using explicit split JSON files keyed by task, seed, and split tag (e.g., `fusion`).
2. **We implement and compare** (i) **Prithvi-EO v2** with **LoRA** on attention projections, (ii) **ResNet-50** and **DINOv2 ViT-B/14** satellite encoders, and (iii) **ResNet-50 / CLIP ViT-B/16 / DINOv2** street encoders with **mean or attention pooling** over views.
3. **We quantify uncertainty** via **multiple random seeds** and recommend reporting **mean ± standard deviation** alongside the best single run.
4. **We discuss limitations** (spatial leakage, label noise, demographic fairness, compute) and document **seed-42** results inline (§4.3) for traceability.

---

## 2. Related work

### 2.1 Geospatial foundation models and multispectral pretraining
**Prithvi-EO 2.0** (IBM, NASA, collaborators) is a **Vision Transformer–style** model pretrained on large **multitemporal, multispectral** Earth observation data (e.g., HLS-like six-band stacks), using **3D spatiotemporal patch embeddings** and **masked autoencoder** objectives suited to cloud and temporal structure. Open weights and documentation are distributed via Hugging Face and GitHub (`ibm-nasa-geospatial`, `NASA-IMPACT/Prithvi-EO-2.0`). **Strength:** strong priors for land surface and seasonality. **Weakness for our setting:** downstream tasks often use **RGB-only** pipelines; **band alignment** and **efficient adaptation** (LoRA, adapters) are needed for fair city-scale regression.

### 2.2 Self-supervised and vision–language models for generic imagery
**DINOv2** learns **discriminative ViT features** from large curated web-scale data without labels; **CLIP** aligns **image and text** towers for zero-shot transfer. **Strength:** robust off-the-shelf encoders for RGB street and satellite crops. **Weakness:** **no multispectral inductive bias**; may underuse physical signal available to EO-specific models.

### 2.3 Parameter-efficient fine-tuning
**LoRA** (low-rank adaptation) injects trainable low-rank matrices into selected linear layers (often **query/key/value** and projections in transformers), greatly reducing trainable parameters versus full fine-tuning. Implemented in **Hugging Face PEFT** and widely used for ViTs. **Strength:** stable adaptation of large backbones on small geospatial/regression datasets. **Weakness:** rank and target-module choices affect capacity; requires reporting.

### 2.4 Spatial regression, street view, and multimodal urban analytics
Prior work uses **street-view imagery** and **overhead imagery** to predict income, safety, and built form. **Strength:** demonstrated correlation with outcomes. **Weakness:** **split leakage** (spatial autocorrelation), **inconsistent cohorts** across modalities, and **single-seed** results.

### Our position
We **bridge** EO-specific (**Prithvi**) and **general vision** (**DINOv2**, **ResNet**, **CLIP**) backbones under a **unified regression head** and **shared validation IDs**, enabling **paired** claims about **when** multispectral FM adaptation helps versus generic ViTs or CNNs.

---

## 3. Methodology

### 3.1 Overview
**Figure 1 (placeholder).** Teaser: *CityLens locations → satellite patch + N street patches → encoder(s) → optional fusion → scalar prediction; sidebar showing “same val IDs for all branches.”*

**Figure 2 (placeholder).** Architecture diagram: tensor shapes through **RGB → 1×1 conv → six bands → Prithvi backbone (LoRA)** vs **timm ResNet/DINOv2**; **street multi-view encoder + pooling**; **late fusion** = concatenate features; **gated fusion** = learned softmax weights over aligned feature chunks.

### 3.2 Problem formulation
For task \(t\) (e.g., log-GDP proxy, health access, building height), let \(\mathcal{D}_t = \{(x_i^{\mathrm{sat}}, \{x_{i,v}^{\mathrm{st}}\}_{v=1}^{N_i}, y_i)\}_{i=1}^{N}\) where \(y_i \in \mathbb{R}\) is the regression target. For **street cohort** experiments, restrict to indices where street views exist. A **split** is a partition of IDs into \(\mathcal{I}^{\mathrm{train}}\) and \(\mathcal{I}^{\mathrm{val}}\) **shared across branches** \(b \in \{\mathrm{sat}, \mathrm{street}, \mathrm{fusion}\}\).

**Objective:** minimize **mean squared error** (or huber variant if used) on the training set:
\[
\mathcal{L} = \frac{1}{|\mathcal{I}^{\mathrm{train}}|} \sum_{i \in \mathcal{I}^{\mathrm{train}}} \bigl( f_\theta^{(b)}(\cdot) - y_i \bigr)^2.
\]

**Metrics:** **R²**, **RMSE** on validation (and test if held out); report **mean ± std** over seeds \(s \in \mathcal{S}\).

### 3.3 Satellite encoders

#### 3.3.1 Prithvi-EO v2 (300M-class) with RGB adaptation
**Backbone:** Prithvi-EO 2.x uses a **ViT-based encoder** with **multispectral, multitemporal** tokenization (3D patch embedding; temporal and positional encodings) pretrained on large EO corpora (see IBM Research / NASA-IMPACT publications and model cards on Hugging Face).

**RGB → six-band stack:** A **learned 1×1 convolution** maps **3-channel RGB** to **6 channels** aligned with Prithvi band ordering (e.g., Blue, Green, Red, NIR narrow, SWIR-1, SWIR-2) as exposed through **TerraTorch**’s registry.

**Tensor layout:** The backbone expects **5D** tensors **(batch, channels, time, height, width)**; a **single time frame** \(T{=}1\) is used for static patches.

**LoRA fine-tuning:** Apply **PEFT LoRA** to attention linear layers (e.g., fused `qkv` and projection modules where present). Hyperparameters to report: **rank \(r\)**, **\(\alpha\)**, **dropout**, **target module names**, and whether **biases** are trained.

#### 3.3.2 CNN and ViT baselines (RGB satellite)
- **ResNet-50:** Deep residual CNN; **global pooled features** → head.  
- **DINOv2 ViT-B/14:** Patch embedding + transformer blocks; **pool CLS or mean patch tokens** per timm implementation → head.

### 3.4 Street encoders and multi-view pooling
For \(N_i\) views, encode each crop with the same backbone; stack features **(batch, views, dim)**.

- **Mean pooling:** Average view embeddings (optionally mask-padded views).  
- **Attention pooling (optional):** Learn a scalar score per view and take a **weighted sum**.

A **two-layer MLP** (`LazyLinear` → ReLU → linear) maps pooled features to \(\hat{y}\).

### 3.5 Fusion
- **Late fusion:** Concatenate **satellite** and **pooled street** feature vectors; MLP regression head.  
- **Gated fusion:** Concatenate features; a small linear layer produces **two nonnegative weights** (softmax) that **blend** dimension-aligned slices of satellite and street features before the head (implementation detail: crop to common minimum width if dimensions differ).

### 3.6 Shared validation splits and multi-seed protocol
1. **Cohort:** Filter to **street-available** records for tasks where street imagery is used or where fusion/street parity is required.  
2. **Split file:** For task \(t\), seed \(s\), validation fraction \(v\), persist JSON with **`train_ids`** and **`val_ids`** under a stable naming scheme, e.g. `{task}_{split_key}_seed{s}_val{v}.json`.  
3. **Paired training:** **Satellite**, **street**, and **fusion** runs for \((t,s)\) **must load the same file** (same `split_key`, e.g. `fusion`).  
4. **Multi-seed:** Repeat for each \(s \in \mathcal{S}\); aggregate **mean ± std** of R² / RMSE and **best epoch** if relevant.

**Why it matters:** Without shared IDs, **apples-to-oranges** validation sets can dominate apparent modality gaps.

**Executable plan (private codebase):** `docs/CLAIMS_AND_REMAINING_RUNS.md`, `docs/PAIRED_SPLITS_AND_MULTI_SEED.md`.

---

## 4. Experiments

### 4.1 Datasets and tasks
**CityLens-style global benchmark:** JSON task files under `Benchmark/`, imagery under `satellite_image/` and `street_view_image/`. Tasks in this study include **`gdp`**, **`acc2health`**, **`build_height`**, **`pop`** (and optionally **`carbon`** — Prithvi rows are **not** locked for carbon; do not cite without a dedicated row).

### 4.2 Implementation details (locked satellite / Prithvi reference, seed 42)

The following match **`docs/PRITHVI_SATELLITE_REFERENCE.md`** (satellite-only `prithvi_rgb_lora`):

| Setting | Value |
|--------|--------|
| Branch | `satellite` |
| Satellite model | `prithvi_rgb_lora` |
| Image size | 224 |
| Batch size | 8 |
| Seed | 42 |
| Target transform | `log1p` (metrics reported in **raw** target space) |
| Learning rate (main) | 2e-4 |
| Backbone LR | 2e-4 |
| Head LR | 1e-3 |
| Weight decay | 1e-2 |
| Validation fraction | 0.1 |

**Other settings:** optimizer / full epoch grid per task as in tables below; frameworks: PyTorch, **timm**, **TerraTorch** / Prithvi registry, **PEFT**. **Hardware:** *[e.g., NVIDIA T4 / L4 / A100]*.

### 4.3 Main quantitative results — seed 42 snapshot *(single seed; not variance)*

**Protocol notes**

- **Satellite Prithvi** rows: **seed 42**, **val fraction 0.1**, **target transform log1p**, metrics in **raw** space — as locked for comparison baselines (`dinov2_sat`, `resnet50_sat`).
- **Street** rows: **street-available cohort** sizes differ from full satellite totals; R² values follow **`docs/GLOBAL_LEARNED_PIPELINE.md`** (subset-matched street training). Pull **RMSE / best epoch** from each run’s `metrics.json` under `Results/global_learned/` if you need them in the paper.
- **Fusion:** narrative reflects the **documented snapshot**; the **full fusion backbone grid** should be pasted into the placeholder table once consolidated from experiments.

#### 4.3.1 Satellite-only — `prithvi_rgb_lora` (seed 42)

| Task | Epoch budget | Best epoch | R² | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `gdp` | 20 | 14 | 0.5808 | 331463584.0 |
| `acc2health` | 30 | 9 | 0.3901 | 9.5502 |
| `build_height` | 30 | 9 | 0.8682 | 2.5345 |
| `pop` | 5 | 2 | -0.0324 | 21641.2461 |

*Do not cite Prithvi on `carbon` without a dedicated locked row.*

#### 4.3.2 Street-only (seed 42, street cohort)

**Cohort sizes** (records with usable street views, post path fix):

| Task | Street-cohort records |
| --- | ---: |
| `gdp` | 429 |
| `acc2health` | 440 |
| `build_height` | 398 |
| `pop` | 402 |

**Best R²** observed for the listed default street backbones:

| Task | Model | R² |
| --- | --- | ---: |
| `gdp` | `resnet50` | 0.3629 |
| `acc2health` | `resnet50` | 0.3299 |
| `build_height` | `resnet50` | 0.3448 |
| `pop` | `dinov2_vitb14` | 0.0058 |

#### 4.3.3 Fusion (seed 42) — narrative summary

- Satellite **Prithvi** leads on **`gdp`** and **`acc2health`**.
- **Fusion** (e.g., DINOv2 street + **late** fusion in the full configuration grid) improves over satellite **only** on **`build_height`** in the documented snapshot.
- **`pop`** remains difficult; fusion can **hurt** vs. satellite.

**Summary table (best modality, snapshot):**

| Task | Best modality (snapshot) | Note |
| --- | --- | --- |
| `gdp` | Satellite (Prithvi) | — |
| `acc2health` | Satellite (Prithvi) | — |
| `build_height` | Fusion (selected config) | Only task where fusion edges satellite |
| `pop` | — | Weak; fusion often worse |

**Space for full fusion grid:** paste task × satellite backbone × street backbone × fusion type after consolidating `metrics.json` exports.

#### 4.3.4 Multi-seed results with shared validation *(required for robust claims)*

| Task | Branch / model | Seed(s) | R² (mean ± std) | RMSE (mean ± std) | Notes |
|------|----------------|---------|-----------------|-------------------|--------|
| *[e.g., gdp]* | *prithvi_rgb_lora* | *42, 43, …* | **\_\_\_\_** | **\_\_\_\_** | *…* |
| *[…]* | *dinov2_sat* | *…* | **\_\_\_\_** | **\_\_\_\_** | *…* |
| *[…]* | *resnet50_sat* | *…* | **\_\_\_\_** | **\_\_\_\_** | *…* |
| *[…]* | *street (specify)* | *…* | **\_\_\_\_** | **\_\_\_\_** | *…* |
| *[…]* | *fusion (late / gated)* | *…* | **\_\_\_\_** | **\_\_\_\_** | *…* |

**Qualitative figure (placeholder):** scatter plots of predicted vs. actual; residual maps if you have geocodes (mind privacy).

### 4.4 Ablations *(suggested)*
- **LoRA rank** \(r \in \{4, 8, 16\}\) on Prithvi.  
- **Pooling:** mean vs. attention on street views.  
- **Fusion:** late vs. gated; **same split file** across ablations.  
- **Without shared split:** *(careful—may be rejected as unethical leakage demo; prefer synthetic commentary)*.

### 4.5 Analysis
- **Failure modes:** small sample cities, label outliers, cloud/occlusion, domain shift.  
- **Cost:** approximate **trainable parameters** with LoRA vs. full fine-tune; **GPU-hours** per task.

---

## 5. Discussion *(dedicated space for multi-seed + shared val)*

### 5.1 Interpretation of multi-seed variance
**[Write here after results are in.]** Discuss whether **ranking of backbones is stable** across seeds, or whether **variance overlaps** (e.g., DINOv2 vs. Prithvi on a given task). Tie to **optimization noise**, **small validation sets**, and **early stopping**.

### 5.2 What shared validation changed
**[Write here.]** Compare *narratively* to any **older** runs where splits differed by modality. Emphasize that **paired protocols** reduce **confounding** from **different held-out cities**.

### 5.3 Limitations
- **Spatial autocorrelation** and **near-duplicate** patches between train and val if splits are random **without** geographic blocking (if you used ID splits only, discuss residual correlation).  
- **Fairness:** outcomes may **encode historical inequity**; models can **amplify bias** when deployed.  
- **Data license and ethics** for street view and satellite imagery.  
- **Compute and carbon** footprint of large encoders.

### 5.4 Societal impact
**[Required-style text for NeurIPS/CVPR-style venues.]** Discuss **dual-use** (surveillance, inequitable resource allocation) and **mitigations** (governance, abstention, audit).

---

## 6. Conclusion
**[1–2 sentences]** Summarize the **protocol** (shared val + multi-seed) and the **empirical takeaway** once tables are filled.

**Future work:** geographic **k-fold** or **city-held-out** evaluation; **per-city** fairness metrics; **temporal** Prithvi inputs; lighter **student** models distilled from Prithvi features.

---

## References *(starter set — normalize with your venue’s BibTeX style)*

1. IBM Research / NASA–IMPACT. **Prithvi-EO 2.0** materials: model cards and technical reports (e.g., “From Pixels to Predictions: Prithvi-EO-2.0…”, AGU 2025 publications); Hugging Face `ibm-nasa-geospatial/Prithvi-EO-2.0-*`; GitHub `NASA-IMPACT/Prithvi-EO-2.0`.  
2. M. Oquab *et al.*, **DINOv2: Learning Robust Visual Features without Supervision**, arXiv:2304.07193, 2023. https://arxiv.org/abs/2304.07193  
3. A. Radford *et al.*, **Learning Transferable Visual Models From Natural Language Supervision** (CLIP), ICML 2021. https://arxiv.org/abs/2103.00020  
4. K. He *et al.*, **Deep Residual Learning for Image Recognition**, CVPR 2016.  
5. E. J. Hu *et al.*, **LoRA: Low-Rank Adaptation of Large Language Models**, ICLR 2022. https://arxiv.org/abs/2106.09685  
6. **Hugging Face PEFT** documentation for LoRA on ViT/transformers. https://huggingface.co/docs/peft  

*(Add CityLens dataset paper, HLS, and any benchmark citations you rely on.)*

---

## Figure / table checklist (A* strategy)

| Item | Purpose |
|------|--------|
| **Figure 1** | Problem + method + gain in one glance |
| **Figure 2** | Encoder + fusion + **shared split** schematic |
| **Table 1** | Main results, **bold** best per task, **mean ± std** |
| **Appendix** | Full hyperparameters, per-seed tables, licenses |

---

*Replace remaining `[brackets]` and placeholders with your final numbers and prose.*
