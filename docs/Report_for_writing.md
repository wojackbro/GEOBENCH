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

**Results (seed 42 — full tables §4.3):** **Satellite:** **`prithvi_rgb_lora`** leads sane baselines on all four tasks; **`gdp` + `resnet50_sat`** is catastrophic (R² ≈ **−4**). **Street:** best encoder varies (**ResNet-50** vs **DINOv2** on `pop`). **Fusion:** **only `build_height`** (DINOv2 street + **late**) beats best satellite; **`gdp` / `acc2health`** still favor satellite Prithvi; **`pop`** fusion scores are negative. **R² / RMSE / MAE** for every run are in **§4.3.1–4.3.3** and **`OFFICIAL_REPORT_SATELLITE_PHASE.pdf`**. *[Replace abstract numbers with mean ± std after multi-seed shared-split reruns.]*

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

**Metrics:** This benchmark is **regression** (scalar targets). Report **R²**, **RMSE**, **MAE** — not classification accuracy.

**Protocol (core):** image **224**, batch **8**, seed **42**, target **`log1p`**, lr **2e-4** / head **1e-3**, weight decay **1e-2**, val fraction **0.1**. Epoch budgets: **`gdp` 20**; **`acc2health`** & **`build_height` 30**; **`pop` 5**. Fusion: satellite **Prithvi** + timm street encoder; **late** = concat features; **gated** = learned gating (see pipeline doc).

**Caveats (from official technical report):** Satellite-only runs may include `clip_vitb16` in the **folder slug**; they do **not** use CLIP street images. Street/fusion use the **remapped street-enabled subset**; satellite val IDs may **differ** from street/fusion — treat the **cross-branch** row as **indicative**, not a strict paired A/B until shared-split reruns land.

**Archived PDF:** identical tables and narrative live in **`docs/OFFICIAL_REPORT_SATELLITE_PHASE.pdf`** (restored from project history).

#### 4.3.1 Satellite-only — full grid *(seed 42)*

| Task | Model | Best epoch | R² | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| `acc2health` | `dinov2_sat` | 20 | 0.0985 | 11.6113 | 8.7746 |
| `acc2health` | `prithvi_rgb_lora` | 9 | 0.3901 | 9.5502 | 7.1910 |
| `acc2health` | `resnet50_sat` | 22 | 0.2124 | 10.8529 | 7.2042 |
| `build_height` | `dinov2_sat` | 18 | 0.6791 | 3.9542 | 2.8124 |
| `build_height` | `prithvi_rgb_lora` | 11 | 0.8599 | 2.6130 | 1.9100 |
| `build_height` | `resnet50_sat` | 26 | 0.8004 | 3.1182 | 2.2806 |
| `gdp` | `dinov2_sat` | 19 | 0.4535 | 3.7845e8 | 2.3281e8 |
| `gdp` | `prithvi_rgb_lora` | 14 | 0.5808 | 3.3146e8 | 1.9811e8 |
| `gdp` | `resnet50_sat` | 5 | **−4.0046** | 1.1452e9 | 3.9336e8 |
| `pop` | `dinov2_sat` | 5 | −0.1840 | 23175.54 | 11363.06 |
| `pop` | `prithvi_rgb_lora` | 2 | −0.0324 | 21641.25 | 10020.26 |
| `pop` | `resnet50_sat` | 2 | −0.2661 | 23965.72 | 11871.83 |

**Bold:** catastrophic baseline — **do not** treat `gdp` + `resnet50_sat` as a fair comparator (see failure analysis in the official PDF).

**Extra Prithvi satellite folders (other epoch budgets):** `acc2health` ep5 → R² **0.3034**; `build_height` ep5 → **0.8326**; `gdp` ep10 → **0.5667**; `gdp` ep5 run (best ep1) → **0.1912**. The separately locked row in **`PRITHVI_SATELLITE_REFERENCE.md`** uses **`build_height` R² = 0.8682** / best epoch **9** (different checkpoint than the row above — cite **one** convention per paper).

*Extension task **`carbon`:** not in this table; add a row only after you lock a protocol.*

#### 4.3.2 Street-only — full grid *(seed 42, remapped subset)*

**Subset sizes:** `gdp` **429**, `acc2health` **440**, `build_height` **398**, `pop` **402**. **Bold** = best R² per task.

| Task | Street encoder | Best epoch | R² | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| `gdp` | **`resnet50`** | 7 | **0.3629** | 4.5277e8 | 3.0879e8 |
| `gdp` | `dinov2_vitb14` | 13 | 0.0342 | 5.5746e8 | 3.4419e8 |
| `gdp` | `clip_vitb16` | 19 | −0.0287 | 5.7530e8 | 3.3851e8 |
| `acc2health` | **`resnet50`** | 21 | **0.3299** | 8.9596 | 6.3076 |
| `acc2health` | `dinov2_vitb14` | 11 | 0.0100 | 10.8903 | 7.4473 |
| `acc2health` | `clip_vitb16` | 3 | −0.0031 | 10.9619 | 7.3645 |
| `build_height` | **`resnet50`** | 10 | **0.3448** | 5.7259 | 4.5244 |
| `build_height` | `dinov2_vitb14` | 22 | 0.3371 | 5.7597 | 4.6357 |
| `build_height` | `clip_vitb16` | 17 | 0.2338 | 6.1918 | 4.9163 |
| `pop` | **`dinov2_vitb14`** | 2 | **0.0058** | 12341.28 | 9162.27 |
| `pop` | `clip_vitb16` | 5 | −0.3057 | 14142.77 | 9081.77 |
| `pop` | `resnet50` | 2 | −0.4352 | 14827.64 | 9242.04 |

#### 4.3.3 Fusion — full grid *(Prithvi satellite + street; seed 42)*

**Definition:** **Prithvi** (sat) + street encoder; **late** or **gated**. Metrics from validation **`metrics.json`**.

**Not run in this snapshot:** `acc2health` + DINO street fusion; CLIP street fusion for **`build_height`** / **`pop`**.

| Task | Street encoder | Fusion | Epochs | Best epoch | R² | RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `acc2health` | `clip_vitb16` | gated | 30 | 19 | 0.1672 | 9.9885 |
| `acc2health` | `clip_vitb16` | late | 30 | 13 | 0.1347 | 10.1811 |
| `acc2health` | `resnet50` | gated | 30 | 3 | 0.1612 | 10.0244 |
| `acc2health` | `resnet50` | late | 30 | 12 | 0.1909 | 9.8452 |
| `build_height` | `dinov2_vitb14` | gated | 30 | 21 | 0.8653 | 2.5963 |
| `build_height` | `dinov2_vitb14` | late | 30 | 23 | **0.8723** | 2.5274 |
| `build_height` | `resnet50` | gated | 30 | 28 | 0.8593 | 2.6532 |
| `build_height` | `resnet50` | late | 30 | 28 | 0.8484 | 2.7541 |
| `gdp` | `clip_vitb16` | gated | 20 | 18 | 0.0151 | 5.6292e8 |
| `gdp` | `clip_vitb16` | late | 20 | 18 | 0.3450 | 4.5909e8 |
| `gdp` | `dinov2_vitb14` | gated | 20 | 2 | 0.0172 | 5.6234e8 |
| `gdp` | `dinov2_vitb14` | late | 20 | 20 | 0.3966 | 4.4061e8 |
| `gdp` | `resnet50` | gated | 20 | 20 | 0.4437 | 4.2306e8 |
| `gdp` | `resnet50` | late | 20 | 6 | 0.3713 | 4.4975e8 |
| `pop` | `dinov2_vitb14` | gated | 5 | 1 | −0.3291 | 14269 |
| `pop` | `dinov2_vitb14` | late | 5 | 5 | −0.4849 | 15082 |
| `pop` | `resnet50` | gated | 5 | 4 | −0.2942 | 14080 |
| `pop` | `resnet50` | late | 5 | 4 | −0.3487 | 14374 |

**Per-task takeaway:** **`gdp`** — best fusion **<** satellite Prithvi; **`acc2health`** — best fusion **<** best satellite & street; **`build_height`** — **DINOv2 street + late** **>** best satellite R² (**only clear fusion win**); **`pop`** — all fusion R² **negative**.

#### 4.3.4 Cross-branch comparison *(indicative; val IDs may not match across branches)*

| Task | Satellite (best R²) | Street (best R²) | Fusion (best R²) | Max R² | Leading modality |
| --- | ---: | ---: | ---: | ---: | --- |
| `gdp` | 0.5808 | 0.3629 | 0.4437 | 0.5808 | Satellite |
| `acc2health` | 0.3901 | 0.3299 | 0.1909 | 0.3901 | Satellite |
| `build_height` | 0.8599 | 0.3448 | 0.8723 | 0.8723 | Fusion |
| `pop` | −0.0324 | 0.0058 | −0.2942 | 0.0058 | Street *(weak)* |

#### 4.3.5 Executive one-liners *(same snapshot)*

| Layer | Headline |
| --- | --- |
| **Satellite** | **`prithvi_rgb_lora`** best among sane satellite runs on all four tasks; **`gdp` + `resnet50_sat`** fails (R² ≈ **−4.0**). |
| **Street** | Best encoder **task-dependent:** **`resnet50`** (`gdp`, `acc2health`, `build_height`); **`dinov2_vitb14`** (`pop`). |
| **Fusion** | Prithvi + street, late/gated: **only `build_height`** beats best satellite on R²; **`gdp` / `acc2health`** still favor satellite Prithvi; **`pop`** fusion R² negative. |

#### 4.3.6 Multi-seed results with shared validation *(required for robust claims)*

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
