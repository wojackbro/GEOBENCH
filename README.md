# GeoBench (documentation & writing bundle)

This repository supports the CityLens / GeoBench geospatial regression line of work.

## Primary reference document

**`paper_curent_draft.docx`** — main paper draft (**canonical** manuscript for this project).

---

## Literature review & sources (models, data, methods)

Use this as a **reading list** for related work and implementation context. Normalize citations for your venue (BibTeX from publisher pages or [DBLP](https://dblp.org/) / [Semantic Scholar](https://www.semanticscholar.org/)).

### A. Geospatial foundation models & Earth observation

| Topic | What to read | Links |
|--------|----------------|--------|
| **Prithvi-EO 2.0** (IBM / NASA / collaborators) | Technical reports, model cards, spatiotemporal ViT + MAE-style EO pretraining | [Hugging Face — `ibm-nasa-geospatial`](https://huggingface.co/ibm-nasa-geospatial) · [GitHub — `NASA-IMPACT/Prithvi-EO-2.0`](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) · [IBM Research publications (search “Prithvi”)](https://research.ibm.com/publications) · [Papers with Code — Prithvi-EO-2.0](https://paperswithcode.com/paper/prithvi-eo-2-0-a-versatile-multi-temporal) |
| **Prithvi / EO tooling in PyTorch** | Registry, multispectral bands, training stacks | [TerraTorch](https://github.com/IBM/terratorch) (IBM) |
| **Harmonized Landsat Sentinel-2 (HLS)** | Physical basis for multispectral inputs used in EO FMs | [NASA LP DAAC — HLS overview](https://www.earthdata.nasa.gov/data/catalog/lpcloud-hlssh.v2.0) · [HLS algorithm theoretical basis (ATBD)](https://lpdaac.usgs.gov/documents/1696/HLS_User_Guide_V2.0.pdf) (PDF) |

### B. General computer vision backbones (used as satellite/street encoders)

| Model | Paper / resource | Links |
|--------|-------------------|--------|
| **ResNet** | Deep residual networks | He *et al.*, *Deep Residual Learning for Image Recognition*, CVPR 2016 — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| **Vision Transformer (ViT)** | Patch-based transformers for images | Dosovitskiy *et al.*, *An Image is Worth 16x16 Words*, ICLR 2021 — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| **Masked Autoencoder (MAE)** | Self-supervised ViT reconstruction (related to many EO FMs) | He *et al.*, CVPR 2022 — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) |
| **DINO → DINOv2** | Self-distillation / large-scale ViT features | Caron *et al.*, *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021 — [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) · Oquab *et al.*, **DINOv2**, 2023 — [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) · [Code — `facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2) |
| **CLIP** | Image–text contrastive pretraining (ViT image tower in `timm`) | Radford *et al.*, ICML 2021 — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) · [OpenCLIP / OpenAI CLIP repos](https://github.com/mlfoundations/open_clip) |
| **`timm` (PyTorch Image Models)** | Unified pretrained weights (`resnet50`, `vit_*_dinov2*`, `vit_*_clip*`, …) | [GitHub — `huggingface/pytorch-image-models`](https://github.com/huggingface/pytorch-image-models) · [Docs](https://huggingface.co/docs/timm) |

### C. Parameter-efficient fine-tuning (Prithvi + LoRA)

| Topic | Paper / docs | Links |
|--------|----------------|--------|
| **LoRA** | Low-rank adapters for large models | Hu *et al.*, ICLR 2022 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **PEFT (Hugging Face)** | Library used with ViTs / transformers | [PEFT documentation](https://huggingface.co/docs/peft) · [LoRA conceptual guide](https://huggingface.co/docs/peft/conceptual_guides/lora) |

### D. Benchmark, dataset & urban / multimodal context

| Topic | Notes | Links |
|--------|--------|--------|
| **CityLens (code & tasks)** | Original benchmark repo, global indicators | [GitHub — `tsinghua-fib-lab/CityLens`](https://github.com/tsinghua-fib-lab/CityLens) |
| **CityLens-Data (HF)** | Dataset hosting | [Hugging Face — `Tianhui-Liu/CityLens-Data`](https://huggingface.co/datasets/Tianhui-Liu/CityLens-Data) |
| **Street-view & overhead imagery → socioeconomic outcomes** | Search “street view income prediction”, “remote sensing socioeconomic”, *Science* / *PNAS* style urban sensing | Use [Google Scholar](https://scholar.google.com/) or [Semantic Scholar](https://www.semanticscholar.org/) with those queries; cite 3–5 **recent** + **foundational** papers in your intro. |
| **Multimodal / late fusion** | Generic late fusion, gating | Survey papers on *multimodal deep learning* (Baltrušaitis *et al.*, TPAMI 2019 is a common survey) — [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) |

### E. Explainability & evaluation rigor

| Topic | Resource | Links |
|--------|----------|--------|
| **Integrated Gradients** | Attribution for CNNs/ViTs | Sundararajan *et al.*, ICML 2017 — [arXiv:1703.01365](https://arxiv.org/abs/1703.01365) · [Captum (PyTorch)](https://github.com/pytorch/captum) |
| **Spatial ML / leakage** | Train/test leakage in geospatial data | Search *“geospatial cross-validation”*, *“spatial leakage machine learning”*; cite e.g. Roberts *et al.*–style spatial CV work as appropriate to your claims. |
| **Multi-seed reporting** | Variance in deep learning | Report mean ± std; see venue reproducibility checklists (e.g. NeurIPS / CVPR). |

### F. Local `docs/` folder (not on GitHub)

The **`docs/`** directory is **gitignored** — it is **not** pushed to this public repository. Keep tables, markdown notes, PDFs, and render scripts **only on your machine**. The **primary reference** for the written paper is still **`paper_curent_draft.docx`** (which *is* tracked unless you ignore it separately).

*Note:* Older commits may still contain `docs/` in history; to scrub from GitHub entirely you’d need history rewriting (e.g. `git filter-repo`). New clones will not receive `docs/`.

---
