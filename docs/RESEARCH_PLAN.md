# Prithvi + CityLens Research Plan

## Overview

Novel research: **geospatial foundation models (Prithvi)** on the **CityLens** benchmark, with transfer learning from UK data. Goal is to establish whether a specialized geospatial model outperforms generic LVLMs on urban socioeconomic prediction.

---

## Phase 1: Setup & Baseline (Weeks 1–2)

| Task | Description | Status |
|------|-------------|--------|
| **Task 1** | Download CityLens dataset from Hugging Face | |
| | **Correct dataset:** `Tianhui-Liu/CityLens-Data` (not abidhossain123) | |
| | https://huggingface.co/datasets/Tianhui-Liu/CityLens-Data | |
| **Task 2** | Set up evaluation environment from [CityLens GitHub](https://github.com/tsinghua-fib-lab/CityLens) | |
| | `conda create -n citylens python=3.10` then `pip install -r requirements.txt` | |
| **Task 3** | Reproduce **one** baseline (e.g., GPT-4o on GDP prediction for one city or all) | |
| | Commands: see CityLens README → Part2.1 Economy (GDP) | |
| **Task 4** | Document reproduction in `docs/BASELINE_LOG.md` (metrics, exact commands) | |

**Deliverable:** Verified setup + baseline log as “control” experiment.

---

## Phase 2: Core Novel Experiments (Weeks 3–8)

| Task | Description |
|------|-------------|
| **Task 5** | Load Prithvi (NASA/IBM), extract features for **all** CityLens satellite images |
| **Task 6** | Fine-tune Prithvi on CityLens (all tasks) → **Experiment 1** |
| **Task 7** | Set up UK data (IMD + Sentinel-2) for transfer learning → **Experiment 2** |
| **Task 8** | Implement simple ensemble: Prithvi + LVLM features → **Experiment 3** |

**Research questions:**

- **RQ1:** Does fine-tuning Prithvi on CityLens outperform off-the-shelf Prithvi and original CityLens LVLM baselines?
- **RQ3:** Does combining Prithvi satellite features + street-view features (ensemble/fusion) improve accuracy?
- **RQ4:** Which cities/tasks are “hard” vs “easy” for each approach? Does Prithvi help more in some geographies?

---

## Phase 3: Analysis & Writing (Weeks 9–12)

| Task | Description |
|------|-------------|
| **Task 9** | Ablation: which tasks/cities benefit most from fine-tuning and fusion |
| **Task 10** | Visualizations: attention maps, error analysis, cross-city |
| **Task 11** | Draft paper with results |

---

## Phase 3: Documentation & Analysis (Ongoing)

**Reminder:** Track everything. Maintain a detailed log of all experiments, hyperparameters, and results (e.g. in `docs/` or a dedicated experiment log). This is the foundation for the paper.

---

## Publication Strategy (Reminder for Later)

- **Paper 1:** Fine-tuning Prithvi on CityLens + ensemble (e.g. IEEE TGRS, ISPRS, ACM TSAS).
- **Paper 2:** Transfer learning UK → global cities (e.g. Computers, Environment and Urban Systems).
- **Paper 3:** Full benchmark: “Beyond LVLMs: Multimodal Foundation Model Framework for Urban Socioeconomic Sensing” (e.g. Nature Cities / Nature Communications).

---

## Dataset Verification

- **CityLens dataset (Hugging Face):** https://huggingface.co/datasets/Tianhui-Liu/CityLens-Data  
- **CityLens code:** https://github.com/tsinghua-fib-lab/CityLens  
- **Paper:** https://arxiv.org/abs/2506.00530  

Zip contents: `satellite_image/`, `street_view_image/`, `Benchmark/`, `Dataset/`.
