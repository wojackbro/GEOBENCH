# Prithvi + CityLens Research

Novel research: geospatial foundation models (Prithvi) on the CityLens benchmark, with transfer learning from UK data.

---

## Run everything in Colab (recommended)

Use **`colab_citylens_full.ipynb`** in Google Colab so you don’t download ~10 GB on your machine:

1. Upload `colab_citylens_full.ipynb` to [Google Colab](https://colab.research.google.com) (or open from Drive).
2. Run all cells in order. The notebook will:
   - Clone the CityLens repo
   - Add path config and patch eval code for Colab
   - Download **Tianhui-Liu/CityLens-Data** from Hugging Face to Colab disk (~10 GB)
   - Install dependencies and run the GDP baseline (GPT-4o)
3. Set your `OPENAI_API_KEY` when prompted (or in Colab Secrets).

No local download or venv needed.

---

## Quick setup (Phase 1) – local

### 1. Download CityLens data (Task 1)

**Correct dataset:** [Tianhui-Liu/CityLens-Data](https://huggingface.co/datasets/Tianhui-Liu/CityLens-Data) on Hugging Face (not abidhossain123).

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install huggingface_hub
python scripts/download_citylens.py
```

Data is extracted to `data/CityLens-Data/`. If the zip has an extra top-level folder, set `CITYLENS_DATA_ROOT` to the folder that directly contains `Benchmark/`, `Dataset/`, `satellite_image/`, `street_view_image/`.

### 2. Set up evaluation environment (Task 2)

**Option A – venv (recommended on macOS):**

```bash
source .venv/bin/activate
pip install -r CityLens/requirements.txt
```

**Option B – conda (if you prefer):**

```bash
conda create -n citylens python=3.10
conda activate citylens
pip install -r CityLens/requirements.txt
```

Set API key for GPT-4o (or your LVLM provider):

```bash
export OPENAI_API_KEY="..."   # for GPT-4o; or see CityLens README for Azure/DeepInfra
```

### 3. Reproduce one baseline (Task 3) – GDP with GPT-4o

From project root:

```bash
source .venv/bin/activate   # or: conda activate citylens
export CITYLENS_DATA_ROOT="$(pwd)/data/CityLens-Data"
cd CityLens
python -m evaluate.global.global_indicator --city_name=all --mode=eval --model_name=gpt-4o --prompt_type=simple --num_process=10 --task_name=gdp
python -m evaluate.global.metrics --city_name=all --model_name=gpt-4o --prompt_type=simple --task_name=gdp
```

**Note:** The repo expects pre-built task JSONs in `Benchmark/` (e.g. `gdp_all.json`) and an `image_urls.csv` for API-based eval. If the zip uses different names, adjust `CityLens/path_config.py` or symlink files. Run `python scripts/verify_data_structure.py` after extracting to see the layout.

### 4. Document (Task 4)

Fill in `docs/BASELINE_LOG.md` with exact commands, metrics (MSE, MAE, R², RMSE), and success/failure notes.

---

## Phase 1 checklist
- [x] Clean workspace, fresh start
- [ ] Task 1: Download CityLens dataset (run `scripts/download_citylens.py`)
- [ ] Task 2: Set up env (venv + `pip install -r CityLens/requirements.txt`, or conda)
- [ ] Task 3: Reproduce one baseline (GPT-4o on GDP; see commands above)
- [ ] Task 4: Document in `docs/BASELINE_LOG.md`

## Phase 2 (Weeks 3–8)
- Task 5: Load Prithvi, extract features for all satellite images
- Task 6: Fine-tune Prithvi on CityLens (Experiment 1)
- Task 7: UK data (IMD + Sentinel-2) for transfer (Experiment 2)
- Task 8: Ensemble Prithvi + LVLM features (Experiment 3)

## Phase 3 (Weeks 9–12)
- Task 9: Ablation (tasks/cities)
- Task 10: Visualizations
- Task 11: Draft paper

## Layout
- `data/` — CityLens and other datasets
- `CityLens/` — Cloned evaluation repo (with `path_config.py` for data root)
- `scripts/` — Download, feature extraction, training
- `docs/` — BASELINE_LOG.md, RESEARCH_PLAN.md, experiment logs
