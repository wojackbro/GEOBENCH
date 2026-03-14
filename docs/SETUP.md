# Setup: CityLens + Prithvi Research

## 1. Dataset (Task 1)

**Source:** [Tianhui-Liu/CityLens-Data](https://huggingface.co/datasets/Tianhui-Liu/CityLens-Data) on Hugging Face.

- Do **not** use `abidhossain123/CityLens-Data`; use **`Tianhui-Liu/CityLens-Data`**.

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install huggingface_hub
python scripts/download_citylens.py
```

This downloads `CityLens-Data.zip` and extracts it to `data/CityLens-Data/`. You should see at least: `Benchmark/`, `Dataset/`, `satellite_image/`, `street_view_image/`.

If the zip layout differs (e.g. one top-level folder inside the zip), move contents so that `data/CityLens-Data/Benchmark/` and the other folders exist directly under `data/CityLens-Data/`.

## 2. Evaluation environment (Task 2)

CityLens code is in `CityLens/` (cloned from [tsinghua-fib-lab/CityLens](https://github.com/tsinghua-fib-lab/CityLens)).

**macOS (venv, recommended):** Use the same `.venv` from step 1, or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r CityLens/requirements.txt
```

**Conda (optional):**

```bash
conda create -n citylens python=3.10
conda activate citylens
pip install -r CityLens/requirements.txt
```

API keys (for baseline reproduction with GPT-4):

- OpenAI: `export OPENAI_API_KEY="..."`
- Or configure Azure / other providers in `CityLens/evaluate/utils.py` and `CityLens/config.py` as needed.

## 3. Data root for evaluation

This repo uses a small path layer so the CityLens scripts can find the downloaded data.

- Set the data root to the **extracted** CityLens data (the folder that contains `Benchmark/`, `Dataset/`, etc.):

```bash
export CITYLENS_DATA_ROOT="$(pwd)/data/CityLens-Data"
```

If you extracted the zip so that there is an extra top-level folder (e.g. `data/CityLens-Data/CityLens-Data/Benchmark/`), set:

```bash
export CITYLENS_DATA_ROOT="$(pwd)/data/CityLens-Data/CityLens-Data"
```

- Run evaluation from the **project root** so that `CityLens/path_config.py` is on the Python path:

```bash
cd /path/to/geo_ai1
export CITYLENS_DATA_ROOT="$(pwd)/data/CityLens-Data"
conda activate citylens
python -m CityLens.evaluate.global.global_indicator --city_name=all --mode=eval --model_name=gpt-4o --prompt_type=simple --num_process=10 --task_name=gdp
```

Or from inside `CityLens/`:

```bash
cd CityLens
export CITYLENS_DATA_ROOT="/path/to/geo_ai1/data/CityLens-Data"
python -m evaluate.global.global_indicator --city_name=all --mode=eval --model_name=gpt-4o --prompt_type=simple --num_process=10 --task_name=gdp
```

The path config expects:

- Task file: `$CITYLENS_DATA_ROOT/Benchmark/{task_name}_{city}.json` (e.g. `gdp_all.json`).
- Results: `$CITYLENS_DATA_ROOT/Results/` (created automatically).
- Image URL CSV: `$CITYLENS_DATA_ROOT/Benchmark/image_urls.csv` (if the benchmark uses URL-based API evaluation).

If the dataset’s `Benchmark/` uses different filenames, either rename them to match or adjust `CityLens/path_config.py`.

## 4. Reproduce one baseline (Task 3)

Example: GPT-4o on GDP (simple prompt), all cities.

```bash
export CITYLENS_DATA_ROOT="$(pwd)/data/CityLens-Data"
# Ensure Benchmark/gdp_all.json exists (from dataset or from gen mode)
python -m evaluate.global.global_indicator --city_name=all --mode=eval --model_name=gpt-4o --prompt_type=simple --num_process=10 --task_name=gdp
python -m evaluate.global.metrics --city_name=all --model_name=gpt-4o --prompt_type=simple --task_name=gdp
```

Then fill `docs/BASELINE_LOG.md` with the exact commands and the metrics (MSE, MAE, R², RMSE).

## 5. Phase 2 (later)

- Prithvi: load NASA/IBM model and extract features for all satellite images.
- Fine-tune Prithvi on CityLens; run ensemble (Prithvi + LVLM/street-view features).
- See `docs/RESEARCH_PLAN.md` for the full plan.
