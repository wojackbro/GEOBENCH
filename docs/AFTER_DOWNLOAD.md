# After the CityLens Download Finishes

When `python scripts/download_citylens.py` completes, it will extract the zip to `data/CityLens-Data/` and print the folder names. Then do the following.

---

## 1. Check the folder structure

```bash
python scripts/verify_data_structure.py
```

You should see `Benchmark/`, `Dataset/`, `satellite_image/`, `street_view_image/`. If the zip had a nested folder (e.g. `CityLens-Data/CityLens-Data/Benchmark`), set the data root to the **inner** folder that directly contains those four:

```bash
export CITYLENS_DATA_ROOT="/Users/abidhossain/Downloads/geo_ai1/data/CityLens-Data/CityLens-Data"
```

Otherwise use:

```bash
export CITYLENS_DATA_ROOT="/Users/abidhossain/Downloads/geo_ai1/data/CityLens-Data"
```

---

## 2. Install CityLens evaluation dependencies

Still in the same terminal, with `.venv` activated:

```bash
pip install -r CityLens/requirements.txt
```

This installs PyTorch, transformers, etc. and may take several minutes.

---

## 3. Set your OpenAI API key (for GPT-4o baseline)

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

(Or configure Azure/DeepInfra in `CityLens/evaluate/utils.py` if you use a different provider.)

---

## 4. Run the GDP baseline (Task 3)

From the **project root** (`geo_ai1`):

```bash
cd /Users/abidhossain/Downloads/geo_ai1
source .venv/bin/activate
export CITYLENS_DATA_ROOT="/Users/abidhossain/Downloads/geo_ai1/data/CityLens-Data"
cd CityLens
python -m evaluate.global.global_indicator --city_name=all --mode=eval --model_name=gpt-4o --prompt_type=simple --num_process=10 --task_name=gdp
python -m evaluate.global.metrics --city_name=all --model_name=gpt-4o --prompt_type=simple --task_name=gdp
```

The first command runs the model on the GDP task (can take a while); the second computes MSE, MAE, R², RMSE.

---

## 5. Document (Task 4)

Copy the metrics from the terminal into `docs/BASELINE_LOG.md` (and note the exact commands you used).

---

**If Benchmark filenames don’t match:** The code expects e.g. `Benchmark/gdp_all.json`. If your zip has different names (e.g. `gdp.json`), either rename/copy the files or adjust `CityLens/path_config.py` to match.
