# Baseline Reproduction Log

**Purpose:** Record reproduced control baselines to verify the environment and support comparisons.

Recommended controls now:
- legacy CityLens API/LVLM baseline if you explicitly want paper-to-paper comparison
- learned low-cost control from `evaluate.global_learned.feature_control`
- learned satellite baseline from `evaluate.global_learned.train --branch satellite`

---

## Environment

- **Date:**
- **Python / conda env:** (e.g. `citylens`, Python 3.10)
- **CityLens repo:** `tsinghua-fib-lab/CityLens` (commit hash if applicable)
- **Data:** CityLens-Data from `Tianhui-Liu/CityLens-Data` (Hugging Face), extracted under `data/CityLens-Data/`

---

## Reproduced Experiment

- **Task:** e.g. `gdp`
- **Family:** `legacy_api_baseline` or `learned_control` or `learned_satellite`
- **Model:** e.g. `gpt-4o`, `resnet50 feature_control`, `prithvi_rgb_lora`
- **Protocol:** e.g. `simple`, `street_feature_control`, `satellite`
- **Cities:** `all`

### Exact commands

```bash
# 1. Generate task data (if not using pre-built Benchmark files)
# python -m evaluate.global.global_indicator --city_name="all" --mode="gen" --task_name="gdp"

# 2a. Legacy API baseline (optional)
# export CITYLENS_DATA_ROOT=/path/to/CityLens-Data
# python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="gdp"
# python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="gdp"

# 2b. Low-cost learned control
# python -m evaluate.global_learned.feature_control --task_name all --street_model resnet50 --image_size 224 --batch_size 8 --seed 42 --skip_if_done

# 2c. Satellite learned baseline
# python -m evaluate.global_learned.train --task_name all --branch satellite --satellite_model prithvi_rgb_lora --epochs 5 --batch_size 8 --image_size 224 --seed 42 --skip_if_done --resume
```

*(Update with the exact commands you used and paths.)*

---

## Results (fill after reproduction)

| Metric | Value |
|--------|--------|
| MSE    | |
| MAE    | |
| R²     | |
| RMSE   | |

*(Compare against CityLens README/paper if you reproduced the legacy baseline, or against your own learned controls if you are staying API-free.)*

---

## Success / Failure

- [ ] **Success:** Metrics match or are close to reported baseline (note any differences).
- [ ] **Failure:** Describe error (e.g. missing paths, API key, data format) and fix applied.

---

## Notes

- The current Colab notebook is API-free by default and focuses on `Results/global_learned/`.
- Learned runs save `config.json`, `metrics.json`, `history.csv` or regression artifacts, and prediction CSVs under `CITYLENS_DATA_ROOT/Results/global_learned/`.
- API keys are only required if you intentionally reproduce the original LVLM baselines.
