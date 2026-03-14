# Baseline Reproduction Log

**Purpose:** Record one reproduced CityLens baseline (e.g. GPT-4o on GDP prediction) to verify the evaluation environment. This is the “control” experiment.

---

## Environment

- **Date:**
- **Python / conda env:** (e.g. `citylens`, Python 3.10)
- **CityLens repo:** `tsinghua-fib-lab/CityLens` (commit hash if applicable)
- **Data:** CityLens-Data from `Tianhui-Liu/CityLens-Data` (Hugging Face), extracted under `data/CityLens-Data/`

---

## Reproduced Experiment

- **Task:** GDP prediction (global indicator)
- **Model:** e.g. `gpt-4o`
- **Prompt type:** `simple` (or `normalized` if you reproduce that)
- **Cities:** `all` (or list if single-city)

### Exact commands

```bash
# 1. Generate task data (if not using pre-built Benchmark files)
# python -m evaluate.global.global_indicator --city_name="all" --mode="gen" --task_name="gdp"

# 2. Run evaluation (from CityLens repo root, with CITYLENS_DATA_ROOT set)
# export CITYLENS_DATA_ROOT=/path/to/geo_ai1/data/CityLens-Data
# python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="gdp"

# 3. Compute metrics
# python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="gdp"
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

*(Compare with paper/README if reported.)*

---

## Success / Failure

- [ ] **Success:** Metrics match or are close to reported baseline (note any differences).
- [ ] **Failure:** Describe error (e.g. missing paths, API key, data format) and fix applied.

---

## Notes

- API keys required for GPT-4o: `OpenAI_API_KEY` or Azure (see CityLens README).
- If the repo uses image URLs instead of local paths, ensure the dataset or Benchmark includes the URL mapping used by the eval script.
