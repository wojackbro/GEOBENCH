# Why "0 valid data" in CityLens on Colab

This doc is specific to **CityLens** (Benchmark/Dataset/Results JSON), not generic train/val/test image datasets.

---

## How CityLens evaluation works

- **Benchmark/** (e.g. `gdp_all.json`) = task file: list of items with `images`, `prompt`, `reference` (ground truth number), and optionally `reference_normalized`.
- **Results/** (e.g. `gdp_all_gemini-1.5-flash_simple.json`) = model output: list of `{ "reference", "reference_normalized", "response" }`.
- **Metrics** = read Results JSON → `extract_float(reference)` and `extract_float(response)` → compute MSE, MAE, R². If either is `None`, that item is skipped. **0 valid data** = every item was skipped.

---

## Common causes and fixes

### 1. Results file missing or empty

- **Symptom:** Metrics say "length of vals: 0 0" or "Results file is EMPTY".
- **Cause:** GDP eval cell was skipped, or failed, or wrote to a different path.
- **Fix:** Run the **GDP eval cell** ("Patch eval_task then run") and wait until it finishes (progress bar 1000/1000). Then run the **Verify dataset** cell to confirm `CITYLENS_DATA_ROOT` and that `Results/gdp_all_gemini-1.5-flash_simple.json` exists.

### 2. Wrong folder structure / path

- **Symptom:** "Repository not found" or "No Benchmark/ and Dataset/ found".
- **Cause:** Data root points to the wrong place (e.g. `/content/CityLens-Data` while data is in `/content/CityLens-Data/CityLens-data` or in Google Drive).
- **Fix:** Run **section 3 (Google Drive + CityLens dataset)** so `CITYLENS_DATA_ROOT` is set to the folder that actually contains `Benchmark/` and `Dataset/`. Run **Verify dataset** to see the detected root.

### 3. Task JSON missing `reference_normalized`

- **Symptom:** Eval crashes with `KeyError: 'reference_normalized'`.
- **Cause:** Benchmark file (e.g. copied from `Dataset/all_global_gdp_task_all.json`) has only `reference`, but the eval code expected `reference_normalized`.
- **Fix:** The codebase is patched to use `d.get("reference_normalized", d.get("reference"))`, so this should no longer crash. If you still see KeyError, ensure you have the latest `CityLens/evaluate/global/global_indicator.py`.

### 4. Model response not parseable as a number

- **Symptom:** Results file has many items but metrics still report "0 valid data"; debug cell shows `_extract_float(response): None` for the first item.
- **Cause:** The model returns long text (e.g. "The estimated GDP is approximately 1.5 billion...") and the first number in the string is not found by the regex, or the format is unexpected.
- **Fix:** `extract_float` in `metrics.py` already tries: (1) direct float, (2) short cleaned string, (3) first number via regex `-?[\d,]+\.?\d*`. If your model consistently uses a different format (e.g. "X.XXe+9"), you may need to extend the regex or parsing in `extract_float`.

### 5. Annotation / key mismatch

- **Symptom:** Debug cell shows "First item keys: [...]" and there is no `reference` or `response`.
- **Cause:** Results JSON was written by different code (e.g. different keys like `ground_truth` / `model_output`).
- **Fix:** Ensure the GDP eval cell uses the same `global_indicator.py` that writes `reference`, `reference_normalized`, and `response`. Do not rename these keys in the eval output.

### 6. Colab extraction / Drive path

- **Symptom:** Data root is wrong after reconnecting or "Run all".
- **Cause:** Section 3 was not run in this session, so `CITYLENS_DATA_ROOT` is unset or points to `/content/` (wiped).
- **Fix:** Always run **section 3** first (mount Drive and set data root). Use the **Verify dataset** cell to confirm where the data is.

---

## Debug checklist (run in Colab)

1. **Verify dataset** cell → confirm `CITYLENS_DATA_ROOT` and that Benchmark/Dataset exist.
2. **Debug: why 0 valid data?** cell → inspect Results JSON length, first item keys, `reference`/`response` types and values, and how many pairs parse to numbers.
3. If Results JSON is **empty** → run GDP eval and wait for completion.
4. If **reference** or **response** is missing or not a number-like value → check eval output format and `extract_float` logic in `evaluate/global/metrics.py`.

---

## Why this is normal

CityLens (and many research benchmarks) assumes a specific folder layout and JSON schema. Small mismatches (path, key names, or number format) lead to 0 valid samples. The fixes above and the new debug cell should narrow down the cause quickly.
