# Paired splits + multi-seed (low compute)

## Idea

- **Same validation IDs** for satellite, street, and fusion on the **street-available cohort**: use one split file per `(task, seed, val_frac)`, e.g. `gdp_fusion_seed42_val0.1.json`.
- **Training** with `--split_key fusion` makes every branch read that file (satellite then trains only on IDs that appear in that split—i.e. the street cohort).

## Commands (from `CityLens/`)

```bash
export CITYLENS_DATA_ROOT=/path/to/CityLens-data
export MPLBACKEND=Agg

# 1) Create split files (no GPU training; instant)
python -m evaluate.global_learned.make_shared_split --task_name gdp --seed 42
python -m evaluate.global_learned.make_shared_split --task_name gdp --seed 43

# 2) Paired runs (repeat per seed)
python -m evaluate.global_learned.train --task_name gdp --branch satellite --satellite_model prithvi_rgb_lora \
  --split_key fusion --seed 42 --epochs 20 --batch_size 8 --skip_if_done --resume

python -m evaluate.global_learned.train --task_name gdp --branch street --street_model resnet50 \
  --split_key fusion --seed 42 --epochs 20 --batch_size 8 --skip_if_done --resume
```

Use the **same** `--seed` for split file selection and `set_seed` inside training.

## Multi-seed

- Run step (1) for each seed in `{42, 43, 44}`.
- Run step (2) for each seed; aggregate `metrics.json` in a spreadsheet.

## Colab / automation

Use your **private** GeoBench checkout: the public repo no longer ships Colab `.ipynb` drivers. Same logic applies (venv + `subprocess` calling `python -m evaluate.global_learned.*`).

### What to run (don’t brute-force everything)

You already have strong **seed-42** tables in **`Report_for_writing.md`** §4.3 (PDF: **`Report_for_writing.pdf`**). **Shared splits** fix a *method* issue (fair val IDs across satellite / street / fusion on the street cohort)—they are not an order to re-train every task × every seed × every modality.

**Sensible default**

1. Pick **1–2 headline tasks** where the story matters (e.g. `gdp`, `build_height`—where satellite vs fusion differed in your snapshot).
2. Use **the same `SEEDS`** for split creation and training. **Single seed:** `[42]`. **Two seeds (default in notebook):** `[42, 43]` → **14** train jobs with fusion. **Three seeds:** `[42, 43, 44]` → **21** jobs. See `CLAIMS_AND_REMAINING_RUNS.md`.
3. Run **satellite + street** under `--split_key fusion`; add **fusion** only if you need a *paired* fusion number under shared val IDs (heaviest).

**When to expand**

- High **variance** across seeds on a headline task → keep more seeds there.
- A task is **missing** from the report or unstable → add that task to `TASKS`.
- Reviewer asks for full grid → then widen `TASKS` / enable `RUN_FUSION` / add backbones.

### Colab checklist (any scale)

1. **Runtime → GPU**.
2. Set **`DATA_ROOT`**, **`GIT_REPO`** if needed.
3. Run cells in order: config → mount → venv → splits → train → optional metrics.
4. Splits: **`make_shared_split`** for each `(seed, task)` in your **`TASKS`** list (fast). Training uses the same **`TASKS`**.
5. **`--skip_if_done --resume`**: safe to restart; finished runs are skipped.
