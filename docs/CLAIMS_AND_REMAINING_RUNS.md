# What you can claim **now** vs proof you still need

**Primary proof = what you already put in the repo:** `docs/PRITHVI_SATELLITE_REFERENCE.md`, `docs/GLOBAL_LEARNED_PIPELINE.md`, and **`docs/Report_for_writing.md`** §4.3 (seed-42 tables merged into the manuscript). PDF export: **`docs/Report_for_writing.pdf`**. Training code and Colab notebooks are **not** in this public tree.

**`metrics.json` on disk** is *not* required for that. It is only useful if you (or a reviewer) want to **reconcile raw run outputs** to the published numbers, or to **check split filenames** in `config.json` when arguing that old runs were already paired. The claims analysis below uses **your documented results**, not hidden files.

---

## 1) Inventory: runs that are **already documented** here

| Modality | Model(s) | Seed | Tasks with numbers in-repo | Where |
|----------|----------|------|----------------------------|--------|
| **Satellite** | `prithvi_rgb_lora` | **42** | `gdp`, `acc2health`, `build_height`, `pop` | `PRITHVI_SATELLITE_REFERENCE.md` (full protocol + table) |
| **Satellite** | `prithvi_rgb_lora` | 42 | **`carbon` — not listed** | — |
| **Street** | `resnet50` (and `dinov2_vitb14` on `pop`) | **42** (implied by pipeline narrative) | `gdp`, `acc2health`, `build_height`, `pop` | `GLOBAL_LEARNED_PIPELINE.md` (R² snapshot only) |
| **Fusion + full grids** | e.g. Prithvi + DINOv2 street + `late` | **42** (per PDF) | Detailed in `Report_for_writing.md` §4.3 / PDF | `Report_for_writing.pdf` |

So: **seed-42 Prithvi satellite** on **four** tasks is the most **detailed** lock file in git. Street/fusion are still **fully citable** via the **PDF** (and the pipeline summary); the only methodological caveat is **paired val IDs** across modalities (see §3), not “missing metrics files.”

---

## 2) Claims you can make with **full** strength *today*

These do **not** require shared splits or extra seeds, *provided* you cite the locked protocol and do not mix in `carbon` for Prithvi.

| Claim | Proof in-repo? |
|--------|----------------|
| Under the listed hyperparameters and **seed 42**, **`prithvi_rgb_lora` satellite** achieves the stated **R² / RMSE** on **`gdp`, `acc2health`, `build_height`, `pop`** | **Yes** — `PRITHVI_SATELLITE_REFERENCE.md` |
| **`pop`** is a **hard** task for the current satellite setup (negative / ~zero R² in that table) | **Yes** — same file |
| You **have not** locked a comparable Prithvi row for **`carbon`** in this repo | **Yes** — absence in lock file |

Anything **beyond satellite-only Prithvi** at seed 42 should be phrased as “we also report …” with a pointer to the **PDF** and/or Drive artifacts, not as tightly locked as the Prithvi table.

---

## 3) Claims that are **only partial / hedged** with current proof

| Claim | Why it’s not “full” yet |
|--------|-------------------------|
| “**Satellite beats street** on `gdp` / `acc2health`” | Satellite numbers are on the **satellite-eligible** pool; street numbers in the pipeline doc are on the **street-enabled subset** (different **N** and, by default, **different val folds** unless you forced one split file). That’s a **directional** comparison, not a strict paired test. |
| “**Fusion beats satellite** on `build_height`” | Same issue: needs **identical val IDs** on the **same** cohort (street-available) to be a clean sentence. |
| “**Robust across random seeds**” | Everything above is **seed 42–centric** in the docs you committed. One seed does **not** support variance claims. |
| “**Carbon** follows the same story as other tasks” | No locked Prithvi row here; PDF may have more—treat as **separate** until aligned with a lock file or appendix table. |

**Optional:** If your PDF tables were produced with **`--split_key fusion`** (or the same split JSON path) for **every** row you compare, cross-modal ordering may already be fair. **Check on Drive:** in each experiment’s `config.json` / logs, the split path should be the same stem, e.g. `.../splits/{task}_fusion_seed42_val0.1.json`, not `{task}_satellite_...` vs `{task}_street_...`.

---

## 4) What you **do not** have enough proof for (without new runs)

| Claim | Missing proof |
|--------|----------------|
| **Strict paired** satellite vs street vs fusion (same val fold, street cohort) | Shared split file + all branches trained with **`--split_key ...`** and same `(task, seed, val_frac)` |
| **Multi-seed stability** (ordering holds on 43, 44, …) | At least **two more seeds** with the **same** protocol |
| **Tight Prithvi-on-carbon** | A run + row in `PRITHVI_SATELLITE_REFERENCE.md` (or appendix) with protocol |

---

## 5) Remaining runs — **default = smallest budget for the strongest *new* claims**

**“New” claims** here = strict **paired** satellite vs street (and optionally fusion) on the **street cohort** with **identical val IDs** (`make_shared_split` + `train.py --split_key fusion`). Your **old** satellite-only Prithvi table still stands on its own.

### Recommended minimum (seed **42** only) — **6 or 7** training runs

| Package | Train jobs | Splits (CPU) | What you can say after |
|---------|------------|--------------|-------------------------|
| **A — Paired sat vs street** | **6** = 3 tasks × 1 seed × 2 branches | 3 JSONs (one seed) | On **`gdp`, `acc2health`, `build_height`**, Prithvi **vs** ResNet-50 street, **same val fold**, street cohort, **seed 42**. |
| **B — + Paired fusion on height** | **+1** → **7** total | same 3 JSONs | Same as A, plus **late fusion** (Prithvi + DINOv2 street) on **`build_height`** only, **same val fold**, seed 42. |

What **6–7 does *not* buy:** any **multi-seed** wording — use **two seeds** (below) or three (extended).

**Cheapest multi-seed:** `SEEDS = [42, 43]` with the same `TASKS` and fusion on `build_height` only → **14** training runs (double the 7-run plan):  
`3 tasks × 2 seeds × 2 branches + 1 fusion task × 2 seeds = 12 + 2 = 14`.  
You can report **both** seeds and note **agreement / spread** (still not a large Monte Carlo study—phrase modestly).

**Notebook default** is **`SEEDS = [42, 43]`** + fusion on → **14** runs. **`RUN_FUSION = False`** → **12** runs (paired **sat vs street** only, two seeds)—you **cannot** add a *new* paired fusion row for `build_height` from that run; cite the **PDF** for fusion or turn fusion on (+2 runs).

Use `SEEDS = [42]` only for **7** runs (one seed + fusion); **6** if fusion off.

### Extended (three seeds)

- `SEEDS = [42, 43, 44]` → **18 + 3 = 21** training runs (same task/branch setup).
- Slightly stronger multi-seed paragraph than two seeds alone.

### Split files

- For each `(task, seed)` in `TASKS × SEEDS`: `{task}_fusion_seed{seed}_val0.1.json` (e.g. six JSONs for two seeds × three tasks).

### Optional extras

| If you need… | Add |
|--------------|-----|
| **`pop`** in the paired story | Add `pop` to `TASKS` (+2 train jobs per seed per sat/street; fusion optional) |
| **`carbon` + Prithvi** | Satellite-only run(s) + update `PRITHVI_SATELLITE_REFERENCE.md` |
| Fusion on **every** task | `FUSION_TASKS = None` in the notebook |

---

## 6) Notebook = executable checklist

In the **private** notebook / repo, the paired multi-seed driver defaults to **two-seed + fusion → 14 runs** (`SEEDS = [42, 43]`). Use `[42]` for **7** runs; `[42, 43, 44]` for **21** runs.
