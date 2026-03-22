# GeoBench (documentation & writing bundle)

This repository is maintained as a **public-facing documentation set** for the CityLens / GeoBench geospatial regression line of work.

**Implementation note:** The **`CityLens/`** codebase, **Colab notebooks** (`.ipynb`), and **`scripts/`** utilities were **removed from this tree** so they are not shared publicly. Keep a **private clone** of your full project (e.g. upstream [GeoBench](https://github.com/wojackbro/GeoBench) / CityLens sources) for training and reproduction.

## Primary writing document

- **`docs/Report_for_writing.md`** — conference-style manuscript shell: abstract, introduction, related work, methodology (all model architectures, **LoRA**, **shared val splits**, **multi-seed**), experiments, discussion placeholders, conclusion, and starter references (with web-backed citations for Prithvi-EO, DINOv2, CLIP, ResNet, LoRA/PEFT).

## PDF (regenerate from markdown)

Run from repo root: **`.venv-pdf/bin/python docs/render_report_pdf.py`** (needs **`pandoc`** + WeasyPrint in `.venv-pdf`).

- **`docs/Report_for_writing.pdf`** — export from **`Report_for_writing.md`**: full manuscript + **complete seed-42 grids** (satellite / street / fusion + cross-branch) in §4.3.
- **`docs/OFFICIAL_REPORT_SATELLITE_PHASE.pdf`** — archived original technical report (same numbers; narrative on failures, XAI, ethics).

## Supporting docs

- `docs/CLAIMS_AND_REMAINING_RUNS.md` — claim boundaries and run plan (references may mention removed notebooks; use private repo to execute).
- `docs/PAIRED_SPLITS_AND_MULTI_SEED.md` — shared split + multi-seed protocol description.
- `docs/GLOBAL_LEARNED_PIPELINE.md` — pipeline narrative (implementation paths refer to removed `CityLens/`; keep for methodology wording).
- `docs/PRITHVI_SATELLITE_REFERENCE.md` — Prithvi-focused notes.

---

*For dataset layout and setup concepts, see `docs/SETUP.md` and `docs/AFTER_DOWNLOAD.md` (they still describe a full `CityLens/` checkout; adapt if you only keep docs publicly).*
