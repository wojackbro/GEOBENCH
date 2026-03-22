# Where the changes are (Gemini integration)

## 1. Repo code (local / Colab after patch)

> **Public repo note:** `CityLens/` and Colab notebooks were removed from this public tree. The table below describes changes that exist in a **full private checkout**.

| File | Change |
|------|--------|
| **`CityLens/evaluate/gemini_adapter.py`** | **New.** Gemini API adapter: `GEMINI_MODEL_NAMES`, `_session_to_gemini_parts()`, `get_response_mllm_api_gemini()`. Uses `GOOGLE_API_KEY` or `GEMINI_API_KEY`. |
| **`CityLens/evaluate/utils.py`** | Imports adapter; at start of `get_response_mllm_api()` returns `get_response_mllm_api_gemini(...)` when `model_name in GEMINI_MODEL_NAMES`. |

**Verify:**
```bash
grep -n "GEMINI_MODEL_NAMES\|get_response_mllm_api_gemini" CityLens/evaluate/utils.py
grep -n "GOOGLE_API_KEY\|get_response_mllm_api_gemini" CityLens/evaluate/gemini_adapter.py
```

## 2. Colab notebook *(removed from public repo; keep in private clone)*

Formerly: `colab_citylens_full.ipynb`.

| Section | Change |
|---------|--------|
| **Section 2** | New cell: “Add Gemini adapter and patch utils.py” – patches `evaluate/utils.py` so Gemini is used when adapter is present. In Colab you must have `evaluate/gemini_adapter.py` (e.g. upload from project or create from `CityLens/evaluate/gemini_adapter.py`). |
| **Section 5** | API key cell asks for **GOOGLE_API_KEY** (Gemini, free at aistudio.google.com/apikey) and sets it. |
| **Eval cells** | `--model_name=gemini-1.5-flash` instead of `gpt-4o`. |

**Verify:** Open the notebook and check Section 5 for “GOOGLE_API_KEY” and the eval cells for “gemini-1.5-flash”.

## 3. Install / deps

| File | Change |
|------|--------|
| **`colab_install_packages.py`** *(private clone only; removed here)* | `"google-generativeai"` added to `PACKAGES`. |
| **Notebook install cell** | Same list includes `google-generativeai` (if using inline package list). |

**Verify (private checkout):** `grep "google-generativeai" colab_install_packages.py`

## 4. Colab: making Gemini work without re-download

- **Data:** Step 3 (download) can be skipped if `/content/CityLens-Data` already exists.
- **Gemini in Colab:** After cloning, `evaluate/gemini_adapter.py` is not in the GitHub repo. So either:
  - **Option A:** Upload `CityLens/evaluate/gemini_adapter.py` from this project into Colab (e.g. into `CityLens/evaluate/`), then run the “Add Gemini adapter” cell to patch `utils.py`, or  
  - **Option B:** In Colab, create `evaluate/gemini_adapter.py` with the same content as in this repo, then run the patch cell.

Then set `GOOGLE_API_KEY`, run install (with `google-generativeai`), and run the eval cells with `model_name=gemini-1.5-flash`.
