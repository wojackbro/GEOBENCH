# GeoBench / CityLens — Geospatial foundation models on urban regression

We benchmark **geospatial foundation models** on **urban regression** using the **CityLens** global tasks (GDP proxy, healthcare accessibility, building height, population, and related indicators). The scientific through-line is twofold: **aligned evaluation**—the same validation place IDs for satellite, street-view, and fusion—and **honest uncertainty**—multi-seed reporting instead of a single lucky seed.

**Canonical manuscript:** `paper_curent_draft.docx`.

---

## Technical workflow

1. **CityLens pipeline**  
   Task JSONs provide targets, one satellite image, and multiple street-view images per location. Experiments are organized by **branch**: satellite-only, street-only, and **fusion** (e.g. late or gated encoder combination).

2. **Prithvi as the geospatial foundation-model anchor**  
   Prithvi-EO 2.0 is pretrained in an EO setting; we treat CityLens as an **adaptation** problem—RGB inputs, channel handling, and **LoRA** fine-tuning—rather than native multispectral HLS evaluation. **Generic vision baselines** (DINOv2, CLIP-family ViTs, ResNet) provide contrast on both modalities.

3. **Training environment**  
   Most training runs in **Google Colab** with Drive mounted as `CITYLENS_DATA_ROOT`, a Python **venv**, and CLI entrypoints under `evaluate.global_learned`. Artifacts land under `Results/global_learned/` (metrics, checkpoints, histories, **split JSONs**).

4. **Shared splits**  
   We use **`make_shared_split`-style** JSON so satellite, street, and fusion are scored on the **same held-out place IDs** on the **street-available cohort** (“paired” / **shared-split** protocol). That alignment is what makes cross-modality ordering defensible.

5. **Multi-seed evaluation**  
   We report aggregates across seeds (e.g. mean ± standard deviation) and treat **two seeds** as evidence of spread, not as a large Monte Carlo study.

6. **Unimodal vs multimodal evaluation**  
   We compare **satellite-only** and **street-only** models directly, then **multimodal** models that combine both streams—so conclusions distinguish “which backbone” from “whether fusion helps.”

7. **Single-branch vs fusion training**  
   Unimodal runs train one branch end-to-end; **fusion** runs train a joint head (e.g. late fusion) so improvements reflect **learned integration**, not a post-hoc ensemble picked on the validation set.

---

## Results, discussion, and conclusions

We study **neighbourhood-scale urban regression from imagery** on CityLens with **Prithvi-EO 2.0 + LoRA** as our primary geospatial FM treatment (**RGB adaptation**, not native multispectral HLS), against generic encoders on **satellite-only**, **street-only**, and **late fusion** setups. **Fair cross-modality comparison requires identical validation units** on the street-available subset; otherwise apparent gains can reflect **split mismatch** or **different effective cohort sizes**, not model quality.

Under our **documented seed-42** grids (full tables in the manuscript and local report PDFs), **satellite Prithvi is strong on GDP and healthcare-related tasks** in that snapshot; **fusion helps most clearly on building height**; **population density remains very difficult** in our RGB setting; and **fusion can hurt** on the hardest cases. That portrait is a useful **single-seed benchmark**; it is **not** synonymous with **cross-seed robustness**.

We then completed a **paired, shared-split, two-seed study** (seeds **42** and **43**) on **GDP**, **healthcare accessibility**, and **building height**, with **fusion trained only for building height** (**14** paired training jobs). With **mean ± standard deviation** across seeds:

- **Building height:** **fusion** (Prithvi satellite + DINOv2 street, late fusion) **outperforms both unimodal branches on average** (mean R² ordering: fusion > satellite > street for that configuration).
- **Healthcare accessibility:** **street (ResNet) outperforms satellite Prithvi on average**, with **large seed-to-seed spread** on the street side.
- **GDP:** **high variance** and **unstable satellite–street ordering** across seeds (including **negative street R²** for one seed in our logs). We **do not** claim a robust modality winner for GDP from **two seeds** alone.

**Broader inferences.** Fusion is **not universally beneficial**—value is **task-dependent**. **Single-seed** geospatial or vision benchmarks can **mis-rank** models when ranking is seed-sensitive. Prithvi-on-CityLens should be framed as **adaptation**, not as a full **native EO spectral** study.

**Methodological takeaway:** we aim to make results easier to audit—**shared validation IDs**, explicit **street-available cohort** definition, **multi-seed uncertainty**, and a **clean separation** between **paired multi-seed rows** and older **legacy / non-paired / different-epoch** exports.

---

## Repository notes

- The **`docs/`** directory is **gitignored** on the public remote (manuscript tables, supplementary PDFs, render scripts stay local). **`paper_curent_draft.docx`** remains the primary writing source unless you ignore it separately.
- **`CityLens/evaluate/global_learned/*.py`** may be gitignored so the public tree stays documentation-forward; use a **private** checkout for full training code.

**Data & upstream benchmark**

- CityLens dataset (example): [CityLens-Data on Hugging Face](https://huggingface.co/datasets/Tianhui-Liu/CityLens-Data)  
- Prithvi family: [ibm-nasa-geospatial on Hugging Face](https://huggingface.co/ibm-nasa-geospatial)
