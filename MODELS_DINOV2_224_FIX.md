# Fix: DINOv2 + fusion crash (`Input height (224) doesn't match model (518)`)

**Cause:** `timm`’s `vit_base_patch14_dinov2.lvd142m` defaults to **518×518** patches. CityLens trains street/satellite at **224×224**, so `materialize_model` / fusion forward fails on the street tower.

**Fix:** Build DINOv2 with `img_size=224` in `TimmEncoder` (see `CityLens/evaluate/global_learned/models.py`).

**Colab / GeoBench:** Copy this repo’s `CityLens/evaluate/global_learned/models.py` over:

`/content/GeoBench/CityLens/evaluate/global_learned/models.py`

Then rerun training.

**Optional:** `pip install --upgrade albumentations` and set `HF_TOKEN` for faster Hugging Face downloads (unrelated to this error).
