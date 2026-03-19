# Prithvi Satellite Reference

This file locks the current satellite-only `prithvi_rgb_lora` reference results so later baseline runs can be compared fairly without further Prithvi retuning.

## Locked settings

- branch: `satellite`
- satellite model: `prithvi_rgb_lora`
- image size: `224`
- batch size: `8`
- seed: `42`
- target transform: `log1p`
- lr: `2e-4`
- backbone lr: `2e-4`
- head lr: `1e-3`
- weight decay: `1e-2`
- val fraction: `0.1`

## Locked reference metrics

| Task | Epoch budget | Best epoch | R2 | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `gdp` | 20 | 14 | 0.5808 | 331463584.0 |
| `acc2health` | 30 | 9 | 0.3901 | 9.5502 |
| `build_height` | 30 | 9 | 0.8682 | 2.5345 |
| `pop` | 5 | 2 | -0.0324 | 21641.2461 |

## Intended use

- Treat these as the fixed Prithvi baselines for the next comparison phase.
- Compare `dinov2_sat` and `resnet50_sat` against the same task splits and training budgets.
- Only revisit Prithvi if comparison runs reveal a clear protocol mismatch or a reproducibility issue.
