# Experiment Notes

## Environment

- Date: 2026-03-03
- Python: 3.12.3
- PyTorch: 2.5.1+cu121
- GPU: NVIDIA GeForce RTX 4090 (24GB)

## Commands

```bash
python scripts/01_generate_dataset.py --config configs/data.yaml
python scripts/05_sanity_check_dataset.py --config configs/data.yaml
python scripts/02_train.py --mode smoke --exp-name exp_001_smoke_v2
python scripts/02_train.py --mode formal --exp-name exp_002_formal
python scripts/03_eval.py --ckpt runs/exp_002_formal/checkpoints/best.pt --split test
python scripts/04_export_plots.py --run-dir runs/exp_002_formal --split test
```

## Dataset Sanity (passed)

- Split sizes: train/val/test = 4200 / 900 / 900
- NF balance per split: train 1400 each, val/test 300 each
- Shape: `X=(B,2,4000)`, `mask=(B,4000)`
- Physical relation check: `max |Tf - NF*Tl| = 0`

## Smoke Run (exp_001_smoke_v2)

- Epochs: 3
- Best val loss: 0.044723
- Test metrics:
  - Tl_MAE = 1.1216e-03
  - Tl_RMSE = 2.7499e-03
  - Tf_MAE = 2.1517e-03
  - Tf_RMSE = 4.3774e-03
  - NF_Acc = 0.9878
  - NF_MacroF1 = 0.9878

## Formal Run (exp_002_formal)

- Epochs: 60
- Best epoch: 56
- Best val loss: 0.009526
- Test metrics:
  - Tl_MAE = 1.1751e-05 s
  - Tl_RMSE = 1.8066e-05 s
  - Tf_MAE = 2.6808e-05 s
  - Tf_RMSE = 3.7967e-05 s
  - NF_Acc = 1.0000
  - NF_MacroF1 = 1.0000

## Additional Analysis (formal test set)

- Physics consistency:
  - mean |Tf_hat - NF_pred*Tl_hat| = 1.7970e-06 s
  - RMSE = 6.8148e-06 s
- Mask quality (`threshold=0.5`):
  - mean IoU = 0.9875
  - p10 IoU = 0.9690
  - min IoU = 0.7265

## Outputs

- Run artifacts: `runs/exp_002_formal/`
- Paper figures: `paper/figures/`
- Paper tables: `paper/tables/`
