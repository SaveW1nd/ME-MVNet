# CISRJ Reproduction Method Test Log (2026-03-05)

## Scope
- Test the reproduced CISRJ separator workflow using:
  - `scripts/20_train_cisrj.py`
  - `scripts/21_eval_cisrj.py`
  - `scripts/22_export_cisrj_report.py`
- Configs:
  - `configs/model_sep_cisrj_repro.yaml`
  - `configs/train_sep_cisrj_repro.yaml`

## Smoke Runs

### Dual Smoke
- Train:
  - `python scripts/20_train_cisrj.py --scenario dual --model-config configs/model_sep_cisrj_repro.yaml --train-config configs/train_sep_cisrj_repro.yaml --mode smoke --exp-name exp_cisrj_repro_dual_smoke_v1`
- Eval:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_repro_dual_smoke_v1/checkpoints/best.pt --scenario dual --model-config configs/model_sep_cisrj_repro.yaml --run-dir runs/exp_cisrj_repro_dual_smoke_v1`
- Result:
  - `SI_SDRi_mean_dB = 5.9831`
  - `SI_SDR_mean_dB = 1.9794`

### Multi Smoke
- Train:
  - `python scripts/20_train_cisrj.py --scenario multi --model-config configs/model_sep_cisrj_repro.yaml --train-config configs/train_sep_cisrj_repro.yaml --mode smoke --exp-name exp_cisrj_repro_multi_smoke_v1`
- Eval:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_repro_multi_smoke_v1/checkpoints/best.pt --scenario multi --model-config configs/model_sep_cisrj_repro.yaml --run-dir runs/exp_cisrj_repro_multi_smoke_v1`
- Result:
  - `SI_SDRi_mean_dB = 5.9168`
  - `SI_SDR_mean_dB = -1.7237`

### Smoke Summary Export
- `python scripts/22_export_cisrj_report.py --dual-run runs/exp_cisrj_repro_dual_smoke_v1 --multi-run runs/exp_cisrj_repro_multi_smoke_v1 --out-csv paper/tables/cisrj_repro_summary_smoke_repro_v1.csv`

## Formal Runs

### Dual Formal
- Train:
  - `python scripts/20_train_cisrj.py --scenario dual --model-config configs/model_sep_cisrj_repro.yaml --train-config configs/train_sep_cisrj_repro.yaml --mode formal --exp-name exp_cisrj_repro_dual_formal_v1`
- Train behavior:
  - early-stopped at epoch 13
  - best at epoch 1
- Eval:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_repro_dual_formal_v1/checkpoints/best.pt --scenario dual --model-config configs/model_sep_cisrj_repro.yaml --run-dir runs/exp_cisrj_repro_dual_formal_v1`
- Result:
  - `SI_SDRi_mean_dB = 6.4833`
  - `SI_SDR_mean_dB = 2.4796`

### Multi Formal
- Train:
  - `python scripts/20_train_cisrj.py --scenario multi --model-config configs/model_sep_cisrj_repro.yaml --train-config configs/train_sep_cisrj_repro.yaml --mode formal --exp-name exp_cisrj_repro_multi_formal_v1`
- Train behavior:
  - early-stopped at epoch 13
  - best at epoch 1
- Eval:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_repro_multi_formal_v1/checkpoints/best.pt --scenario multi --model-config configs/model_sep_cisrj_repro.yaml --run-dir runs/exp_cisrj_repro_multi_formal_v1`
- Result:
  - `SI_SDRi_mean_dB = 8.9150`
  - `SI_SDR_mean_dB = 1.2746`

### Formal Summary Export
- `python scripts/22_export_cisrj_report.py --dual-run runs/exp_cisrj_repro_dual_formal_v1 --multi-run runs/exp_cisrj_repro_multi_formal_v1 --out-csv paper/tables/cisrj_repro_summary_formal_repro_v1.csv`

## Key Conclusion
- Reproduction pipeline is executable end-to-end.
- Current reproduced-formal separation remains below paper Table-4 references:
  - Dual gap: `-14.42 dB` (to 20.9 dB)
  - Multi gap: `-8.88 dB` (to 17.8 dB)
- Both formal runs converged to best at epoch 1, indicating optimization instability / mismatch with paper training dynamics.
