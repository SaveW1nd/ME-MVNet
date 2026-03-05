# CISRJ Repro Test on Scenario-Specific Datasets (2026-03-05)

## Objective
- Test whether using dual-only and multi-only datasets narrows the gap to paper SI-SDRi.

## Data configs
- dual-only: `configs/data_composite_dual.yaml` (`output_dir=data/raw/cisrj_seppe_dual_v1`)
- multi-only: `configs/data_composite_multi.yaml` (`output_dir=data/raw/cisrj_seppe_multi_v1`)

## Model/train configs
- `configs/model_sep_cisrj_repro.yaml`
- `configs/train_sep_cisrj_repro.yaml`

## Runs

### Dual formal
- Train:
  - `python scripts/20_train_cisrj.py --scenario dual --data-config configs/data_composite_dual.yaml --model-config configs/model_sep_cisrj_repro.yaml --train-config configs/train_sep_cisrj_repro.yaml --mode formal --exp-name exp_cisrj_repro_dual_formal_ds_v2`
- Eval:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_repro_dual_formal_ds_v2/checkpoints/best.pt --scenario dual --data-config configs/data_composite_dual.yaml --model-config configs/model_sep_cisrj_repro.yaml --run-dir runs/exp_cisrj_repro_dual_formal_ds_v2`
- Result:
  - `SI_SDRi_mean_dB=6.3018`
  - `SI_SDR_mean_dB=2.4829`
  - active sources: `2000`

### Multi formal
- Train:
  - `python scripts/20_train_cisrj.py --scenario multi --data-config configs/data_composite_multi.yaml --model-config configs/model_sep_cisrj_repro.yaml --train-config configs/train_sep_cisrj_repro.yaml --mode formal --exp-name exp_cisrj_repro_multi_formal_ds_v2`
- Eval:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_repro_multi_formal_ds_v2/checkpoints/best.pt --scenario multi --data-config configs/data_composite_multi.yaml --model-config configs/model_sep_cisrj_repro.yaml --run-dir runs/exp_cisrj_repro_multi_formal_ds_v2`
- Result:
  - `SI_SDRi_mean_dB=8.6730`
  - `SI_SDR_mean_dB=1.2705`
  - active sources: `3000`

## Comparison to previous mixed-data formal repro
- Previous (mixed-data, v1):
  - dual: `6.4833`
  - multi: `8.9150`
- Scenario-specific (this run):
  - dual: `6.3018`
  - multi: `8.6730`
- Observation:
  - No improvement from scenario-specific datasets; performance slightly lower.

## Export
- `paper/tables/cisrj_repro_summary_formal_repro_splitds_v2.csv`
