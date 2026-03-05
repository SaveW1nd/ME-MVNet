# Scenario-Split Evaluation Framework Log (2026-03-05)

## Goal
- Keep the mixed training framework as default.
- Add a safe evaluation workflow that reports metrics separately for `dual` and `multi`.

## Rollback and baseline protection
- Removed dual-only configs from active `configs/`:
  - `configs/train_sep_cisrj_focus_dual_v1.yaml`
  - `configs/train_sep_cisrj_focus_dual_ft_v2.yaml`
- Backups preserved at:
  - `runs/framework_snapshots/train_sep_cisrj_focus_dual_v1.yaml.bak`
  - `runs/framework_snapshots/train_sep_cisrj_focus_dual_ft_v2.yaml.bak`
- Mixed-framework snapshot saved at:
  - `runs/framework_snapshots/20260305_mixed_v1`
  - note: `runs/experiment_logs/mixed_framework_snapshot_20260305.md`

## Code changes

### 1) Split evaluation support in script 14
- File: `scripts/14_eval_seppe.py`
- Added:
  - `--scenario {all,dual,multi}` argument
  - scenario subset filtering by `K_active`
  - scenario-aware output directories:
    - `tables/scenario_dual/`
    - `tables/scenario_multi/`
    - `predictions/scenario_dual/`
    - `predictions/scenario_multi/`
- Default behavior remains unchanged when `--scenario all`.

### 2) One-click split evaluation script
- File: `scripts/23_eval_seppe_by_scenario.py`
- Function:
  - calls script 14 for selected scenarios (default `dual,multi`)
  - aggregates results into:
    - `tables/test_metrics_by_scenario.csv`
    - `tables/test_metrics_by_scenario.json`

## Validation run
- Command:
  - `python scripts/23_eval_seppe_by_scenario.py --ckpt runs/exp_joint_formal_need2_e2_v1/checkpoints/best.pt --split test --run-dir runs/exp_joint_formal_need2_e2_v1 --sep-config configs/model_sep_sf11_grouped_wide.yaml`
- Log:
  - `runs/experiment_logs/scenario_split_eval_need2_e2_v1.log`
- Output summary:
  - `runs/exp_joint_formal_need2_e2_v1/tables/test_metrics_by_scenario.csv`

## Current split results (E2 test)
- dual:
  - `A_total=0.5780`
  - `Tl_MAE_us=0.14045`
  - `Ts_MAE_us=0.41444`
- multi:
  - `A_total=0.40333`
  - `Tl_MAE_us=0.22330`
  - `Ts_MAE_us=0.78240`
