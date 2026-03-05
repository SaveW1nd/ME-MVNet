# Scenario Datasets Build Log (2026-03-05)

## Goal
- Keep existing mixed dataset framework unchanged.
- Add paper-style scenario-specific datasets:
  - dual-only (`K_active=2`)
  - multi-only (`K_active=3`)

## Code changes
- Updated: `src/data/isrj_composite_generator.py`
  - Added optional `grid.k_active_fixed` support (`2` or `3`).
  - Backward compatible: if not provided, still uses mixed mode via `k_active_dual_ratio`.
  - Added `scenario_mode` in saved `meta.json`.

## New configs
- `configs/data_composite_dual.yaml`
  - output: `data/raw/cisrj_seppe_dual_v1`
  - `grid.k_active_fixed: 2`
- `configs/data_composite_multi.yaml`
  - output: `data/raw/cisrj_seppe_multi_v1`
  - `grid.k_active_fixed: 3`

## Commands
- `python scripts/10_generate_dataset_composite.py --config configs/data_composite_dual.yaml`
- `python scripts/10_generate_dataset_composite.py --config configs/data_composite_multi.yaml`

## Build logs
- `runs/experiment_logs/gen_data_composite_dual_v1.log`
- `runs/experiment_logs/gen_data_composite_multi_v1.log`

## Validation
- Dual dataset:
  - train `{2: 5000}`
  - val `{2: 1000}`
  - test `{2: 1000}`
  - slot-3 checks:
    - `NF[:,2]` unique = `[0]`
    - nonzero `J[:,2]` count = `0`
- Multi dataset:
  - train `{3: 5000}`
  - val `{3: 1000}`
  - test `{3: 1000}`
  - slot-3 active count:
    - train `5000`, val `1000`, test `1000`

## Note
- Existing mixed dataset (`data/raw/cisrj_seppe_v1`) is untouched.
