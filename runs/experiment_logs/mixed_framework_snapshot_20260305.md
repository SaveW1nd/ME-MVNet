# Mixed Framework Snapshot

- Snapshot time: 2026-03-05
- Snapshot id: 20260305_mixed_v1
- Purpose: preserve current mixed 2ISRJ+3ISRJ baseline before scenario-split evaluation experiments.

## Canonical mixed settings
- Dataset mix control: `configs/data_composite.yaml` -> `grid.k_active_dual_ratio: 0.5`
- Separation config: `configs/train_sep_cisrj_map.yaml`
- Joint config: `configs/train_joint.yaml`
- Eval config: `configs/eval_composite.yaml`

## Rolled back from dual-only attempt
- Moved out of active config dir:
  - `configs/train_sep_cisrj_focus_dual_v1.yaml`
  - `configs/train_sep_cisrj_focus_dual_ft_v2.yaml`
- Backup location:
  - `runs/framework_snapshots/train_sep_cisrj_focus_dual_v1.yaml.bak`
  - `runs/framework_snapshots/train_sep_cisrj_focus_dual_ft_v2.yaml.bak`

## Snapshot path
- `runs/framework_snapshots/20260305_mixed_v1`
