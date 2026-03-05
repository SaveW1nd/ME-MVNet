# Hardfix Mainline + Single-Variable (Periodic Cond) Experiment Log

Date: 2026-03-06

## Objective
- Roll back PE head to `need3_hardfix` mainline behavior.
- Keep only one new mechanism for A/B: periodic autocorr embedding as extra conditioning input.
- Use real-time monitor with 18-epoch gate before full formal.

## Code Changes
- `src/models/gateformer.py`
  - Restored hardfix-style 2-query head (Tl + NF).
  - Removed anchor residual decoding, Ts direct/struct branches, struct-NF fusion.
  - Added one optional input: `z_periodic` (single added mechanism).
- `src/models/penet.py`
  - Restored hardfix PE flow and Ts decoding via expected NF.
  - Added `PeriodicBranchPE` to produce only `z_periodic` embedding.
  - No anchor/confidence outputs.
- `configs/model_pe.yaml`
  - Restored hardfix defaults.
  - Added `periodic_branch.enabled=true` with `out_dim=64` for one-variable validation.
- `configs/train_joint.yaml`
  - Set `w_phys=0.0`, `w_anchor=0.0` (disabled for this one-variable run).

## Screening (18 epochs)
Command:
```powershell
python scripts/26_monitor_train_joint.py \
  --sep-ckpt runs/exp_sep_formal_datafix_v1/checkpoints/best.pt \
  --train-config runs/tmp_monitor/train_joint_screen18.yaml \
  --mode formal \
  --base-exp-name exp_joint_hardfix1var_screen18_v1 \
  --attempts 3 \
  --seed-list 20260304,20260314,20260324 \
  --bad-check-epoch 18 \
  --min-best-a-total 0.20 \
  --plateau-patience 50 \
  --accept-a-total 0.20 \
  --poll-seconds 5
```
Result:
- attempt01 completed
- best_A_total=0.3385@epoch17, last_A_total=0.3354@epoch18
- passed gate, then moved to full formal

Summary:
- `runs/experiment_logs/exp_joint_hardfix1var_screen18_v1_monitor_20260306_005446.md`
- `runs/experiment_logs/exp_joint_hardfix1var_screen18_v1_a01.log`

## Full Formal (80 epochs upper bound)
Command:
```powershell
python scripts/26_monitor_train_joint.py \
  --sep-ckpt runs/exp_sep_formal_datafix_v1/checkpoints/best.pt \
  --train-config configs/train_joint.yaml \
  --mode formal \
  --base-exp-name exp_joint_hardfix1var_formal_v1 \
  --attempts 1 \
  --seed-list 20260304 \
  --bad-check-epoch 18 \
  --min-best-a-total 0.20 \
  --plateau-patience 12 \
  --accept-a-total 0.45 \
  --poll-seconds 5
```
Result:
- attempt01 completed
- best_A_total=0.4964@epoch58

Summary:
- `runs/experiment_logs/exp_joint_hardfix1var_formal_v1_monitor_20260306_010545.md`
- `runs/experiment_logs/exp_joint_hardfix1var_formal_v1_a01.log`
- model dir: `runs/exp_joint_hardfix1var_formal_v1_a01`

## Test Evaluation (all/dual/multi)
Commands:
```powershell
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_hardfix1var_formal_v1_a01/checkpoints/best.pt --scenario all   --split test --run-dir runs/exp_joint_hardfix1var_formal_v1_a01
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_hardfix1var_formal_v1_a01/checkpoints/best.pt --scenario dual  --split test --run-dir runs/exp_joint_hardfix1var_formal_v1_a01
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_hardfix1var_formal_v1_a01/checkpoints/best.pt --scenario multi --split test --run-dir runs/exp_joint_hardfix1var_formal_v1_a01
```
Logs:
- `runs/experiment_logs/exp_joint_hardfix1var_formal_v1_a01_eval_all.log`
- `runs/experiment_logs/exp_joint_hardfix1var_formal_v1_a01_eval_dual.log`
- `runs/experiment_logs/exp_joint_hardfix1var_formal_v1_a01_eval_multi.log`

## Key Metrics
- all: A_total=0.4948, NF_Acc=0.8180
- dual: A_total=0.5950, NF_Acc=0.8573
- multi: A_total=0.4280, NF_Acc=0.7787

## Compare vs hardfix baseline
Comparison file:
- `paper/tables/need3_hardfix_onevar_periodic_compare.csv`

A_total delta:
- all: +0.0076 (0.4872 -> 0.4948)
- dual: +0.0060 (0.5890 -> 0.5950)
- multi: +0.0087 (0.4193 -> 0.4280)

Observation:
- One-variable periodic conditioning yields a small but consistent A_total improvement across all/dual/multi.
- NF-specific metrics are mixed (A_NF slightly down), indicating the gain mainly comes from Tl/Ts joint tolerance hit behavior rather than pure NF classification gain.
