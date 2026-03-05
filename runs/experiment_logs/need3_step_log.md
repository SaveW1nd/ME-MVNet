# Need3 Step Log (2026-03-05)

## 1) Code changes
- Removed NF class weighting in joint loss:
  - `src/models/losses_seppe.py`: `F.cross_entropy(..., weight=...)` -> standard `F.cross_entropy(...)`.
- Removed `nf_class_weights` from configs:
  - `configs/train_joint.yaml`
  - `configs/train_joint_cisrj_repro.yaml`
- Added GateFormer module:
  - `src/models/gateformer.py`
  - Includes gate-logit downsampling (`adaptive_avg_pool1d`), transformer encoder, query cross-attention, Tl/NF heads.
- Replaced PE param head with GateFormer:
  - `src/models/penet.py`
  - Kept Raw/TF/Mech branches and gate head; replaced old `fusion_mlp -> tl/nf head` path.
- Updated PE config:
  - `configs/model_pe.yaml`: replaced `fusion` block with `gateformer` block.
- Model export update:
  - `src/models/__init__.py`: export `GateFormer`.

## 2) Unit tests
Command:
```bash
python -m pytest tests/test_penet_shapes.py tests/test_losses.py tests/test_cisrj_sn_shapes.py -q
```
Result:
- 3 passed, 1 warning.

## 3) Joint smoke train
Command:
```bash
python scripts/13_train_seppe_joint.py \
  --sep-ckpt runs/exp_cisrj_sep_dual_formal_v1/checkpoints/best.pt \
  --sep-config configs/model_sep_cisrj_map.yaml \
  --mode smoke \
  --exp-name exp_joint_need3_smoke_v2
```
Result:
- run_dir: `runs/exp_joint_need3_smoke_v2`
- best_epoch: 3
- best_A_total (val): 0.063961
- elapsed_seconds: 121.485

Per-epoch val summary:
- epoch1: A_total=0.0020, NF_acc=0.3830
- epoch2: A_total=0.0147, NF_acc=0.3370
- epoch3: A_total=0.0640, NF_acc=0.4137

## 4) Test evaluation (all/dual/multi)
Common ckpt:
- `runs/exp_joint_need3_smoke_v2/checkpoints/best.pt`

### all
Command:
```bash
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_need3_smoke_v2/checkpoints/best.pt --sep-config configs/model_sep_cisrj_map.yaml --run-dir runs/exp_joint_need3_smoke_v2
```
Overall:
- NF_Acc: 0.4237
- A_Tl: 0.1948
- A_Ts: 0.1584
- A_NF: 0.3696
- A_total: 0.0516

### dual
Command:
```bash
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_need3_smoke_v2/checkpoints/best.pt --sep-config configs/model_sep_cisrj_map.yaml --run-dir runs/exp_joint_need3_smoke_v2 --scenario dual
```
Overall:
- NF_Acc: 0.5080
- A_Tl: 0.2120
- A_Ts: 0.1580
- A_NF: 0.4150
- A_total: 0.0620

### multi
Command:
```bash
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_need3_smoke_v2/checkpoints/best.pt --sep-config configs/model_sep_cisrj_map.yaml --run-dir runs/exp_joint_need3_smoke_v2 --scenario multi
```
Overall:
- NF_Acc: 0.3393
- A_Tl: 0.1833
- A_Ts: 0.1587
- A_NF: 0.3393
- A_total: 0.0447

## 5) Notes
- This is a smoke run (3 epochs), used to verify training/eval chain after architecture switch.
- Formal run is still required for Need3 acceptance criteria comparison against Need2-E2.

## 6) Formal run (Need3 main, same protocol as Need2-E2)
Date: 2026-03-05

Command:
```bash
python scripts/13_train_seppe_joint.py \
  --sep-ckpt runs/exp_sep_formal_datafix_v1/checkpoints/best.pt \
  --sep-config configs/model_sep_sf11_grouped_wide.yaml \
  --pe-config configs/model_pe.yaml \
  --train-config configs/train_joint.yaml \
  --mode formal \
  --exp-name exp_joint_formal_need3_gateformer_v1
```

Artifacts:
- run dir: `runs/exp_joint_formal_need3_gateformer_v1`
- train log: `runs/experiment_logs/need3_joint_formal_v1.log`

Best validation:
- `best_epoch=35`
- `best_A_total=0.446372`

## 7) Test evaluation (all / dual / multi)
Common ckpt:
- `runs/exp_joint_formal_need3_gateformer_v1/checkpoints/best.pt`

### all
- `A_total=0.4520`
- `A_Ts=0.4884`
- `A_Tl=0.6432`
- `A_NF=0.8340`
- `NF_Acc=0.8220`
- `Ts_MAE_us=0.6281`

### dual
- `A_total=0.5510`
- `A_Ts=0.5830`
- `A_Tl=0.7410`
- `A_NF=0.9050`
- `NF_Acc=0.8573`
- `Ts_MAE_us=0.4389`

### multi
- `A_total=0.3860`
- `A_Ts=0.4253`
- `A_Tl=0.5780`
- `A_NF=0.7867`
- `NF_Acc=0.7867`
- `Ts_MAE_us=0.7542`

Eval logs:
- `runs/experiment_logs/need3_joint_eval_all_v1.log`
- `runs/experiment_logs/need3_joint_eval_dual_v1.log`
- `runs/experiment_logs/need3_joint_eval_multi_v1.log`

## 8) Plot/Table export and comparison
- Export cmd:
```bash
python scripts/15_export_plots_seppe.py --run-dir runs/exp_joint_formal_need3_gateformer_v1 --paper-dir paper/need3_v1
```
- Export log: `runs/experiment_logs/need3_joint_export_v1.log`
- Need2-E2 vs Need3 comparison table:
  - `paper/tables/need3_vs_need2e2.csv`

## 9) Quick comparison to Need2-E2 (same protocol)
- all: `A_total 0.4732 -> 0.4520` (down `-0.0212`)
- dual: `A_total 0.5780 -> 0.5510` (down `-0.0270`)
- multi: `A_total 0.4033 -> 0.3860` (down `-0.0173`)
- but `NF_Acc` increased in all/dual; `Ts_MAE` improved in all/multi.
