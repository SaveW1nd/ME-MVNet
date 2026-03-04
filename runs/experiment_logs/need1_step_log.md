# need1 实验执行日志

- 日期: 2026-03-04
- 目标: 按 `need1.md` 分步验证 SepNet 三项改动，并完成稳定性修复与复现实验。
- 统一日志目录: `runs/experiment_logs/`

## 0) Baseline（改动前）

### Stage-1 SepNet
- 命令: `python scripts/12_train_sepnet.py --mode formal --exp-name exp_sep_formal_v1`
- 结果: best_epoch=50, best_val(L_sep)=2.592068
- 指标: best_val(SI_SDR_jam)=-4.875344
- 日志: `runs/exp_sep_formal_v1/logs/train.log`

### Stage-2 Joint + Test
- 命令: `python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal_v1/checkpoints/best.pt --mode formal --exp-name exp_joint_formal_v1`
- 结果: best_epoch=34, best_val(A_total)=0.232556
- Test: A_total=0.2244, NF_Acc=0.6737, Tl_MAE_us=0.28647, Ts_MAE_us=1.21453
- 指标文件: `runs/exp_joint_formal_v1/tables/test_metrics.json`

## 1) Step-1（只做改动1+2）

- 改动1: `src/models/sepnet.py` masks `sigmoid -> softmax(dim=1)`
- 改动2: `src/models/sepnet.py` residual 均分 -> 按4路能量比例分配

### Stage-1
- 命令: `python scripts/12_train_sepnet.py --mode formal --exp-name exp_sep_formal_step12_v1`
- 结果: best_epoch=49, best_val(L_sep)=-0.775763
- 指标: best_val(SI_SDR_jam)=-5.455174
- 结论: 比 baseline 更差。
- 日志: `runs/exp_sep_formal_step12_v1/logs/train.log`

## 2) Step-2（加入改动3）

- 改动3: `src/models/losses_seppe.py` background loss: `-SI_SDR -> 0.1*MSE`

### Stage-1
- 命令: `python scripts/12_train_sepnet.py --mode formal --exp-name exp_sep_formal_step123_v1`
- 结果: best_epoch=50, best_val(L_sep)=11.498278
- 指标: best_val(SI_SDR_jam)=-4.423282（较 baseline 提升）
- 现象: 中途出现过 NaN 日志（未加防护时）。
- 日志: `runs/exp_sep_formal_step123_v1/logs/train.log`

### Stage-2
- 命令: `python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal_step123_v1/checkpoints/best.pt --mode formal --exp-name exp_joint_formal_step123_v1`
- 结果: best_epoch=28, best_val(A_total)=0.178723
- 现象: 后半程多次 NaN，性能明显下降。
- 日志: `runs/exp_joint_formal_step123_v1/logs/train.log`

### Test
- 指标文件: `runs/exp_joint_formal_step123_v1/tables/test_metrics.json`
- Test: A_total=0.1776, NF_Acc=0.6530, Tl_MAE_us=0.30350, Ts_MAE_us=1.20458

## 3) Step-3（稳定性修复 + 重跑）

### 3.1 代码修复
- `src/models/sisdr.py`: `si_sdr` 输出 `nan_to_num` 限幅。
- `src/models/pit_perm.py`: `active_cost/silence_cost` 增加 `nan_to_num`。
- `src/models/sepnet.py`: 能量权重与输出源增加 `nan_to_num`。
- `src/models/penet.py`, `src/models/memvnet.py`: 机制分支特征提取改为 `float32`，增大 eps，增加 `nan_to_num`。
- `src/train/trainer_sepnet.py`, `src/train/trainer_seppe.py`:
  - 增加 non-finite loss 跳过逻辑。
  - 增加 `N_skip` 统计并写入 epoch 日志。
- 新增稳定配置:
  - `configs/train_sep_need1_stable.yaml`（`amp=false`, `lr=8e-4`）
  - `configs/train_joint_need1_stable.yaml`（`amp=false`, `lr_sep=2e-4`, `lr_pe=8e-4`）

### 3.2 Smoke 验证
- 命令: `python scripts/12_train_sepnet.py --train-config configs/train_sep_need1_stable.yaml --mode smoke --exp-name exp_sep_smoke_step123_stable_v1`
- 结果: best_epoch=3, best_val(L_sep)=25.399483
- 关键日志: 全程 `skip=0`
- 原始输出: `runs/experiment_logs/need1_sep_smoke_stable_v1.log`

### 3.3 Stage-1 Formal（稳定版）
- 开始时间: 2026-03-04 16:26:01
- 命令: `python scripts/12_train_sepnet.py --train-config configs/train_sep_need1_stable.yaml --mode formal --exp-name exp_sep_formal_step123_stable_v1`
- 结果: best_epoch=49, best_val(L_sep)=11.740825
- 指标: best_val(SI_SDR_jam)=-4.462929
- 关键日志: 50 epoch 全程 `skip=0`，无 NaN
- 训练日志: `runs/exp_sep_formal_step123_stable_v1/logs/train.log`
- 原始输出: `runs/experiment_logs/need1_sep_formal_stable_v1.log`

### 3.4 Stage-2 Formal（稳定版）
- 开始时间: 2026-03-04 16:31:48
- 命令: `python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal_step123_stable_v1/checkpoints/best.pt --train-config configs/train_joint_need1_stable.yaml --mode formal --exp-name exp_joint_formal_step123_stable_v1`
- 结果: best_epoch=21, best_val(A_total)=0.231874
- 关键日志: 全程 `skip=0`，无 NaN
- 训练日志: `runs/exp_joint_formal_step123_stable_v1/logs/train.log`
- 原始输出: `runs/experiment_logs/need1_joint_formal_stable_v1.log`

### 3.5 Test 评估 + 图表导出
- 评估命令: `python scripts/14_eval_seppe.py --ckpt runs/exp_joint_formal_step123_stable_v1/checkpoints/best.pt --split test --run-dir runs/exp_joint_formal_step123_stable_v1`
- 导出命令: `python scripts/15_export_plots_seppe.py --run-dir runs/exp_joint_formal_step123_stable_v1 --paper-dir paper`
- Test 指标:
  - A_total=0.2296
  - NF_Acc=0.6607
  - Tl_MAE_us=0.28025
  - Ts_MAE_us=1.13462
- 指标文件: `runs/exp_joint_formal_step123_stable_v1/tables/test_metrics.json`
- 原始输出:
  - `runs/experiment_logs/need1_joint_eval_stable_v1.log`
  - `runs/experiment_logs/need1_joint_export_stable_v1.log`

## 4) 对比结论

- 相比 Step-2（未稳定）:
  - Stage-2 val A_total: `0.178723 -> 0.231874`（显著恢复）
  - Test A_total: `0.1776 -> 0.2296`（显著恢复）
- 相比 baseline:
  - Test A_total: `0.2244 -> 0.2296`（+0.0052）
  - Test Tl_MAE_us: `0.28647 -> 0.28025`（改善）
  - Test Ts_MAE_us: `1.21453 -> 1.13462`（改善）
  - Test NF_Acc: `0.6737 -> 0.6607`（小幅下降）
- 当前最稳配置下，训练过程已无 NaN/Inf 污染（`skip=0`）。
- need1 的三项核心改动在“稳定性修复后”可实现总体指标提升，但 `SI_SDR_jam` 仍未达到 `>= 0 dB` 目标，后续需针对分离主干继续优化。

## 5) 本轮产物清单

- 新增配置:
  - `configs/train_sep_need1_stable.yaml`
  - `configs/train_joint_need1_stable.yaml`
- 关键新实验目录:
  - `runs/exp_sep_smoke_step123_stable_v1`
  - `runs/exp_sep_formal_step123_stable_v1`
  - `runs/exp_joint_formal_step123_stable_v1`
- 聚合日志:
  - `runs/experiment_logs/need1_step_log.md`
- 原始命令日志:
  - `runs/experiment_logs/need1_sep_smoke_stable_v1.log`
  - `runs/experiment_logs/need1_sep_formal_stable_v1.log`
  - `runs/experiment_logs/need1_joint_formal_stable_v1.log`
  - `runs/experiment_logs/need1_joint_eval_stable_v1.log`
  - `runs/experiment_logs/need1_joint_export_stable_v1.log`
