# need2 实验执行日志（坐标统一修复）

- 日期: 2026-03-04
- 目标: 修复 `X` 标准化与 `J` 未同尺度的监督不一致问题，并验证对 `SI_SDR_jam` 与 `A_total` 的影响。
- 代码改动: `src/data/dataset_npz_composite.py`
- 运行配置: 复用稳定配置 `configs/train_sep_need1_stable.yaml` 与 `configs/train_joint_need1_stable.yaml`

## 修复说明
- 之前: `X -> (X-mean)/std`，`J` 保持原尺度。
- 现在: `X -> (X-mean)/std`，同时 `J -> J/std`（同一通道标准差）。
- 目的: 保证分离监督与 mixture-consistency 在同一坐标系中。

## Stage-1 Smoke
- 时间: 2026-03-04 17:24:16
- 命令: python scripts/12_train_sepnet.py --train-config configs/train_sep_need1_stable.yaml --mode smoke --exp-name exp_sep_smoke_need2_normframe_v1

## Stage-1 Smoke（active-SI-SDR loss）
- 时间: 2026-03-04 17:26:20
- 命令: python scripts/12_train_sepnet.py --train-config configs/train_sep_need1_stable.yaml --mode smoke --exp-name exp_sep_smoke_need2_activejam_v1

## Stage-1 Formal（active-SI-SDR loss）
- 时间: 2026-03-04 17:27:04
- 命令: python scripts/12_train_sepnet.py --train-config configs/train_sep_need1_stable.yaml --mode formal --exp-name exp_sep_formal_need2_activejam_v1

## Stage-2 Formal（active-SI-SDR loss）
- 时间: 2026-03-04 17:31:53
- 命令: python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal_need2_activejam_v1/checkpoints/best.pt --train-config configs/train_joint_need1_stable.yaml --mode formal --exp-name exp_joint_formal_need2_activejam_v1

## Test 评估（need2）
- 时间: 2026-03-04 17:45:49
- 命令: python scripts/14_eval_seppe.py --ckpt runs/exp_joint_formal_need2_activejam_v1/checkpoints/best.pt --split test --run-dir runs/exp_joint_formal_need2_activejam_v1

## 图表导出（need2）
- 时间: 2026-03-04 17:46:16
- 命令: python scripts/15_export_plots_seppe.py --run-dir runs/exp_joint_formal_need2_activejam_v1 --paper-dir paper

## 结果汇总

### Stage-1 结论
- `need2`（active-SI-SDR）Stage1 formal: best `SI_SDR_jam=-4.9395 dB`。
- 与上一轮稳定最优（`-4.4629 dB`）相比未提升。
- 仅做“坐标统一修复”在 smoke 中也未显示正向趋势（`-10.17 dB`）。

### Stage-2 验证与 Test
- Stage2 val best `A_total=0.232380`（接近 baseline `0.232556`，略高于上一轮稳定 `0.231874`）。
- Test（need2）:
  - `A_total=0.2256`
  - `NF_Acc=0.6747`
  - `Tl_MAE_us=0.2780`
  - `Ts_MAE_us=1.1375`

### 与历史结果对比
- 对比 baseline test (`A_total=0.2244`): `+0.0012`（小幅提升）
- 对比上一轮稳定最优 test (`A_total=0.2296`): `-0.0040`（下降）
- 结论: need2 这组改动没有刷新当前最优 test A_total，也未改善 Stage1 `SI_SDR_jam`。

## 产物路径
- Stage1 smoke: `runs/exp_sep_smoke_need2_normframe_v1`
- Stage1 formal: `runs/exp_sep_formal_need2_activejam_v1`
- Stage2 formal: `runs/exp_joint_formal_need2_activejam_v1`
- 原始日志:
  - `runs/experiment_logs/need2_sep_smoke_v1.log`
  - `runs/experiment_logs/need2_sep_smoke_activejam_v1.log`
  - `runs/experiment_logs/need2_sep_formal_activejam_v1.log`
  - `runs/experiment_logs/need2_joint_formal_activejam_v1.log`
  - `runs/experiment_logs/need2_joint_eval_activejam_v1.log`
  - `runs/experiment_logs/need2_joint_export_activejam_v1.log`
