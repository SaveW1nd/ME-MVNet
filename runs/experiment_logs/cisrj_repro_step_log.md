# CISRJ-SN 复现实验执行日志（2026-03-05）

- 目标: 基于当前工程新增的 `cisrj_sn` 分离器完成可运行复现实验，产出 dual/multi 的 smoke 与 formal 分离指标。
- 相关脚本:
  - `scripts/20_train_cisrj.py`
  - `scripts/21_eval_cisrj.py`
  - `scripts/22_export_cisrj_report.py`

## 代码修复

- 时间: 2026-03-05 03:03
- 文件: `scripts/22_export_cisrj_report.py`
- 修复: 增加项目根目录 `sys.path` 注入，解决 `ModuleNotFoundError: No module named 'src'`。

## 实验执行记录

### 1) Dual Smoke 训练
- 命令:
  - `python scripts/20_train_cisrj.py --scenario dual --model-config configs/model_sep_cisrj_map.yaml --train-config configs/train_sep_cisrj_map.yaml --mode smoke --exp-name exp_cisrj_sep_dual_smoke_v1`
- 日志: `runs/experiment_logs/cisrj_sep_dual_smoke_v1.log`
- 结果:
  - `best_epoch=3`
  - `best_val=-11.8676`
  - checkpoint: `runs/exp_cisrj_sep_dual_smoke_v1/checkpoints/best.pt`

### 2) Dual Smoke 评测
- 命令:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_sep_dual_smoke_v1/checkpoints/best.pt --scenario dual --model-config configs/model_sep_cisrj_map.yaml --run-dir runs/exp_cisrj_sep_dual_smoke_v1`
- 日志: `runs/experiment_logs/cisrj_sep_dual_smoke_v1_eval.log`
- 结果:
  - `SI_SDRi_mean_dB=10.1246`
  - `SI_SDR_mean_dB=6.1209`
  - `Count_active_sources=1000`

### 3) Multi Smoke 训练
- 命令:
  - `python scripts/20_train_cisrj.py --scenario multi --model-config configs/model_sep_cisrj_map.yaml --train-config configs/train_sep_cisrj_map.yaml --mode smoke --exp-name exp_cisrj_sep_multi_smoke_v1`
- 日志: `runs/experiment_logs/cisrj_sep_multi_smoke_v1.log`
- 结果:
  - `best_epoch=3`
  - `best_val=0.6438`
  - checkpoint: `runs/exp_cisrj_sep_multi_smoke_v1/checkpoints/best.pt`

### 4) Multi Smoke 评测
- 命令:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_sep_multi_smoke_v1/checkpoints/best.pt --scenario multi --model-config configs/model_sep_cisrj_map.yaml --run-dir runs/exp_cisrj_sep_multi_smoke_v1`
- 日志: `runs/experiment_logs/cisrj_sep_multi_smoke_v1_eval.log`
- 结果:
  - `SI_SDRi_mean_dB=7.2404`
  - `SI_SDR_mean_dB=-0.4000`
  - `Count_active_sources=1500`

### 5) Dual Formal 训练
- 命令:
  - `python scripts/20_train_cisrj.py --scenario dual --model-config configs/model_sep_cisrj_map.yaml --train-config configs/train_sep_cisrj_map.yaml --mode formal --exp-name exp_cisrj_sep_dual_formal_v1`
- 日志: `runs/experiment_logs/cisrj_sep_dual_formal_v1.log`
- 结果:
  - 训练在 `epoch 11` 早停
  - `best_epoch=1`
  - `best_val=-9.8391`
  - checkpoint: `runs/exp_cisrj_sep_dual_formal_v1/checkpoints/best.pt`

### 6) Dual Formal 评测
- 命令:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_sep_dual_formal_v1/checkpoints/best.pt --scenario dual --model-config configs/model_sep_cisrj_map.yaml --run-dir runs/exp_cisrj_sep_dual_formal_v1`
- 日志: `runs/experiment_logs/cisrj_sep_dual_formal_v1_eval.log`
- 结果:
  - `SI_SDRi_mean_dB=8.9964`
  - `SI_SDR_mean_dB=4.9927`
  - `Count_active_sources=1000`

### 7) Multi Formal 训练
- 命令:
  - `python scripts/20_train_cisrj.py --scenario multi --model-config configs/model_sep_cisrj_map.yaml --train-config configs/train_sep_cisrj_map.yaml --mode formal --exp-name exp_cisrj_sep_multi_formal_v1`
- 日志: `runs/experiment_logs/cisrj_sep_multi_formal_v1.log`
- 结果:
  - 训练在 `epoch 45` 早停
  - `best_epoch=35`
  - `best_val=-11.1807`
  - checkpoint: `runs/exp_cisrj_sep_multi_formal_v1/checkpoints/best.pt`

### 8) Multi Formal 评测
- 命令:
  - `python scripts/21_eval_cisrj.py --ckpt runs/exp_cisrj_sep_multi_formal_v1/checkpoints/best.pt --scenario multi --model-config configs/model_sep_cisrj_map.yaml --run-dir runs/exp_cisrj_sep_multi_formal_v1`
- 日志: `runs/experiment_logs/cisrj_sep_multi_formal_v1_eval.log`
- 结果:
  - `SI_SDRi_mean_dB=11.3881`
  - `SI_SDR_mean_dB=3.7476`
  - `Count_active_sources=1500`

## 汇总导出

- Smoke 汇总:
  - 命令: `python scripts/22_export_cisrj_report.py --dual-run runs/exp_cisrj_sep_dual_smoke_v1 --multi-run runs/exp_cisrj_sep_multi_smoke_v1 --out-csv paper/tables/cisrj_repro_summary_smoke_map_v1.csv`
  - 日志: `runs/experiment_logs/cisrj_repro_export_smoke_map_v1.log`

- Formal 汇总:
  - 命令: `python scripts/22_export_cisrj_report.py --dual-run runs/exp_cisrj_sep_dual_formal_v1 --multi-run runs/exp_cisrj_sep_multi_formal_v1 --out-csv paper/tables/cisrj_repro_summary_formal_map_v1.csv`
  - 日志: `runs/experiment_logs/cisrj_repro_export_formal_map_v1.log`

- 主汇总（覆盖为 formal）:
  - 命令: `python scripts/22_export_cisrj_report.py --dual-run runs/exp_cisrj_sep_dual_formal_v1 --multi-run runs/exp_cisrj_sep_multi_formal_v1 --out-csv paper/tables/cisrj_repro_summary.csv`
  - 日志: `runs/experiment_logs/cisrj_repro_export_main.log`

## 当前结论（map 配置）

- Dual:
  - smoke `SI_SDRi=10.12 dB` > formal `8.996 dB`，存在后期退化。
- Multi:
  - formal `SI_SDRi=11.39 dB` > smoke `7.24 dB`，训练可持续改善。
- 与论文 Table-4 参考值差距（formal）:
  - dual: `-11.90 dB`
  - multi: `-6.41 dB`

## 下一步建议

- 优先修复 dual 稳定性:
  - 对齐论文学习率策略（plateau 降 LR）。
  - 将损失权重切到 paper-like（仅主分离损失）做 dual 专项对照。
  - 固定 dual-only 训练并监控每轮活跃源覆盖率与错误分配比例。
