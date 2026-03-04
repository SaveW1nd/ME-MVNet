# separation_focus_log.md

- 日期: 2026-03-04
- 目标: 只优化 Stage-1 分离质量（`SI_SDR_jam`），暂停后续 joint/eval。
- 当前基线（最优历史）: `runs/exp_sep_formal_step123_stable_v1`，best val `SI_SDR_jam=-4.4629 dB`。

## 实验 SF-1: 降低 background loss 权重
- 代码:
  - `src/models/losses_seppe.py`
  - 变更: `L_sep = L_jam + 0.03 * MSE_bg`（由 `0.1` 降到 `0.03`）
- 预期: 减少背景项对分离主目标的牵引，优先提升 jammer SI-SDR。

### SF-1 Stage1 Smoke
- 时间: 2026-03-04 17:52:06
- 命令: python scripts/12_train_sepnet.py --train-config configs/train_sep_need1_stable.yaml --mode smoke --exp-name exp_sep_smoke_sf1_bg003_v1

### SF-1 Stage1 Formal
- 时间: 2026-03-04 17:52:37
- 命令: python scripts/12_train_sepnet.py --train-config configs/train_sep_need1_stable.yaml --mode formal --exp-name exp_sep_formal_sf1_bg003_v1

## 实验 SF-2: SepNet 容量增强
- 模型配置: configs/model_sep_sf2_wide.yaml (C=192, blocks=10, cycle=4, dropout=0.05)
- 训练配置: configs/train_sep_sf2_wide.yaml (epochs=60, lr=6e-4, batch=48)
- 预期: 提升分离表征能力，继续拉升 SI_SDR_jam。

### SF-2 Stage1 Smoke
- 时间: 2026-03-04 17:57:37
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf2_wide.yaml --mode smoke --exp-name exp_sep_smoke_sf2_wide_v1

### SF-2 Stage1 Formal
- 时间: 2026-03-04 17:58:29
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf2_wide.yaml --mode formal --exp-name exp_sep_formal_sf2_wide_v1

## 实验 SF-3: 从 SF-2 best ckpt 低学习率微调
- 脚本增强: scripts/12_train_sepnet.py 新增 --init-ckpt 支持
- 训练配置: configs/train_sep_sf3_finetune.yaml (lr=2e-4, epochs=40)
- 初始化: runs/exp_sep_formal_sf2_wide_v1/checkpoints/best.pt
- 目标: 在已有分离能力上继续提升 SI_SDR_jam。

### SF-3 Stage1 Formal
- 时间: 2026-03-04 18:11:56
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf3_finetune.yaml --init-ckpt runs/exp_sep_formal_sf2_wide_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf3_finetune_v1

## 实验 SF-4: jam_only 分离头（3 路 jammer + 残差背景）
- 代码: src/models/sepnet.py 新增 jam_only 可选模式
- 配置: configs/model_sep_sf4_jamonly_wide.yaml
- 目的: 将表示容量和 mask 竞争完全聚焦到 jammer。

### SF-4 Stage1 Smoke
- 时间: 2026-03-04 18:21:17
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf4_jamonly_wide.yaml --train-config configs/train_sep_sf2_wide.yaml --mode smoke --exp-name exp_sep_smoke_sf4_jamonly_v1

## 实验 SF-5: 背景权重进一步降到 0.01（微调）
- 代码: src/models/losses_seppe.py, L_sep = L_jam + 0.01*MSE_bg`n- 初始化: runs/exp_sep_formal_sf2_wide_v1/checkpoints/best.pt
- 配置: model_sep_sf2_wide + train_sep_sf3_finetune

### SF-5 Stage1 Formal
- 时间: 2026-03-04 18:22:37
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf3_finetune.yaml --init-ckpt runs/exp_sep_formal_sf2_wide_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf5_bg001_finetune_v1

## 实验 SF-6: SepNet-XL 快速筛选
- 模型: configs/model_sep_sf6_xl.yaml (C=256, blocks=12)
- 训练: configs/train_sep_sf6_xl.yaml (batch=32, lr=5e-4)

### SF-6 Stage1 Smoke
- 时间: 2026-03-04 18:31:16
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf6_xl.yaml --train-config configs/train_sep_sf6_xl.yaml --mode smoke --exp-name exp_sep_smoke_sf6_xl_v1

## 实验 SF-7: 二次低学习率微调
- 配置: configs/train_sep_sf7_finetune.yaml (lr=1e-4, epochs=40)
- 初始化: runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt

### SF-7 Stage1 Formal
- 时间: 2026-03-04 18:32:34
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf7_finetune.yaml --init-ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf7_finetune_v1

## 阶段性汇总（只看 Stage-1 分离）

- baseline（step123稳定版）: best val `SI_SDR_jam=-4.4629 dB`
- SF-1（bg=0.03）: best val `-4.2926 dB`
- SF-2（宽模型 C192/B10）: best val `-2.7276 dB`
- SF-3（从 SF-2 低lr微调）: best val `-2.5610 dB`
- SF-5（bg=0.01 + 微调）: best val `-2.4927 dB`  **当前最佳**
- SF-7（二次微调）: best val `-2.5280 dB`（未超过 SF-5）

结论：目前最有效组合为“宽模型 + 低背景权重 + 单次微调”，已将 val `SI_SDR_jam` 从 `-4.46` 提升到 `-2.49 dB`。

## 实验 SF-8: 分离行为诊断（不是只看 dB）
- 新脚本: scripts/16_analyze_sep_behavior.py
- 命令: python scripts/16_analyze_sep_behavior.py --ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt --model-config configs/model_sep_sf2_wide.yaml --split val --run-dir runs/exp_sep_formal_sf5_bg001_finetune_v1
- 输出:
  - tables/sep_diagnostics.json
  - tables/sep_case_scores.csv
  - figures/sep_diagnostics/*.png
- 关键观察:
  - jam_sisdr_diag_mean_db=-2.477
  - jam_sisdr_offdiag_mean_db=-38.320（串扰整体很低）
  - collapse_cos_mean=0.431（仍有中等程度源形态相似）
  - silence_ratio_mean=0.200（inactive 槽泄漏仍明显）
  - bg_leak_corr_mean=0.924（背景与 jammer 包络相关性偏高，提示背景分离不干净）

## 实验 SF-9: 行为约束损失（inactive抑制+背景解耦）
- 代码: src/models/losses_seppe.py, src/train/trainer_sepnet.py
- 新增项: L_sep_sil, L_sep_orth
- 总损失: L_sep = L_jam + 0.01*L_bg + 0.2*L_sil + 0.05*L_orth
- 初始化: runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt

### SF-9 Stage1 Smoke
- 时间: 2026-03-04 18:51:13
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf7_finetune.yaml --init-ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt --mode smoke --exp-name exp_sep_smoke_sf9_behaviorloss_v1

### SF-9 Stage1 Smoke（重跑）
- 时间: 2026-03-04 18:52:02
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf7_finetune.yaml --init-ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt --mode smoke --exp-name exp_sep_smoke_sf9_behaviorloss_v2

### SF-9 Stage1 Formal
- 时间: 2026-03-04 18:53:00
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf7_finetune.yaml --init-ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf9_behaviorloss_v1

## 实验 SF-10: 行为约束加权（强抑制）
- 配置: configs/train_sep_sf10_behavior_strong.yaml
- 权重: w_bg=0.01, w_sil=0.50, w_orth=0.20
- 初始化: runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt

### SF-10 Stage1 Smoke
- 时间: 2026-03-04 19:02:15
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf10_behavior_strong.yaml --init-ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt --mode smoke --exp-name exp_sep_smoke_sf10_behaviorstrong_v1

### SF-10 Stage1 Formal
- 时间: 2026-03-04 19:03:09
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf2_wide.yaml --train-config configs/train_sep_sf10_behavior_strong.yaml --init-ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf10_behaviorstrong_v1

## 实验 SF-11: 分组解码器（架构）
- 代码: src/models/sepnet.py 新增 decoder_grouped，可按槽独立解码
- 配置: configs/model_sep_sf11_grouped_wide.yaml
- 训练: configs/train_sep_sf11_arch_smoke.yaml（先去掉行为loss项，只检验架构本身）

### SF-11 Stage1 Smoke
- 时间: 2026-03-04 19:11:59
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf11_arch_smoke.yaml --mode smoke --exp-name exp_sep_smoke_sf11_grouped_v1

### SF-11 Stage1 Formal
- 时间: 2026-03-04 19:12:55
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf11_arch_smoke.yaml --mode formal --exp-name exp_sep_formal_sf11_grouped_v1

## 实验 SF-12: 针对 SF-11 的行为修正微调
- 目标: 降低 SF-11 的背景污染与源塌缩
- 方式: per-slot 背景去相关 + 多样性约束（L_sep_div）
- 配置: configs/train_sep_sf12_refine.yaml
- 初始化: runs/exp_sep_formal_sf11_grouped_v1/checkpoints/best.pt

### SF-12 Stage1 Smoke
- 时间: 2026-03-04 19:26:08
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf12_refine.yaml --init-ckpt runs/exp_sep_formal_sf11_grouped_v1/checkpoints/best.pt --mode smoke --exp-name exp_sep_smoke_sf12_refine_v1

### SF-12 Stage1 Formal
- 时间: 2026-03-04 19:26:49
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf12_refine.yaml --init-ckpt runs/exp_sep_formal_sf11_grouped_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf12_refine_v1

## 观察驱动结论更新（SF-11 / SF-12）

### 核心现象
- SF-11（分组解码器）显著提升分离有效性：best val `SI_SDR_jam` 从 `-2.49` 提升到 `-1.89 dB`。
- SF-12（在 SF-11 上加 per-slot 解耦与多样性约束）进一步提升到 `-1.70 dB`。

### 行为指标对比（val）
- `jam_sisdr_diag_mean_db`: SF5 `-2.477` -> SF11 `-1.872` -> SF12 `-1.692`（持续改善）
- `silence_ratio_mean`: SF5 `0.2001` -> SF11 `0.1369` -> SF12 `0.1207`（inactive 泄漏显著改善）
- `collapse_cos_mean`: SF5 `0.4313` -> SF11 `0.4583` -> SF12 `0.4528`（较 SF11 略回落，但仍高于 SF5）
- `bg_leak_corr_mean`: SF5 `0.9239` -> SF11 `0.9435` -> SF12 `0.9469`（背景污染问题加重，仍是主矛盾）

### 设计决策
- 保留: `decoder_grouped=true`（对分离和 inactive 抑制收益明确）。
- 保留: SF-12 的行为约束作为可调配置（可继续调权重）。
- 下一步应专门处理背景污染，而不是继续盲目加大分离主干容量。

## 实验 SF-13: 背景专用解耦分支（架构）
- 代码: SepNet 新增 bg_dedicated + bg_residual_ratio
- 模型配置: configs/model_sep_sf13_bgdedicated.yaml
- 训练配置: configs/train_sep_sf13_arch.yaml（仅测架构，不加行为loss）

### SF-13 Stage1 Smoke
- 时间: 2026-03-04 19:35:22
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf13_bgdedicated.yaml --train-config configs/train_sep_sf13_arch.yaml --mode smoke --exp-name exp_sep_smoke_sf13_bgdedicated_v1

### SF-13 Stage1 Smoke（partial init from SF-11）
- 时间: 2026-03-04 19:36:31
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf13_bgdedicated.yaml --train-config configs/train_sep_sf13_arch.yaml --init-ckpt runs/exp_sep_formal_sf11_grouped_v1/checkpoints/best.pt --init-partial --mode smoke --exp-name exp_sep_smoke_sf13_bgdedicated_warm_v1

### SF-13 Stage1 Formal（warm init from SF-11）
- 时间: 2026-03-04 19:37:14
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf13_bgdedicated.yaml --train-config configs/train_sep_sf13_arch.yaml --init-ckpt runs/exp_sep_formal_sf11_grouped_v1/checkpoints/best.pt --init-partial --mode formal --exp-name exp_sep_formal_sf13_bgdedicated_warm_v1

## 实验 SF-14: 背景净化（提升 w_bg）
- 主线: SF-12 架构与行为约束保持不变
- 调整: w_bg 0.01 -> 0.05
- 配置: configs/train_sep_sf14_bgclean.yaml
- 初始化: runs/exp_sep_formal_sf12_refine_v1/checkpoints/best.pt

### SF-14 Stage1 Formal
- 时间: 2026-03-04 19:48:14
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf14_bgclean.yaml --init-ckpt runs/exp_sep_formal_sf12_refine_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf14_bgclean_v1

## 实验 SF-15: 提高背景解耦权重（w_orth）
- 目标: 降低 bg_leak_corr
- 配置: configs/train_sep_sf15_bgorth.yaml
- 初始化: runs/exp_sep_formal_sf14_bgclean_v1/checkpoints/best.pt

### SF-15 Stage1 Formal
- 时间: 2026-03-04 19:52:57
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf15_bgorth.yaml --init-ckpt runs/exp_sep_formal_sf14_bgclean_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf15_bgorth_v1

## 实验 SF-16: 背景对真值干扰去相关（w_bgtrue）
- 新项: L_sep_bgtrue（b_hat 与 aligned j_true 的相关性惩罚）
- 配置: configs/train_sep_sf16_bgtrue.yaml
- 初始化: runs/exp_sep_formal_sf15_bgorth_v1/checkpoints/best.pt

### SF-16 Stage1 Formal
- 时间: 2026-03-04 19:56:56
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf16_bgtrue.yaml --init-ckpt runs/exp_sep_formal_sf15_bgorth_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf16_bgtrue_v1

## 实验 SF-17: 提高真值去相关权重（w_bgtrue=0.5）
- 配置: configs/train_sep_sf17_bgtrue_high.yaml
- 初始化: runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt

### SF-17 Stage1 Formal
- 时间: 2026-03-04 19:59:47
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf17_bgtrue_high.yaml --init-ckpt runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf17_bgtrue_high_v1

## 近期结论（SF-13~SF-17，观察驱动）

- SF-13（背景专用分支）在 warm start 下虽可收敛，但行为指标劣化：
  - `silence_ratio` 上升到 `0.2397`
  - `bg_leak_corr` 上升到 `0.9560`
  - 结论: 当前实现不保留。

- SF-14（回到 SF-12 主线，提高 w_bg）:
  - `silence_ratio` 降到 `0.0901`
  - `bg_leak_corr` 为 `0.9494`（仍高）

- SF-15（提高 w_orth）:
  - `jam_sisdr_diag_mean_db` 提升到 `-1.6468`
  - `bg_leak_corr` 仅微降到 `0.9493`

- SF-16（新增对齐真值干扰去相关 L_sep_bgtrue）:
  - `jam_sisdr_diag_mean_db=-1.6176`
  - `silence_ratio=0.0868`
  - `collapse_cos=0.4454`
  - `bg_leak_corr=0.9488`
  - 结论: 当前综合最优（分离有效性 + 行为健康）。

- SF-17（进一步提高 w_bgtrue）:
  - `jam_sisdr_diag_mean_db` 继续升到 `-1.5855`
  - 但 `bg_leak_corr` 反弹到 `0.9504`
  - 结论: 过强约束会牺牲背景纯净性，不取。

### 当前推荐检查点（只针对 Stage-1 分离）
- `runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt`
- 诊断报告: `runs/exp_sep_formal_sf16_bgtrue_v1/tables/sep_diagnostics.json`

## 实验 SF-18: residual 背景分配抑制（bg_residual_scale）
- 模型: configs/model_sep_sf18_bgresidual.yaml（bg_residual_scale=0.1）
- 训练: configs/train_sep_sf18_bgresidual.yaml
- 初始化: runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt

### SF-18 Stage1 Formal
- 时间: 2026-03-04 20:48:26
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf18_bgresidual.yaml --train-config configs/train_sep_sf18_bgresidual.yaml --init-ckpt runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf18_bgresidual_v1

## 实验 SF-19: residual 背景分配平衡版（scale=0.3）
- 模型: configs/model_sep_sf19_bgresidual03.yaml
- 训练: configs/train_sep_sf19_bgresidual03.yaml
- 初始化: runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt

### SF-19 Stage1 Formal
- 时间: 2026-03-04 20:51:10
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf19_bgresidual03.yaml --train-config configs/train_sep_sf19_bgresidual03.yaml --init-ckpt runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf19_bgresidual03_v1

## 实验 SF-20: 直接压制背景包络相关性（w_bgenv）
- 新项: L_sep_bgenv（b_hat_amp 与 j_true_amp 相关性惩罚）
- 配置: configs/train_sep_sf20_bgenv.yaml
- 初始化: runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt

### SF-20 Stage1 Formal
- 时间: 2026-03-04 20:54:31
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf20_bgenv.yaml --init-ckpt runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf20_bgenv_v1

## 实验 SF-21: 目标同尺度归一化（normalize_targets=true）
- 目的: 修复 X/J 标度不一致导致的背景相关性偏置
- 配置: configs/train_sep_sf21_normtargets.yaml
- 模型: configs/model_sep_sf11_grouped_wide.yaml
- 初始化: runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt

### SF-21 Stage1 Formal
- 时间: 2026-03-04 20:59:06
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf21_normtargets.yaml --init-ckpt runs/exp_sep_formal_sf16_bgtrue_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf21_normtargets_v1

## 实验 SF-22: normalize_targets 主线低学习率续训
- 配置: configs/train_sep_sf22_normtargets_ft.yaml
- 初始化: runs/exp_sep_formal_sf21_normtargets_v1/checkpoints/best.pt

### SF-22 Stage1 Formal
- 时间: 2026-03-04 21:01:59
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf22_normtargets_ft.yaml --init-ckpt runs/exp_sep_formal_sf21_normtargets_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf22_normtargets_ft_v1

## 实验 SF-23: 提高包络相关惩罚（w_bgenv=0.2）
- 配置: configs/train_sep_sf23_bgenv_high.yaml
- 初始化: runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt

### SF-23 Stage1 Formal
- 时间: 2026-03-04 21:04:40
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf23_bgenv_high.yaml --init-ckpt runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf23_bgenv_high_v1

## 实验 SF-24: 背景重建强化（w_bg=0.2）
- 配置: configs/train_sep_sf24_bgheavy.yaml
- 初始化: runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt

### SF-24 Stage1 Formal
- 时间: 2026-03-04 21:07:30
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf24_bgheavy.yaml --init-ckpt runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf24_bgheavy_v1

## 实验 SF-25: 背景权重折中（w_bg=0.1）
- 配置: configs/train_sep_sf25_bgmid.yaml
- 初始化: runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt

### SF-25 Stage1 Formal
- 时间: 2026-03-04 21:09:42
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf25_bgmid.yaml --init-ckpt runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf25_bgmid_v1

## SF-18~SF-25 阶段结论（专项修复）

### 关键发现
1. `X` 归一化但 `J` 不归一化会系统性抬高背景相关性指标：
   - mismatch frame `bg_vs_jam_corr_mean ≈ 0.806`
   - consistent frame `bg_vs_jam_corr_mean ≈ 0.178`
2. 因此后续背景污染专项应在 `normalize_targets=true` 条件下评估与训练。

### 新主线（normalize_targets=true）结果
- SF-21: `SI_SDR_jam(best val)=-1.8336`, `bg_leak_corr=0.94366`
- SF-22: `SI_SDR_jam(best val)=-1.7540`, `bg_leak_corr=0.94322`（当前平衡最好）
- SF-24 (`w_bg=0.2`): `bg_leak_corr=0.93726`（最干净，但分离退化）
- SF-25 (`w_bg=0.1`): `bg_leak_corr=0.94246`（折中）

### 当前建议
- 若优先“分离有效性 + 兼顾污染”: 选 `SF-22`
  - `runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt`
- 若优先“背景纯净”: 选 `SF-24`
  - `runs/exp_sep_formal_sf24_bgheavy_v1/checkpoints/best.pt`

## 实验 SF-26: 背景权重中高（w_bg=0.15）
- 配置: configs/train_sep_sf26_bg015.yaml
- 初始化: runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt

### SF-26 Stage1 Formal
- 时间: 2026-03-04 21:12:23
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf26_bg015.yaml --init-ckpt runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf26_bg015_v1

## 实验 SF-27: 背景权重插值（w_bg=0.12）
- 配置: configs/train_sep_sf27_bg012.yaml
- 初始化: runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt

### SF-27 Stage1 Formal
- 时间: 2026-03-04 21:14:27
- 命令: python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf27_bg012.yaml --init-ckpt runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt --mode formal --exp-name exp_sep_formal_sf27_bg012_v1

## SF-21~SF-27 Pareto 总结（normalize_targets=true 主线）

- SF-22: `jam_diag=-1.743`, `bg_leak=0.9432`, `silence=0.0838`
- SF-25 (w_bg=0.10): `jam_diag=-1.845`, `bg_leak=0.9425`, `silence=0.0691`
- SF-27 (w_bg=0.12): `jam_diag=-1.889`, `bg_leak=0.9415`, `silence=0.0689`
- SF-26 (w_bg=0.15): `jam_diag=-1.941`, `bg_leak=0.9398`, `silence=0.0646`

解读:
- `w_bg` 上升会稳定压低 `bg_leak_corr`，但会牺牲 jammer 分离质量。
- 当前阈值导向（`bg_leak<=0.94`）下，SF-26 是可行解。
- 若优先综合分离有效性，SF-22 仍是更强基线。

## 单样本可视化测试（用户请求）
- 时间: 2026-03-04 21:40:17
- 模型: runs/exp_sep_formal_sf22_normtargets_ft_v1/checkpoints/best.pt
- 数据: test split, sample_index=137
- 结果: jam_sisdr_case=-10.5059 dB, bg_leak_corr=0.9990
- 分源 SI-SDR: [19.17, -79.91, 29.22] dB（第2路出现严重崩塌）
- 输出文件:
  - runs/analysis_single_case_sf22/sample_case_plot.png
  - runs/analysis_single_case_sf22/sample_report.json

## 专项修复：active jammer 零能量问题定位与修复
- 时间: 2026-03-04 22:04:00 ~ 22:10:00
- 触发: 用户质疑 “是否 ISRJ 实现导致某些 active 槽没有能量”。

### 诊断
- 关键代码: `src/data/isrj_composite_generator.py::generate_one_jammer`
- 问题路径:
  1. jammer 生成后执行 `shift_right_zero_pad`
  2. 延迟 `delay` 原先在 `[0, N)` 均匀采样
  3. 在零填充右移下，部分样本会把整段 jammer 完全移出观测窗
  4. 于是出现 `NF>0` 但 `J` 与 `G` 实际全零
- 复现实验（修复前）:
  - 单 jammer Monte-Carlo 零能量率: 约 `11%~19%`
  - 现有 train 集合: `active slots=12500`, `zero-energy active=2004`

### 修复
- 修改文件: `src/data/isrj_composite_generator.py`
- 新增 `_sample_safe_right_delay(mask, rng)`:
  - 依据 active mask 的最后非零索引计算 `max_delay`
  - 仅在保证“至少一个 active 样本仍在窗内”的范围内采样右移
- 结果: 保留随机延迟建模，同时消除 active 槽被整体移出窗口的问题。

### 质量闸门补充
- `tests/test_composite_generator.py` 新增断言:
  - `NF>0` 的槽位必须 `jam_energy>1e-10`
  - `NF>0` 的槽位必须 `gate_count>0`
- `scripts/11_sanity_check_composite.py` 新增硬检查:
  - 输出并校验 `zero-energy active` 与 `zero-gate active` 计数
  - 非零即失败

### 回归验证
- 单测: `pytest -q tests/test_composite_generator.py` -> `1 passed`
- 统计脚本（修复后）: 单 jammer 零能量率全为 `0.0`
- 旧数据复检（修复前生成）: 失败，`2004/12500` active 零能量
- 重新生成数据:
  - 命令: `python scripts/10_generate_dataset_composite.py --config configs/data_composite.yaml`
  - 复检: `python scripts/11_sanity_check_composite.py --config configs/data_composite.yaml --data-dir data/raw/cisrj_seppe_v1 --num-plot 1`
  - 结果: train/val/test 的 `zero-energy active=0`, `zero-gate active=0`

## 修复后复训验证（新数据）
- 时间: 2026-03-04 22:25 ~ 22:36
- 目标: 在修复后的数据上重新验证 Stage-1 分离有效性与行为指标。

### DataFix Stage1 Smoke（从头训练）
- 命令:
  - `python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf21_normtargets.yaml --mode smoke --exp-name exp_sep_smoke_datafix_v1`
- 日志: `runs/experiment_logs/sf_datafix_sep_smoke_v1.log`
- 输出:
  - `runs/exp_sep_smoke_datafix_v1/checkpoints/best.pt`
- 关键结果:
  - `best_epoch=3`
  - `best_val(L_sep)=-0.5301`
  - `val SIjam@epoch3=1.016 dB`

### DataFix Stage1 Formal（从头训练）
- 命令:
  - `python scripts/12_train_sepnet.py --model-config configs/model_sep_sf11_grouped_wide.yaml --train-config configs/train_sep_sf21_normtargets.yaml --mode formal --exp-name exp_sep_formal_datafix_v1`
- 日志: `runs/experiment_logs/sf_datafix_sep_formal_v1.log`
- 输出:
  - `runs/exp_sep_formal_datafix_v1/checkpoints/best.pt`
- 关键结果:
  - `best_epoch=12`
  - `best_val(L_sep)=-7.4604`
  - `val SIjam@epoch12=3.574 dB`

### DataFix 行为诊断（val）
- 命令:
  - `python scripts/16_analyze_sep_behavior.py --ckpt runs/exp_sep_formal_datafix_v1/checkpoints/best.pt --model-config configs/model_sep_sf11_grouped_wide.yaml --split val --run-dir runs/exp_sep_formal_datafix_v1 --normalize-targets`
- 日志: `runs/experiment_logs/sf_datafix_sep_behavior_diag_v1.log`
- 输出:
  - `runs/exp_sep_formal_datafix_v1/tables/sep_diagnostics.json`
  - `runs/exp_sep_formal_datafix_v1/tables/sep_case_scores.csv`
  - `runs/exp_sep_formal_datafix_v1/figures/sep_diagnostics/*.png`
- 关键指标:
  - `jam_sisdr_diag_mean_db=3.563`
  - `jam_sisdr_offdiag_mean_db=-21.177`
  - `jam_sisdr_gap_db=23.890`
  - `collapse_cos_mean=0.538`
  - `silence_ratio_mean=0.404`
  - `bg_leak_corr_mean=0.876`

### DataFix 单样本复测（test idx=137）
- 输出目录: `runs/analysis_single_case_datafix`
- 文件:
  - `sample_report.json`
  - `sample_case_plot.png`
- 结果:
  - `K_active=3, NF=[2,1,2]`
  - `jam_sisdr_case=10.082 dB`
  - `slot_sisdr=[8.114, 6.116, 16.017] dB`
  - `bg_leak_corr=0.921`

## DataFix 阶段结论
- 确认此前“有标注但无能量”的主问题来自数据生成实现，而非模型本身。
- 修复并重生数据后，分离训练/行为指标与单样本可视化均明显恢复，先前的严重崩塌样本（test#137）已消失。
- 下一阶段应继续围绕行为指标做针对优化，优先压低 `silence_ratio` 与 `collapse_cos`，同时维持当前 `bg_leak_corr` 水平。

## 整体结果测试（端到端 Joint + Test Eval）
- 时间: 2026-03-04 22:37 ~ 22:56
- 目的: 在修复后的数据上完成整体链路验证（Stage-2 训练 + test 集总体评估 + 图表导出）。

### Joint Smoke
- 命令:
  - `python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal_datafix_v1/checkpoints/best.pt --sep-config configs/model_sep_sf11_grouped_wide.yaml --pe-config configs/model_pe.yaml --train-config configs/train_joint_need1_stable.yaml --mode smoke --exp-name exp_joint_smoke_datafix_v1`
- 日志: `runs/experiment_logs/sf_datafix_joint_smoke_v1.log`
- 结果:
  - `best_epoch=3`
  - `best_A_total=0.0698`

### Joint Formal
- 命令:
  - `python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal_datafix_v1/checkpoints/best.pt --sep-config configs/model_sep_sf11_grouped_wide.yaml --pe-config configs/model_pe.yaml --train-config configs/train_joint_need1_stable.yaml --mode formal --exp-name exp_joint_formal_datafix_v1`
- 日志: `runs/experiment_logs/sf_datafix_joint_formal_v1.log`
- 结果:
  - early stop at epoch 40
  - `best_epoch=28`
  - `best_A_total=0.2918`
  - checkpoint: `runs/exp_joint_formal_datafix_v1/checkpoints/best.pt`

### Test 集整体评估
- 命令:
  - `python scripts/14_eval_seppe.py --ckpt runs/exp_joint_formal_datafix_v1/checkpoints/best.pt --split test --data-config configs/data_composite.yaml --sep-config configs/model_sep_sf11_grouped_wide.yaml --pe-config configs/model_pe.yaml --eval-config configs/eval_composite.yaml --run-dir runs/exp_joint_formal_datafix_v1`
- 日志: `runs/experiment_logs/sf_datafix_joint_eval_v1.log`
- 指标（overall）:
  - `Tl_MAE_us=0.2204`
  - `Ts_MAE_us=0.9114`
  - `NF_Acc=0.7977`
  - `NF_macroF1=0.7896`
  - `A_Tl=0.5416`
  - `A_Ts=0.3324`
  - `A_NF=0.7940`
  - `A_total=0.3032`
- 文件:
  - `runs/exp_joint_formal_datafix_v1/predictions/test_pred.npz`
  - `runs/exp_joint_formal_datafix_v1/tables/test_metrics.json`
  - `runs/exp_joint_formal_datafix_v1/tables/test_metrics_by_jnr.csv`
  - `runs/exp_joint_formal_datafix_v1/tables/test_metrics_by_kactive.csv`

### 图表导出（run + paper）
- 命令:
  - `python scripts/15_export_plots_seppe.py --run-dir runs/exp_joint_formal_datafix_v1 --split test --paper-dir paper --num-cases 3`
- 日志: `runs/experiment_logs/sf_datafix_joint_export_v1.log`
- 导出结果:
  - `runs/exp_joint_formal_datafix_v1/figures/*`
  - `paper/figures/*`
  - `paper/tables/*`

## 本轮测试结论
- 修复数据生成后，端到端整体链路稳定可训练，且 test 总体指标达到 `A_total=0.3032`。
- 难点仍在 `Ts` 精度（`A_Ts=0.3324` 明显低于 `A_NF` 与 `A_Tl`），后续整体优化应优先针对 `Ts` 头和多任务权重。
