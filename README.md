# ME-MVNet

复合 ISRJ 分离与逐源参数估计主线仓库。

## 主架构

当前默认主模型只保留以下组件：

1. `SepNet(periodicctx)`
2. `Raw IQ branch`
3. `TF branch`
4. `GateFormer`
5. `Ts = (NF + 1) * Tl`

`mechanism branch` 仅保留为消融能力，不属于默认主架构。

## 当前正式主线结果

基于 `overlap085` 数据设定、`no-mech` 主架构：

- `all`: `0.7040`
- `dual`: `0.8490`
- `multi`: `0.6073`

## 仓库中真正需要的核心文件

### 配置

- `configs/data_composite.yaml`
- `configs/model_sep.yaml`
- `configs/model_pe.yaml`
- `configs/train_sep.yaml`
- `configs/train_joint.yaml`
- `configs/eval_composite.yaml`

### 训练与评测入口

- `scripts/10_generate_dataset_composite.py`
- `scripts/11_sanity_check_composite.py`
- `scripts/12_train_sepnet.py`
- `scripts/13_train_seppe_joint.py`
- `scripts/14_eval_seppe.py`

### 主模型

- `src/models/sepnet.py`
- `src/models/penet.py`
- `src/models/gateformer.py`
- `src/models/builders.py`
- `src/models/blocks_1d.py`
- `src/models/blocks_2d.py`
- `src/models/losses_seppe.py`
- `src/models/pit_perm.py`
- `src/models/sisdr.py`

### 数据与训练支持

- `src/data/isrj_composite_generator.py`
- `src/data/dataset_npz_composite.py`
- `src/data/stft.py`
- `src/train/trainer_sepnet.py`
- `src/train/trainer_seppe.py`

## 标准流程

### 1. 生成数据

```bash
python scripts/10_generate_dataset_composite.py --config configs/data_composite.yaml
python scripts/11_sanity_check_composite.py --config configs/data_composite.yaml
```

### 2. Stage-1 训练 Separator

```bash
python scripts/12_train_sepnet.py --mode formal --exp-name exp_sep_formal
```

### 3. Stage-2 联合训练

```bash
python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal/checkpoints/best.pt --mode formal --exp-name exp_joint_formal
```

### 4. 正式评测

```bash
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_formal/checkpoints/best.pt --split test --run-dir runs/exp_joint_formal
```

## 结果解释

正式指标采用：

- `A_NF`
- `A_Tl`
- `A_Ts`
- `A_total`

其中 `A_total` 表示单个 active source 同时满足 `NF / Tl / Ts` 正确。

## 仓库约定

- 默认路径只服务于当前主架构。
- 实验分支代码允许保留，但不应进入默认配置与默认文档。
- `runs/`、临时脚本、实验日志、阶段性说明文件不属于仓库主内容。
