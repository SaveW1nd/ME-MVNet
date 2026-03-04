# ME‑MVSepPE 开发指导（路线B：复合ISRJ分离 + 逐源测参）

你是一个开发代理（AI coder）。你的目标是在现有仓库基础上新增一条完整可复现流水线：

1. 生成 **复合（2~3个）ISRJ + 背景回波 + AWGN** 数据集（含每个源的真值分量与参数）
2. 训练 **SepNet** 把复合信号分离成逐源 ISRJ
3. 训练 **PE‑Net** 对每个分离源做 **门控估计 + 参数估计**
4. 评估与出图：对齐 Wang‑style 指标：**A_{Tl},A_{Ts},A_{NF},A_{total}** + MAE/RMSE
5. 导出论文图表到 `paper/`

必须遵守本文档的**参数定义、数据格式、模块 I/O 形状、loss 组成与训练阶段**。不得引入参考信号，不得修改参数定义。

## 0. 系统定义（固定）

### 输入（网络看到的只有这个）

- 混合复基带序列 `x`：`(B, 2, N)`，2通道为 I/Q（float32）

### 混合组成（生成器内部）

$$
x=\sum_{k=1}^{K} j_k \;+\; b
$$



- K=3（固定输出槽位数）
- j_k：第 k 个 ISRJ 分量（最多 3 个）
- b：背景项（回波 + AWGN，属于 residual/background）

### 输出（系统最终输出）

对每个槽位 k=1..3 输出：

- 分离波形：\hat{j}_k（`(B,2,N)`）
- 门控：\hat{g}_k（`(B,N)`，0~1）
- 参数：\hat{T}_{l,k}、\hat{N}_{F,k}、\hat{T}_{s,k}

参数关系固定：
$$
T_{s,k}=(N_{F,k}+1)\cdot T_{l,k}
$$


实现中 **网络只回归 T_l**，只分类 N_F，并由上式计算 T_s，保证一致性无需额外一致性 loss。

## 1. 数据集规格（固定）

### 1.1 体制参数（固定）

采用 Wang‑style 的一组体制参数（用于复合分离任务）：

- `FS = 100e6` Hz
- `TP = 15e-6` s（脉宽，发射脉冲长度）
- `PRI = 40e-6` s（一个记录长度）
- `N = FS * PRI = 4000` 点
- LFM 调频率：`KR = B / TP`
- `B = 10e6` Hz（先固定 10 MHz，后续可扩展随机）

### 1.2 ISRJ 核心参数取值

每个 jammer 源独立采样参数：

- `NF ∈ {1,2,3}`（三分类）
- `Tl ∈ [1, 3] µs`（连续均匀采样）
  - `Nl = round(Tl * FS)`，范围约 100..300（整数）
- `Ts = (NF+1)*Tl`
- 每源 JNR：`JNR_dB ∈ [-10, 20]`（连续均匀采样）

### 1.3 “dual/triple” 复合样本定义（固定）

- 每个样本生成 `K_active ∈ {2,3}`，并写入字段 `K_active`
- 当 `K_active=2`：第 3 路源定义为 silence（全零）
  - `j_3=0`
  - `g_3=0`
  - `NF_3=0`（新增“空源”类别）
  - `Tl_3=0, Ts_3=0`
- 当 `K_active=3`：三路源均为真实 jammer

因此 NF 分类 head 必须是 **4 类**：`NF ∈ {0,1,2,3}`，其中 0 表示空源。

### 1.4 数据集规模与划分（固定）

- train/val/test = **5000 / 1000 / 1000**
- `K_active` 采样比例固定为：dual:triple = **1:1**（各占 50%）

## 2. 数据生成器（必须实现的严格规则）

新增生成器：`src/data/isrj_composite_generator.py`

### 2.1 发射脉冲（PRI 内）

生成长度 N=4000 的 `s_full`：

- 前 `Np = round(TP*FS)=1500` 点为基带 LFM：
  - `t = (n - Np/2)/FS`
  - `s[n] = exp(j*pi*KR*t^2)`
- 后续 `n>=Np` 置 0（PRI 余量）

### 2.2 单源 ISRJ（每个 jammer）

给定 `Nl, NF`，定义周期长度：

- `Nu = (NF+1)*Nl`

定义“采样窗口起点 offset”：

- `offset ∈ [0, Np - Nl]` 均匀采样整数

（保证采样切片一定从非零 LFM 区域采样）

定义每周期采样与转发（只对 sampling 发生在 LFM 区域的周期生成）：

- `K_cycles = floor((Np - offset) / Nu)`
- 对每个周期 c：
  - `base = offset + c*Nu`
  - `slice = s_full[base : base+Nl]`
  - 对 `m=1..NF`：
    - `st = base + m*Nl`
    - `ed = st + Nl`
    - `j[st:ed] += slice`
    - `g[st:ed] = 1`

随后施加 jammer 的**全局时延**（模拟不同虚假距离/重叠）：

- `delay ∈ [0, N-1]` 均匀采样整数
- `j = shift_right_zero_pad(j, delay)`
- `g = shift_right_zero_pad(g, delay)`

再施加 jammer 的**多普勒/频偏**（作为 nuisance，提升鲁棒性）：

- `fd ∈ [-0.2B, 0.2B]` 连续均匀
- `j *= exp(j*2*pi*fd*n/FS)`

### 2.3 背景项 b（回波 + AWGN）

背景回波 `e`（不测参）：

- 生成一个 “回波” chirp：`e` 等于 `s_full` 延迟 `delay_echo` 后的版本（`delay_echo ∈ [0, N-1]`）
- 回波幅度由 `SNR_dB = 0` 固定（相对噪声）

噪声 `n`：

- 复 AWGN：`n ~ CN(0, PN)`
- 每个样本固定一个 `PN`，在合成时先生成 `n`

将每路 jammer 按各自 JNR 缩放：

- 对第 k 路：
  - 计算 `Pj0 = mean(|j_k|^2)`（非零后）
  - 目标功率 `Pj = PN * 10^(JNR_k/10)`
  - 缩放 `j_k *= sqrt(Pj / (Pj0 + eps))`

最后：

- `b = e + n`
- `x = sum_k j_k + b`
- 存储 `X = [Re(x), Im(x)]`

## 3. 数据文件格式（NPZ，固定字段）

每个 split：`data/raw/cisrj_seppe_v1/{train,val,test}.npz`

字段必须包含：

- `X`: float32 `(num, 2, N)`
- `J`: float32 `(num, 3, 2, N)`  三路 jammer 真值（silence 路为 0）
- `G`: uint8   `(num, 3, N)`     三路 forwarding mask 真值（silence 路为 0）
- `Tl_us`: float32 `(num, 3)`    单位 µs（silence 路为 0）
- `Ts_us`: float32 `(num, 3)`    单位 µs（silence 路为 0）
- `NF`: int32 `(num, 3)`         取值 {0,1,2,3}
- `JNR_dB`: float32 `(num, 3)`   silence 路可写 0
- `K_active`: int32 `(num,)`     取值 2 或 3
- `meta_json`: str               保存 config 与参数定义

## 4. 模型架构（固定：两段式）

新增模型组合名称：**ME‑MVSepPE**

### 4.1 SepNet：分离前端（只做分离）

文件：`src/models/sepnet.py`

**输入**：`x` `(B,2,N)`

**输出**：

- `j_hat` `(B,3,2,N)`：三路 jammer 分离结果
- `b_hat` `(B,2,N)`：背景分量（回波+噪声）
- 保证混合一致性：`sum(j_hat)+b_hat = x`（通过 residual projection 实现）

结构固定为：

- Encoder：1D Conv（输入通道2，输出 C，stride=4）
- Separator：TCN 堆叠，输出 4 路 masks（3 jam + 1 bg）
- Decoder：1D ConvTranspose 重建到 2 通道波形
- Mixture consistency：残差平均分配到 4 路

### 4.2 PE‑Net：逐源门控测参（复用你们现有 ME‑MVNet 思想）

文件：`src/models/penet.py`（内部可复用 `memvnet.py` 的 Raw/TF/Mech 分支）

PE‑Net 对每个 `j_hat[:,k]` 独立运行一次（共享权重）

**输入**：单源 `j_k` `(B,2,N)`

**输出**：

- `g_hat` `(B,N)` sigmoid
- `Tl_hat_us` `(B,)` softplus（直接输出 µs）
- `NF_logits` `(B,4)` softmax 类别 `[0,1,2,3]`
- `Ts_hat_us` `(B,)`：由 `Ts = (E[NF]+1)*Tl` 计算（训练用 softmax 期望）

PE‑Net 内部固定 3 视图：

- Raw‑IQ branch：1D stem + TCN + Transformer（沿用）
- TF branch：STFT log‑mag + 2D CNN encoder（沿用）
- Mechanism branch：从 |j| 提取统计+自相关特征（逐源提取）

门控头：

- 从 Raw 特征序列解码到长度 N 的 `g_hat`

参数头：

- concat `[z_iq, z_tf, z_mech, pooled(H ⊙ g_hat)]` 后输出 `Tl_hat_us` 与 `NF_logits`

## 5. Loss 设计（固定：3 类 loss）

文件：`src/models/losses_seppe.py`

总 loss 只有三类：

### 5.1 分离 loss：PIT‑SI‑SDR（jam） + 背景 SI‑SDR

- 对每个样本，枚举 3! 个排列 π（K=3 固定，枚举比匈牙利更稳）
- 代价：
  - active jammer：`cost = -SI_SDR(j_hat_k, j_true_pi(k))`
  - silence jammer（NF=0 的路）：`cost = mean(|j_hat_k|^2)`（能量惩罚）
- 取最小总代价的排列 `π*`

定义：

- `L_sep = sum_k cost_k + (-SI_SDR(b_hat, b_true))`

### 5.2 门控 loss：BCE（逐源）

- 对齐排列 `π*` 后计算：
- `L_gate = sum_k BCEWithLogits(g_hat_k, g_true_pi*(k))`

### 5.3 参数 loss：Tl 回归 + NF 分类（逐源）

- 对齐排列 `π*`
- Tl 只对 `NF>0` 的 active 源计算回归：
  - `L_Tl = sum_{NF>0} Huber(Tl_hat_us, Tl_true_us)`
- NF 对所有源计算分类（包含 0 类）：
  - `L_NF = sum_k CE(NF_logits, NF_true)`

### 总损失（固定系数）

$$
L = L_{sep} + 0.5(L_{Tl}+L_{NF}) + 0.2L_{gate}
$$



## 6. 训练流程（固定：两阶段）

### Stage 1：只训练 SepNet（分离收敛）

- 训练目标：`L_sep`
- epoch：`E1 = 50`
- 输出 checkpoint：`runs/exp_xxx_sep/checkpoints/best.pt`

### Stage 2：联合训练（SepNet + PE‑Net）

- 加载 Stage1 最佳 SepNet
- 训练目标：完整 `L`
- epoch：`E2 = 80`

## 7. 评估指标（必须实现，禁止只报 NF acc）

文件：`src/eval/metrics_seppe.py`

评估时同样对每个样本求 `π*`（用分离 cost），对齐后统计：

### 连续误差

对所有 active 源（NF>0）：

- `Tl_MAE_us`, `Tl_RMSE_us`
- `Ts_MAE_us`, `Ts_RMSE_us`

### 分类

对所有源（含 0 类）：

- `NF_Acc`, `NF_macroF1`

### Wang‑style 容差准确率（核心）

对所有 active 源：

- A_{Tl}：`|Tl_hat - Tl| <= 0.15 µs` 的比例
- A_{Ts}：`|Ts_hat - Ts| <= 0.25 µs` 的比例
- A_{NF}：`NF_hat == NF` 的比例
- A_{total}：三者同时满足的比例

额外分桶评估（必须）：

- 按 `JNR_dB`（对每个源）分桶输出上述指标曲线
- 按 `K_active`（2/3）分别输出指标

## 8. 仓库改动（必须按此落地）

在现有结构上新增文件：

### configs/

- `data_composite.yaml`
- `model_sep.yaml`
- `model_pe.yaml`
- `train_sep.yaml`
- `train_joint.yaml`
- `eval_composite.yaml`

### scripts/

- `10_generate_dataset_composite.py`
- `11_sanity_check_composite.py`
- `12_train_sepnet.py`
- `13_train_seppe_joint.py`
- `14_eval_seppe.py`
- `15_export_plots_seppe.py`

### src/data/

- `isrj_composite_generator.py`
- `dataset_npz_composite.py`

### src/models/

- `sepnet.py`
- `penet.py`
- `losses_seppe.py`
- `pit_perm.py`（枚举 3! 排列的 helper）
- `sisdr.py`（SI‑SDR 实现）

### src/train/

- `trainer_sepnet.py`
- `trainer_seppe.py`

### tests/

- `test_composite_generator.py`
- `test_sepnet_shapes.py`
- `test_penet_shapes.py`
- `test_pit_perm.py`

## 9. 必须提供的命令（最终可复现）

## 10. 验收标准（必须通过）

### 数据层

- NPZ 字段与 shape 完全符合第 3 节
- dual 样本满足：第三路 `J==0`、`G==0`、`NF==0`
- `Ts_us == (NF+1)*Tl_us` 对 active 源严格成立

### 训练层

- Stage1：SepNet 的 `L_sep` 下降，平均 SI‑SDR 提升为正
- Stage2：`A_total` 随 epoch 上升；同时报告 NF acc 与 A_total

### 评估层

- 输出：
  - `runs/exp_xxx/tables/test_metrics.json`
  - `runs/exp_xxx/tables/test_metrics_by_jnr.csv`
  - `runs/exp_xxx/tables/test_metrics_by_kactive.csv`
  - `runs/exp_xxx/figures/` 下包含：
    - Tl/Ts scatter（µs）
    - NF confusion matrix（含 0 类）
    - A_total vs JNR 曲线
    - 分离示例波形（混合、GT 三路、预测三路）

## 11. 禁止事项

- 禁止使用参考信号作为模型输入
- 禁止引入额外一致性 loss（Ts 由结构计算保证一致）
- 禁止仅用 NF acc 作为结论指标（必须同时给出 A_total）

如果你把这份指导交给 AI 开发者，它会知道：**要新写哪些文件、每个模块输入输出是什么、loss 怎么写、怎么训练、怎么评估、跑哪些命令、最后应该产出哪些图表**。

你希望我把这份指导进一步“落地到配置文件模板（data_composite.yaml / train_joint.yaml）”，把每个字段和值也固定下来吗？这样 AI 可以直接照 YAML 开发，不需要再自己决定超参命名。