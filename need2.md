下面是基于你们 **26662a5**（datafix 后分离已变正、A_total≈0.30）这条主线的**下一步方案**。目标非常明确：**把 A_Ts / A_total 再往上推**，同时进一步降低 “active → 0（漏检）” 的 NF 错误。改动保持小、结构叙事不变（仍是 SepNet 分离 + PE-Net 逐源门控测参）。

---

# 下一步方案总览（Need3）

## 核心判断

现在 **SepNet 已经能分（SI-SDR>0）**，整体瓶颈转移到了 **Ts 精度与 active→0 漏检**：

* `A_NF≈0.79`、`A_Tl≈0.54`，但 `A_Ts≈0.33`（卡住 A_total）
* Ts 本质由 `Ts=(NF+1)*Tl` 决定，所以 **Ts 过不了容差= Tl 精度不够 + NF 偶尔错** 的组合结果。

---

# 改动方案（交给 AI 改代码）

## 改动 1：在 joint 阶段显式加入 **Ts 监督 loss**（仍不预测 Ts，Ts 由结构计算）

### 目的

直接把优化目标对齐到评估里最卡的指标 `A_Ts`，让网络在训练时“以 Ts 为准”自动调节 Tl/NF。

### 改哪些文件

* `src/models/losses_seppe.py`
* `configs/train_joint.yaml`（增加/开启权重）

### 具体怎么改

在 `losses_seppe.py` 的参数损失部分（对齐 perm 后），已有：

* `L_Tl`（只对 active 源）
* `L_NF`（对全部源含 0 类）
* `L_gate`

现在新增：

* `L_Ts`（只对 active 源），其中：
  * `NF_probs = softmax(NF_logits)`（4类：0/1/2/3）
  * `E_NF = Σ NF_value[i]*NF_probs[i]`，`NF_value=[0,1,2,3]`
  * `Ts_hat_us = (E_NF + 1) * Tl_hat_us`
  * `L_Ts = Huber(Ts_hat_us, Ts_true_us)`

并把参数损失改成：

$$
L_{param} = L_{Tl} + L_{NF} + w_{ts} L_{Ts}
$$

配置里固定：

* `w_ts = 1.0`

> 这一步不会引入“多余一致性 loss”，因为 Ts 仍然由结构计算，不增加新的自由度。

---

## 改动 2：NF 的交叉熵加入 **类别权重**，专门打击 “active→0（漏检）”

### 目的

你们现在最伤 `A_Ts/A_total` 的 NF 错误模式是把 active 判成 0。把 active 类的 true-class 权重抬高，能显著减少这种漏检。

### 改哪些文件

* `src/models/losses_seppe.py`（NF CE 的地方）
* `configs/train_joint.yaml`（传权重）

### 具体怎么改

在 `CrossEntropyLoss` 里加入 `weight`，固定为：

* `NF_class_weights = [0.5, 1.5, 1.5, 1.5]`

（0 类权重低，1/2/3 类权重高）

实现方式：

这会让 “真 active 被判成 0” 的损失变大，从而减少漏检。

---

## 改动 3：把 Tl 的回归改成 **“按 (NF+1) 加权”**（直接对齐 Ts 的放大效应）

### 目的

同样的 Tl 误差，在 NF=3 时会被放大 4 倍进入 Ts 误差；因此 Tl 回归应对高 NF 更敏感。

### 改哪些文件

* `src/models/losses_seppe.py`

### 具体怎么改

对齐 perm 后，对 active 源（NF_true>0）计算 Tl 的 Huber，然后乘权重：

* `w_k = (NF_true_k + 1)`（单位无关）
* `L_Tl = mean( w_k * huber(Tl_hat_us, Tl_true_us) )`

这一步结构上非常清晰：**Ts 的误差来源就是 (NF+1)*Tl**，训练时按同样比例强调 Tl。

---

# 训练与验证流程（固定执行顺序）

为保证可控，按下面顺序做三次实验（每次只改一项，便于归因）：

### 实验 E1（只加 Ts loss）

* 开启 `w_ts=1.0`
* 其它不变
* 关注：`A_Ts`、`Ts_MAE_us`、`A_total`

### 实验 E2（在 E1 基础上加 NF class weights）

* `NF_class_weights=[0.5,1.5,1.5,1.5]`
* 关注：confusion matrix 里 **真1/2/3→预测0** 是否明显减少；`A_NF`、`A_total`

### 实验 E3（在 E2 基础上加 Tl 加权）

* `L_Tl` 乘 `(NF_true+1)`
* 关注：高 NF（尤其 NF=3）下的 Ts 误差是否显著下降；`A_Ts`、`A_total`

---

# 必须新增的诊断输出（让你们快速判断是不是“Ts 卡住”被解决）

在 `scripts/14_eval_seppe.py` 或 `src/eval/metrics_seppe.py` 增加一份 breakdown CSV：

* `metrics_by_nf.csv`：按真 NF=1/2/3 分别统计
  * Tl_MAE_us、Ts_MAE_us、A_Tl、A_Ts、A_NF、A_total

* `metrics_cond_nf_correct.csv`：分两组统计
  * NF 预测正确时的 Ts_MAE / A_Ts
  * NF 预测错误时的 Ts_MAE / A_Ts

这样你能立刻判断 Ts 错误到底主要来自 NF 错还是 Tl 精度不足。

---

# 交给 AI 的“执行清单”（最短可操作版）

1. `losses_seppe.py`：
   * 新增 `L_Ts`（用 `Ts_hat=(E_NF+1)*Tl_hat`）
   * NF CE 加 class weights
   * Tl Huber 乘 `(NF_true+1)`

2. `configs/train_joint.yaml`：
   * 添加 `w_ts: 1.0`
   * 添加 `nf_class_weights: [0.5,1.5,1.5,1.5]`

3. `metrics_seppe.py`：
   * 输出 `metrics_by_nf.csv`、`metrics_cond_nf_correct.csv`

4. 跑三组实验（E1/E2/E3），每组输出完整 `paper/tables` 和 `paper/figures`

---

如果你把 `losses_seppe.py` 里当前的函数名（例如 `compute_param_loss` / `compute_total_loss`）那几行贴出来，我可以把上面三项改动写成**逐行 patch**（精确到插入点与变量名），AI 直接复制粘贴就能提交新 commit。