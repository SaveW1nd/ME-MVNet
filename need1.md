下面给你一套**“只改 2–3 处代码、但能显著抬 SepNet 分离质量（进而抬 A_total）”**的方案。核心目标很明确：**把 SepNet 的 jammer SI-SDR 从负值拉到 ≥0 dB**，先解决“源能量被背景/别的源吞掉”的问题（你现在最大错误模式就是 active 源被 PE-Net 判成 0）。这一点从你当前 SepNet 里 **mask 用 sigmoid、残差均分**就能解释。

---

## 方案总览（3 个改动，全部都在你现在的框架里）

1. **SepNet 的 mask 从 sigmoid 改成 source-softmax（4 路竞争）**
2. **Mixture consistency 的 residual 从“平均分配”改成“按源能量比例分配”**
3. **SepNet 的 background loss 从 SI-SDR 改为 MSE，并固定小权重（0.1）**

这三项只动 `sepnet.py` + `losses_seppe.py`，其它不动。

---

# 改动 1：SepNet masks 用 softmax 竞争（避免多源同时激活）

你现在在 `src/models/sepnet.py` 里是：

* `mask_logits = ... view(bsz,4,c,l)`
* `masks = torch.sigmoid(mask_logits)`
* 每一路 mask 都是独立的，**不会互相竞争**，同一个 TF/时域片段可能同时被多个源“拿走”。

**改法（只改一行）：**

把

改成

原因：softmax 让每个 `(c,l)` 位置上 4 个源的 mask **加起来=1**，网络必须做“归属选择”，这会直接提升分离的可辨识性（尤其在重叠时）。

---

# 改动 2：Residual 不能均分，要按能量比例分（避免把源“抹平”）

你现在的 mixture consistency 是：

也就是说 residual 被平均分给 4 路。

这会导致一个很典型的问题：**哪怕 SepNet 学到了一点源结构，最后也会被 residual 均分“重新搅在一起”**，SI-SDR 很难上 0 dB。

**改法（替换这段残差分配逻辑）：**

在 `sources = decoded.view(bsz,4,2,n)` 之后（你现在已经有这个张量），改成：

效果：残差会更多地补到“本来就应该承载能量”的那一路，**不会把所有路强行拉回到同一个平均解**。这一步通常对 SI-SDR 提升非常明显，尤其是你目前 jam SI-SDR 还在负值区间的时候。

---

# 改动 3：background 的 SI-SDR 换成 MSE（并固定 0.1 权重）

你现在 `compute_sep_loss()` 里对 background 用的是：

* `b_true = X - sum(j_true)`
* `l_bg = -si_sdr(b_hat, b_true).mean()`
* `l_sep = l_jam + l_bg` 

background 里包含 **echo + AWGN**，SI-SDR 这种“尺度不敏感”的度量在噪声项上会带来不稳定梯度，而且会让 stage-1 训练“更难对齐”。

**改法（改成 MSE 并小权重）：**

把 `compute_sep_loss` 中这一段：

改为：

理由：MSE 对噪声/回波残差是更直接的逐点约束；固定 0.1 权重保证 stage-1 的优化重心仍然是 jammer 分离（l_jam）。

> 这也不会破坏你 trainer 里统计 background SI-SDR 的日志（它只是指标，不是 loss）。

---

## 你改完后，应该看哪些“立刻能验证”的信号

在 **Stage-1 SepNet** 的训练日志里（`trainer_sepnet.py` 会记录 `SI_SDR_jam`）：

* 你现在 best val 的 jam SI-SDR 仍是负值（之前你发的分析里也提到），改完后目标是：

**val 的 `SI_SDR_jam` 进入正值（≥0 dB）**

一旦 SepNet 达到这个水平，Stage-2 的 **NF 漏检（active→0）会显著减少**，A_Ts、A_total 都会跟着上。

---

## 我建议你按这个顺序提交与验证

1. 先只做 **改动1+改动2**（都在 `sepnet.py`），跑 Stage-1 训练，看 `SI_SDR_jam` 是否明显抬升。
2. 再加 **改动3**（loss 里的 background MSE），看 Stage-1 收敛是否更平滑、best 更高。
3. 然后跑 Stage-2 joint，看 `A_NF` 与 `A_total` 在高 JNR 桶（18–20 dB）是否明显上升（你当前这些桶的提升最容易观测）。
