# 总体方案：删 NF 权重 + 加 GateFormer（Transformer）读门控周期

## 目标

1. **取消 NF 类别权重**，恢复标准 CE
2. **把 PE-Net 的参数头升级为 Transformer（GateFormer）**：输入是逐源的时序特征 + 门控序列，Transformer 用长程注意力直接捕捉“重复/周期”，输出 Tl/NF，再由结构计算 Ts
3. 保持你们现有叙事不变：**分离 → 门控 → 测参**，loss 仍然是 3 类（sep / gate / param），外加你们 Need2 的 Ts loss（这个已经证明有效）

------

# 改动 1：彻底删除 NF 类别权重（你要求的第一步）

## 要改

- `configs/train_joint.yaml`：删除 `nf_class_weights` 字段
- `src/models/losses_seppe.py`：NF 的 CE 改回无权重

## 标准写法

```python
ce = torch.nn.CrossEntropyLoss()
L_NF = ce(NF_logits.reshape(-1, num_classes), NF_true.reshape(-1))
```

这一步做完，0 类误报会自然下降，但 active→0 漏检可能回升；下面的 GateFormer 专门用来补这块。

------

# 改动 2：在 PE-Net 里加 Transformer（GateFormer），用“门控周期结构”直接推 Tl/NF

你们现在 PE-Net 大致是：Raw/TF/Mech 三视角 → 拼接 → MLP 输出 Tl 与 NF。
我把它改成：**仍保留三视角，但参数读取改为 Transformer 查询门控结构**（能讲清楚：ISRJ 参数本质就是门控周期）。

## 2.1 GateFormer 的输入与输出（写死）

### 输入（逐源）

- `H_k`: Raw 分支时序特征，shape `(B, L, D)`（注意：L 是下采样长度，如 1000；D 如 256）
- `g_logit_k`: 门控 head 输出的 logit，shape `(B, N)`（未 sigmoid）
- `z_tf_k`: TF 分支 embedding，shape `(B, Dtf)`
- `z_mech_k`: Mech 分支 embedding，shape `(B, Dm)`

### 输出（逐源）

- `Tl_hat_us`: `(B,)`（softplus 正数，单位 µs）
- `NF_logits`: `(B, 4)`（0/1/2/3 四分类）
- `Ts_hat_us`: `(B,) = (E[NF]+1)*Tl_hat_us`（结构计算，不新增 head）

------

## 2.2 GateFormer 的结构（清晰可画）

**关键点：Transformer 不去分离信号，它只做“周期模式读取”。**

### Step A：把门控序列变成低分辨率 token，并与 H_k 对齐

- 将 `g_logit_k` 用 average pooling 降采样到长度 L：
  `g_ds = AvgPool1d(kernel=N/L, stride=N/L)(g_logit_k)` → `(B, L)`
- 拼到时序特征上：
  `U = concat([H_k, g_ds[...,None]], dim=-1)` → `(B, L, D+1)`
- 线性投影回 D：`U = Linear(D+1→D)(U)`

### Step B：Transformer Encoder 处理 U

- 2–4 层 TransformerEncoder（heads=4 或 8，d_model=D，dropout=0.1）
- 加固定可学习位置编码 `pos_emb`（length=L）

输出：`U'` shape `(B, L, D)`

### Step C：两个“参数查询 token”

- `q_Tl`、`q_NF` 两个 learnable query token，shape `(B, 2, D)`
- 用 cross-attention（TransformerDecoder layer 或自己写一次 cross-attn）让 query 去 attend `U'`

得到：

- `h_Tl` `(B,D)`
- `h_NF` `(B,D)`

### Step D：输出头

- `Tl_hat_us = softplus(MLP_Tl(h_Tl))`
- `NF_logits = MLP_NF(h_NF)`（输出 4 类 logits）

同时把 TF/Mech embedding 注入 query（让估参不只靠门控）：

- `cond = Linear(concat(z_tf_k, z_mech_k) → D)`
- `q_Tl = q_Tl + cond`，`q_NF = q_NF + cond`

------

## 2.3 为什么这比“类别权重”更本质、更可讲

- Tl/NF/Ts 是 **门控的周期参数**，不是 chirp 内容参数
- 门控结构在复合场景里被残余互串/噪声污染时，单纯 MLP 更容易只抓局部统计；Transformer 的优势是能在全局范围聚合“重复间隔与周期性一致性”
- 取消类别权重后，模型不再“靠偏置赌 active”，而是靠长程周期证据判定 NF，解释性更强

------

# 改动 3：loss 保持清晰（不堆叠），只调整 param 部分

你们现在最有效的一条是 **Ts loss**（Need2 已经证明能显著抬 A_total）。这条保留。

最终 param loss（逐源对齐 perm 后）：

- `L_gate = BCEWithLogits(g_logit, g_true)`
- `L_Tl = Huber(Tl_hat_us, Tl_true_us)`（active 源）
- `L_NF = CE(NF_logits, NF_true)`（含 0 类）
- `L_Ts = Huber(Ts_hat_us, Ts_true_us)`（active 源）

总 loss 不变成“很多项”，还是清晰的三块：
[
L = L_{sep} + \lambda_p(L_{Tl}+L_{NF}+L_{Ts}) + \lambda_g L_{gate}
]
你们现有系数直接沿用：

- `lambda_p=0.5`
- `lambda_g=0.2`

------

# AI 执行清单（按这个顺序改，保证一次能跑通）

## Step 1：删 NF 权重

-  删 config 字段
-  `losses_seppe.py` CE 改成无权重

## Step 2：实现 GateFormer 模块

新增文件：

-  `src/models/gateformer.py`

包含：

- downsample gate logits 到 L
- transformer encoder
- 两个 query token cross-attn
- 输出 Tl_hat_us / NF_logits

## Step 3：把 PE-Net 的 param head 替换成 GateFormer

修改：

-  `src/models/penet.py`
  - 保留原本 Raw/TF/Mech 分支与 gating head
  - 将原本 `fusion_mlp → Tl/NF head` 替换为 `GateFormer(H_k, g_logit_k, z_tf_k, z_mech_k)`

## Step 4：训练与验收

-  用你们当前 datafix 数据集跑 joint formal
-  重点看：
  - `A_Ts` 与 `A_total` 是否至少维持 Need2-E2 的水平（0.47 附近）
  - 0 类误报是否回落（真0→预测0 上升）
  - active→0 漏检是否不明显反弹

