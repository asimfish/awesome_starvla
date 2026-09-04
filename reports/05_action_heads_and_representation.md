# 05 · 动作头与动作表示：原理、证据与选型

StarVLA 生态的三篇论文共享同一个骨干（Qwen3-VL-4B）和同一套代码，这让"动作头"这个变量第一次可以被单独隔离出来比较。本文把四种头的数学形式、三篇论文给出的全部对照数字，以及"动作空间怎么设计"的证据汇总在一处，最后给一个选型指南。

## 1. 四种头的形式化

记骨干在时刻 $t$ 的多模态 hidden state 为 $H_t$，待预测的动作块为 $A_t=(a_t,\dots,a_{t+K-1})$，$K$ 为 chunk 长度。

### FAST：自回归离散 token

把连续动作块经 FAST tokenizer（DCT 压缩 + BPE）编成 token 序列 $z_{1:M}=\mathrm{Tok}(A_t)$，在 LLM 自己的词表空间里做 next-token 预测：

$$
\mathcal{L}_{\text{FAST}}=-\sum_{m=1}^{M}\log p_\theta(z_m\mid H_t,z_{<m})
$$
推理时逐 token 生成再逆 tokenizer。优点：不改 VLM 架构；代价：连续控制经过离散瓶颈，且解码是串行的。

### OFT：并行连续回归

在输入序列末尾拼 $K$ 个 action query token，取其 hidden state $h^{\text{act}}_{t,k}$ 用小 MLP $g_\phi$ 逐个回归：

$$
\hat a_{t+k}=g_\phi(h^{\text{act}}_{t,k}),\qquad
\mathcal{L}_{\text{OFT}}=\frac{1}{Kd_a}\sum_{k}\left\|g_\phi(h^{\text{act}}_{t,k})-a_{t+k}\right\|_1
$$
一次前向输出整个 chunk。它同时是一个诊断工具：如果 OFT 表现好，说明骨干表征里线性可读地暴露了连续控制所需的信息。代价是点估计，不能表达多模态动作分布。

### PI：flow-matching 动作专家

学一个把高斯噪声搬运到示教动作块的向量场。取 $\epsilon\sim\mathcal N(0,I)$、$\tau\in[0,1]$，线性概率路径 $A_t^\tau=(1-\tau)\epsilon+\tau A_t$，目标速度 $u=A_t-\epsilon$：

$$
\mathcal{L}_{\text{PI}}=\mathbb{E}\left[\left\|v_\phi(A_t^\tau,\tau;H_t)-(A_t-\epsilon)\right\|_2^2\right]
$$
推理从噪声出发做 $N$ 步欧拉积分。StarVLA 实现为逐层 cross-attention 到多层 VL hidden state 的 DiT。

### GR00T：双系统 flow matching

同样的 flow-matching 目标，但 DiT 运动模块被显式分离为 System 1，额外条件于本体感受状态 $s_t$ 和本体标识 $e$：

$$
\mathcal{L}_{\text{GR00T}}=\mathbb{E}\left[\left\|v_\phi(A_t^\tau,\tau;H_t,s_t,e)-(A_t-\epsilon)\right\|_2^2\right]
$$
VLM 作为 System 2 只提供场景与指令的 token；状态与噪声动作在独立运动模块里嵌入、处理、解码回本体原生动作空间。

## 2. 三篇论文的对照数字

### 2.1 数据充足时：连续头相当，离散头落后

| 来源 | 设定 | FAST | OFT | PI | GR00T |
|---|---|---|---|---|---|
| StarVLA 报告表 2 | LIBERO avg，30K 步 | 95.4 | 96.6 | 95.7 | 96.5 |
| StarVLA-α 表 2 | LIBERO avg | 97.8 | 98.8 | 98.1 | 98.7 |
| StarVLA-α 表 2 | SimplerEnv WidowX | 35.6 | 64.6 | 65.9 | 65.3 |
| StarVLA-α 表 2 | RoboTwin clean* | 72.5 | 88.2 | 88.1 | 88.0 |
| StarVLA-α 表 2 | RoboCasa-GR1 | 45.0 | 53.8 | 48.9 | 52.8 |
| VLAct 表 2 | RoboTwin Data Scaling Clean（VLAct 骨干） | – | 92.5 | 93.0 | 89.6 |

FAST 在 WidowX 上落后 29 个点、RoboTwin 落后 16 个点；三种连续头之间差距在 LIBERO 与 RoboTwin 上不超过 1 个点，GR1 上 PI 低 5 个点。

### 2.2 数据稀少或扰动强时：回归头领先

| 来源 | 设定 | OFT | PI | GR00T |
|---|---|---|---|---|
| VLAct 表 2 | RoboTwin Base Clean（VLAct 骨干） | 80.5 | 77.0 | 76.0 |
| VLAct 表 2 | RoboTwin Base Random（VLAct 骨干） | **41.5** | 23.7 | 22.9 |
| VLAct 表 9 | RoboTwin，从零微调 | 61.7 | 60.5 | 51.2 |

只用 50 条 clean 轨迹/任务时，两种生成式头在 Random 测试集上比回归头低 18 个点。这是"头无关"结论的明确边界，也是路线图 D3 的出发点。

### 2.3 预训练头如何影响下游换头（VLAct 表 8/9）

| 预训练头 | 下游 PI 微调 | 相对从零 |
|---|---|---|
| 无 | 60.5 | – |
| OFT | 55.1 | −5.4 |
| OFT + GR00T | 63.1 | +2.6 |
| OFT + PI + GR00T | 77.0 | +16.5 |

| 下游头 | 从零 | 同头单头预训练 | 三头预训练 | 三头 − 单头 |
|---|---|---|---|---|
| OFT | 61.7 | 78.8 | 80.5 | +1.7 |
| PI | 60.5 | 75.4 | 77.0 | +1.6 |
| GR00T | 51.2 | 71.7 | 76.0 | +4.3 |

单头预训练会让未见过的头低于从零起点（decoder lock-in）；加第二个头就翻正；三头对同头也有净增益，GR00T 受益最大。

## 3. 动作空间设计的证据

| 问题 | 方案 | 证据 | 结论 |
|---|---|---|---|
| 多本体动作维度不同怎么办 | 每本体独立头 vs 零填充到 32 维 vs RDT 物理统一空间 | StarVLA-α 表 6：GR1 53.5 / **57.3** / 52.3 | 零填充 + 让 VLM 自己分辨本体，优于两种"专门设计" |
| 零填充之上还能做什么 | 独立头 vs 统一头 vs 部分统一（夹爪共享、手臂分开、非激活维 mask） | VLAct 表 12：RoboTwin 78.5 / 79.5 / **80.5**；LIBERO-Plus 81.1 / 81.4 / **82.6** | 物理同义维度显式共享再拿 1 个点 |
| 周期关节角 | 原始回归 vs 数据侧 wrap 到 $[-\pi,\pi]$ vs + 残差 wrap loss | VLAct 表 10：75.5 / 78.6 / **80.5** | +5 个点，是 VLAct 单项消融里最大的一项 |
| 关节 vs 末端、绝对 vs delta | delta / relative 动作 | StarVLA-α 表 4：低数据段 +1～6 个点，数据充足后归零 | 参数化的收益随数据量衰减 |
| 本体感受与历史帧 | +proprio / +2 帧历史 | StarVLA-α 表 4：proprio 在 RoboTwin 50×50 上 +10.5；历史帧多数设定略降 | 状态有用，朴素堆帧无用 |

## 4. 选型指南

| 场景 | 建议 | 理由 |
|---|---|---|
| 单本体、数据充足（每任务 ≥ 数百条）、追求简单 | OFT | 与生成式头持平，一次前向，实现最小 |
| 低数据（每任务 ≤ 50 条）或强视觉扰动 | OFT | RoboTwin Base/Random 领先 18 个点 |
| 需要表达多模态动作分布（多解任务、高频接触） | PI 或 GR00T | 回归头是点估计 |
| 需要状态条件、多本体、独立运动模块的部署形态 | GR00T | 状态与本体标识进 System 1，VLM 可独立更新 |
| 只能改 LLM 输出层、不能加模块 | FAST | 唯一不改架构的方案，接受 15–30 个点的代价 |
| 做持续预训练（塑造骨干供他人复用） | **多头共监督**（至少两个连续头） | 单头预训练会造成 decoder lock-in |
| 多本体训练 | 零填充 + 把物理同义维（夹爪）对齐 + 非激活维 mask | StarVLA-α 表 6 与 VLAct 表 12 叠加 |
| 关节角控制 | 数据侧 wrap + 残差 wrap loss，必做 | +5 个点，零成本 |

## 5. 悬而未决

- 三头预训练的计算开销没有公开数字（路线图 E2）。
- 生成式头在低数据段落后的原因未被诊断：是收敛慢、目标过拟合，还是去噪步数不足（路线图 D3）。
- "头多样性"是否可以推到非动作头（未来帧预测、空间 QA），是路线图 B1/B4 的问题。
- 部分统一布局在人形与灵巧手上如何扩展（路线图 C1）。

代码层面每种头的类、forward 路径与 loss 实现见 [02 · 代码库解析](02_starvla_codebase_analysis.md)。
