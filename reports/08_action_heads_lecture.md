# 08 · 讲稿：四种动作头（FAST / OFT / PI / GR00T）到底是什么

> 用途：一次 60 分钟的组会讲解，面向已经了解 VLM 但没有做过 VLA 的听众。每节开头给"讲述提示"（用什么直觉切入、板书什么），正文是可以直接照读的讲稿，方框内是公式速查。配套幻灯片：[`report/action_heads_lecture_slides.pdf`](../report/action_heads_lecture_slides.pdf)。

## 0. 材料清单

| 头 | 原始论文 | 出处 | 英文 PDF | 中文翻译 |
|---|---|---|---|---|
| FAST | Pertsch et al., *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, [arXiv 2501.09747](https://arxiv.org/abs/2501.09747)，2025-01 | Physical Intelligence | [papers/en](../papers/en/2501.09747_FAST.pdf) | [papers/zh](../papers/zh/2501.09747_FAST_zh.pdf) |
| OFT | Kim, Finn, Liang, *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*（OpenVLA-OFT），[arXiv 2502.19645](https://arxiv.org/abs/2502.19645)，2025-02 | Stanford | [papers/en](../papers/en/2502.19645_OpenVLA_OFT.pdf) | [papers/zh](../papers/zh/2502.19645_OpenVLA_OFT_zh.pdf) |
| PI | Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*，[arXiv 2410.24164](https://arxiv.org/abs/2410.24164)，2024-10 | Physical Intelligence | arXiv 非独占许可，不随仓库分发 | 同左；中译版只保存在本机 `~/Desktop/research/papers/heads_local/` |
| GR00T | Bjorck et al., *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*，[arXiv 2503.14734](https://arxiv.org/abs/2503.14734)，2025-03 | NVIDIA | [papers/en](../papers/en/2503.14734_GR00T_N1.pdf) | [papers/zh](../papers/zh/2503.14734_GR00T_N1_zh.pdf) |

背景论文（不翻译，讲稿中会解释所需部分）：Lipman et al., *Flow Matching for Generative Modeling*，[arXiv 2210.02747](https://arxiv.org/abs/2210.02747)；OpenVLA，[arXiv 2406.09246](https://arxiv.org/abs/2406.09246)。StarVLA 中四个头的实现文件在 §7 列出。

四篇论文中 FAST、OpenVLA-OFT、GR00T N1 为 CC BY 4.0，原文与译文均已入库；π0 为 arXiv 非独占许可，仓库只放链接。

## 1. 开场（5 分钟）

**讲述提示**：先不讲任何一个头，先把"问题"钉死。板书一行：VLM 会输出 token，机器人要的是连续、高频、多维的动作序列。四个头就是四种把前者变成后者的办法。

一个 VLA（Vision-Language-Action）模型由两部分组成：一个在互联网图文上预训练过的视觉-语言模型（VLM）负责"看懂场景、听懂指令"，一个动作头负责把 VLM 的内部表征变成机器人能执行的控制量。VLM 那一半大家都熟，真正让 VLA 与聊天模型不同的是动作头。

动作头要解决的问题有三个特点：

1. **连续**。关节角、末端位姿、夹爪开合都是实数，VLM 的原生输出是离散 token。
2. **高频且成块**。现代策略一次预测一个"动作块"（action chunk）：未来 H 步的动作一起输出，H 从 8 到 50 不等，控制频率 10–50 Hz。一次前向要产出 H × D 个实数（D 是动作维度，单臂 7、双臂 14、人形更多）。
3. **多模态分布**。同一场景下可能有多种同样合理的动作（从左边绕还是从右边绕），点估计会取平均，产生两边都不是的动作。

四条路线一句话：

- **FAST**：不改 VLM，把动作块压缩成一串离散 token，让 VLM 像生成文字一样生成动作。
- **OFT**：在 VLM 末尾拼一组占位 token，用一个小 MLP 把它们的隐状态直接回归成连续动作，一次前向出整块。
- **PI（π0）**：给 VLM 配一个专门的"动作专家"网络，用 flow matching 从噪声逐步生成动作块，能表达多模态分布。
- **GR00T**：同样用 flow matching，但把它放进一个显式分离的"快系统"（DiT），VLM 作为"慢系统"只提供条件；状态和本体标识进快系统。

接下来每个头讲四件事：它想解决什么、核心机制、训练与推理怎么做、在 StarVLA 代码里长什么样。

## 2. 共同记号

| 记号 | 含义 |
|---|---|
| $o_t$ | 时刻 $t$ 的观测：多视角 RGB 图像 + 语言指令（部分头还含本体感受状态 $s_t$） |
| $A_t = (a_t, a_{t+1}, \dots, a_{t+H-1})$ | 动作块，$H$ 为块长，每个 $a \in \mathbb{R}^D$ |
| $H_t$ | VLM 处理 $o_t$ 后的隐状态序列（最后一层或多层） |
| $z$ | 骨干交给动作头的潜表征（StarVLA 报告里的写法） |
| $\tau \in [0,1]$ | flow matching 的"时间"，与机器人时刻 $t$ 无关 |
| $\epsilon \sim \mathcal N(0, I)$ | 高斯噪声，形状与 $A_t$ 相同 |

## 3. FAST：把动作当时间序列压缩（12 分钟）

**讲述提示**：从"为什么最直接的办法会失败"讲起。板书两条正弦曲线，一条采样 5 Hz，一条 50 Hz，问听众：如果模型每次只预测下一个采样点，哪条更好学？

### 3.1 它要解决的问题

最早的 VLA（RT-2、OpenVLA）用最朴素的办法把动作变成 token：每个时刻、每个维度单独分成 256 个箱，一个箱一个 token。7 维动作一步就是 7 个 token，一个 50 步的块要 350 个 token。这在低频数据（Bridge、RT-1，约 5 Hz）上能用，到高频数据（DROID 15 Hz 以上）就学不动了。

FAST 论文用一个玩具实验把原因说清楚：自回归模型的学习信号来自"给定前面所有 token，下一个 token 带来的新信息"。对平滑信号，采样频率越高，相邻两步的差别越小，下一个 token 的边际信息趋近于零；模型最省力的解法是复制上一个 token，于是学出来的策略几乎不动。这不是数据太难，是 token 化方式把信息稀释了。

### 3.2 核心机制：DCT + BPE

直觉是 JPEG。图像像素在空间上平滑，所以离散余弦变换（DCT）后能量集中在少数低频系数，把高频小系数丢掉就是压缩。动作在时间上也平滑，同样的招数可以用在每个动作维度的时间序列上。

FAST 的五步（论文图 4、算法 1）：

1. **归一化**：用训练集每个维度的 1% 和 99% 分位数把动作映射到 $[-1, 1]$。用分位数而不是最值，是为了不被大数据集里偶发的离群动作带偏，也让不同本体、不同尺度的数据集能共用一个 tokenizer。
2. **逐维 DCT**：对块内每个维度的 $H$ 个值做 DCT，得到 $D \times H$ 的系数矩阵。
3. **缩放取整**：系数乘一个尺度因子 $\gamma$ 后取整。$\gamma$ 是唯一的超参数，控制"有损程度 vs 压缩率"。取整后矩阵大部分是零。
4. **展平**：把矩阵拉成一维整数序列，**低频系数在前**、各维度交错。这样序列开头就是动作块的整体形状，后面才是细节。
5. **BPE**：在整数序列上训练一个 byte-pair encoding tokenizer（词表 1024），把成串的零和常一起出现的系数组合并成一个 token。这是整条流水线里唯一需要"训练"的部分，训练只要几分钟。

结果：一个 1 秒的动作块每条手臂大约 30 个 token（论文表 I），比分箱少一个量级。这些 token 直接覆写 VLM 词表里最少使用的那部分，VLM 的架构和训练目标完全不变——还是 next-token 交叉熵。

$$
\mathcal{L}_{\text{FAST}} = -\sum_{m=1}^{M} \log p_\theta\big(z_m \mid H_t, z_{<m}\big), \qquad z_{1:M} = \mathrm{Tok}_{\text{FAST}}(A_t)
$$

推理时自回归地生成 token，再走逆变换：BPE 解码 → 还原系数矩阵 → 逆 DCT → 反归一化。

**FAST+**：作者在约 100 万条真实机器人轨迹上训练了一个通用 tokenizer，覆盖单臂、双臂、移动机器人和各种控制频率，可以拿来直接用，不必为每个数据集重训。

### 3.3 结果与代价

- 在 DROID（15 Hz）上，用 FAST 训练的 π0 骨干做到了首个"零样本"桌面操作评测；在 LIBERO、20 Hz 的餐桌清理、T 恤折叠上与 diffusion 版 π0 相当。
- 训练效率：达到同等性能所需 GPU 小时比 diffusion 版 π0 少约 5 倍——因为目标就是普通的交叉熵，不需要在每个样本上采样噪声和时间步。
- 代价：**推理慢**。动作要一个 token 一个 token 地生成，几十个 token 的自回归解码比一次前向的回归头或几步去噪的生成头都慢；这也是后面 OFT 要解决的问题。
- 另一个代价：解码可能产生不合法的 token 序列（BPE 无法还原），需要兜底处理。

### 3.4 在 StarVLA 中

`starVLA/model/framework/VLM4A/QwenFast.py`：动作经 `physical-intelligence/fast` tokenizer 变成 `<robot_action_k>` 字串放进 assistant 回合；loss 就是 VLM 的 next-token 交叉熵；推理调用 `generate()`，然后按 Qwen3-VL-Action 词表中预留的 2048 个 action token 的 id 区间抽出 token 解码。它是四个头里唯一不给 VLM 加任何新模块的。

### 3.5 一句话记住

FAST 把"动作 token 化"从每步每维分箱换成时间序列压缩，用 DCT 去掉冗余、用 BPE 变成稠密 token；它保住了 VLM 的原生接口，付出的是推理速度和一层离散化的信息损失。StarVLA-α 表 2 里 FAST 在 WidowX 上比连续头低 29 个点，VLAct 的先导实验也显示 FAST 预训练能迁移但保留 FAST 头本身表现差——离散瓶颈是结构性的。

## 4. OFT：并行解码 + 连续回归（10 分钟）

**讲述提示**：这一节是"工程上把三件显而易见的事一起做对"。先算一笔账：OpenVLA 一步 7 个 token 串行解码，3–5 Hz；如果要 25 Hz 的双臂控制，差了一个量级。

### 4.1 它要解决的问题

OpenVLA-OFT 的出发点不是新架构，是问：拿一个现成的自回归 VLA（OpenVLA，7B），微调时哪些设计决定了速度和成功率？作者系统比较了三组选择——生成方式（自回归 vs 并行）、动作表示（离散 vs 连续）、学习目标（next-token 交叉熵 vs L1 回归 vs diffusion）——然后把最优组合打包成 OFT 配方。

### 4.2 核心机制

**并行解码。** 自回归模型每生成一个 token 要跑一次解码器前向，7 维动作就是 7 次，一个 K 步的块就是 $K \times D$ 次。OFT 的改法：往输入里放一组"空动作嵌入"（可以理解为占位 token），把解码器的因果注意力掩码换成双向注意力，一次前向同时得到所有位置的隐状态。

**动作块。** 有了并行解码，预测 K 步只需多放 K 组占位 token，吞吐提高 K 倍而延迟几乎不变。论文在 LIBERO 用 K=8，在 ALOHA 用 K=25（对应 25 Hz 控制下的一秒）。

**连续输出 + L1。** 占位 token 的最终隐状态不再经过 softmax 选箱，而是送进一个小 MLP 直接回归归一化后的连续动作，用 L1 loss 训练：

$$
\hat a_{t+k} = g_\phi\big(h^{\text{act}}_{t,k}\big),\qquad
\mathcal{L}_{\text{OFT}} = \frac{1}{K d_a}\sum_{k=0}^{K-1}\left\|\hat a_{t+k} - a_{t+k}\right\|_1
$$

作者也试了 diffusion 目标，发现在他们的设定下 L1 回归成功率相当、而训练和推理都简单得多。

**FiLM（OFT+）。** 在 ALOHA 上加了腕部相机后，策略容易只看图像、忽略语言（比如抓到勺子后不管指令让它做什么都倒进同一个碗）。解决办法是在视觉特征上做特征级线性调制（FiLM）：用语言嵌入生成缩放和偏置，强迫视觉通路携带指令信息。带 FiLM 的版本叫 OpenVLA-OFT+。

### 4.3 结果

- LIBERO 四套件平均从原版 OpenVLA 的 76.5% 提到 97.1%，当时的最好结果。
- 动作生成吞吐提升 26 倍；在 ALOHA 上用 25 步的块做到 43 倍，能跟上 25 Hz 控制。
- 真机 ALOHA 双臂任务上超过按默认配方微调的 π0 与 RDT-1B，也超过从零训练的 ACT 与 Diffusion Policy。

### 4.4 在 StarVLA 中

`starVLA/model/framework/VLM4A/QwenOFT.py`：在指令末尾拼一句形如"预测接下来 N 个动作"的提示并附 N 个 🔍 占位字符，取最后一层隐状态中这 N 个位置，送入两层残差 MLP（`MLPResNet`，H → 2H → D），loss 是带 mask 的 L1：$(|\hat a - a| \cdot m).\mathrm{sum}() / m.\mathrm{sum}()$。单次前向，无迭代。这是 StarVLA-α 的默认配置，也是 VLAct 的默认下游头。

### 4.5 一句话记住

OFT 证明：给一个足够强的 VLM，最简单的"占位 token + MLP + L1"就能拿到顶尖成功率和最快的推理。它的边界在表达能力——输出是点估计，多解任务上会取平均；但 StarVLA-α 和 VLAct 的数据表明，在数据充足或低数据 + 强扰动两种设定下，它都至少不输给生成式头（RoboTwin Base/Random 下 OFT 41.5 vs GR00T 22.9、PI 23.7）。

## 5. PI（π0）：用 flow matching 生成动作块（12 分钟）

**讲述提示**：这一节的难点是 flow matching 本身。不要从概率论讲，用"把一团噪声搬到一段轨迹"的图像讲：起点是高斯噪声，终点是真实动作块，网络学的是每个中间位置该往哪个方向走多远。

### 5.1 它要解决的问题

π0 的目标是通用的灵巧操作：叠衣服、收拾桌子、装袋，这些任务需要 50 Hz 的高频控制和长达几十步的动作块，而且同一状态下合理的动作不止一种。离散 token 有信息损失，点估计回归会把多解平均掉，所以作者选择了生成式建模——具体是 flow matching，它是 diffusion 的一个更简洁的变体。

### 5.2 flow matching 的最小版本

把真实动作块记为 $A$，噪声记为 $\epsilon$。在两者之间画一条直线，用 $\tau \in [0,1]$ 标记位置：

$$
A^{\tau} = \tau A + (1-\tau)\,\epsilon
$$

$\tau=0$ 是纯噪声，$\tau=1$ 是真实动作。沿这条直线走，每单位 $\tau$ 的位移是常数：

$$
u = A - \epsilon
$$

训练一个网络 $v_\theta(A^\tau, \tau; o_t)$ 去预测这个位移（"速度场"），损失就是均方误差：

$$
\mathcal{L}_{\text{PI}} = \mathbb{E}_{A,\ \epsilon,\ \tau}\left[\left\| v_\theta(A^{\tau}, \tau; o_t) - (A - \epsilon)\right\|_2^2\right]
$$

推理时从噪声出发，沿学到的速度场走 N 步（前向欧拉）：

$$
A^{\tau+\delta} = A^{\tau} + \delta\, v_\theta(A^{\tau}, \tau; o_t)
$$

π0 用 N=10 步。两个实践细节：(i) 训练时 $\tau$ 不是均匀采样，而是从一个偏向小 $\tau$（更接近噪声）的 Beta 分布采样，让网络多练"从很糟的起点纠正"这一段；(ii) 动作块整体一起去噪，块内各步之间用双向注意力，所以输出的 H 步动作是相互协调的。

和 diffusion 的关系：都是"从噪声迭代生成"，flow matching 用直线路径和速度目标，公式更简单、步数更少，StarVLA 里 PI 与 GR00T 两个头共用这一套目标。

### 5.3 动作专家：第二组权重

flow matching 需要一个网络吃噪声动作 $A^\tau$、时间 $\tau$ 和观测条件。π0 的做法是在 VLM 旁边加一个**动作专家**（action expert）：它和 VLM 走同一个 transformer 的注意力，但拥有**独立的一套权重**，只处理机器人状态和动作 token；图像和文本 token 仍由 VLM 的权重处理。这相当于一个两成员的混合专家（MoE）：按 token 类型路由到不同的权重。作者的理由是动作 token 的统计性质与语言 token 差异太大，混用一套权重会拖累两边。

规模：VLM 骨干用 PaliGemma（3B，互联网图文预训练），动作专家从零初始化约 3 亿参数，总计 3.3B。动作块 H=50，控制频率最高 50 Hz。

### 5.4 训练配方

预训练数据约 1 万小时的自有灵巧操作数据（7 种机器人配置、68 个任务）加上开源 OXE。作者强调两阶段：预训练数据要"广而杂"，让模型见过各种状态并学会从错误里恢复；后训练数据要"精而顺"，教模型把一个任务做得流畅。两者缺一都不行——只有精数据的模型不会纠错，只有杂数据的模型做不利索。

### 5.5 在 StarVLA 中

`starVLA/model/framework/VLM4A/QwenPI_v3.py`：不复用 PaliGemma，而是把 Qwen3-VL 最后 36 层的隐状态各经一个 LayerNorm + Linear 投影到 1024 维，逐层 cross-attention 到一个 36 层的 DiT（全 cross-attention，不交错 self-attention）；本体状态量化成 256 桶写进文本；flow matching 目标同上，$\tau$ 用 Beta(1.5, 1) 变换采样；推理 4 步欧拉。参数量 4.44B（骨干）+ 5.39 亿（DiT）+ 0.95 亿（投影）。

### 5.6 一句话记住

π0 把"动作生成"从分类或回归换成了生成式建模，能表达多模态、高频、长块的动作分布，代价是每次推理要跑 N 步去噪、并且多出一套动作专家的权重。VLAct 的表 8 显示：只用 OFT 单头预训练的骨干，换 PI 头微调反而低于从零（60.5 → 55.1），说明生成式头对骨干表征的"读法"和回归头不同——这是多头共监督的动机。

## 6. GR00T：双系统里的 flow matching（10 分钟）

**讲述提示**：GR00T 和 π0 都是 flow matching，讲清楚"差别在哪"比重复公式更重要。板书两栏：π0 的动作专家与 VLM 共享注意力、同频运行；GR00T 的 DiT 是独立模块、通过 cross-attention 读 VLM 特征、可以跑得比 VLM 快得多。

### 6.1 它要解决的问题

GR00T N1 是面向人形机器人的开源基础模型。人形本体维度高、需要高频闭环控制，而 VLM 的推理速度跟不上。作者借用"快思考 / 慢思考"的比喻做了一个双系统：**System 2** 是 VLM，负责理解场景和指令，约 10 Hz；**System 1** 是一个 diffusion transformer（DiT），负责实时生成动作，约 120 Hz。两者联合端到端训练，但推理时可以异步。

### 6.2 核心机制

**System 2。** 用 NVIDIA 的 Eagle-2 VLM（SmolLM2 语言模型 + SigLIP-2 图像编码器）。一个重要的工程发现：不用最后一层，而用**第 12 层**的隐状态作为条件——中间层保留了更多空间和视觉细节，末层过于语义化。公开的 GR00T-N1-2B 总参数 2.2B，其中 VLM 1.34B。

**System 1。** 一个带自适应层归一化（adaLN，用它注入去噪时间步 $\tau$）的 DiT。块内交替两种注意力：self-attention 在"噪声动作 token + 本体状态 token"上做，cross-attention 去读 System 2 输出的视觉-语言 token。训练目标仍是 flow matching：

$$
\mathcal{L}_{\text{GR00T}} = \mathbb{E}\left[\left\| v_\theta(A^{\tau}, \tau;\ H_t,\ s_t,\ e) - (A - \epsilon)\right\|_2^2\right]
$$

多出来的条件 $s_t$（本体感受状态）和 $e$（本体标识）是它与 π0 的关键区别之一。

**跨本体的编解码。** 不同机器人的状态和动作维度不同，GR00T 给每个本体配一对小 MLP：状态编码器把 $s_t$ 投到统一维度，动作解码器把 DiT 输出映射回该本体的原生动作空间。DiT 主体是共享的。动作块 16 步，推理 4 步去噪，在 L40 上一块动作 63.9 ms。

**数据金字塔与潜动作。** 底层是网页数据和人类第一视角视频，中层是合成的"神经轨迹"（用视频生成模型扩充）与仿真数据，顶层是真机数据。没有动作标签的视频怎么用？训练一个 VQ-VAE，从相邻两帧 $(x_t, x_{t+H})$ 学出一个"潜动作" $z_t$，把它当作一种额外的本体（LAPA），用同样的 flow matching 目标训练。这样人类视频也能进同一个训练循环。

### 6.3 在 StarVLA 中

`starVLA/model/framework/VLM4A/QwenGR00T.py`：用 Qwen3-VL **最后一层**隐状态作为 cross-attention 的 K/V（StarVLA 没有复刻"取第 12 层"）；DiT-B（768 维、12 头）16 层，交替 self/cross；状态经 MLP 编码后拼在序列最前；flow matching 目标同 PI，4 步欧拉，训练时每个样本重复 8 次不同噪声（`repeated_diffusion_steps=8`）。VLAct 的多头版本里 GR00T 头以 `state_dim: 0` 构建、状态改走文本，以便与其他头共用混合本体的 batch。

### 6.4 π0 与 GR00T 的差别（听众最常问）

| | π0 的动作专家 | GR00T 的 System 1 |
|---|---|---|
| 与 VLM 的耦合 | 同一注意力、独立权重（MoE 式） | 独立 DiT，通过 cross-attention 读 VLM 特征 |
| 取哪层特征 | 与 VLM 逐层交互 | 单层（论文第 12 层；StarVLA 用末层） |
| 运行频率 | 与 VLM 同步 | 可异步，System 1 远快于 System 2 |
| 状态与本体 | 状态 token 进专家 | 状态 + 本体标识经本体特定 MLP 进 DiT |
| 块长 / 步数 | H=50，10 步 | H=16，4 步 |
| 跨本体 | 统一填充动作空间 | 每本体一对编解码 MLP |

### 6.5 一句话记住

GR00T 把 flow matching 放进一个可以独立高频运行的模块，并用本体特定的编解码 MLP 处理多本体，同时把没有动作标签的视频通过潜动作纳入训练。StarVLA-α 表 2 里它与 OFT、π 在数据充足时相当；VLAct 表 9 里它是多头预训练受益最大的头（+4.3）。

## 7. 横向对比（8 分钟）

**讲述提示**：这张表是整场讲解的落点。让听众能在"接口 / 目标 / 推理 / 表达力 / 代码类"五个维度上一眼分清四个头。

| 维度 | FAST | OFT | PI（π0） | GR00T |
|---|---|---|---|---|
| 与 VLM 的接口 | 复用词表，动作 = token | 占位 token 的隐状态 → MLP | 独立权重的动作专家 | 独立 DiT，cross-attention 读 VLM |
| 动作表示 | 离散（DCT + BPE） | 连续，点估计 | 连续，分布 | 连续，分布 |
| 训练目标 | next-token 交叉熵 | L1 | flow matching MSE | flow matching MSE（+ 状态、本体条件） |
| 推理 | 自回归解码几十个 token | 一次前向 | N 步欧拉（π0 10 步；StarVLA 4 步） | N 步欧拉（4 步） |
| 新增参数 | 0 | 一个小 MLP | ~3 亿（π0）/ 5.4 亿 + 0.95 亿（StarVLA PI_v3） | DiT-B 16 层 + 本体 MLP |
| 多模态动作分布 | 可以（分类分布） | 不能 | 可以 | 可以 |
| 高频长块 | 依赖压缩比 | 天然支持 | 天然支持 | 天然支持 |
| 跨本体 | 统一 tokenizer（FAST+） | 零填充 / mask | 统一填充 | 本体特定 MLP |
| StarVLA 类 | `QwenFast` | `QwenOFT` | `QwenPI_v3` | `QwenGR00T` |

**同一骨干、同一数据下的数字**（StarVLA-α 表 2，Qwen3-VL-4B）：

| | LIBERO avg | WidowX | RoboTwin clean* | RoboCasa-GR1 |
|---|---|---|---|---|
| FAST | 97.8 | 35.6 | 72.5 | 45.0 |
| OFT（MLP） | 98.8 | 64.6 | 88.2 | 53.8 |
| PI | 98.1 | 65.9 | 88.1 | 48.9 |
| GR00T | 98.7 | 65.3 | 88.0 | 52.8 |

结论有两层：离散头明显落后；三种连续头在数据充足时相当。但 VLAct 补了两条边界：低数据 + 强扰动下回归头领先生成式头 18 个点（RoboTwin Base/Random：OFT 41.5 vs GR00T 22.9 / PI 23.7）；单头预训练会让骨干"锁进"该头的解码几何，换头微调反而低于从零（decoder lock-in），三头共监督才能得到对头不敏感的骨干。

## 8. 选型与常见问题（5 分钟）

**Q：我只有一个机械臂、几百条示教，用哪个？** OFT。实现最小、推理最快，成功率不输生成式头。

**Q：任务有明显的多解（比如从哪一侧抓），OFT 会怎样？** 回归头输出条件均值，可能落在两个解之间。这时换 PI 或 GR00T，它们输出的是分布中的一个样本。

**Q：想复用别人的 VLM checkpoint、不想加任何模块？** FAST 是唯一不改架构的方案，接受推理慢和 15–30 个点的代价。

**Q：多本体、需要状态输入、要把运动模块单独部署？** GR00T。状态和本体标识进 System 1，DiT 可以独立于 VLM 更新和加速。

**Q：我在做持续预训练，要交付一个骨干给别人用？** 至少两个连续头一起监督（VLAct 表 8：OFT 单头让 PI 微调 −5.4，加 GR00T 头后 +2.6）。

**Q：flow matching 和 diffusion 到底什么关系？** 同一族方法。diffusion 学"去噪"（预测噪声），flow matching 学"位移"（预测 $A - \epsilon$），路径是直线，公式和采样都更简单。在动作生成里两者效果接近，flow matching 步数更少。

**Q：为什么 GR00T 取 VLM 第 12 层而不是最后一层？** 中间层保留更多空间与视觉细节。这与 VLAct "冻结下半层保护低层视觉表征"的发现是同一件事的两面。

## 9. 阅读顺序

1. 先读 OpenVLA-OFT（[中译](../papers/zh/2502.19645_OpenVLA_OFT_zh.pdf)）第 IV 节：三组设计决策的对照实验最直观。
2. 再读 FAST（[中译](../papers/zh/2501.09747_FAST_zh.pdf)）第 IV–V 节：玩具实验 + 算法 1。
3. π0 第 IV 节（模型）与附录 B（flow matching 细节）：只看公式和图 3 即可；预训练配方部分留到做数据时再读。
4. GR00T N1（[中译](../papers/zh/2503.14734_GR00T_N1_zh.pdf)）第 2 节（模型）与 3.1（数据金字塔、潜动作）。
5. 回到 StarVLA 的四个框架文件对照代码（[02 · 代码库解析](02_starvla_codebase_analysis.md) 第 4 章），再看 [05 · 动作头与动作表示](05_action_heads_and_representation.md) 的证据汇总。

## 附：公式速查卡

| 头 | 前向 | 损失 |
|---|---|---|
| FAST | $z_{1:M} = \mathrm{BPE}(\mathrm{round}(\gamma\,\mathrm{DCT}(\mathrm{norm}(A))))$ | $-\sum_m \log p_\theta(z_m \mid H_t, z_{<m})$ |
| OFT | $\hat a_{t+k} = g_\phi(h^{\text{act}}_{t,k})$ | $\frac{1}{Kd_a}\sum_k \|\hat a_{t+k} - a_{t+k}\|_1$ |
| PI | $A^\tau = \tau A + (1-\tau)\epsilon$；推理 $A \leftarrow A + \delta\, v_\theta$ | $\mathbb{E}\|v_\theta(A^\tau,\tau;o_t) - (A-\epsilon)\|_2^2$ |
| GR00T | 同 PI，条件多了 $s_t, e$；本体 MLP 编解码 | $\mathbb{E}\|v_\theta(A^\tau,\tau;H_t,s_t,e) - (A-\epsilon)\|_2^2$ |

记号约定见 §2；π0 论文原文用 $\tau A + (1-\tau)\epsilon$，VLAct 论文写成 $(1-\tau)\epsilon + \tau A$，是同一条直线。
