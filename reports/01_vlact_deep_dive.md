# 01 · VLAct 精读：以表示为中心的 VLA 持续预训练

| 项目 | 内容 |
|---|---|
| 论文 | Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models |
| arXiv | [2608.27550](https://arxiv.org/abs/2608.27550)，2026-08 |
| 作者 | Senqiao Yang†, Chengyao Wang†, Yuxin Chen, Zixuan Wang, Longxiang Tang, Haokun Gui, Jinhui Ye, Changsheng Lu, Xiaoyang Wu, Mingkang Zhu；导师 Pengguang Chen, Shu Liu, Zhuotao Tian, Hengshuang Zhao, Bei Yu, Jiaya Jia |
| 项目页 | https://starvla.github.io/VLAct |
| 代码库 | 基于 [StarVLA](https://github.com/starVLA/starVLA)；骨干 Qwen3-VL-4B；16 GPU |
| 本仓库文件 | [英文 PDF](../papers/en/2608.27550_VLAct.pdf) · [中文 PDF](../papers/zh/2608.27550_VLAct_zh.pdf) |

## 0. 一句话

机器人轨迹比图文数据难扩展一个量级，所以在**固定机器人数据预算**下，VLA 持续预训练的目标应该从"拟合更多动作"换成"把有限轨迹蒸馏成可迁移的视觉-动作表征"。VLAct 用三个都很轻的组件（冻结浅层 + caption 混训、三头共监督、部分统一动作空间）做到了这一点：**只换骨干权重、其余全部固定**，下游成功率提升 7.6–21.4 个点，在 RoboCasa-GR1 这个预训练从未见过的人形本体上，20% 数据就超过全量 GR00T-N1.6。

## 1. 问题设定

作者的出发点是一组事实，而非一个假设：

- 网页图文可以爬取，机器人轨迹只能靠遥操作或专门采集协议产生（DROID、RoboCoin、MolmoAct 都是这样来的）。
- 机器人策略要泛化的空间是组合且连续的：场景 × 物体 × 任务目标 × 本体 × 接触动力学。即便是几十万条轨迹的数据集，覆盖仍然稀疏且不均匀。
- 因此持续预训练不能指望"穷尽覆盖"。给定固定数据预算，决定下游表现的是这些轨迹**如何塑造骨干表征**，而不只是有多少条。

由此引出论文的核心视角：VLM 骨干不是"从通用视觉-语言预训练继承来的固定部件"，而是 VLA 的**一阶设计变量**。

### 术语澄清：这里的"持续预训练"

从一个已经预训练好的 VLM 出发，在广泛、多本体的机器人轨迹上训练，然后再做下游任务特定微调。π0、π0.5、GR00T N1/N1.5 都是这个设定，只是它们叫"VLA pre-training"。VLAct 用更精确的词，但正文里在语境清楚时仍简称 pre-training。这一区分很重要：本文不训练任何从零开始的基础模型。

## 2. 先导实验：动作监督如何重塑骨干

第 2 节的 pilot study 是全文最有启发性的部分。固定 Qwen3-VL-4B，只改变预训练与微调阶段使用的动作头，在 LIBERO-Plus 和 RoboTwin-Clean 上观察。四种头的定义见附录 I.1：

| 头 | 接口 | 训练目标 | 推理 |
|---|---|---|---|
| FAST | 离散自回归 token | next-token 交叉熵 | 逐 token 生成后逆 tokenizer |
| OFT | 并行连续回归（MLP 读 K 个 action query 的 hidden state） | L1 | 一次前向 |
| PI | flow-matching action expert | \(\|v_\phi(A^\tau,\tau;H)-(A-\epsilon)\|^2\) | N 步积分 |
| GR00T | 双系统：VLM 慢推理 + DiT 快运动模块 | 同 flow matching，额外条件于状态与本体标识 | N 步去噪 |

两个失效模式：

**(a) 离散监督能迁移，但丢信息。** FAST-token 预训练的骨干接 GR00T 头微调，比从零 GR00T 微调略好——说明离散动作 token 确实往骨干里注入了可迁移的结构。但如果保留 FAST 头本身，成功率远低于任何连续头，且预训练也补不回来。离散化丢掉了操作所需的细粒度幅度与时序信息。

**(b) 单一连续头会造成 head-specific 表征坍缩。** OFT 预训练 → OFT 微调大涨；同一个 OFT 预训练骨干 → PI 或 GR00T 微调却**低于从零微调**。附录表 8 给了数字：PI 从零微调 60.5，OFT-only 预训练后 55.1（−5.4），OFT+GR00T 双头预训练后 63.1（+2.6），三头含 PI 预训练后 77.0。作者把这个现象命名为 **decoder lock-in**：动作信息仍在骨干里，但被组织成只有预训练那个头才好解码的几何形态。同头成功率高估了骨干的可复用性。

结论：可迁移的 VLA 骨干需要同时满足"保留细粒度动作信息"和"能被多种头解码"。这直接引出 3.3 节的多头共监督。

## 3. 方法：三个组件 + 一个微调协议

所有组件**只在持续预训练阶段使用**，目的是塑造骨干；下游用户可以自由选择动作头。

### 3.1 保护 VLM 先验

- **浅层保护**：冻结整个视觉编码器 + LLM 下半层，只更新上半层 LLM 和动作头。理由是低层负责底层视觉处理与早期视觉-语言对齐（附录图 6 用逐层 attention 可视化支撑：低层关注广泛视觉/空间信息，深层聚焦语义与任务相关区域）。下游微调时全模型解冻。
- **caption 混训**：每个 minibatch 同时含机器人样本和辅助 VLM 样本，\(\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{action}}+0.5\,\mathcal{L}_{\text{VLM-CE}}\)。消融了五类辅助数据（caption、BBox-QA、Point-QA、Spatial-QA、纯文本指令），caption 是最强的单一锚点。

### 3.2 多头连续共监督

三个连续头 OFT、PI、GR00T 并联在同一骨干上，接收同一潜表征 \(z\)、预测同一 ground-truth 动作块 \(a\)：

\[
\mathcal{L}_{\text{action}}=\mathcal{L}_{\text{OFT}}+\mathcal{L}_{\text{PI}}+\mathcal{L}_{\text{GR00T}}
\]

三个头共享一次骨干前向，额外开销只是头本身。作者刻意不引入新头或对齐模块，而是把**头的多样性本身当作监督信号**：三种不同的解码偏置迫使骨干把动作信息编码成多种参数化都能读取的形式。附录 E 显示这不只改善跨头迁移，同头微调也涨 1.6–4.3 个点。

### 3.3 跨本体的部分统一动作空间

三种设计对比（图 4）：每个本体一个独立头（监督被隔离）；朴素全统一（低维本体零填充到公共维度，把物理含义不同的坐标对齐到一起）；**部分统一**（VLAct 采用）。

具体布局是一个固定 20 维输出向量（附录 I.2）：

| 维度 | 含义 | 表示 |
|---|---|---|
| 1–6 | 双臂本体左臂（AgileX） | 绝对关节角 |
| 7–12 | 双臂本体右臂 | 绝对关节角 |
| 13–18 | 单臂本体（Franka） | delta 末端位姿 |
| 19 | **共享夹爪**：Franka 单夹爪与 AgileX 左夹爪 | 归一化到 [0,1] |
| 20 | 双臂本体右夹爪 | 归一化到 [0,1] |

每个样本只在本体的激活维度上计算 loss，非激活维 mask 掉。没有本体适配器、路由模块或本体条件解码器。

**wrap-aware loss**：绝对关节角是周期量，标准回归把 179° 和 −179° 当成差 358°。数据侧先把角度 wrap 到 \([-\pi,\pi]\)：\(a_{\text{wrap}}=(a+\pi)\bmod 2\pi-\pi\)；loss 侧再对残差 wrap：\(\delta_{\text{wrap}}=((\hat a-a)+\pi)\bmod 2\pi-\pi\)，\(\mathcal{L}_{\text{wrap}}=|\delta_{\text{wrap}}|\)，加到每个头各自的目标上。对 PI/GR00T，它作用在最终生成的动作样本上，而不是中间的噪声或速度目标。只用于绝对关节角维度。

### 3.4 数据、训练与微调协议

- 预训练数据全开源：DROID（v1.0.0 + v1.0.1）、InternData-A1、RoboCoin、MolmoAct，加 caption 数据。两个本体：Franka 单臂（7 维：6 维 delta EE + 1 夹爪）与 AgileX 双臂（14 维：12 关节角 + 2 夹爪）。
- 数据清洗（附录 H）：删无效任务名；delta EE 动作按 FPS 换成每秒量，平移速度 > 0.5 或旋转速度 > 1.0 的步标记无效，**按步 mask 而非丢整条轨迹**，仅当一个 chunk 内无效比例 > 0.5 才丢弃；关节角先删 \([-2\pi,2\pi]\) 之外的值再 wrap；夹爪去极值后逐数据集 min-max 归一化。
- 代码库 StarVLA，骨干 Qwen3-VL-4B，16 GPU。
- **微调协议**：丢弃预训练头和 caption 流，重新随机初始化任务头，全参数解冻，与每个基线使用完全相同的下游数据、优化器、训练预算。所有 VLAct 对比中，唯一变量是骨干权重。

## 4. 主结果

### 4.1 汇总表

| 基准 | 设定 | VLAct | 同骨干基线 Qwen3VL-OFT | 最强外部对手 |
|---|---|---|---|---|
| LIBERO-Plus | 7 类扰动均值 | **82.6** | 75.0 | ABot-M0 80.5 |
| VLA-Arena | 11 套件加权 | **54.8** | 33.4 | π0.5 44.3 |
| RoboTwin 2.0 Base | Clean / Random | **80.5 / 41.5** | 61.7 / 10.5 | X-VLA 70.0 / 39.0 |
| RoboTwin 2.0 Data Scaling | Clean / Random | **92.5 / 90.8** | 88.2 / 88.3 | HoloBrain-0 91.9 / 92.3；Fast-WAM 91.9 / 91.8 |
| DOMINO | SR / MS | **18.50 / 34.20** | 10.86 / 30.49 | π0.5 9.63 / 26.17 |
| RoboCasa-GR1（未见本体） | 全量 | **54.0** | 48.8 | GR00T-N1.6 47.6 |
| RoboCasa-GR1 | 20% 数据 | **49.5** | — | 已超全量 GR00T-N1.6 |
| RoboDojo（ARX X5，未见本体） | 分数 / 成功率 | 10.66 / 7.60 | StarVLA-α 6.40 / 3.24 | DM0.5 24.90 / 19.34 |
| 真机 Franka 单臂短程 | 4 任务均值 | **92.5** | 77.5 | — |
| 真机 Franka 双臂协调 | 5 任务均值 | **72.0** | 44.0 | — |

### 4.2 逐基准要点

- **LIBERO-Plus**：最大增益在 Camera（+26.9）、Robot、Noise、Layout 四个维度，Language 维度反而比基线低 5.5（81.5 vs 87.0）。说明表示中心预训练主要强化了视觉-空间鲁棒性，对语言扰动没有帮助甚至略有代价。
- **RoboTwin 2.0**：Base 设定只用 50 条 clean 轨迹/任务却在 Random 测试集上从 10.5 涨到 41.5，clean→random 的泛化提升最能体现骨干的作用。三种下游头在 Data Scaling 下 Clean 分别为 OFT 92.5、GR00T 89.6、PI 93.0，相差 3.4 以内，支撑"迁移的是骨干表征而非特定头"。但 Base/Random 下 OFT 41.5 vs GR00T 22.9 vs PI 23.7，头之间差距仍然很大。
- **真机**：预训练只用单臂数据，却把双臂协调从 44.0 拉到 72.0，叠裤子 +30、叠毛巾 +30。长程 scoop beans 从 33.3 到 80.0——基线经常跳过"舀豆子"直接空勺倒。OOD 物体替换场景基线跌到 46–47，VLAct 保持在 82–83。
- **RoboDojo**：35 个策略中成功率第 6、分数第 8。超过所有四个显式标注的 WAM（X-WAM 7.69/3.83 最强），也超过 Xiaomi-Robotics-0、GalaxeaVLA G0、LingBot-VLA、ABot-M0。相对同骨干的 StarVLA-α 涨 4.26 分 / 4.36 个点，增益主要在 Precision 和 Long-Horizon；**Memory 维度只有 0.66 / 0.56，几乎为零**。

## 5. 消融汇总

| 组件 | 设定 | 结果 | 来源 |
|---|---|---|---|
| 浅层保护 | 全骨干更新 → 冻视觉编码器 → 冻视觉编码器 + 下半 LLM | LIBERO-Plus 78.9 → 81.3 → 82.6；RoboTwin 77.1 → 79.3 → 80.5 | 表 6 |
| 辅助数据 | robot-only → +caption | 75.0 → 82.6（caption 单源最强；Spatial-QA、Point-QA、BBox-QA、纯文本指令均有提升；全混合 82.5 略低于 caption-only，因固定步数下稀释了 caption 采样） | 图 8 |
| 多头 vs 单头（跨头） | PI 微调：从零 60.5 / OFT-only 55.1 / OFT+GR00T 63.1 / 三头 77.0 | decoder lock-in 被双头缓解 | 表 8 |
| 多头 vs 单头（同头） | OFT 78.8 → 80.5；PI 75.4 → 77.0；GR00T 71.7 → 76.0 | +1.7 / +1.6 / +4.3 | 表 9 |
| 动作空间 | 独立头 → 统一头 → 统一表征 | RoboTwin 78.5 → 79.5 → 80.5；LIBERO-Plus 81.1 → 81.4 → 82.6 | 表 12 |
| 关节角处理 | 原始回归 → 数据侧 wrap → +wrap loss | RoboTwin Clean 75.5 → 78.6 → 80.5 | 表 10 |
| 异构数据 | +20K RealOmin UMI 轨迹（仅轻量过滤） | LIBERO-Plus 82.6 → 83.7 | 表 11 |

一个容易被忽略的结论：**纯文本指令数据**（Nemotron SFT）也能提升下游操作成功率。它与机器人感知、控制、视觉 grounding 都无关，所以增益只能解释为"把可训练层拉回基础模型的工作区间 + 让梯度更多样"，而不是任务层面的直接迁移。这为"辅助数据 = 表征保护而非额外监督"的解释提供了对照实验。

## 6. 与 StarVLA-α 的对话：预训练是双刃剑，还是配方问题

同一团队四个月前的 [StarVLA-α](03_starvla_alpha.md)（2604.11757）报告了一个相反方向的结论：在 Qwen3-VL-4B + MLP 头上做朴素动作预训练是"双刃剑"——OXE 预训练把 RoboTwin Clean 50×50 从 50.3 拉到 30.2，把 RoboCasa-GR1 24×10 从 9.8 拉到 1.2；即便是同域的 InternData-A1，也只在 RoboTwin 低数据段有帮助，在 GR1 上仍然掉点。

VLAct 恰好回应了这个矛盾：用的数据同样包含 InternData-A1，同样的骨干，但把"朴素拟合动作"换成"保护先验 + 多头 + 统一动作空间"后，GR1（预训练未见）从 48.8 涨到 54.0，20% 数据就到 49.5。两篇放在一起读，结论是：**动作预训练是否有用，取决于它如何塑造表征，而不是数据量本身**。这也是 VLAct 标题"Beyond Data Scaling"的实证含义。

另一个值得对照的点：StarVLA-α 表 6 显示"简单零填充到 32 维"优于 RDT 动作空间和多头设计（GR1 57.3 vs 52.3 / 53.5），而 VLAct 表 12 显示"部分统一"优于"朴素统一"优于"独立头"。两者并不冲突——StarVLA-α 的对比对象是"每本体独立头"和 RDT 的物理统一空间，VLAct 进一步指出：在零填充之上，把物理同义维度（夹爪）显式对齐、把物理不同义维度（不同运动学的手臂）分开，还能再拿 1 个点。

## 7. 批判性评价

### 做得好的地方

1. **归因干净。** 所有对比中动作头、初始化、下游数据、优化器、预算全部相同，只换骨干权重。7.6–21.4 个点的增益可以直接归到持续预训练配方上。
2. **组件足够简单，能被验证也能被推翻。** 没有新模块、新 head、新 loss family。浅层冻结是一个正则表达式，多头是三个已有 head 并联，动作布局是一张 20 维的表。任何人在 StarVLA 上都能复现。
3. **pilot study 把"动作头选择"这个长期争论换了问法。** 从"哪个头最好"变成"如何让骨干对头不敏感"。decoder lock-in 是一个可测量的现象，表 8 给出了最小可复现的对照。
4. **成本诚实。** 开源数据 + 16 GPU，与榜上工业系统的差距被明确标注，并没有宣称绝对 SOTA。

### 局限与未回答的问题

1. **规模单一。** 只有 4B。作者自己指出更大 VLM 的最优配方可能不同。StarVLA-α 表 10 显示 4B→8B 增益 <1%，但那是无预训练设定，持续预训练下的 scaling 曲线仍是空白。
2. **Memory 维度接近零。** RoboDojo 上 0.66 / 0.56，与 DM0.5 的 47.74 / 47.44 差了两个量级。单帧输入、无历史的设计天然不适合需要记忆的任务，这个短板配方本身不解决。
3. **"头无关"有边界。** RoboTwin Base/Random 下 OFT 41.5 而 GR00T 22.9、PI 23.7；LIBERO-Plus 的 Language 维度掉了 5.5 个点。低数据 + 强扰动时，头之间的差距没有被抹平。
4. **Data Scaling 对比不可控。** 训练算力、调度、checkpoint 选择在各方法间不统一（作者已标注）。RoboDojo 榜单也不按算力归一化。
5. **真机每任务 10 次 rollout。** 初始状态固定共享保证了可比性，但 10 次的方差仍然大，差距在 10–20 个点以内的结论应谨慎对待。
6. **三个头共训的代价没有量化。** 论文说"只增加头的计算"，但 PI 和 GR00T 各自带一个 DiT，预训练阶段的显存和吞吐开销没有给出数字。
7. **同名基线数字不一致。** VLAct 表 2 的内部基线 Qwen3VL-OFT 在 RoboTwin Base Clean 为 61.7，而 StarVLA 仓库 `examples/simBenchmarks/Robotwin/README.md` 报告的同配置 OFT 基线为 50.38（48/50 任务），StarVLA-α 表 1 为 50.3。三处是不同训练轮次，比较 VLAct 的增益幅度时应以论文内部同轮次的对照为准（详见 [06 · 基准生态](06_benchmarks_landscape.md) §2.3、§3）。
8. **20 维布局是手工设计的。** 它只覆盖两类本体（6-DoF 双臂关节角 + 6-DoF 单臂 delta EE）。加入人形、灵巧手、移动底盘时怎么扩展，论文没有讨论。GR1 和 ARX X5 的迁移是在下游微调时重新初始化头做到的，不是零样本。

## 8. 可复现性

- 代码与模型：论文声明全部开源（StarVLA 训练脚本 + checkpoint）。写作本报告时（2026-09）StarVLA 主仓库尚无名为 VLAct 的 example 目录，建议关注 https://starvla.github.io/VLAct 与 HuggingFace `StarVLA` 组织。
- 数据全部公开：DROID、InternData-A1、RoboCoin、MolmoAct、LLaVA-ReCap-CC3M、LLaVA-OneVision、RefCOCO、COCO-ReM、PixMo-Points、RoboPoint、SenseNova-SI-800K、Nemotron-SFT-Instruction-Following-Chat-v2。
- 配方各项在 StarVLA 代码中的现状（已有 / 部分 / 缺失）见 [02 · StarVLA 代码库解析](02_starvla_codebase_analysis.md) 第 9 章。

## 9. 对后续研究的直接启示

- 把"骨干如何被动作监督重塑"当作可测量对象：decoder lock-in、表征漂移（RefCOCO IoU 随训练步数下降）、跨头线性可读性，都可以做成诊断指标。
- 多头共监督是一个廉价的正则化器，可以推广到"多任务头"（动作 + 未来帧预测 + 空间 QA）。
- 部分统一动作空间的"物理同义维度对齐"原则，可以推到人形与灵巧手：哪些维度该共享、哪些该 mask，本身是一个可学习的问题。
- Memory 与长程是现有配方未触及的空白，也是 RoboDojo 榜单上拉开差距的地方。

具体的研究路线图见 [07 · 研究路线图](07_research_roadmap.md)。
