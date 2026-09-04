# 09 · EventVLA 解读：给 StarVLA-OFT 装上稀疏视觉证据记忆

| 项目 | 内容 |
|---|---|
| 论文 | EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies |
| arXiv | [2606.20092](https://arxiv.org/abs/2606.20092)，2026-06（CC BY 4.0） |
| 代码 | 官方 [InternRobotics/EventVLA](https://github.com/InternRobotics/EventVLA)；本仓库以子模块收录分支 [asimfish/EventVLA](https://github.com/asimfish/EventVLA)（`code/EventVLA/`，含 RoboTwin-MeM 基准） |
| 模型 / 数据 | [HF ganlinyang/EventVLA](https://huggingface.co/ganlinyang/EventVLA/tree/main)、[HF RoboTwin-MeM](https://huggingface.co/datasets/ganlinyang/RoboTwin-MeM) |
| 基础模型 | StarVLA 的 `QwenOFT`（Qwen3-VL + MLP 回归头）；checkpoint 布局即 StarVLA 格式 |
| 本仓库文件 | [英文 PDF](../papers/en/2606.20092_EventVLA.pdf) · [中文 PDF](../papers/zh/2606.20092_EventVLA_zh.pdf) |

## 0. 一句话

标准 VLA 是马尔可夫策略：只看当前帧，一旦任务关键线索被遮挡或消失就无从决策。EventVLA 在 StarVLA-OFT 上加了一个**稀疏视觉证据记忆**：两条规则锚点（初始帧 + 最近 K 帧）负责场景布局与运动连续性，一个与动作头并联的轻量**关键帧证据记忆（KEM）头**从同一份隐状态预测"未来 H 步里哪一步会是关键事件"，命中就把那一帧原图写进一个有界 FIFO 缓冲。在作者新建的 RoboTwin-MeM 记忆基准上，基线 QwenOFT 3.8%，只加锚点 18.0%，加 KEM 后 **75.2%**；真机 ARX 双臂四个记忆任务 60–90%。它直接回应了本仓库路线图 D1 与基准报告 §5.4 指出的"Memory 是所有 StarVLA 系模型的死角"。

## 1. 问题：非马尔可夫的操作任务

作者把长程操作形式化为非马尔可夫决策过程：$a_t = \pi(o_t, M_{t-1}, \ell)$，多出来的 $M_{t-1}$ 是一个外部视觉记忆缓冲。为什么需要它？典型场景：掀开盖子看一眼里面是什么颜色的方块再盖回去，之后要按颜色顺序操作——关键证据只短暂出现过一次。RoboDojo 榜单上几乎所有策略在 Memory 维度接近零（VLAct 0.66 / 0.56，见 [01](01_vlact_deep_dive.md) §4.2），就是这类任务。

作者把已有的记忆方案分成三类并各指出短板：双系统（高层 VLM 规划 + 低层控制）延迟高、误差会传播；循环 / 压缩记忆有信息瓶颈；无选择的历史缓冲把大量冗余帧塞进上下文。EventVLA 的立场是：**记忆应当稀疏、按事件写入，并且由策略自己决定写什么**。

## 2. 方法

### 2.1 两条规则锚点（Visual Anchors）

$A_t = \{o_0\} \cup \{o_{t-K}, \dots, o_{t-1}\}$。初始帧 $o_0$ 是永久的空间锚，记住东西被挪动前的布局；最近 $K$ 帧提供运动与任务进度线索。这部分不需要学习。消融显示两者都不可缺：在 RMBench 上去掉初始帧掉到 33.7%，去掉短期窗掉到 23.8%（全量 67.8%）。

### 2.2 关键帧证据记忆（KEM）头

- **输入**：与动作头完全相同的最后一层隐状态 $h_t \in \mathbb{R}^{H \times d}$（H 为动作块长度）。这些位置本来就编码了"接下来 H 步要做什么"，所以 KEM 天然带有对未来执行计划的预判。
- **输出**：$\hat p_t = \sigma(\mathrm{MLP}(h_t)) \in [0,1]^H$，第 $i$ 个分量是"未来第 $i$ 步是关键帧"的概率。按块预测而不是逐步分类，是因为关键事件可能在块中间一闪而过，逐步分类器会错过。
- **写入**：某个 $\hat p_t^i \ge \tau_{\text{commit}}$ 时，把 $t+i$ 时刻的原图写进事件缓冲 $E_t$；缓冲有上限 $N_{\max}$，FIFO 淘汰。推理时先用一维 NMS + 时间冷却把连续的高概率段压成离散的写入事件，避免同一事件写入多帧。
- **拼接**：$o_t$、$A_t$、$E_{t-1}$ 按时间顺序排成一个图像序列送进 VLM。记忆就是"多几张图"，架构不变。

### 2.3 训练

- 关键帧的真值来自一条离线的 Qwen3-VL 自动标注流水线，不需要人工标时间戳。
- 用时间平滑的软标签和按序列平均的 BCE 监督 KEM：$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{kem}}$，与动作损失端到端联合优化。
- 训练时的记忆内容从"真值关键帧"逐步过渡到"模型自己预测的关键帧"（teacher-to-student 课程），弥合训练与推理的分布差。

## 3. RoboTwin-MeM 基准

建在 RoboTwin 2.0 上的 8 个双臂任务，每个任务用参数 $n$（1–5）显式标出**成功必须记住几个转瞬即逝的关键帧**：Rearrange Blocks Hard（n=1）、Put Back Block Hard（2）、Pick Objects in Order（3）、Pick the Unhidden Block（3）、Cover Blocks Hard（4）、Reproduce Route（4）、Find Seal and Seal Stamp（1–4）、Press Button Keyframe（2–5）。每回合平均 430–1544 步。这是目前 StarVLA 生态里唯一把"要记多少"作为可控变量的基准，正是 [06 · 基准生态](06_benchmarks_landscape.md) §5.4 说缺的东西。

## 4. 结果

### 4.1 RMBench（记忆只依赖持久布局与固定动作风格）

| 方法 | 均值 |
|---|---|
| QwenOFT（基线，无记忆） | 5.6 |
| π0.5 | 10.4 |
| Mem-0（双系统） | 42.0 |
| MemoryVLA（QwenOFT 版） | 41.7 |
| **EventVLA，仅锚点** | **67.8** |

只靠两条规则锚点就是最好成绩，说明这类基准考的是"记住布局"，不是中间事件。

### 4.2 RoboTwin-MeM（考中间证据）

| 方法 | 8 任务均值 |
|---|---|
| QwenOFT | 3.8 |
| π0.5 | 7.8 |
| MemER / Mem-0 | 10.5 / 0.0 |
| MemoryVLA（QwenOFT 版） | 10.8 |
| EventVLA 仅锚点 | 18.0 |
| **EventVLA 锚点 + KEM** | **75.2** |

消融里最有信息量的几行：把原图缓冲换成隐式记忆库掉到 24.9（说明"存原图、让 VLM 重新看"比存特征有效）；硬标签替代软标签掉到 48.8；去掉 NMS 掉到 53.4；缓冲上限 $N_{\max}=2$ 掉到 32.0；把动作块从默认缩到 15 步掉到 13.6——KEM 的"预见"距离依赖足够长的块。

### 4.3 不伤常规任务，真机可用

RoboTwin 2.0 常规（马尔可夫）任务上，EventVLA 相对 QwenOFT 基线 Easy 80.0 → 83.8、Hard 78.0 → 81.6。真机 ARX ACONE 双臂四个记忆任务（找被藏起的方块、按读到的次数抓取、按指示顺序抓取），每任务 20 次：EventVLA 90 / 60 / 90 / 75，π0.5 0–10，记忆增强基线 πMEM 30–50。

## 5. 评价

**做得好的地方**

1. 记忆机制与 StarVLA 的接口完全兼容：多一个并联的 MLP 头、多几张输入图，训练循环和 OFT 头不变。这与 VLAct 的多头共监督在结构上同源——都是"在同一份隐状态上再挂一个头"，只是监督信号不同（关键帧概率 vs 另一种动作解码）。
2. 把"记忆写什么"变成可学习、可消融的对象，并且用一个参数化的基准把"要记几件事"作为自变量。
3. 存原图而非特征的选择被消融支持，也解释了为什么 RoboDojo 上单帧模型的 Memory 分接近零：不是骨干不够强，是根本没看到需要的帧。

**局限与未回答的问题**

1. 缓冲有界（作者自述）：超过 10 分钟、事件密集的任务会饱和并淘汰早期证据；层次化或压缩记忆是下一步。
2. 关键帧真值依赖 Qwen3-VL 离线标注，标注质量的上限决定 KEM 的上限；论文没有报告标注错误率。
3. 只在 OFT 头上验证。KEM 与 flow-matching 头（PI / GR00T）共存时，隐状态 $h_t$ 的语义是否仍然"包含未来计划"，需要实验。
4. 与 VLAct 配方的叠加未做：VLAct 是持续预训练阶段的骨干配方，EventVLA 是下游架构改动，两者正交，可以直接组合（见 §6）。

## 6. 与本仓库其他材料的关系

- **路线图 D1（记忆与长程）**：EventVLA 就是 D1 路线 (i)"压缩历史"之外的第四条路——稀疏事件记忆；RoboTwin-MeM 是 D1 应当采用的评测。已在 [07 · 路线图](07_research_roadmap.md) D1 中更新。
- **基准 §5.4**：RoboTwin-MeM（8 任务、n 可控）与 RMBench 填上了"没有好记忆基准"的空白，已在 [06](06_benchmarks_landscape.md) 中补注。
- **可直接做的实验**：用 VLAct 骨干替换 EventVLA 的 Qwen3-VL 初始化，其余不变，看 RoboTwin-MeM 是否叠加提升；以及把 KEM 头加进 [`code/vlact_ext`](../code/vlact_ext/) 的 `QwenMultiHead`，检验多头共监督下 KEM 的预测是否更准。
- **代码入口**：`code/EventVLA/EventVLA/`（模型与训练）、`code/EventVLA/RoboTwin-Mem/`（基准）；模型 checkpoint 与 StarVLA `from_pretrained` 兼容。
