# 03 · StarVLA-α 解读：把 VLA 系统的复杂度降下来

| 项目 | 内容 |
|---|---|
| 论文 | StarVLA-α: Reducing Complexity in Vision-Language-Action Systems |
| arXiv | [2604.11757](https://arxiv.org/abs/2604.11757)，2026-04；ECCV 2026 |
| 作者 | Jinhui Ye†, Ning Gao†, Senqiao Yang, Jinliang Zheng, Zixuan Wang, Yuxin Chen, Pengguang Chen, Yilun Chen‡, Shu Liu, Jiaya Jia‡（HKUST / XJTU / CUHK / THU / 通义实验室 / SmartMore） |
| 本仓库文件 | [英文 PDF](../papers/en/2604.11757_StarVLA_alpha.pdf) · [中文 PDF](../papers/zh/2604.11757_StarVLA_alpha_zh.pdf) |

## 0. 一句话

一个刻意做到最简的 VLA 基线——Qwen3-VL-4B + 读取专用 action token 隐状态的 MLP 头、原始 RGB + 指令输入、训练集 z-score 归一化、不用状态/历史帧/机器人预训练——在 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1 四个基准上打平或超过 π0.5、GR00T-N1.6、OpenVLA-OFT；把四个基准的数据合在一起训一个 generalist，在 GR1 上反而更高（57.3 vs 53.8）；真机 RoboChallenge ARX5 11 任务成功率 33.6 vs π0.5 的 12.7。它的价值不是数字，而是提供了一块**去掉混杂因素的对照板**，用来重新审视 VLA 社区习以为常的设计。

## 1. 动机：VLA 领域的"巴别塔"

论文开篇的诊断（图 1）：现有 VLA 系统在架构（视觉塔、语言骨干、动作专家）、预训练数据、本体配置、基准专用工程四个维度上同时变化，报告的提升无法归因到具体的建模创新。VLM 领域已经收敛出标准配方（LLaVA 系列），VLA 还没有。StarVLA-α 的目标是方法论上的清晰，而不是架构新颖性。

## 2. 设计：最小充分性假设

**假设**：一个强 VLM 配一个轻量动作头，就能拿到通常归功于复杂设计的大部分收益。

- **最小数据处理**：所有环境共用一条数据管线；输入原始 RGB + 语言指令；动作只用训练集统计做零均值单位方差归一化；评测严格按各基准官方协议。
- **干净架构**：Qwen3-VL 原生处理视觉与语言，省掉挑选和拼接 CLIP/SigLIP/DINO 的步骤；顶层接一个 MLP，读取指定 action token 的隐状态，回归一个动作块。
- **统一基准集成**：异构性只允许存在于"薄适配器"里（观测格式、动作接口、评测入口），同一模型和配方跑遍所有基准。

训练细节（附录 C）：骨干 lr 1e-5、动作头 lr 1e-4、cosine 调度、每 GPU batch 16、最多 100k 步；LIBERO 8×A100，SimplerEnv / GR1 / RoboTwin-Clean 16×A100，RoboTwin Clean+Rand 48×A100，全基准联训 64×A100。

## 3. 主结果（表 1）

| 方法 | LIBERO avg | SimplerEnv WidowX / Google VA / VM | RoboTwin clean / clean* / random* | RoboCasa-GR1 |
|---|---|---|---|---|
| OpenVLA-OFT | 97.1 | 31.3 / 54.3 / 63.0 | – | – |
| π0 | 94.1 | 27.1 / 54.8 / 58.8 | 46.4 / 65.9 / 58.4 | – |
| π0.5 | 96.9 | 46.9 / 68.4 / 72.7 | 60.2 / 82.7 / 76.8 | 37.0 |
| GR00T-N1.6 | 97.0 | 62.0 / 65.3 / 67.7 | – | 47.6 |
| StarVLA-α（specialist） | **98.8** | 64.6 / 70.2 / **76.0** | 50.3 / 88.2 / 88.3 | 53.8 |
| StarVLA-α（generalist） | 97.8 | **65.2** / 69.8 / 74.3 | – / **88.7** / 87.8 | **57.3** |

\* 表示 clean + random 数据都用于训练。RoboTwin Base（仅 clean 50×50）上 π0.5 的 60.2 高于 StarVLA-α 的 50.3——这是全表中最简基线输给对手的唯一一格，低数据 + 双臂关节角是它的弱点。

## 4. 三个"常识"的重新检验（第 3 节）

### 4.1 动作头设计重要吗？（表 2）

同一骨干换四种头：

| 头 | LIBERO | WidowX | RoboTwin clean* | GR1 |
|---|---|---|---|---|
| MLP（OFT 式） | 98.8 | 64.6 | 88.2 | 53.8 |
| FAST（离散 token） | 97.8 | 35.6 | 72.5 | 45.0 |
| GR00T（双系统流匹配） | 98.7 | 65.3 | 88.0 | 52.8 |
| π（流匹配专家） | 98.1 | 65.9 | 88.1 | 48.9 |

结论：连续 > 离散（FAST 在 WidowX 上低 29 个点）；三种连续头之间差距很小。骨干够强时，动作头的额外复杂度是不必要的。这一发现是 VLAct pilot study 的直接前身——VLAct 进一步追问"同头表现好是否等于骨干可复用"，答案是否。

### 4.2 现有的动作预训练重要吗？（表 3）

| 中间预训练 | 轨迹数 | RoboTwin Clean 50×50 | +Random×500 | GR1 24×10 | GR1 24×1000 |
|---|---|---|---|---|---|
| 无 | – | 50.3 | 88.2 | 9.8 | 53.8 |
| + OXE | 232.6k | 30.2 | 83.6 | **1.2** | 27.8 |
| + InternData-A1 | 630k | **63.6** | 88.6 | 2.8 | 35.4 |
| + RoboTwin-Rand | 25k | **79.7** | 88.8 | 2.2 | 33.3 |

这是全文最重要的一张表。跨域数据（OXE）全面拖后腿；同域数据（InternData-A1、RoboTwin-Rand）只在目标域的低数据段有帮助，换到 GR1 一律掉点，最惨的把 9.8 打到 1.2。作者的措辞是"双刃剑，应谨慎使用"。

这张表和四个月后 VLAct 的结果构成了一组对照：VLAct 用同样的 InternData-A1（加 DROID、RoboCoin、MolmoAct），换成表示中心配方后，GR1 从 48.8 涨到 54.0。**朴素动作拟合会伤害跨本体迁移，问题在配方不在数据。** 详见 [01 · VLAct 精读](01_vlact_deep_dive.md) 第 6 节。

### 4.3 数据工程必要吗？（表 4）

本体感受、历史帧、delta 动作、relative 动作四项：低数据段（RoboTwin Clean 50×50、GR1 24×10）有 1–10 个点的小幅提升，本体感受在 RoboTwin 低数据段 +10.5 最明显；数据充足后全部回到基线水平；历史帧在多数设定下反而略降。

## 5. All-in-one Generalist 评测（第 4 节）

作者提出一个评测范式主张：像 LLM 领域一样，用**一个模型跑所有基准**来检验泛化，而不是每个基准单独微调。实现方式极简——把所有本体的动作零填充到 32 维，联合训练，不做任务特定设计。

分析部分的四个发现：

- **本体专用动作设计不必要**（表 6）：零填充 vs RDT 动作空间 vs 多动作头，GR1 上 57.3 / 52.3 / 53.5，Google VM 上 74.3 / 71.4 / 67.8。
- **模型规模**（表 10）：2B→4B 在 WidowX +18.1、GR1 +6.6；4B→8B 增益 <1%。4B 是当前训练规模下的甜点。
- **初始化质量**（表 9）：随机初始化 GR1 28.8，Florence-2 39.2，Qwen2.5-VL 53.6，Qwen3-VL 57.3，Qwen3.5 56.1。骨干先验直接决定跨基准泛化上限。
- **batch size**（表 11）：64→1024，GR1 从 40.0 涨到 59.2，RoboTwin clean 从 80.4 到 88.8，几乎单调。作者认为 batch 多样性是 generalist 训练里最重要的优化因素，影响比模型规模更大、更一致。

## 6. 真机（第 5 节 + 附录 E/F）

- **RoboChallenge Table-30**（ARX5，11 任务）：SR 33.6 / 进度分 54.5，π0.5 为 12.7 / 27.6，π0 为 3.6 / 14.7。put cup on coaster 100% SR，fold dishcloth 0%。
- **Franka OOD**（附录 F）：三个任务（垃圾分类、拾取彩蛋、蛋托格子放置）ID 均值 85.3，OOD 均值 76.3；蛋托放置在未见行列组合上从 91.3 掉到 68.8，是最敏感的一项。

## 7. 评价

### 贡献

1. 提供了一块可复现、去混杂的对照板。后续 StarVLA 生态的多篇工作（VLAct、ST4VLA）都以它为基线。
2. 表 3 的"预训练双刃剑"是对领域默认假设的一次有数据支撑的质疑，直接催生了 VLAct 的问题定义。
3. "generalist 评测范式"和 batch size 结论对工程实践有直接指导意义。

### 局限

1. 所有结论建立在 Qwen3-VL-4B 上；"动作头不重要"在弱骨干（Florence-2）或更大骨干上是否成立没有验证。
2. 表 3 只比较了三种预训练数据源，没有区分"数据分布不匹配"与"训练方式不当"两种解释——VLAct 后来证明是后者，但 StarVLA-α 本身的表述容易被读成"预训练没用"。
3. RoboTwin Base 设定 50.3 低于 π0.5 的 60.2，低数据 + 关节角控制是最简基线的明显短板，论文没有深入分析。
4. 附录 E 引用了一张不存在的表（"Table ??"），RoboChallenge 其他三个平台（UR5、Franka、ALOHA）的数字实际未给出。
5. 32 维零填充在只有四类本体时够用；本体数量增加后，填充维度和不同本体间坐标语义冲突的问题会放大，VLAct 表 12 已经显示"部分统一"比"朴素统一"多 1 个点。

## 8. 与本仓库其他材料的关系

- 代码实现：`starVLA/model/framework/VLM4A/QwenOFT.py` 即 StarVLA-α 默认配置，见 [02 · 代码库解析](02_starvla_codebase_analysis.md)。
- 后续演进：[01 · VLAct](01_vlact_deep_dive.md) 在同一骨干上回答"如何让预训练不再是双刃剑"。
- 基准细节：[06 · 基准生态](06_benchmarks_landscape.md)。
