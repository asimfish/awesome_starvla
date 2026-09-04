# 07 · 研究路线图：在 StarVLA / VLAct 之上做什么

这份文档回答一个问题：读完 StarVLA 技术报告、StarVLA-α、VLAct 三篇论文并看过代码之后，下一步的研究该往哪里走。每个方向给出问题、已有证据、具体做法、在 StarVLA 代码中的落点、评测方案和风险。方向按"表征诊断 → 预训练目标 → 动作空间 → 能力短板 → scaling → 评测方法"六类组织，最后给一个可以直接执行的六个月计划。

## 0. 三篇论文留下的地图

| 已被证实 | 证据 | 留下的空白 |
|---|---|---|
| 强 VLM + MLP 头 + 最少数据工程就是一个强基线 | StarVLA-α 表 1：LIBERO 98.8、GR1 53.8 | 只在 4B 验证；低数据 + 关节角控制是短板（RoboTwin Base 50.3 < π0.5 60.2） |
| 连续头 ≫ 离散头；三种连续头在数据充足时相当 | StarVLA-α 表 2；VLAct pilot | 低数据 + 强扰动时头差距仍大（VLAct RoboTwin Base/Random：OFT 41.5 vs GR00T 22.9） |
| 朴素动作预训练伤害跨本体迁移 | StarVLA-α 表 3：OXE 把 GR1 24×10 从 9.8 打到 1.2 | 为什么伤害？哪一层的表征被破坏？ |
| 保护先验 + 多头 + 部分统一动作空间能把预训练变成净收益 | VLAct：GR1 48.8→54.0，20% 数据超全量 GR00T-N1.6 | 每个组件的作用机制仍是经验性的；规模、本体数量、头的种类都没有扫过 |
| 单一动作头会让骨干表征坍缩到该头的解码几何（decoder lock-in） | VLAct 表 8：PI 微调 60.5→55.1（OFT-only 预训练） | 没有直接的表征度量，只有下游成功率这一间接证据 |
| 动作-only 微调让 VLM 在 20K 步内忘掉 grounding；共训能同时提升操作 | StarVLA 报告第 6 节（ST4VLA）：Google VM 66.1→84.6 | 辅助数据的配比、调度、与动作损失的梯度冲突尚未系统研究 |
| generalist 联训 + 大 batch 优于 specialist | StarVLA-α 表 5、表 11：GR1 53.8→57.3，batch 64→1024 使 GR1 40.0→59.2 | generalist × 持续预训练的组合没有做 |
| 记忆是所有 StarVLA 系模型的死角 | RoboDojo：VLAct Memory 0.66/0.56，DM0.5 47.74/47.44 | 单帧输入的架构假设没有被挑战 |

## 1. 方向清单

每个方向的编号在后文的优先级矩阵和六个月计划中复用。

### A. 表征诊断与度量

**A1 · 骨干可复用性诊断套件。**
问题：decoder lock-in 目前只能通过"换头微调看成功率"间接观察，一次实验要几十 GPU 小时。
做法：在冻结骨干上训练轻量线性/MLP 探针，从 action query 隐状态预测动作，报告不同头几何下的探针误差；用 CKA / 线性回归可解释方差度量预训练前后各层表征变化；把 RefCOCO-g IoU、POPE、MME 做成训练过程中的周期性探针（ST4VLA 已展示可行）。
落点：新增 `starVLA/training/trainer_utils/probes.py`，在 `train_starvla*.py` 的评估钩子里调用；VLM 能力探针可复用 `train_starvlm.py` 的评测数据接口。
评测：诊断指标与下游跨头迁移成功率的相关性（目标：Spearman ρ > 0.7）。
风险：探针本身可能对头几何敏感，需要多种探针取平均。

**A2 · "哪层该冻"的自动化。**
问题：VLAct 冻结"视觉编码器 + LLM 下半层"是一个粗粒度的手工选择（表 6：三档消融）。
做法：用 A1 的逐层漂移度量在训练早期决定每层的学习率衰减系数（layer-wise LR decay）或 LoRA rank；对比三种方案：硬冻结、LLRD、上半层 LoRA。
落点：`TrainerUtils.freeze_backbones` 已支持正则/名单冻结，`build_param_lr_groups` 已支持按模块分组 lr，把"按层深度生成 lr 组"加进去即可。
评测：LIBERO-Plus 的 Camera / Robot / Noise 维度（VLAct 增益最大处），以及 A1 的漂移曲线。

### B. 预训练目标设计

**B1 · 头多样性作为正则化的推广。**
问题：VLAct 用 OFT + PI + GR00T 三个连续头。头的"多样性"是否可以更极端、更便宜？
做法：(i) 加入 FAST 离散头作为第四个辅助头，检验离散监督注入的"粗粒度结构"是否与连续头互补（VLAct pilot 显示 FAST 预训练→GR00T 微调略优于从零）；(ii) 加入未来帧预测头（与 WM4A 汇合，见 B3）；(iii) 加入空间 QA 头（RoboPoint 式的点预测）。做 2^k 因子实验找出哪些头组合真正有效。
落点：`starVLA/model/framework/VLM4A/` 下新建 `QwenMultiHead.py`，在 forward 里把多个 `action_model` 的 loss 相加；每个头的开关和权重走 YAML。
评测：VLAct 表 8/9 协议——每个预训练变体接四种下游头微调。
风险：头越多，预训练显存越大；需要先测量三头方案的实际开销（E2）。

**B2 · 辅助数据的自动配比与调度。**
问题：VLAct 固定 0.5 的 VLM loss 权重、固定 caption 采样比；全混合（82.5）略低于 caption-only（82.6），作者归因为稀释。
做法：把 A1 的漂移信号作为控制量：漂移超过阈值就提高 VLM 数据比例，反之降低（类似 curriculum / bandit 调度）；对比固定比例、线性退火、漂移驱动三种策略。
落点：`train_starvla_cotrain.py` 的双 dataloader 循环里加一个调度器，控制每步 VLM batch 的采样概率和 `loss_scale.vlm`。
评测：LIBERO-Plus 总分 + RefCOCO-g 保持率，画 Pareto 前沿。

**B3 · 统一潜动作 vs 显式多头。**
问题：多头共监督是"让表征对多种解码器可读"；另一条路是学一个头无关的潜动作空间（LAPA、UniVLA），再让各头从潜空间解码。
做法：在同一骨干上比较 (a) VLAct 三头显式监督，(b) 潜动作 VQ 目标 + 单解码头，(c) 两者叠加。
落点：`examples/modelExtensions/DiscreteDiffusion` 已有离散动作建模的例子可以借用 codebook 部分。
评测：跨头迁移（VLAct 表 8 协议）+ 未见本体迁移（RoboCasa-GR1 数据效率曲线）。

**B4 · 世界模型作为辅助信号。**
问题：StarVLA 报告的"广义 VLA 视角"认为 VLM 路线和世界模型路线只差辅助信号。WM4A 已把 Cosmos-Predict2 / Wan2.2 当骨干，但"VLM 骨干 + 未来帧预测辅助头"这一组合没有做。
做法：在 Qwen3-VL 骨干上加一个轻量的未来帧/未来特征预测头（预测 DINO 或 SigLIP 特征而非像素，降低成本），与动作头共训。
落点：`docs/WM4A.md` 描述的接口 + `starVLA/model/framework/WM4A/`；特征预测头可放在 `starVLA/model/modules/`。
评测：DOMINO（动态场景，需要预测）和 RoboTwin Random。

### C. 动作空间与跨本体

**C1 · 部分统一动作空间的可学习化。**
问题：VLAct 的 20 维布局手工指定"夹爪共享、手臂分开"，只覆盖 6-DoF 双臂关节角和 6-DoF 单臂 delta EE。人形（GR1）、灵巧手、移动底盘进来后要重画。
做法：用一个可学习的"维度对齐矩阵"决定各本体动作维度到共享坐标的映射，以稀疏性正则鼓励物理同义维度共享；或者用本体描述文本（DoF、末端类型、控制模式）作为条件让骨干自己路由。对照组：VLAct 手工布局、StarVLA-α 32 维零填充、每本体独立头。
落点：`starVLA/dataloader/lerobot_datasets.py` 的动作拼装与 mask 逻辑；动作头输出维与 mask 在 `action_model` 配置中定义。
评测：加入 GR1 数据做三本体预训练，看 RoboCasa-GR1 与 RoboDojo（ARX X5）迁移。

**C2 · 几何一致的动作参数化。**
问题：wrap-aware loss 只处理了周期关节角（RoboTwin 75.5→80.5，+5 个点，是 VLAct 单项消融里最大的）。旋转表示（欧拉 / 四元数 / Rot6D）、关节空间与末端空间的混合仍按普通欧氏量回归。
做法：系统比较旋转表示 × 对应的测地线 loss；对 delta EE 的旋转分量用 SO(3) 上的距离；检验 StarVLA-α 表 4 中 delta / relative 动作在低数据段的小幅收益是否被 loss 选择放大。
落点：`starVLA/dataloader` 的动作转换 + 每个 head 的 loss 函数。
评测：RoboTwin Base（关节角）、LIBERO-Plus Robot 维度（EE 控制）。

**C3 · 零样本 / 极少样本本体迁移。**
问题：VLAct 在 GR1 上 20% 数据 49.5，但零样本未测；头是重新初始化的。
做法：保留预训练的统一头，用本体描述做条件（C1 的路由方案），测 0 / 1% / 5% / 20% 数据曲线；对比"重新初始化头"与"复用统一头"。
评测：RoboCasa-GR1、RoboDojo 数据效率曲线。

### D. 能力短板

**D1 · 记忆与长程。**
问题：RoboDojo Memory 维度 0.66 / 0.56，榜首 DM0.5 47.74 / 47.44。StarVLA-α 表 4 显示简单堆两帧历史反而降分。
做法：四条路线并行小规模验证：(i) 压缩历史 token（把过去 k 帧的 VLM 特征池化成少量 memory token 拼进上下文）；(ii) 双系统里让 System 2 维护显式状态（语言化的任务进度），System 1 只看当前帧；(iii) 在 GR00T 头里加入过去动作块作为条件；(iv) **稀疏事件记忆**——[EventVLA](09_eventvla.md) 已在 StarVLA-OFT 上验证：初始帧 + 最近 K 帧的规则锚点，加一个与动作头并联的关键帧预测头（KEM），命中就把原图写进有界 FIFO 缓冲；RoboTwin-MeM 上 QwenOFT 3.8 → 75.2。它的消融表明"存原图让 VLM 重看"优于存特征，这应作为 (i) 的对照。
落点：`QwenGR00T.py` 的条件输入；dataloader 需支持返回历史帧（`umi_datasets.py` / `lerobot_datasets.py` 的 frame 索引逻辑）。
评测：**RoboTwin-MeM**（8 个双臂任务，需记忆的关键帧数 n=1–5 可控，见 [09](09_eventvla.md) §3）与 RMBench 作为主指标；RoboDojo Memory 与 Long-Horizon 子集作外部核验；VLAct 真机 scoop beans 这类多阶段任务。首个具体实验：用 VLAct 骨干替换 EventVLA 的 Qwen3-VL 初始化，看两者是否叠加。

**D2 · 语言泛化。**
问题：LIBERO-Plus Language 维度 VLAct 81.5 低于同骨干基线 87.0，是唯一掉分的维度。
做法：预训练时对任务名做 LLM 改写增广（同义指令、不同粒度）；把纯文本指令数据（VLAct 已证明有正向作用）与机器人样本做指令级对齐共训。
落点：`train_starvla_cotrain.py` 的 VLM 数据流；改写脚本可放 `scripts/`。
评测：LIBERO-Plus Language 维度、VLA-Arena Extrapolation。

**D3 · 低数据 + 强扰动下的头差距。**
问题：RoboTwin Base/Random：OFT 41.5、GR00T 22.9、PI 23.7。生成式头在低数据段明显落后于回归头，与"三头相当"的一般结论矛盾。
做法：检验是训练步数不足（去噪头收敛慢）、还是 flow-matching 目标在小数据上过拟合；尝试从 OFT 头蒸馏到 PI/GR00T 头（回归先验作为教师），或减少去噪步数。
评测：RoboTwin Base 上按训练步数画曲线。

### E. 系统与 scaling

**E1 · 持续预训练的规模曲线。**
问题：StarVLA-α 说 4B→8B 增益 <1%，但那是无预训练；VLAct 只做了 4B。
做法：Qwen3-VL 2B / 4B / 8B（可能加 Qwen3.5 9B）× {无预训练, 朴素预训练, VLAct 配方}，固定下游协议。
评测：LIBERO-Plus、RoboTwin Base、RoboCasa-GR1 20% 数据。
资源：这是路线图里最贵的一项，8B 三头预训练需要 32+ GPU。

**E2 · 三头共训的开销测量与降本。**
问题：VLAct 声称"只增加头的计算"，但 PI 和 GR00T 各带一个 DiT，没有数字。
做法：用 StarVLA 报告第 8 节的方法（issue #158 的测量脚本）测三头 vs 单头的 samples/s 和显存；尝试头间共享 DiT 主干、头 dropout（每步随机只激活一个头）。
落点：`Makefile` / 测试脚本 + 效率 issue 的复现。

**E3 · 固定 GPU 预算下的数据 vs 配方 Pareto。**
问题：论文标题说 beyond data scaling，但没有画"数据量 × 配方"的二维图。
做法：预训练数据 25% / 50% / 100% × {朴素, VLAct}，固定 16 GPU 步数。
评测：GR1 20% 数据成功率。

### F. 评测方法学

**F1 · 骨干基准（backbone benchmark）。**
问题：VLAct 的"只换骨干权重"协议是目前最干净的归因方式，但没有被标准化。
做法：在 StarVLA `examples/eval_protocol.md` 之上定义一个"骨干评测协议"：固定下游头、数据、优化器、步数，只替换 `qwen_vl_interface` 权重；提供脚本一键跑 LIBERO-Plus + RoboTwin Base + GR1-20%。
落点：`examples/simBenchmarks/*/train_files/` 的 YAML 模板 + 一个新的 `examples/backboneBench/`。

**F2 · generalist × 持续预训练。**
问题：StarVLA-α 的 all-in-one generalist 和 VLAct 的持续预训练是两条独立的线。
做法：VLAct 骨干 → all-in-one 联训四基准，看是否叠加。
评测：StarVLA-α 表 5 协议。

**F3 · 真机统计功效。**
问题：每任务 10 次 rollout，10–20 个点以内的差距不可靠。
做法：用已发布的真机数据做 bootstrap，给出在 n=10 / 20 / 50 下能分辨的最小差距；在仓库里提供计算脚本。

## 2. 优先级矩阵

影响 = 对"持续预训练配方"这一核心问题的推进程度；成本 = GPU 小时与工程量。

| 方向 | 影响 | 成本 | 建议 |
|---|---|---|---|
| A1 诊断套件 | 高（所有后续方向的度量基础） | 低 | **先做** |
| E2 三头开销 | 中 | 低 | 先做，一周内出数字 |
| F1 骨干基准 | 高 | 低 | 先做，产出可复用脚本 |
| A2 自动冻结 | 中 | 低 | 第二批 |
| B1 头多样性推广 | 高 | 中 | 第二批 |
| D1 记忆 | 高 | 中 | 第二批（RoboDojo 最大短板） |
| C1 可学习动作空间 | 高 | 中 | 第三批 |
| B2 辅助数据调度 | 中 | 中 | 第三批 |
| D3 低数据头差距 | 中 | 低 | 第三批 |
| B4 世界模型辅助头 | 高 | 高 | 第四批 |
| C2 几何一致参数化 | 中 | 低 | 第四批 |
| E1 规模曲线 | 高 | 很高 | 有资源再做 |
| B3 / C3 / D2 / E3 / F2 / F3 | 中 | 低–中 | 穿插 |

## 3. 六个月执行计划

前提：16 张 A100/H800 级 GPU，StarVLA 代码库，VLAct 数据配方全部开源可下载。

| 月份 | 目标 | 交付物 |
|---|---|---|
| 1 | 复现 VLAct 基线；实现 A1 探针与 F1 骨干基准脚本；测 E2 开销 | 复现报告（LIBERO-Plus 82.6 ± ?）、`probes.py`、`examples/backboneBench/`、开销表 |
| 2 | A2 自动冻结 + B1 头组合因子实验（4 头 × 2^4 子集中挑 6 个） | 漂移曲线 vs 冻结策略图；头组合 → 跨头迁移矩阵 |
| 3 | D1 记忆三路线小规模验证；D3 低数据头差距诊断 | RoboDojo Memory 子集结果；训练步数曲线 |
| 4 | C1 可学习动作空间，加入 GR1 做三本体预训练 | 三本体 → GR1 / ARX X5 迁移曲线，对照 VLAct 手工布局 |
| 5 | B2 调度器 + B4 特征预测辅助头 | Pareto 前沿图；DOMINO / RoboTwin Random 结果 |
| 6 | 整合最优组合，跑完整评测（LIBERO-Plus、VLA-Arena、RoboTwin、DOMINO、GR1、RoboDojo 提交），写论文 | 一篇"VLAct 之后"的方法论文 + 向 StarVLA 提 PR |

每个月的实验都遵守 F1 协议：只换骨干，下游一切固定。所有数字进 `assets/` 的 CSV，图由脚本生成。

## 4. 风险与对策

- **复现偏差。** VLAct 代码与 checkpoint 发布进度未知；若拿不到官方权重，先用 StarVLA-α 配置 + 论文附录 H 的数据清洗规则自行复现，把复现差距作为第一个报告。
- **算力不足。** 三头预训练在 16 GPU 上的吞吐是瓶颈（E2 先测）；必要时用头 dropout 或 2B 骨干做方法验证，4B 只跑最终配置。
- **基准饱和与噪声。** LIBERO 已饱和（>98%），用 LIBERO-Plus / VLA-Arena 替代；SimplerEnv 方差大，按 StarVLA 报告的做法跑 5 次取均值；RoboDojo 榜单不归一化算力，只用于相对比较。
- **归因陷阱。** 任何新组件都要同时报告"同头"和"跨头"两组数字（VLAct 表 8/9 协议），避免重蹈"单头成功率高估复用性"的覆辙。

## 5. 与本仓库其他材料的关系

- 证据来源：[01 · VLAct 精读](01_vlact_deep_dive.md)、[03 · StarVLA-α](03_starvla_alpha.md)、[04 · 技术报告](04_starvla_codebase_report.md)
- 每个方向在代码中的落点细节：[02 · 代码库解析](02_starvla_codebase_analysis.md)，尤其第 9 章"VLAct 配方对照"
- 评测选择依据：[06 · 基准生态](06_benchmarks_landscape.md)
