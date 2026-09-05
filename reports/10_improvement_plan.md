# 10 · 改进方案：在 StarVLA / VLAct 之上搭一条可复现的研究线

> 这是执行计划，不是综述。它把 [07 · 路线图](07_research_roadmap.md) 里的方向收敛成四个研究问题、八个工作包、一张实验矩阵和四个决策门，并规定代码放在哪、怎么验收。方案分两个阶段：**阶段 A（无 GPU，当前）**——把所有诊断工具、训练策略、辅助头、评测协议写成 CPU 可测的代码与配置；**阶段 B（有 GPU）**——按里程碑跑实验，每个里程碑有明确的通过 / 放弃判据。

## 0. 目标与边界

**总目标**：回答"固定机器人数据与算力预算下，什么样的持续预训练配方能产出对下游头、本体、任务都可迁移的 VLA 骨干"，并把答案做成 StarVLA 上可复现的代码 + 协议。

**已有基础**（本仓库）：
- [`code/vlact_ext`](../code/vlact_ext/)：VLAct 六项配方在 StarVLA 上的完整实现（多头框架 `QwenMultiHead`、wrap-aware loss、20 维统一布局、冻结规则），61 个 CPU 测试通过。
- [`code/EventVLA`](../code/EventVLA/)：基于 StarVLA-OFT 的稀疏视觉证据记忆 + RoboTwin-MeM 基准。
- 证据基础：[01](01_vlact_deep_dive.md)、[03](03_starvla_alpha.md)、[05](05_action_heads_and_representation.md)、[06](06_benchmarks_landscape.md)、[09](09_eventvla.md)。

**边界**：不追求新头、新骨干；不做 RL；不改 StarVLA 已有文件（全部以扩展包形式提供，可整体作为 PR 提交）。

## 1. 四个研究问题与可检验假设

| # | 研究问题 | 假设 | 检验方式 | 对应 WP |
|---|---|---|---|---|
| Q1 | 骨干的"可复用性"能不能不跑下游微调就直接度量？ | **H1**：在冻结骨干上训练的跨头线性 / 小 MLP 探针误差，以及各层相对 VLM 初始权重的 CKA 漂移，与"换头微调成功率"强相关（Spearman ρ > 0.7）。 | 对 ≥ 6 个预训练变体（无 / OFT 单头 / 三头 / 三头 + 冻结 / 三头 + caption / VLAct 全配方）同时计算探针指标与跨头微调成功率，做相关分析。 | WP1 |
| Q2 | "头多样性即正则化"能否推到非动作头？ | **H2**：在三个动作头之外加未来视觉特征预测头或关键帧预测头做辅助监督，同头与跨头迁移不降，DOMINO（动态）与 RoboTwin-MeM（记忆）提升。 | VLAct 表 8/9 协议 + DOMINO + RoboTwin-MeM，2^k 因子设计挑 6 个组合。 | WP3 |
| Q3 | 先验保护能否自动化、粒度更细？ | **H3**：按层漂移自适应的分层学习率衰减（LLRD）在 LIBERO-Plus 的 Camera / Robot / Noise 维度 ≥ 硬冻结（VLAct 表 6 的 82.6），且不损失 Language 维度（VLAct 掉了 5.5）。 | 硬冻结 vs 固定 LLRD vs 漂移驱动 LLRD vs 上半层 LoRA，四组对照。 | WP2、WP4 |
| Q4 | 骨干配方与记忆架构是否叠加？ | **H4**：用 VLAct 骨干初始化 EventVLA（其余不变），RoboTwin-MeM 与 RMBench 均高于 Qwen3-VL 初始化。 | EventVLA 官方协议，两组初始化 × 3 seeds。 | WP7 |

## 2. 工作包

状态标记：**A** = 阶段 A 交付（CPU 可测的代码 / 配置）；**B** = 需要 GPU。

| WP | 名称 | 交付物 | 验收标准 | 依赖 | 阶段 |
|---|---|---|---|---|---|
| WP0 | VLAct 基线复现 | `QwenMultiHead` + `configs/vlact_pretrain_example.yaml` 跑通 16 GPU 预训练；LIBERO-Plus / RoboTwin Base / GR1-20% 三个数字 | 与论文差距 ≤ 2 个点；否则记录差距来源 | vlact_ext | B |
| WP1 | 诊断套件 `starvla_lab.probes` | 跨头线性 / MLP 探针（闭式岭回归 + 小 MLP）、层间线性 CKA、相对参考模型的漂移曲线、VLM 能力探针钩子（接 `train_starvlm.py` 的评测数据） | CPU 单测：探针在合成数据上恢复已知线性映射；CKA 对同一表征 = 1、对正交旋转不变；钩子在 mock 训练循环里按步触发 | 无 | A |
| WP2 | 自动冻结 / LLRD `starvla_lab.schedules.llrd` | 分层学习率衰减参数组（与 `freeze_rules` 共用路径解析）；按漂移信号在线调整层学习率的调度器 | 单测：衰减系数按层深单调；与冻结规则组合时冻结层 lr=0；调度器在给定漂移序列下输出确定的 lr 轨迹 | vlact_ext.freeze_rules | A |
| WP3 | 辅助头扩展 `starvla_lab.heads` | (a) 未来视觉特征预测头：从 action query 隐状态预测下一步（或第 k 步）视觉特征（DINO / SigLIP 特征而非像素），余弦 + MSE 损失；(b) 关键帧预测头：按块预测未来 H 步关键帧概率，软标签 BCE，NMS + 冷却写入策略（自己实现，与 EventVLA 协议兼容）；(c) 两者作为可选头注册进 `QwenMultiHead`（loss 加权求和、可开关） | 单测：损失在完美预测时为 0、mask 生效；写入策略在合成概率序列上产生预期的稀疏事件；接进 `QwenMultiHead` 后 forward 返回 `loss_featpred` / `loss_keyframe` 且总 loss 可反传 | vlact_ext.multihead_framework | A |
| WP4 | 辅助数据调度器 `starvla_lab.schedules.aux` | VLM 数据采样比与 `loss_scale.vlm` 的三种策略：固定 / 线性退火 / 漂移驱动（漂移超阈值提高比例） | 单测：三种策略在给定漂移序列下的输出轨迹；边界（比例 ∈ [min, max]） | WP1 漂移指标 | A |
| WP5 | 骨干基准协议 `starvla_lab.bench.backbone_bench` | "只换骨干"协议：固定下游头 / 数据 / 优化器 / 步数 / seed，输入一组骨干 checkpoint，生成 StarVLA 训练与评测命令清单（LIBERO-Plus、RoboTwin Base clean+random、GR1 10/20/50/100%），以及结果汇总表模板 | 单测：给定 3 个假 checkpoint 生成 3 × 基准 × seed 条命令，yaml 覆盖项只含骨干路径；dry-run 通过 | 无 | A |
| WP6 | 开销测量 `starvla_lab.bench.overhead` | 单头 vs 三头 vs 三头 + 辅助头的前向 / 反向时间、峰值显存、samples/s 的测量脚本（对齐 StarVLA issue #158 的口径）；头 dropout（每步随机只激活一个头）作为降本选项 | CPU 上用 mock 骨干 smoke 通过；输出 CSV | vlact_ext | A（数字 B） |
| WP7 | 记忆叠加实验 | EventVLA 以 VLAct 骨干初始化的配置与启动脚本；RoboTwin-MeM / RMBench 评测脚本引用 | 配置能被 EventVLA 代码加载（dry-run） | WP0、EventVLA | A（配置）/ B |
| WP8 | 可学习动作布局 | 20 维手工布局 → 可学习维度对齐矩阵（稀疏正则）或本体描述条件路由 | 留待 M4 | vlact_ext.unified_action_layout | B |

## 3. 实验矩阵（阶段 B）

**固定协议（F1）**：下游头 OFT；数据 / 优化器 / 步数 / batch 固定；checkpoint 规则固定为"最后一个"；3 seeds 报均值 ± 标准差；同时报"同头"（OFT）与"跨头"（PI、GR00T 各微调一次）两组数字。基准：LIBERO-Plus（主）、RoboTwin 2.0 Base（clean + random 都报）、RoboCasa-GR1 数据比例曲线（10 / 20 / 50 / 100%）；WP3 加 DOMINO 与 RoboTwin-MeM。

| ID | 变体 | 回答 | 预训练 GPU 小时（16×A100 估） | 下游微调轮数 |
|---|---|---|---|---|
| R0 | 无预训练（Qwen3-VL-4B 直接微调） | 基线 | 0 | 3 基准 × 3 头 × 3 seed |
| R1 | OFT 单头朴素预训练 | 对照（StarVLA-α 表 3 的"双刃剑"） | ~600 | 同上 |
| R2 | 三头，全参数 | 隔离多头 | ~900 | 同上 |
| R3 | 三头 + 硬冻结 + caption（VLAct 全配方） | WP0 复现 | ~900 | 同上 |
| R4 | R3 但 LLRD 替代硬冻结 | Q3 | ~900 | 同上 |
| R5 | R4 + 漂移驱动 LLRD + 辅助数据调度 | Q3 | ~900 | 同上 |
| R6 | R3 + 未来特征预测头 | Q2 | ~1000 | 同上 + DOMINO |
| R7 | R3 + 关键帧头 | Q2 | ~1000 | 同上 + RoboTwin-MeM |
| R8 | R3 + 两者 | Q2 | ~1100 | 同上 |
| R9 | EventVLA ← R3 骨干 vs ← Qwen3-VL | Q4 | 0（复用 R3） | RoboTwin-MeM + RMBench × 3 seed |

R0–R8 全部计算 WP1 的探针指标，回答 Q1。总预训练约 7,200 GPU 小时（16 卡约 19 天连续），下游微调另计；若预算不足，先跑 R0 / R1 / R3 / R7 / R9 五组。

## 4. 里程碑与决策门

| 月 | 里程碑 | 决策门 |
|---|---|---|
| 1 | WP0 复现 + WP1 / WP5 / WP6 上线，跑 R0、R1、R3 | **G1**：R3 与论文差距 ≤ 2 点，否则先修复现再往下走 |
| 2 | R2、R4、R5；Q1 相关分析（用 R0–R5 六个变体） | **G2**：探针指标与跨头成功率 ρ > 0.7 → 后续实验用探针做早停与筛选；否则探针只作辅助报告 |
| 3 | R6、R7、R8（WP3）；R9（WP7） | **G3**：任一辅助头在同头 / 跨头不降且目标基准 +3 点以上 → 进入论文主线；否则记录为负结果 |
| 4 | WP8 可学习布局，加 GR1 做三本体预训练 | **G4**：对照 VLAct 手工布局 ≥ 持平且 GR1 20% 数据提升 → 保留；否则回到手工布局 |
| 5 | 整合 G2–G4 通过的组件，全基准评测，RoboDojo 提交 | — |
| 6 | 成文 + 向 StarVLA 提 PR（`vlact_ext` + `starvla_lab`） | — |

## 5. 代码组织与规范

```
code/
├── vlact_ext/                 # 已有：VLAct 配方（不再大改，只修 bug）
└── starvla_lab/               # 本方案的研究包
    ├── probes/                #   WP1：action_probe.py, cka.py, drift.py, hooks.py
    ├── schedules/             #   WP2 / WP4：llrd.py, aux_scheduler.py
    ├── heads/                 #   WP3：feature_prediction_head.py, keyframe_head.py, register.py
    ├── bench/                 #   WP5 / WP6：backbone_bench.py, overhead_bench.py
    ├── configs/               #   实验矩阵 R0–R9 的 yaml 与 sweep 定义
    ├── tests/                 #   全部 CPU 可跑
    └── README.md
experiments/
├── run_matrix.csv             # R0–R9 × 基准 × seed 的运行清单（状态列）
└── results/                   # 每次运行一个 JSON，由 bench 脚本汇总成表
```

规范：与 `vlact_ext` 一致——不修改 StarVLA 源码；依赖注入便于 mock；类型标注；每个模块 ≥ 1 个边界测试；含中文的文件用 heredoc 写入；实验结果只进 `experiments/results/`，图表由脚本生成。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| WP0 复现失败（VLAct 官方权重 / 脚本未发布） | 以 StarVLA-α 配置 + VLAct 附录 H 的清洗规则自行复现，把差距作为第一份报告；G1 允许 2 点误差 |
| 三头 + 辅助头预训练显存爆 | WP6 先测；头 dropout（每步只激活一个头）或 2B 骨干做方法验证 |
| 探针与下游不相关（G2 失败） | 探针降级为报告项，不影响后续 WP；Q1 记为负结果并分析原因 |
| 辅助头与动作头梯度冲突 | 记录各头梯度范数与余弦；必要时用 GradNorm / PCGrad 式加权，作为 WP3 的备选 |
| 基准噪声淹没差异 | 3 seeds；LIBERO-Plus 10,030 实例作主指标；SimplerEnv 不作主指标 |

## 7. 阶段 A 本轮交付清单

- [x] 本方案
- [x] `starvla_lab/probes`：跨头探针（闭式岭回归 + MLP）、线性 CKA、`DriftTracker`、`ProbeRunner` 钩子 + 23 个测试
- [x] `starvla_lab/schedules`：`layerwise_lr_decay_groups`（复用 freeze_rules）、`DriftDrivenLLRD`、`AuxDataScheduler` + 12 个测试
- [x] `starvla_lab/heads`：`FutureFeaturePredictionHead`、`KeyframeHead`（软标签 BCE、NMS / 冷却写入策略、课程）、`QwenMultiHeadLab` + 51 个测试
- [x] `starvla_lab/bench`：`backbone_bench`（只换骨干协议、命令渲染、变化键审计、结果聚合）、`overhead_bench`（秒/步、samples/s、峰值显存、头 dropout）+ 9 个测试
- [x] `starvla_lab/configs`（`protocol_f1.yaml`、`matrix_R0_R9.yaml`）+ `scripts/build_run_matrix.py` → `experiments/run_matrix.csv`（162 次下游微调，预训练预算 7,300 GPU 小时）
- [x] `starvla_lab/README.md`、`experiments/README.md`

阶段 A 完成：`python3 -m pytest code/starvla_lab/tests -q` → 95 passed。下一步是阶段 B 的 WP0（需要 16 GPU），入口见 [`code/starvla_lab/README.md`](../code/starvla_lab/README.md) 与 [`experiments/README.md`](../experiments/README.md)。
