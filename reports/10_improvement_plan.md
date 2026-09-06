# 10 · 改进方案：在 StarVLA / VLAct 之上搭一条可复现的研究线

> 这是执行计划，不是综述。**v2（2026-09-05，设计评审后修订）**：修订内容见文末「评审修订记录」。它把 [07 · 路线图](07_research_roadmap.md) 里的方向收敛成四个研究问题、八个工作包、一张实验矩阵和四个决策门，并规定代码放在哪、怎么验收。方案分两个阶段：**阶段 A（无 GPU，当前）**——把所有诊断工具、训练策略、辅助头、评测协议写成 CPU 可测的代码与配置；**阶段 B（有 GPU）**——按里程碑跑实验，每个里程碑有明确的通过 / 放弃判据。

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
| Q1 | 骨干的"可复用性"能不能不跑下游微调就直接度量？ | **H1**：**主指标**（预注册）：冻结骨干上、以 OFT 动作查询位隐状态拟合的跨头线性探针 R²；**辅指标**：各层相对 VLM 初始权重的 CKA 漂移均值。主指标与"换头微调成功率"的 Spearman ρ > 0.7，且 95% 自助置信区间下界 > 0.4。 | 样本点 = 变体（R0–R8）× 每个变体保存的 5 个中间 checkpoint（20k–100k 步）× seed（核心变体 3 个），约 90 个点；每个点都跑同头与跨头微调。**n=6 的变体级相关不作为判据。** | WP1、WP5 |
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
| WP3b | 数据准备 `starvla_lab.data` | (a) `data/future_features.py`：冻结特征提取器（SigLIP / DINO，注入的 callable）逐轨迹缓存 `[T, d_feat]`，训练期 `FutureFeatureTransform` 按 `(trajectory_id, step)` 附加 `future_features` 与 mask；(b) `data/keyframe_labels.py`：夹爪阈值穿越 + 停顿的启发式关键帧标注（无标注基线）、`FunctionLabeler` 接任意 VLM 标注函数、`KeyframeLabelTransform` 转成块内相对步 | 单测：缓存复用不重算、缺帧 mask=0、启发式事件位置正确、块外事件被裁掉 | WP3 | A |
| WP4 | 辅助数据调度器 `starvla_lab.schedules.aux` | VLM 数据采样比与 `loss_scale.vlm` 的三种策略：固定 / 线性退火 / 漂移驱动（漂移超阈值提高比例） | 单测：三种策略在给定漂移序列下的输出轨迹；边界（比例 ∈ [min, max]） | WP1 漂移指标 | A |
| WP5 | 骨干基准协议 `starvla_lab.bench.backbone_bench` | "只换骨干"协议：固定下游头 / 数据 / 优化器 / 步数 / seed，输入一组骨干 checkpoint，生成 StarVLA 训练与评测命令清单（LIBERO-Plus、RoboTwin Base clean+random、GR1 10/20/50/100%），以及结果汇总表模板 | 单测：给定 3 个假 checkpoint 生成 3 × 基准 × seed 条命令，yaml 覆盖项只含骨干路径；dry-run 通过 | 无 | A |
| WP6 | 开销测量 `starvla_lab.bench.overhead` | 单头 vs 三头 vs 三头 + 辅助头的前向 / 反向时间、峰值显存、samples/s 的测量脚本（对齐 StarVLA issue #158 的口径）；头 dropout（每步随机只激活一个头）作为降本选项 | CPU 上用 mock 骨干 smoke 通过；输出 CSV。**GPU 数字已测**（2026-09-05，1×A100，Qwen3-VL-4B，batch 8，VLAct 冻结）：三头 = OFT 单头的 1.54× 时间、27.2 GB 峰值；头 dropout 1.21×；屏蔽 OFT 查询位无额外开销；见 [`experiments/results/wp6_overhead/`](../experiments/results/wp6_overhead/README.md) | vlact_ext | A + B（开销数字） |
| WP7 | 记忆叠加实验 | EventVLA 以 VLAct 骨干初始化的配置与启动脚本；RoboTwin-MeM / RMBench 评测脚本引用 | 配置能被 EventVLA 代码加载（dry-run） | WP0、EventVLA | A（配置）/ B |
| WP9 | 训练循环集成 `starvla_lab.train` | `train_starvla_lab.py`：镜像 StarVLA 的 `train_starvla.py` / `train_starvla_cotrain.py` 主流程，只替换三处——`datasets.vla_data.data_fraction < 1` 时按轨迹子采样（`install_fraction_hook`）、`trainer.lab.llrd.enabled` 时用分层衰减参数组建优化器、`_train_step` 外包一层 `LabHooks`（辅助数据调度写回 `trainer.loss_scale.vlm`、头 dropout 设 `active_heads`、每 N 步探针 → 漂移 → 驱动 LLRD / 调度器、JSONL 记录）；配置统一在 `trainer.lab.*` | 单测：mock 训练器下三种钩子按步生效、`calibrate_only` 只记录不干预、LLRD 与 `LambdaLR` 协同、模式选择 | WP1、WP2、WP4、vlact_ext | A（接线）/ B（真实运行） |
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

R0–R8 全部计算 WP1 的探针指标，回答 Q1。

**分级设计与全量预算（评审后修订）。** 上一版只算了预训练。按 StarVLA 报告实测的 1.77 s/步（8×A100，batch 16），一次 30k 步下游微调约 120–130 GPU 小时，原设计 162 次 × 3 种头 ≈ 58,000 GPU 小时，不可行。现改为两级：

| 级别 | 变体 | seeds | GR1 数据比例 | 跨头微调 |
|---|---|---|---|---|
| core | R0、R1、R3、R8 | 0、1、2 | 10 / 20 / 50 / 100% | PI、GR00T 各 1 seed，只在 LIBERO-Plus + RoboTwin Base |
| extra | R2、R4、R5、R6、R7 | 0 | 20 / 100% | 不做 |

由 `scripts/build_run_matrix.py` 生成，`experiments/budget.md` 为权威数字：

| 项目 | GPU 小时 | 次数 |
|---|---:|---:|
| 预训练 R1–R8 | 7,300 | 8 |
| 下游微调（OFT 头） | 11,790 | 92 |
| 跨头微调 PI / GR00T | 2,000 | 16 |
| **合计** | **约 21,000** | |

16 卡连续约 55 天。预算不足时的最小子集：R0 / R1 / R3 / R9 + 核心级下游（约 9,000 GPU 小时）。

## 4. 里程碑与决策门

| 月 | 里程碑 | 决策门 |
|---|---|---|
| 1 | WP0 复现 + WP9 接线验证 + WP5 / WP6 上线，跑 R0、R1、R3（R3 开 `probes.calibrate_only` 记录漂移曲线，据此标定 LLRD / 调度器的阈值） | **G1**：R3 与论文差距 ≤ 2 点，否则先修复现再往下走；R3 的漂移曲线必须完整（每 2k 步一条） |
| 2 | R2、R4、R5；Q1 相关分析（变体 × 5 个中间 checkpoint × seed，约 90 个点） | **G2**：预注册主指标（跨头线性探针 R²）与跨头成功率 ρ > 0.7 且置信下界 > 0.4 → 后续实验用探针做早停与筛选；否则探针只作辅助报告，Q1 记负结果 |
| 3 | R6、R7、R8（WP3）；R9（WP7） | **G3**：任一辅助头在同头 / 跨头不降且目标基准 +3 点以上 → 进入论文主线；否则记录为负结果 |
| 4 | WP8 可学习布局，加 GR1 做三本体预训练 | **G4**：对照 VLAct 手工布局 ≥ 持平且 GR1 20% 数据提升 → 保留；否则回到手工布局 |
| 5 | 整合 G2–G4 通过的组件，全基准评测，RoboDojo 提交 | — |
| 6 | 成文 + 向 StarVLA 提 PR（`vlact_ext` + `starvla_lab`） | — |

## 5. 代码组织与规范

```
code/
├── vlact_ext/                 # 已有：VLAct 配方（只修 bug；本轮加了 active_heads 头 dropout 开关）
└── starvla_lab/               # 本方案的研究包（127 个 CPU 测试）
    ├── probes/                #   WP1：action_probe.py, cka.py, drift.py, hooks.py
    ├── schedules/             #   WP2 / WP4：llrd.py, aux_scheduler.py
    ├── heads/                 #   WP3：feature_prediction_head.py, keyframe_head.py, register.py
    ├── data/                  #   WP3b / F1：subsample.py, future_features.py, keyframe_labels.py
    ├── train/                 #   WP9：lab_config.py, integration.py, train_starvla_lab.py
    ├── bench/                 #   WP5 / WP6：backbone_bench.py, overhead_bench.py
    ├── configs/               #   protocol_f1.yaml（真实 StarVLA 路径与评测命令模板）、matrix_R0_R9.yaml（分级 + trainer.lab.*）
    ├── tests/
    └── README.md
experiments/
├── run_matrix.csv             # 主矩阵（OFT 头，92 次）
├── run_matrix_QwenPI_v3.csv / run_matrix_QwenGR00T.csv   # 跨头矩阵（各 8 次）
├── budget.md                  # 全量 GPU 小时预算（脚本生成）
└── results/                   # 每次运行一个 JSON
```

规范：与 `vlact_ext` 一致——不修改 StarVLA 源码；依赖注入便于 mock；类型标注；每个模块 ≥ 1 个边界测试；含中文的文件用 heredoc 写入；实验结果只进 `experiments/results/`，图表由脚本生成。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| WP0 复现失败（VLAct 官方权重 / 脚本未发布） | 以 StarVLA-α 配置 + VLAct 附录 H 的清洗规则自行复现，把差距作为第一份报告；G1 允许 2 点误差 |
| 三头 + 辅助头预训练显存爆 | WP6 先测；头 dropout（每步只激活一个头）或 2B 骨干做方法验证 |
| 探针与下游不相关（G2 失败） | 探针降级为报告项，不影响后续 WP；Q1 记为负结果并分析原因。F0 首跑已表明两种探针能分开"改写了多少"（保留度）与"读出了多少"（可读性）：300 步 regime 里三头模型顶层改写 240× 却无可读性收益、保留度低 2 个百分点；G2 要在 R3 的 checkpoint 上看这两条曲线是否随规模分离 |
| 辅助头与动作头梯度冲突 | 记录各头梯度范数与余弦；必要时用 GradNorm / PCGrad 式加权，作为 WP3 的备选 |
| 基准噪声淹没差异 | 3 seeds；LIBERO-Plus 10,030 实例作主指标；SimplerEnv 不作主指标 |
| 漂移控制阈值无依据 | M1 的 R3 以 `calibrate_only` 跑出漂移曲线后再定 `drift_high / drift_low`；阈值写进 R5 的配置并在报告里给出标定图 |
| **单卡 bf16 路径的更新量化（F3 实测，2026-09-06）** | StarVLA 以 bf16 加载 VLM，`STARVLA_DISABLE_DEEPSPEED=1` 时 AdamW 直接更新 bf16 权重（无 fp32 主权重）；lr 1e-5 的单步更新对 \|w\| > 2e-3 的权重不足半个 ulp，被舍入丢弃，梯度方向一致的头（三头、flow-matching）更容易越过门槛——F0 里 OFT 与三头 240× 的漂移差可能部分是舍入门槛效应；LLRD 把低层 lr 压到 1e-7 后只剩偶发整 ulp 跳变（F3a 的锯齿漂移与更差损失）。对策：`trainer.lab.backbone_fp32`（可训练骨干参数 fp32、计算 bf16 autocast，对应 DeepSpeed 的 fp32 主权重）。**F4 已证实**：同配置下损失好 26%、漂移大一个数量级；单卡冒烟一律开此开关，bf16 数字只作历史对照；多卡 DeepSpeed 路径不受影响 |
| 度量口径不一致 | 固定探测批：`probes.probe_data_mix` 指定的跨场景混合里按指令轮询抽 64 个样本（`probes.probe_batch_size`），纯 VLM prompt、换回预训练 `embed_tokens`、token 级 CKA（≤ 4096 个固定 token 位置）为主、mean-pool 为次，OFT 查询位隐状态拟合探针；所有变体共用同一批与同一 `QwenBackboneProbe`；记录以已完成的更新次数为步号 |
| **漂移度量被 `embed_tokens` 主导（F0 实测，2026-09-06）** | F0 在 LIBERO-goal 上的受控诊断（[`experiments/results/f0_libero_goal_smoke/`](../experiments/results/f0_libero_goal_smoke/README.md) §3）：微调只改了 `embed_tokens` 的 42–46 行（相对变化 ~2e-5），却贡献了 mean-pool CKA 漂移的 98%，冻结层也随之"漂移"；换回预训练嵌入后 OFT 微调的骨干漂移只有 0.0002，三头模型 0.0038（第 35 层 0.019）。修正：(a) 探针提取时临时换回预训练 `embed_tokens`，或把 `embed_tokens` 加入冻结集合并作为消融（VLAct 未说明）；(b) 主指标改为 token 级 CKA（几千 token 为样本），mean-pool 只作辅助；(c) 探针批必须跨场景 / 跨任务（≥ 64 样本，LIBERO 四套 + RoboTwin 混合），单场景批的 Gram 矩阵近退化；(d) M1 的阈值标定用修正后的度量重做。**已落实（F0 v3，2026-09-06）**：(a)(b)(c) 实现为 `starvla_lab.probes.QwenBackboneProbe` + `probes.probe_data_mix`，接进 `train_starvla_lab`；重跑后冻结层漂移精确为 0，单头 OFT 300 步只在第 35 层动 2e-4，三头模型第 35 层 5e-2、第 33 层以下 < 1e-4，曲线单调饱和；冻结 `embed_tokens` 的消融损失不变（0.247 vs 0.251），读数与"探针时换回"一致。(d) 的量级：单头逐层 < 3e-4，三头只有第 34–35 层 > 1e-3，库默认阈值 0.10 / 0.05 差两个数量级，R5 起点改为 1e-2 / 1e-3 |
| 评测脚本参数约定各异 | 每个基准的评测命令是 `protocol_f1.yaml` 里的模板字符串（`{ckpt_file}`、`{seed}`、`{run_id}` 占位），与 StarVLA 真实脚本的参数一一对应 |

## 7. 阶段 A 交付清单（评审后更新）

- [x] 本方案（v2）
- [x] `starvla_lab/probes`：跨头探针（闭式岭回归 + MLP）、线性 CKA、`DriftTracker`、`ProbeRunner` 钩子
- [x] `starvla_lab/schedules`：`layerwise_lr_decay_groups`（复用 freeze_rules）、`DriftDrivenLLRD`、`AuxDataScheduler`
- [x] `starvla_lab/heads`：`FutureFeaturePredictionHead`、`KeyframeHead`（软标签 BCE、NMS / 冷却写入策略、课程）、`QwenMultiHeadLab`
- [x] `starvla_lab/data`：`TrajectorySubset` + `install_fraction_hook`（数据比例曲线）、未来特征缓存与 transform、启发式 / 可插拔关键帧标注与 transform
- [x] `starvla_lab/train`：`trainer.lab.*` 配置、LLRD 优化器构建、`LabHooks` + `attach_to_trainer`、`train_starvla_lab.py`（单 / 双 loader 两种模式）
- [x] `starvla_lab/bench`：只换骨干协议（评测命令模板、`{ckpt_file}` 占位、分级 seeds、GPU 小时合计）、开销测量、头 dropout（已接进 `QwenMultiHead.active_heads`）
- [x] `starvla_lab/configs` + `scripts/build_run_matrix.py` → `experiments/run_matrix*.csv` + `experiments/budget.md`
- [x] `starvla_lab/README.md`、`experiments/README.md`

- [x] 与真实 StarVLA 的 CPU 集成：`scripts/setup_cpu_env.sh`（py3.12 环境）+ `scripts/smoke_starvla_integration.py`（真实三头工厂 + `QwenMultiHead` + 全部 `LabHooks` 钩子；`flow_matching_loss` 与原头 `forward` 逐位相等）

阶段 A 完成：`python3 -m pytest code/starvla_lab/tests -q` → 127 passed（含 v3 探针 15 个、F2/F4 接线 2 个）；`python3 -m pytest code/vlact_ext/tests -q` → 60 passed, 1 skipped（系统 python3.9，mock 骨干）；py3.12 + StarVLA 可导入时两包合跑 → 169 passed, 2 skipped，冒烟脚本通过。

**阶段 B 已开始（1 卡，2026-09-05/06）**：
- [x] WP6 开销数字：三头 = OFT 单头 1.54× 时间、27.2 GB（[`experiments/results/wp6_overhead/`](../experiments/results/wp6_overhead/README.md)）
- [x] WP9 真实数据接线：`train_starvla_lab` + `QwenOFT` / `QwenMultiHead` 在 LIBERO-goal 上各 300 步跑通；三头模型里 OFT 头损失 0.243 vs 单头 0.244；漂移度量的定义问题及修正见 [`experiments/results/f0_libero_goal_smoke/`](../experiments/results/f0_libero_goal_smoke/README.md)
- [x] 用修正后的漂移度量重跑（F0 v3，三条运行各 300 步，1 卡）：`QwenBackboneProbe`（换回嵌入 + token 级 CKA + 跨场景分层探针批 + 按更新次数编号）；OFT 第 35 层 2e-4、三头 5e-2、冻结层精确 0；`embed_tokens` 冻结消融无代价；M1 阈值量级写进 R5（[`experiments/results/f0_libero_goal_smoke/`](../experiments/results/f0_libero_goal_smoke/README.md) §3.5）
- [x] WP1 跨头线性探针在 F0 的 5 个最终模型 + 预训练 VLM 上首跑（`scripts/cross_head_probe.py`，1 卡 15 分钟，2,048 样本跨两套）：预训练 VLM 已线性编码约一半动作方差（R² 0.50），五个微调骨干都没有提高它，三头模型第 35 层反而低 0.01；token 级保留度 OFT 99.5% vs 三头 97.6%（反向 96.7%），v2 复现一致——顶层改写是小幅侵蚀而非写入，G2 的线性探针基线已立（[`experiments/results/f0_libero_goal_smoke/`](../experiments/results/f0_libero_goal_smoke/README.md) §3.6）
- [x] F2 冻结骨干迁移（16 条运行，1 卡）：预训练 / OFT 微调 / 三头微调 / OFT+冻结嵌入四种骨干全部冻结，各接全新 OFT 头与 PI 头在 LIBERO-spatial（未见）与 LIBERO-goal（已调）上训 300 步。没有有害的 decoder lock-in（OFT 骨干接新 PI 头 = 预训练骨干），三头骨干对同任务新 OFT 头最好（0.216 vs 0.226 vs 0.267）但优势不迁到新场景，PI 头对骨干不敏感（[`experiments/results/f2_frozen_backbone_transfer/`](../experiments/results/f2_frozen_backbone_transfer/README.md)）
- [x] F3 学习率控制（2 条运行，1 卡）：漂移驱动 LLRD 端到端工作（第 35 层越过 1e-2 后倍率减半到下限 0.05，最终漂移 −24%，头损失代价 0.004–0.011）但触发太晚；静态 LLRD 0.85 不硬冻结明显更差。发现单卡 bf16 路径没有 fp32 主权重、lr 1e-5 的更新对多数权重不足半个 ulp——F0/F3 漂移的绝对量级与 OFT/三头倍数需用 F4 校准（[`experiments/results/f3_llrd/`](../experiments/results/f3_llrd/README.md) §3）
- [x] F4 fp32 主权重校准（2 条运行，1 卡）：同配置下单头 OFT 损失 0.247 → 0.183、三头 0.242/0.324/0.416 → 0.178/0.293/0.353；第 35 层漂移 OFT 0.0003 → 0.026、三头 0.052 → 0.484——bf16 单卡路径一直在吞骨干更新，F0/F2/F3 的绝对读数需重新解释，三头/单头顶层改写倍数由 240× 修正为 19×，"三头不伤 OFT 头"仍成立（[`experiments/results/f4_fp32_master_weights/`](../experiments/results/f4_fp32_master_weights/README.md)）；`f0_libero_goal_smoke.yaml` 默认 `backbone_fp32: true` 并冻结 `embed_tokens`
- [x] F4 checkpoint 上的跨头探针（含 H1 预注册的 OFT 查询位可读性）：查询位 L35 goal 0.462（预训练）→ 0.562（OFT）→ 0.581（三头），跨套 goal→spatial −1.5 → −0.08 → −0.03，保留度 1.0 → 0.870 → 0.794——微调把动作信息写进骨干、三头写得更多且更可迁移、代价是改写预训练表征；bf16 checkpoint 上"无写入"的结论作废（[F4 §3](../experiments/results/f4_fp32_master_weights/README.md)）
- [ ] F2 冻结骨干迁移在 fp32 骨干上的复测（pub2 运行中）；F3 的 LLRD 两臂在 fp32 下复跑
- [ ] R3 标定曲线（VLAct 全配方、2000 步一探针）→ 定 R5 的 `drift_high / drift_low`（起点已按 F3 改为 1e-3 / 1e-4）

## 8. 已知偏差与解释约束

- `vlact_ext` 的 GR00T 头以 `state_dim: 0` 构建、本体状态改走文本，与 GR00T 论文"状态进 System 1"及 StarVLA 原生 `QwenGR00T` 不同；比较时须注明，它不是同一个头。
- R2 起 `mask_oft_queries_for_fm_heads: true`，PI / GR00T 头看不到 OFT 的查询位；R1 / R0 不涉及。跨头结论只在此设定下成立。
- `QwenMultiHeadLab` 通过覆写 `_encode` 缓存骨干输出再在 `forward` 中取用；这依赖 `vlact_ext` 父类的私有方法签名，父类改动时需同步。
- 只换骨干协议里下游统一用 `starvla_lab.train.train_starvla_lab`（`trainer.lab.mode: single`），与 StarVLA 原生 `train_starvla.py` 的差别仅在数据比例钩子；比例为 1 时行为一致。

## 9. 评审修订记录（2026-09-05）

| # | 评审发现 | 处理 |
|---|---|---|
| 1 | 探针 / 调度器是库，未接进训练循环；`trainer.llrd.*` 等键无消费者 | 新增 WP9 `starvla_lab.train`；配置键统一为 `trainer.lab.*`，由 `train_starvla_lab.py` 消费 |
| 2 | 协议引用的 yaml / 评测脚本是占位名 | `protocol_f1.yaml` 改为真实路径；评测命令改为每基准模板（LIBERO-plus 环境变量、RoboTwin `start_eval.sh -m demo_clean/demo_randomized`、GR1 位置参数） |
| 3 | `data_fraction` 无实现 | `data/subsample.py`：按轨迹的确定性子集 + `install_fraction_hook` 挂到 `make_LeRobotSingleDataset` |
| 4 | 预训练脚本未指定、`datasets.vlm_data: null` 不可用 | 矩阵每个变体增加 `train_script`；R1 / R2 用单 loader 脚本 |
| 5 | 下游算力未预算 | 分级设计；`budget.md` 给出 21,000 GPU 小时全量与 9,000 的最小子集 |
| 6 | G2 统计功效不足、指标未预注册 | Q1 改为约 90 个点（变体 × checkpoint × seed）、预注册主指标、置信下界判据 |
| 7 | WP3 缺数据流水线 | 新增 WP3b `starvla_lab.data`（特征缓存、关键帧标注） |
| 8 | FM 头默认看到 OFT 查询位 | R2+ 设 `mask_oft_queries_for_fm_heads: true` |
| 9 | GR00T 头语义偏离 | 写入 §8 已知偏差 |
| 10 | `QwenMultiHeadLab` 侧信道缓存 | 保留，写入 §8；父类改动时同步 |
| 11 | 头 dropout 未接线 | `QwenMultiHead` 增加 `active_heads`，`LabHooks` 每步设置 |
| 12 | 漂移阈值无依据 | M1 的 R3 以 `calibrate_only` 标定；写入 §6 |
| 13 | 度量粒度未定 | §6 固定探测批与 `extract_fn`（最初为 `train_starvla_lab.qwen_layer_extract_fn` 逐层 mean-pool；F0 v2 诊断后改为 `probes.QwenBackboneProbe`：换回预训练嵌入、token 级 CKA 为主） |

## 10. 阶段 B 首批 1 卡实验的综合结论（2026-09-06）

五组实验（WP6 开销、F0 v2/v3、跨头探针、F2、F3）都在 Qwen3-VL-4B + LIBERO-goal、300 步、1×A100 的小 regime 里完成，合计约 40 条运行、8 小时卡时。它们不能替代 R0–R9，但已经改写了方案的几处假设与工具定义：

| 研究问题 | 方案原假设 | 小 regime 下的证据 | 对后续的影响 |
|---|---|---|---|
| Q1 可复用性能否不跑下游就度量 | H1：冻结骨干上的跨头线性探针 R² 与换头成功率强相关 | **fp32 checkpoint（F4）上**：H1 查询位可读性 0.462（预训练）→ 0.562（OFT）→ 0.581（三头），跨套 goal→spatial −1.5 → −0.08 → −0.03，保留度 1.0 → 0.87 → 0.79——微调把动作信息线性地写进骨干，三头写得更多、更可迁移，代价是改写预训练表征。bf16 checkpoint 上"无写入、只有侵蚀"的读数是欠训练骨干的产物 | 查询位可读性作 G2 主指标（`cross_head_probe.py --query_layers`），pooled 与保留度作辅助；F2 的"冻结骨干 + 新头 300 步"作廉价代理，其 fp32 复测决定它是否与探针同向 |
| Q3 先验保护能否自动化 | H3：漂移驱动 LLRD ≥ 硬冻结 | 控制器端到端可用、机制正确（第 35 层越阈后倍率减到下限，漂移 −24%，头损失代价 ≤ 0.011），但触发迟滞；静态 LLRD 0.85 不硬冻结在 bf16 单卡路径下明显更差 | R5 阈值起点改为 1e-3 / 1e-4；R4/R5 必须在有 fp32 主权重的路径（DeepSpeed 或 `backbone_fp32`）上跑，否则低 lr 层的更新会被 bf16 舍入吞掉（§6） |
| VLAct 配方 (c) 多头共监督 | 三头不伤单头、骨干更换头友好 | 三头模型里 OFT 头损失与单头相同（fp32：0.178 vs 0.183）、开销 1.45–1.54×；fp32 下三头骨干的查询位表征跨场景迁移最好（goal→spatial −0.03 vs OFT −0.08），pooled 可读性最高（0.566 vs 0.537）。bf16 下的 F2"更换头无差别"结论基于欠训练骨干，fp32 复测进行中 | "多头共监督让写入的表征变通用"在小规模上有了第一条正面证据（预注册指标）；R3 vs R1 用同一探针 + F2 协议放大检验 |
| VLAct 配方 (a) 冻结下半层 | 视觉编码器 + LLM 前 18 层冻结即可保护先验 | 300 步内两种模型都只动第 33 层以上，第 18–32 层漂移 < 1e-4；`embed_tokens` 冻不冻对损失无影响、对探针影响巨大 | 冻结集合建议加入 `embed_tokens`（VLAct 未说明）；探针一律换回预训练嵌入 |
| 度量口径 | mean-pool 隐状态 CKA、训练集前 64 样本 | v2 的曲线 98% 是嵌入假象；修正后（token 级、换回嵌入、跨场景分层批、按更新次数编号）曲线单调、噪声底为 0 | 已写入 §6 与 `QwenBackboneProbe`；R3 的标定曲线用新口径 |
| 单卡冒烟路径本身 | 与多卡路径等价 | 无 fp32 主权重 → lr 1e-5 更新对多数权重不足半个 ulp。**F4 证实**：fp32 主权重下 OFT 损失 0.247 → 0.183、第 35 层漂移 0.0003 → 0.026，三头 0.052 → 0.484；三头/单头倍数 240× → 19×，OFT 也在明显改写顶层 | `backbone_fp32` 成为单卡默认；F0 §3.6 跨头探针、F2、F3 基于 bf16 checkpoint 的绝对读数待在 F4 checkpoint 上重测（已排队） |

**下一步（按优先级）**：F2 在 fp32 骨干上的复测（运行中）→ F3 两臂 fp32 复跑 → R3（VLAct 全配方、LIBERO 四套 + caption 共训、DeepSpeed）用新口径标定，同时用 F2 协议做 G2 的第一批点。
