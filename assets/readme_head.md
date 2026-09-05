# Awesome StarVLA Resources

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![papers](https://img.shields.io/badge/papers-120-blue)
![reports](https://img.shields.io/badge/reports-11-red)
![zh-PDF](https://img.shields.io/badge/zh--PDF-7-green)
![tests](https://img.shields.io/badge/CPU_tests-186_passed-brightgreen)
![license](https://img.shields.io/badge/license-CC_BY_4.0-8a2be2)

围绕 **StarVLA 代码库生态**与 **VLA 持续预训练 / 动作头 / 跨本体表示学习**的论文列表 + 调研 + 代码仓库。核心对象是同一团队、同一骨干（Qwen3-VL-4B）、同一代码库的三篇工作：[StarVLA 技术报告](https://arxiv.org/abs/2604.05014)（基础设施）、[StarVLA-α](https://arxiv.org/abs/2604.11757)（去混杂的对照基线）、[VLAct](https://arxiv.org/abs/2608.27550)（表示中心的持续预训练配方），以及基于 StarVLA 做记忆的 [EventVLA](https://arxiv.org/abs/2606.20092)。除 120 篇文献编目外，仓库还提供 11 份中文深度报告、7 篇论文的保版式中文翻译、两个不改 StarVLA 源码即可使用的扩展包（VLAct 配方 + 改进方案研究包，186 个 CPU 测试）、EventVLA 子模块，以及已经跑出来的第一批 GPU 数字。

> **一句话结论**：VLM 骨干是 VLA 的一阶设计变量，预训练不是双刃剑，配方决定符号——朴素动作拟合会把机器人预训练变成负资产（OXE 预训练让 RoboCasa-GR1 24×10 从 9.8 掉到 1.2），保护 VLM 先验 + 多头共监督 + 部分统一动作空间把同样的数据源变成净收益（20% 数据超过全量 GR00T-N1.6）；下一步最值钱的是表征诊断工具、"只换骨干"的基准协议，以及所有人都接近零的 Memory 维度。

*Maintained by [asimfish](https://github.com/asimfish)。欢迎 PR 与 issue，规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。*

## Contents

- [1. Quick Start](#1-quick-start) — 按目的选入口；7 篇核心论文
- [2. Reports](#2-reports) — 11 份中文深度报告
- [3. Code & Experiments](#3-code--experiments) — `vlact_ext`、`starvla_lab`、EventVLA 子模块、已实测的 GPU 数字
- [4. Papers](#4-papers) — 120 篇文献，11 个分类：[StarVLA Family](#starvla-family) · [Generalist VLA Policies](#generalist-vla-policies) · [Action Heads](#action-heads--action-representation) · [VLM Backbones](#vlm-backbones-for-vla) · [Pre-training & Co-training](#representation-centric-pre-training--co-training) · [Cross-Embodiment & Data](#cross-embodiment--robot-data) · [World Models](#world-models-for-action) · [Benchmarks](#benchmarks--evaluation) · [RL Post-training](#rl-post-training-for-vla) · [Human Video → Robot](#human-video--robot) · [Surveys](#surveys)
- [5. Cheat Sheets](#5-cheat-sheets) — StarVLA 代码库速览、基准选择、研究路线图
- [6. Repository Layout & Build](#6-repository-layout--build)
- [7. License & Citation](#7-license--citation)

## [1. Quick Start](#contents)

| 目的 | 入口 |
|---|---|
| 15 分钟了解全貌 | [29 页 Beamer 幻灯片](report/awesome_starvla_slides.pdf)（XeLaTeX 源码同目录）或 [18 页 PPTX](report/awesome_starvla_slides.pptx)（PowerPoint 可编辑） |
| 通读 | [62 页合订报告 PDF](report/awesome_starvla_full_report.pdf) · [HTML](report/awesome_starvla_full_report.html)（报告 01–10 + 两张总览图） |
| 1 小时抓住主线 | [01 · VLAct 精读](reports/01_vlact_deep_dive.md) → [05 · 动作头与动作表示](reports/05_action_heads_and_representation.md) → [07 · 研究路线图](reports/07_research_roadmap.md) |
| 搞懂四种动作头 | [08 · 讲稿](reports/08_action_heads_lecture.md)（60 分钟组会讲稿）+ [20 页讲解幻灯片](report/action_heads_lecture_slides.pdf) + 四篇源论文中译（`papers/zh/`） |
| 上手 StarVLA 代码 | [02 · 代码库解析](reports/02_starvla_codebase_analysis.md)（第 9 章：VLAct 配方在代码中的落点；第 11 章：最短上手命令）→ [06 · 基准生态](reports/06_benchmarks_landscape.md)（第 5 章：基准选择）→ [`code/vlact_ext/README.md`](code/vlact_ext/README.md) → [§3.3 本地跑通](#33-本地跑通不需要-gpu-和权重) |
| 动手做改进 | [10 · 改进方案](reports/10_improvement_plan.md)（研究问题 → 工作包 → 实验矩阵 → 决策门）→ [`code/starvla_lab/README.md`](code/starvla_lab/README.md) → [`experiments/README.md`](experiments/README.md)；动 EventVLA 之前先读 [11 · 代码审计](reports/11_eventvla_code_audit.md) |
| 系统研读 | 按 04 → 03 → 01 → 02 → 05 → 06 → 07 → 09 → 11 → 10 的顺序读 `reports/`，配 `papers/zh/` 对照原文 |

**7 篇核心论文**（英文原版与中文翻译均在 `papers/`；一句话摘要见第 4 节对应条目；⭐ = StarVLA 团队或基于 StarVLA 代码库）：

| 论文 | 角色 | 链接 |
|---|---|---|
| ⭐ StarVLA: A Lego-like Codebase for VLA Model Developing | 基础设施 | [arXiv](https://arxiv.org/abs/2604.05014) · [code](https://github.com/starVLA/starVLA) · [PDF](papers/en/2604.05014_StarVLA_codebase.pdf) · [中译](papers/zh/2604.05014_StarVLA_codebase_zh.pdf) · [解读 04](reports/04_starvla_codebase_report.md) |
| ⭐ StarVLA-α: Reducing Complexity in VLA Systems（ECCV 2026） | 去混杂的对照基线 | [arXiv](https://arxiv.org/abs/2604.11757) · [PDF](papers/en/2604.11757_StarVLA_alpha.pdf) · [中译](papers/zh/2604.11757_StarVLA_alpha_zh.pdf) · [解读 03](reports/03_starvla_alpha.md) |
| ⭐ VLAct: Representation-Centric Continued Pre-training for VLA | 持续预训练配方，本仓库核心 | [arXiv](https://arxiv.org/abs/2608.27550) · [project](https://starvla.github.io/VLAct) · [PDF](papers/en/2608.27550_VLAct.pdf) · [中译](papers/zh/2608.27550_VLAct_zh.pdf) · [解读 01](reports/01_vlact_deep_dive.md) |
| ⭐ EventVLA: Event-Driven Visual Evidence Memory | StarVLA-OFT 上的稀疏视觉记忆 | [arXiv](https://arxiv.org/abs/2606.20092) · [code](code/EventVLA)（[官方](https://github.com/InternRobotics/EventVLA)）· [PDF](papers/en/2606.20092_EventVLA.pdf) · [中译](papers/zh/2606.20092_EventVLA_zh.pdf) · [解读 09](reports/09_eventvla.md) · [代码审计 11](reports/11_eventvla_code_audit.md) |
| FAST: Efficient Action Tokenization for VLA | 离散动作头源论文 | [arXiv](https://arxiv.org/abs/2501.09747) · [PDF](papers/en/2501.09747_FAST.pdf) · [中译](papers/zh/2501.09747_FAST_zh.pdf) |
| OpenVLA-OFT: Fine-Tuning VLA Models — Optimizing Speed and Success | OFT 头源论文 | [arXiv](https://arxiv.org/abs/2502.19645) · [PDF](papers/en/2502.19645_OpenVLA_OFT.pdf) · [中译](papers/zh/2502.19645_OpenVLA_OFT_zh.pdf) |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | GR00T DiT 头源论文 | [arXiv](https://arxiv.org/abs/2503.14734) · [PDF](papers/en/2503.14734_GR00T_N1.pdf) · [中译](papers/zh/2503.14734_GR00T_N1_zh.pdf) |

π0（[arXiv 2410.24164](https://arxiv.org/abs/2410.24164)，PI 头源论文）为 arXiv 非独占许可，只放链接不放 PDF。

## [2. Reports](#contents)

全部为中文，遵循[写作规范](CONTRIBUTING.md#2-报告写作规范)：直接陈述、每个数字有出处（论文表号 / 代码行号）、术语保留英文。

| # | 报告 | 一句话 | 行数 |
|---|---|---|---|
| 01 | [VLAct 精读](reports/01_vlact_deep_dive.md) | 问题设定、decoder lock-in、三组件配方与微调协议、主结果与消融、8 条局限、可复现性 | ~170 |
| 02 | [StarVLA 代码库解析](reports/02_starvla_codebase_analysis.md) | 逐目录解析（配置 / 数据 / 四头 / 训练 / 部署 / 25 个 examples）、VLAct 六项配方在代码中"已有 / 部分 / 缺失"与 diff 级改动建议、5 处已核实 bug、上手路径 | ~660 |
| 03 | [StarVLA-α 解读](reports/03_starvla_alpha.md) | 最简基线、三个"常识"的重新检验（表 2/3/4）、generalist 评测范式、真机、局限 | ~110 |
| 04 | [StarVLA 技术报告解读](reports/04_starvla_codebase_report.md) | 两条契约、四种实例、三种训练模式、server–client 评测、LIBERO 基线的步数 / epoch、计算效率 | ~140 |
| 05 | [动作头与动作表示](reports/05_action_heads_and_representation.md) | 四种头的形式化、三篇论文全部对照数字、动作空间设计证据、选型指南 | ~120 |
| 06 | [基准生态](reports/06_benchmarks_landscape.md) | 13 个仿真基准 + 5 个真机流程逐个拆解、评测公平性、面向持续预训练研究的基准选择、没有好基准的维度 | ~310 |
| 07 | [研究路线图](reports/07_research_roadmap.md) | 6 类 18 个方向（问题 / 证据 / 做法 / 代码落点 / 评测 / 风险）、优先级矩阵、六个月计划 | ~180 |
| 08 | [讲稿：四种动作头](reports/08_action_heads_lecture.md) | FAST / OFT / PI / GR00T 各一节（问题、机制、训练与推理、结果、StarVLA 实现）、横向对比、选型问答、公式速查 | ~300 |
| 09 | [EventVLA 解读](reports/09_eventvla.md) | 非马尔可夫任务、视觉锚点 + 关键帧证据记忆头、RoboTwin-MeM 基准、RMBench 67.8 / RoboTwin-MeM 3.8→75.2 / 真机 60–90、可直接做的叠加实验 | ~100 |
| 10 | [改进方案 v2](reports/10_improvement_plan.md) | 四个研究问题与预注册假设、十个工作包与验收标准、分级 R0–R9 实验矩阵与 GPU 预算、四个决策门、设计评审的 13 条修订 | ~165 |
| 11 | [EventVLA 代码审计](reports/11_eventvla_code_audit.md) | 逐文件核对 `code/EventVLA`：论文 vs 代码 17 项对照（N_max、λ、NMS / 冷却、课程、图像顺序、每 chunk 事件数）、oracle 标签、评测协议里的硬编码、上游 16 个 issue 的复现坑、P0–P2 改进落点 | ~170 |

## [3. Code & Experiments](#contents)

三个代码目录都**不修改 StarVLA 源码**：前两个是拷入即用的扩展包，第三个是子模块。

| 目录 | 内容 | 验证 |
|---|---|---|
| [`code/vlact_ext/`](code/vlact_ext/) | VLAct 六项配方里 StarVLA 缺失 / 半支持的四项：`QwenMultiHead` 多头共监督框架（OFT + GR00T + PI 三头，`action_loss = Σ w_h·L_h`，`predict_action(head=…)` 路由）、wrap-aware L1、20 维部分统一动作布局 transform、正则 / 区间冻结规则；附完整的 VLAct 预训练示例 yaml | 61 个 CPU 测试，mock 骨干，约 30 s |
| [`code/starvla_lab/`](code/starvla_lab/) | [10 · 改进方案](reports/10_improvement_plan.md) 阶段 A 的研究包：`probes/`（跨头探针、CKA、漂移追踪）、`schedules/`（分层学习率衰减、漂移驱动的冻结与辅助数据调度）、`heads/`（未来特征预测头、关键帧头、`QwenMultiHeadLab`）、`data/`（按轨迹子采样、辅助头离线数据）、`train/`（`train_starvla_lab.py` 入口）、`bench/`（"只换骨干"协议、开销测量、头 dropout）、`configs/`（R0–R9 矩阵） | 125 个 CPU 测试 |
| [`code/EventVLA/`](code/EventVLA/) | git 子模块 → [asimfish/EventVLA](https://github.com/asimfish/EventVLA)（fork 自官方）：EventVLA 模型、训练与评测代码 + RoboTwin-MeM 基准（8 个记忆任务）。审计见 [报告 11](reports/11_eventvla_code_audit.md) | 上游 2 个测试文件 |
| [`experiments/`](experiments/) | 运行清单与结果台账：`run_matrix*.csv`（主矩阵 92 次 + 跨头 16 次）、`budget.md`（预训练 7,300 + 下游 13,800 ≈ 21,000 GPU 小时）、`results/<run>/`（每次运行一个 JSON + README） | — |

### 3.1 已实测的数字

| 实验 | 结论 | 详情 |
|---|---|---|
| WP6 · 三头共监督的训练开销（1×A100-80GB，Qwen3-VL-4B 真实权重，batch 8） | OFT 单头 1.27 s/step / 15.4 GB；PI 单头 1.64 / 22.0；三头 OFT + GR00T + PI 1.95 / 27.2（1.54× 单头，不是 3×）；三头 + 头 dropout 1.54 / 25.0 | [`experiments/results/wp6_overhead/`](experiments/results/wp6_overhead/README.md) |
| F0 · LIBERO-goal 真实数据微调冒烟（各 300 步，1×A100，v2 + v3 共 5 条运行） | `QwenMultiHead` 的 OFT 头 L1 0.242–0.243 vs `QwenOFT` 0.244–0.251——三头不伤单头；v2 里训练中记录的逐层 1−CKA"漂移"几乎全部来自 `embed_tokens` 里几十行 prompt 词嵌入的更新被单场景探针批放大，WP1 的度量因此改为"换回预训练嵌入 + token 级 CKA + 跨场景探针批"；v3 用修正后的探针重跑：冻结层精确 0，单头 OFT 只在第 35 层动 2e-4，三头模型第 35 层 5e-2（其余层 < 1e-3），冻结 `embed_tokens` 无代价；WP1 跨头线性探针首跑：预训练 VLM 已线性编码约一半动作方差，五个微调骨干都没提高它，三头模型的顶层改写是小幅侵蚀（token 级保留度 97.6% vs OFT 99.5%）而非写入 | [`experiments/results/f0_libero_goal_smoke/`](experiments/results/f0_libero_goal_smoke/README.md) |
| CPU · 与真实 StarVLA 的集成 | `QwenMultiHead` 用 StarVLA 真实的三个头工厂前向 / 反传 / 逐头预测；`flow_matching_loss` 与原头逐位相等（atol 1e-6）；冻结规则 + LLRD 参数组、头 dropout、探针驱动调度全部走通 | `scripts/smoke_starvla_integration.py` |

仍需 GPU：LIBERO / RoboTwin 仿真评测、DeepSpeed 多卡显存、R3 标定曲线、任何 ≥ 10k 步的训练效果数字——这些是 [07 · 路线图](reports/07_research_roadmap.md) 第 1 个月"复现 VLAct"的起点。

### 3.2 运行测试

```bash
python3 -m pytest code/vlact_ext/tests -q        # 60 passed, 1 skipped（系统 python3.9，CPU）
python3 -m pytest code/starvla_lab/tests -q      # 125 passed
python3 scripts/build_run_matrix.py --print-commands 2
```

### 3.3 本地跑通（不需要 GPU 和权重）

StarVLA 要求 Python ≥ 3.10，所以真实集成单独建环境；`starVLA_code/` 是 [starVLA/starVLA](https://github.com/starVLA/starVLA) 的 checkout，放在本仓库旁边：

```bash
bash scripts/setup_cpu_env.sh                          # uv 建 .venv-starvla（py3.12）+ CPU torch + StarVLA 可编辑安装
PYTHONPATH=code:../starVLA_code .venv-starvla/bin/python -m pytest code/vlact_ext/tests code/starvla_lab/tests -q   # 184 passed, 2 skipped
PYTHONPATH=code:../starVLA_code .venv-starvla/bin/python scripts/smoke_starvla_integration.py                       # 约 15 s
```

GPU 侧：`scripts/gpu_overhead_bench.py`、`scripts/probe_diagnostics.py`、`scripts/cross_head_probe.py` 与 `scripts/cluster/{sync_to_node,setup_gpu_env,run_overhead_bench,run_f0_smoke,run_cross_head_probe}.sh` 可在任一有 Qwen3-VL-4B 权重的单卡机器上复现 §3.1 的数字。

## [4. Papers](#contents)

120 篇文献，11 个分类。条目格式参考 [Thinklab-SJTU/awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co)：**标题.** 会议 / arXiv, 年份. 链接；斜体作者；一句中文摘要（含数字与结论）及与 StarVLA / VLAct 的关系。⭐ = StarVLA 团队或直接基于 StarVLA 代码库的工作。113 条 arXiv 链接经 arXiv API 逐条核验标题，其余 7 条为无 arXiv 的官方技术博客 / 模型卡 / 数据集页。源文件：[`assets/papers_curated.md`](assets/papers_curated.md)。

![Figure 1 · Timeline](assets/fig1_timeline.svg)

*图 1 · 120 篇文献的时间线：11 个分类分泳道，按 arXiv 首版年月定位（7 条无 arXiv 编号的官方页面未画出），★ 为 StarVLA 团队或基于 StarVLA 代码库的工作。*

![Figure 2 · Taxonomy](assets/fig2_taxonomy.svg)

*图 2 · VLA 持续预训练的设计空间：骨干与先验保护 / 动作头 / 动作空间 / 数据 / 训练系统 / 评测 / 开放问题七个维度，每格是三篇论文给出的一条证据（表号见 `reports/`）。两图由 `scripts/make_figures.py` 从 `assets/papers_curated.md` 生成。*

