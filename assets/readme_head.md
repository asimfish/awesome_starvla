# Awesome StarVLA Resources

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![papers](https://img.shields.io/badge/papers-120-blue)
![reports](https://img.shields.io/badge/reports-7-red)
![zh-PDF](https://img.shields.io/badge/zh--PDF-3-green)
![slides](https://img.shields.io/badge/slides-29p-orange)
![full report](https://img.shields.io/badge/full_report-44p-8a2be2)
![license](https://img.shields.io/badge/license-CC_BY_4.0-8a2be2)

![Figure 1 · Timeline](assets/fig1_timeline.svg)

*图 1 · 120 篇文献的时间线：11 个分类分泳道，按 arXiv 首版年月定位（无 arXiv 编号的 7 条官方页面未画出），★ 为 StarVLA 团队或基于 StarVLA 代码库的工作。2025 下半年到 2026 上半年集中了通用 VLA 策略与世界模型两条线的爆发。*

![Figure 2 · Taxonomy](assets/fig2_taxonomy.svg)

*图 2 · VLA 持续预训练的设计空间：骨干与先验保护 / 动作头 / 动作空间 / 数据 / 训练系统 / 评测 / 开放问题七个维度，每格是三篇论文给出的一条证据（表号可在 `reports/` 中查到）。两图由 `scripts/make_figures.py` 从 `assets/papers_curated.md` 生成。*

围绕 **StarVLA 代码库生态**与 **VLA（Vision-Language-Action）持续预训练 / 动作头 / 跨本体表示学习**的论文与资源列表 + 系统性调研仓库（2026-09 完成）。核心对象是三篇同一团队、同一骨干（Qwen3-VL-4B）、同一代码库的工作：[StarVLA 技术报告](https://arxiv.org/abs/2604.05014)（基础设施）、[StarVLA-α](https://arxiv.org/abs/2604.11757)（去混杂的对照基线）、[VLAct](https://arxiv.org/abs/2608.27550)（表示中心的持续预训练配方）。与一般 awesome 列表不同，本仓库同时提供：

- **7 份中文深度报告**（`reports/`）：VLAct 精读、StarVLA 代码库逐文件解析（含 VLAct 配方在代码中"已有 / 部分 / 缺失"的对照与 diff 级改动建议）、StarVLA-α 与技术报告解读、动作头综合对比、13 个基准的评测生态、研究路线图（6 类 18 个方向 + 六个月执行计划）
- **3 篇论文英文原版 + 保版式中文翻译 PDF**（`papers/`，翻译由 [super_translate](https://github.com/asimfish/super_translate) 生成；三篇论文均为 CC BY 4.0）
- **29 页 Beamer 幻灯片**（`report/awesome_starvla_slides.pdf`，XeLaTeX 源码同目录）+ **44 页合订全文报告**（`report/awesome_starvla_full_report.pdf`，7 份报告 + 两张总览图）
- **120 篇文献编目**（第 4 节，113 条 arXiv 链接经 arXiv API 逐条核验标题，其余 7 条为无 arXiv 的官方技术博客 / 模型卡 / 数据集页）

一句话结论：**VLM 骨干是 VLA 的一阶设计变量，预训练不是双刃剑，配方决定符号**——朴素动作拟合会把机器人预训练变成负资产（OXE 预训练让 RoboCasa-GR1 24×10 从 9.8 掉到 1.2），保护 VLM 先验 + 多头共监督 + 部分统一动作空间把同样的数据源变成净收益（20% 数据超过全量 GR00T-N1.6）；下一步最值钱的是表征诊断工具、"只换骨干"的基准协议，以及所有人都接近零的 Memory 维度。

StarVLA 团队或直接基于 StarVLA 代码库构建的工作以 ⭐ 标记。

*Maintained by [asimfish](https://github.com/asimfish). 欢迎 PR 与 issue，规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。*

## [Content](#content)

<table>
<tr><td colspan="2"><a href="#1-start-here">1. Start Here</a></td></tr>
<tr><td colspan="2"><a href="#2-core-readings">2. Core Readings</a></td></tr>
<tr><td colspan="2"><a href="#3-reports">3. Reports（中文深度报告）</a></td></tr>
<tr><td colspan="2"><a href="#4-papers">4. Papers</a></td></tr>
<tr>
	<td>&emsp;<a href="#starvla-family">4.1 StarVLA Family</a></td>
	<td>&emsp;<a href="#generalist-vla-policies">4.2 Generalist VLA Policies</a></td>
</tr>
<tr>
	<td>&emsp;<a href="#action-heads--action-representation">4.3 Action Heads &amp; Action Representation</a></td>
	<td>&emsp;<a href="#vlm-backbones-for-vla">4.4 VLM Backbones for VLA</a></td>
</tr>
<tr>
	<td>&emsp;<a href="#representation-centric-pre-training--co-training">4.5 Representation-Centric Pre-training &amp; Co-training</a></td>
	<td>&emsp;<a href="#cross-embodiment--robot-data">4.6 Cross-Embodiment &amp; Robot Data</a></td>
</tr>
<tr>
	<td>&emsp;<a href="#world-models-for-action">4.7 World Models for Action</a></td>
	<td>&emsp;<a href="#benchmarks--evaluation">4.8 Benchmarks &amp; Evaluation</a></td>
</tr>
<tr>
	<td>&emsp;<a href="#rl-post-training-for-vla">4.9 RL Post-training for VLA</a></td>
	<td>&emsp;<a href="#human-video--robot">4.10 Human Video → Robot</a></td>
</tr>
<tr>
	<td>&emsp;<a href="#surveys">4.11 Surveys</a></td>
	<td></td>
</tr>
<tr><td colspan="2"><a href="#5-starvla-codebase-at-a-glance">5. StarVLA Codebase at a Glance</a></td></tr>
<tr><td colspan="2"><a href="#6-benchmarks-cheat-sheet">6. Benchmarks Cheat Sheet</a></td></tr>
<tr><td colspan="2"><a href="#7-research-roadmap">7. Research Roadmap</a></td></tr>
<tr><td colspan="2"><a href="#8-repository-layout">8. Repository Layout</a></td></tr>
<tr><td colspan="2"><a href="#9-translation--build-pipeline">9. Translation &amp; Build Pipeline</a></td></tr>
<tr><td colspan="2"><a href="#10-license--credits">10. License &amp; Credits</a></td></tr>
</table>

## [1. Start Here](#content)

| 时间预算 | 路线 |
|---|---|
| 15 分钟 | [`report/awesome_starvla_slides.pdf`](report/awesome_starvla_slides.pdf)（29 页，含 4 页备份） |
| 通读 | [`report/awesome_starvla_full_report.pdf`](report/awesome_starvla_full_report.pdf)（44 页 A4，7 份报告合订 + 图 1 时间线 / 图 2 设计空间）· [HTML 版](report/awesome_starvla_full_report.html) |
| 1 小时 | [01 · VLAct 精读](reports/01_vlact_deep_dive.md) → [05 · 动作头与动作表示](reports/05_action_heads_and_representation.md) → [07 · 研究路线图](reports/07_research_roadmap.md) |
| 准备上手代码 | [02 · StarVLA 代码库解析](reports/02_starvla_codebase_analysis.md)（第 9 章是 VLAct 配方在代码中的落点，第 11 章是最短上手命令序列） → [06 · 基准生态](reports/06_benchmarks_landscape.md)（第 5 章是基准选择建议） |
| 系统研读 | 按 04 → 03 → 01 → 02 → 05 → 06 → 07 的顺序读 `reports/`，配 `papers/zh/` 中文 PDF 对照原文 |

## [2. Core Readings](#content)

三篇论文与一个代码库，是本仓库全部报告的一手材料。

1. ⭐ **StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing.** arXiv, 2026\. [paper](https://arxiv.org/abs/2604.05014), [code](https://github.com/starVLA/starVLA), [PDF](papers/en/2604.05014_StarVLA_codebase.pdf), [PDF-zh](papers/zh/2604.05014_StarVLA_codebase_zh.pdf), [解读](reports/04_starvla_codebase_report.md)
_StarVLA Community & Von Neumann Institute, HKUST_ — 骨干–动作头两条契约；FAST / OFT / π / GR00T 四头 × VLM / 世界模型两类骨干；LIBERO 30K 步（9.5 epoch）达 96.6，OpenVLA-OFT 用 175K 步（223 epoch）才多 0.5；8→256 GPU 扩展效率 79%。
2. ⭐ **StarVLA-α: Reducing Complexity in Vision-Language-Action Systems.** ECCV, 2026\. [paper](https://arxiv.org/abs/2604.11757), [PDF](papers/en/2604.11757_StarVLA_alpha.pdf), [PDF-zh](papers/zh/2604.11757_StarVLA_alpha_zh.pdf), [解读](reports/03_starvla_alpha.md)
_Jinhui Ye, Ning Gao, Senqiao Yang, et al., Yilun Chen, Shu Liu, Jiaya Jia_ — Qwen3-VL-4B + MLP 头 + 最少数据工程的对照基线：LIBERO 98.8、GR1 53.8；三个"常识"的重新检验——连续头 ≫ 离散头且三种连续头相当、朴素动作预训练是双刃剑（OXE 把 GR1 24×10 从 9.8 打到 1.2）、数据工程收益随数据量归零；generalist 联训 + 大 batch（64→1024 使 GR1 40.0→59.2）。
3. ⭐ **Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models (VLAct).** arXiv, 2026\. [paper](https://arxiv.org/abs/2608.27550), [project](https://starvla.github.io/VLAct), [PDF](papers/en/2608.27550_VLAct.pdf), [PDF-zh](papers/zh/2608.27550_VLAct_zh.pdf), [解读](reports/01_vlact_deep_dive.md)
_Senqiao Yang, Chengyao Wang, Yuxin Chen, et al., Hengshuang Zhao, Bei Yu, Jiaya Jia_ — 冻视觉编码器 + LLM 下半层与 caption 混训保护先验；OFT + PI + GR00T 三头共监督消除 decoder lock-in（单头预训练让 PI 微调 60.5→55.1，加一头翻正到 63.1）；20 维部分统一动作空间 + wrap-aware loss（+5 点）。只换骨干权重：LIBERO-Plus 82.6、VLA-Arena 54.8、RoboTwin 2.0 92.5、GR1 20% 数据 49.5 > 全量 GR00T-N1.6 47.6；开源数据 + 16 GPU。
4. ⭐ **StarVLA 代码库（GitHub）.** [starVLA/starVLA](https://github.com/starVLA/starVLA)，MIT，[代码解析](reports/02_starvla_codebase_analysis.md)
— 261 个 Python 文件 / 57.6k 行；28 个注册框架（VLM4A 20、WM4A 6、VM4A 2）；11 个动作头文件；13 个仿真基准 + 5 个真机示例；7 个测试文件、无 CI。

## [3. Reports](#content)

| # | 报告 | 内容 | 行数 |
|---|---|---|---|
| 01 | [VLAct 精读](reports/01_vlact_deep_dive.md) | 问题设定、pilot study 与 decoder lock-in、三组件 + 微调协议、主结果与消融汇总、与 StarVLA-α 的对话、批判性评价（8 条局限）、可复现性 | ~170 |
| 02 | [StarVLA 代码库解析](reports/02_starvla_codebase_analysis.md) | 目录与设计哲学、配置系统、数据管线、模型层（四头实现细节）、训练、部署、25 个 examples、agent skill、**VLAct 六项配方在代码中的状态与 diff 级改动建议**、代码质量问题（已核实的 5 处 bug 与文档漂移）、上手路径 | ~660 |
| 03 | [StarVLA-α 解读](reports/03_starvla_alpha.md) | 最简基线设计、主结果、三个常识的重新检验（表 2/3/4）、generalist 评测范式、真机、局限 | ~110 |
| 04 | [StarVLA 技术报告解读](reports/04_starvla_codebase_report.md) | 两条契约、四种实例、三种训练模式、server–client 评测、LIBERO 基线（含步数 / epoch）、ST4VLA 共训案例、计算效率、"广义 VLA 视角" | ~140 |
| 05 | [动作头与动作表示](reports/05_action_heads_and_representation.md) | 四种头的形式化、三篇论文全部对照数字、动作空间设计证据、选型指南 | ~120 |
| 06 | [基准生态](reports/06_benchmarks_landscape.md) | 13 个仿真基准 + 5 个真机流程逐个拆解（协议、数字、入口脚本、已知问题）、评测公平性、真机协议、**面向持续预训练研究的基准选择建议**、没有好基准的维度 | ~310 |
| 07 | [研究路线图](reports/07_research_roadmap.md) | 三篇论文留下的地图、6 类 18 个方向（问题 / 证据 / 做法 / 代码落点 / 评测 / 风险）、优先级矩阵、六个月执行计划、风险与对策 | ~180 |

## [4. Papers](#content)

条目格式参考 [Thinklab-SJTU/awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co)：**标题.** 会议 / arXiv, 年份. 链接；斜体作者；一句中文摘要（含数字与结论）及与 StarVLA / VLAct 的关系。完整列表另存于 [`assets/papers_curated.md`](assets/papers_curated.md)。

