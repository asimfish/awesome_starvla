# Awesome StarVLA Resources

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![papers](https://img.shields.io/badge/papers-120-blue)
![reports](https://img.shields.io/badge/reports-8-red)
![zh-PDF](https://img.shields.io/badge/zh--PDF-6-green)
![slides](https://img.shields.io/badge/slides-29p_beamer_%2B_18p_pptx-orange)
![full report](https://img.shields.io/badge/full_report-50p-8a2be2)
![code](https://img.shields.io/badge/vlact__ext-61_tests_passed-brightgreen)
![license](https://img.shields.io/badge/license-CC_BY_4.0-8a2be2)

![Figure 1 · Timeline](assets/fig1_timeline.svg)

*图 1 · 120 篇文献的时间线：11 个分类分泳道，按 arXiv 首版年月定位（无 arXiv 编号的 7 条官方页面未画出），★ 为 StarVLA 团队或基于 StarVLA 代码库的工作。2025 下半年到 2026 上半年集中了通用 VLA 策略与世界模型两条线的爆发。*

![Figure 2 · Taxonomy](assets/fig2_taxonomy.svg)

*图 2 · VLA 持续预训练的设计空间：骨干与先验保护 / 动作头 / 动作空间 / 数据 / 训练系统 / 评测 / 开放问题七个维度，每格是三篇论文给出的一条证据（表号可在 `reports/` 中查到）。两图由 `scripts/make_figures.py` 从 `assets/papers_curated.md` 生成。*

围绕 **StarVLA 代码库生态**与 **VLA（Vision-Language-Action）持续预训练 / 动作头 / 跨本体表示学习**的论文与资源列表 + 系统性调研仓库（2026-09 完成）。核心对象是三篇同一团队、同一骨干（Qwen3-VL-4B）、同一代码库的工作：[StarVLA 技术报告](https://arxiv.org/abs/2604.05014)（基础设施）、[StarVLA-α](https://arxiv.org/abs/2604.11757)（去混杂的对照基线）、[VLAct](https://arxiv.org/abs/2608.27550)（表示中心的持续预训练配方）。与一般 awesome 列表不同，本仓库同时提供：

- **8 份中文深度报告**（`reports/`）：VLAct 精读、StarVLA 代码库逐文件解析（含 VLAct 配方在代码中"已有 / 部分 / 缺失"的对照与 diff 级改动建议）、StarVLA-α 与技术报告解读、动作头综合对比、13 个基准的评测生态、研究路线图（6 类 18 个方向 + 六个月执行计划）、**四种动作头（FAST / OFT / PI / GR00T）的 60 分钟讲稿**（配 20 页讲解幻灯片）
- **6 篇论文英文原版 + 保版式中文翻译 PDF**（`papers/`，翻译由 [super_translate](https://github.com/asimfish/super_translate) 生成）：StarVLA 三篇 + 四个动作头的源论文中的 FAST、OpenVLA-OFT、GR00T N1（六篇均为 CC BY 4.0；π0 为 arXiv 非独占许可，仓库只放链接）
- **29 页 Beamer 幻灯片**（`report/awesome_starvla_slides.pdf`，XeLaTeX 源码同目录）+ **18 页原生可编辑 PPTX**（`report/awesome_starvla_slides.pptx`，由 [ppt-master](https://github.com/hugohe3/ppt-master) Quick Generate 生成，全部为原生形状与表格）+ **44 页合订全文报告**（`report/awesome_starvla_full_report.pdf`，8 份报告 + 两张总览图）
- **120 篇文献编目**（第 4 节，113 条 arXiv 链接经 arXiv API 逐条核验标题，其余 7 条为无 arXiv 的官方技术博客 / 模型卡 / 数据集页）
- **VLAct 缺失组件的 StarVLA 扩展代码**（`code/vlact_ext/`，约 1,600 行）：多头共监督框架 `QwenMultiHead`、wrap-aware L1、20 维部分统一动作布局 transform、正则 / 区间冻结规则，附 61 个 CPU 单元测试与完整的 VLAct 预训练示例 yaml；不改 StarVLA 任何已有文件即可拷入使用

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
<tr>
	<td>&emsp;<a href="#51-vlact-extension-for-starvla-codevlact_ext">5.1 VLAct Extension for StarVLA（code/vlact_ext）</a></td>
	<td></td>
</tr>
<tr><td colspan="2"><a href="#6-benchmarks-cheat-sheet">6. Benchmarks Cheat Sheet</a></td></tr>
<tr><td colspan="2"><a href="#7-research-roadmap">7. Research Roadmap</a></td></tr>
<tr><td colspan="2"><a href="#8-repository-layout">8. Repository Layout</a></td></tr>
<tr><td colspan="2"><a href="#9-translation--build-pipeline">9. Translation &amp; Build Pipeline</a></td></tr>
<tr><td colspan="2"><a href="#10-license--credits">10. License &amp; Credits</a></td></tr>
</table>

## [1. Start Here](#content)

| 时间预算 | 路线 |
|---|---|
| 15 分钟 | [`report/awesome_starvla_slides.pdf`](report/awesome_starvla_slides.pdf)（29 页 Beamer，含 4 页备份）或 [`report/awesome_starvla_slides.pptx`](report/awesome_starvla_slides.pptx)（18 页精简版，PowerPoint 可编辑） |
| 通读 | [`report/awesome_starvla_full_report.pdf`](report/awesome_starvla_full_report.pdf)（50 页 A4，8 份报告合订 + 图 1 时间线 / 图 2 设计空间）· [HTML 版](report/awesome_starvla_full_report.html) |
| 1 小时 | [01 · VLAct 精读](reports/01_vlact_deep_dive.md) → [05 · 动作头与动作表示](reports/05_action_heads_and_representation.md) → [07 · 研究路线图](reports/07_research_roadmap.md) |
| 搞懂四个动作头 | [08 · 讲稿：FAST / OFT / PI / GR00T](reports/08_action_heads_lecture.md)（60 分钟组会讲稿，含直觉、公式、训练/推理、StarVLA 实现、选型问答）+ [讲解幻灯片](report/action_heads_lecture_slides.pdf)（20 页）+ 四篇源论文中译（`papers/zh/`） |
| 准备上手代码 | [02 · StarVLA 代码库解析](reports/02_starvla_codebase_analysis.md)（第 9 章是 VLAct 配方在代码中的落点，第 11 章是最短上手命令序列） → [06 · 基准生态](reports/06_benchmarks_landscape.md)（第 5 章是基准选择建议） → [`code/vlact_ext/README.md`](code/vlact_ext/README.md)（怎么把多头 / wrap loss / 统一布局拷进 StarVLA） |
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

四种动作头的源论文（配 [08 · 讲稿](reports/08_action_heads_lecture.md) 与 [讲解幻灯片](report/action_heads_lecture_slides.pdf) 阅读）：

5. **FAST: Efficient Action Tokenization for Vision-Language-Action Models.** arXiv, 2025\. [paper](https://arxiv.org/abs/2501.09747), [PDF](papers/en/2501.09747_FAST.pdf), [PDF-zh](papers/zh/2501.09747_FAST_zh.pdf)
_Karl Pertsch et al._（Physical Intelligence）— 分位数归一化 → 逐维 DCT → 缩放取整 → 低频优先展平 → BPE（词表 1024），1 秒动作块约 30 个 token/臂；朴素分箱在高频下每个 token 的边际信息趋零；训练 GPU 小时比 diffusion 版 π0 少约 5×，推理为自回归解码。
6. **Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT).** arXiv, 2025\. [paper](https://arxiv.org/abs/2502.19645), [PDF](papers/en/2502.19645_OpenVLA_OFT.pdf), [PDF-zh](papers/zh/2502.19645_OpenVLA_OFT_zh.pdf)
_Moo Jin Kim, Chelsea Finn, Percy Liang_（Stanford）— 空动作嵌入 + 双向注意力实现并行解码，动作块使吞吐 ×K，MLP 直出连续动作 + L1 回归；LIBERO 76.5 → 97.1，吞吐 26×（ALOHA 25 步块 43×）；FiLM 解决多视角下忽略语言。
7. **π0: A Vision-Language-Action Flow Model for General Robot Control.** arXiv, 2024\. [paper](https://arxiv.org/abs/2410.24164)（arXiv 非独占许可，PDF 不随仓库分发）
_Kevin Black et al._（Physical Intelligence）— PaliGemma 3B + 从零初始化的约 3 亿参数动作专家（独立权重、MoE 式路由），flow matching 目标 $\|v_\theta - (A-\epsilon)\|^2$，τ 从偏向噪声端的 Beta 分布采样，推理 10 步欧拉；H=50、最高 50 Hz；约 1 万小时自有数据 + OXE，7 种本体 68 任务。
8. **GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.** arXiv, 2025\. [paper](https://arxiv.org/abs/2503.14734), [PDF](papers/en/2503.14734_GR00T_N1.pdf), [PDF-zh](papers/zh/2503.14734_GR00T_N1_zh.pdf)
_Johan Bjorck et al._（NVIDIA）— 双系统：Eagle-2 VLM（第 12 层特征，≈10 Hz）作 System 2，adaLN DiT（交替 self / cross-attention，≈120 Hz，块长 16，4 步去噪）作 System 1；本体特定 MLP 编解码状态与动作；数据金字塔 + VQ-VAE 潜动作把无标签视频纳入训练；2.2B 参数。

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
| 08 | [讲稿：四种动作头](reports/08_action_heads_lecture.md) | FAST（DCT + BPE 时间序列压缩）、OFT（并行解码 + L1 回归）、PI（flow matching 动作专家）、GR00T（双系统 DiT）各一节：要解决的问题、核心机制、训练与推理、论文结果、StarVLA 实现；横向对比、选型问答、阅读顺序、公式速查 | ~300 |

## [4. Papers](#content)

条目格式参考 [Thinklab-SJTU/awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co)：**标题.** 会议 / arXiv, 年份. 链接；斜体作者；一句中文摘要（含数字与结论）及与 StarVLA / VLAct 的关系。完整列表另存于 [`assets/papers_curated.md`](assets/papers_curated.md)。

