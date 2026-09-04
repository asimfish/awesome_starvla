
## [5. StarVLA Codebase at a Glance](#content)

数字来自本地快照（分支 `starVLA_dev`，HEAD `d81fc66`，2026-09-04）的实际统计，细节与行号引用见 [02 · 代码库解析](reports/02_starvla_codebase_analysis.md)。

| 层 | 关键文件 | 一句话 |
|---|---|---|
| 框架 | `starVLA/model/framework/{VLM4A,WM4A,VM4A}/*.py` | 28 个注册框架；`build_framework(cfg)` 查 `FRAMEWORK_REGISTRY`；每个文件末尾带 `__main__` smoke test |
| 契约 | `starVLA/model/framework/base_framework.py` | `forward(examples) → {"action_loss"}`；`predict_action(examples) → {"normalized_actions"}`；两者都直接吃原始样本 dict |
| 动作头 | `starVLA/model/modules/action_model/` | 11 个文件：MLP（OFT，masked L1）、FAST（2048 个 action token 占用词表区间）、Layerwise-FM（PI，36 层 cross-DiT，4 步 Euler）、FM DiT-B（GR00T，16 层，状态前置） |
| VLM 接口 | `starVLA/model/modules/vlm/` | `get_vlm_model` 分派 9 个分支：Qwen2.5-VL / Qwen3-VL / Qwen3.5 / Gemma-4 / Molmo2 / MiniCPM-V / Florence-2 / Cosmos-Reason2 / VILA |
| 数据 | `starVLA/dataloader/` | GR00T LeRobot 管线移植（v2.0/v3.0，q99 / mean_std / min_max 归一化，`DATASET_NAMED_MIXTURES` 加权混合）；VLM LLaVA-json；UMI 适配 |
| 训练 | `starVLA/training/` | `train_starvla.py`（SFT）、`train_starvla_cotrain.py`（VLA + VLM 双 loader 双 backward，`loss_scale.vlm`）、`train_starvlm.py`、`train_starvln.py`；`freeze_modules` 精确点路径；分模块 lr；只存 state_dict |
| 部署 | `deployment/model_server/` | WebSocket / ZMQ 策略服务器；基准侧 12 个 `model2*_interface.py` 适配器 |
| 生态 | `examples/` | 13 仿真基准 / 5 真机 / 6 模型扩展 / 1 UMI 人类数据 |

**VLAct 六项配方在代码中的状态**：(b) caption 共训与 (f) 丢头重训 **已有**；(a) 浅层冻结与 (d) 20 维部分统一布局 **部分**（冻结需列 18 条精确路径，mask 只有 OFT 头消费）；(c) 多头共监督与 (e) wrap-aware loss **缺失**。复现 VLAct 的工程量集中在一个新框架 `QwenMultiHead` 与一个动作空间 transform。

## [6. Benchmarks Cheat Sheet](#content)

面向"VLA 持续预训练 / 表示学习"研究的选择（依据见 [06 · 基准生态](reports/06_benchmarks_landscape.md) 第 5 章）：

| 优先级 | 基准 | 考察 | 要点 |
|---|---|---|---|
| 1 | LIBERO-plus | 零样本鲁棒性（Camera / Robot / Language / Light / Background / Noise / Layout） | 训练集固定为标准 LIBERO，提升只能来自骨干；10,030 实例 |
| 2 | RoboTwin 2.0 **Base** | 少样本（50 clean/任务）+ clean→random | clean 与 random 必须同时报；Data Scaling 只作补充曲线 |
| 3 | RoboCasa-GR1 数据比例曲线 | 未见人形本体的样本效率 | 10% / 20% / 50% / 100%，基线也要跑同样比例点 |
| 4 | VLA-Arena | L0→L1/L2 结构外推 + 安全代价 | 官方 30 episodes/任务，StarVLA 默认 10 |
| 5 | RoboDojo | 第三方裁判 + Generalization / Precision / Long-Horizon / Memory / Open 五维 | 适合最终报告；Memory 维度几乎人人接近零 |
| 不建议 | LIBERO 标准版、SimplerEnv、MetaWorld、CALVIN D→D、BEHAVIOR-1K | | 饱和（95–98）、方差大、或评测成本极高 |

报告协议：固定下游预算并写明 checkpoint 规则；≥3 seeds；区分见过 / held-out 本体（VLAct 只有 GR1 与 ARX X5 是真 held-out）；附表示层诊断（跨头迁移矩阵）。

## [7. Research Roadmap](#content)

完整版见 [07 · 研究路线图](reports/07_research_roadmap.md)。六类 18 个方向：

| 类别 | 方向 | 出发点 |
|---|---|---|
| A 表征诊断 | A1 骨干可复用性探针套件；A2 自动化"哪层该冻" | decoder lock-in 只能间接观察；冻结只有 3 档消融 |
| B 预训练目标 | B1 头多样性推广（FAST / 未来帧 / 空间 QA 头）；B2 辅助数据调度；B3 潜动作 vs 多头；B4 世界模型辅助头 | 三头同头 +1.6～4.3；全混合被稀释；"广义 VLA 视角" |
| C 动作空间 | C1 可学习的部分统一布局；C2 几何一致参数化（SO(3)）；C3 零样本本体迁移 | 20 维手工表；wrap loss +5；GR1 20% 才 49.5 |
| D 能力短板 | D1 记忆与长程；D2 语言泛化；D3 低数据下生成式头落后 | Memory 0.66；Language −5.5；Random 下 OFT 41.5 vs GR00T 22.9 |
| E 系统 scaling | E1 规模曲线；E2 三头开销与降本；E3 数据 × 配方 Pareto | 只有 4B；开销未量化 |
| F 评测方法 | F1 骨干基准协议；F2 generalist × 持续预训练；F3 真机统计功效 | "只换骨干"未标准化；两条线未合并；n=10 |

先做（高影响、低成本）：A1 诊断套件、F1 骨干基准脚本、E2 三头开销测量。六个月计划以 16 GPU 为前提，每月一个交付物，最后一个月整合最优组合、全基准评测并向 StarVLA 提 PR。

## [8. Repository Layout](#content)

```
awesome_starvla/
├── README.md                       # 本文件：论文列表 + 导读
├── CONTRIBUTING.md                 # 条目格式、核验要求、报告写作规范
├── LICENSE                         # CC BY 4.0（报告、README、幻灯片）
├── papers/
│   ├── en/                         # 三篇论文英文原版 PDF（arXiv，CC BY 4.0）
│   └── zh/                         # 保版式中文翻译 PDF + 翻译缓存 + QA 报告
├── reports/                        # 7 份中文深度报告（01–07）
├── report/
│   ├── awesome_starvla_slides.tex  # Beamer 源码（XeLaTeX + ctex，16:9）
│   ├── awesome_starvla_slides.pdf  # 29 页
│   ├── awesome_starvla_full_report.html / .pdf   # 44 页合订全文报告
├── assets/
│   ├── papers_curated.md           # 120 条文献编目（README 第 4 节的源）
│   ├── starvla_code_facts.md       # 代码库硬事实卡片（数字、路径、接口签名）
│   ├── fig1_timeline.svg           # 图 1：120 篇文献时间线（由 make_figures.py 生成）
│   ├── fig2_taxonomy.svg           # 图 2：设计空间分类树
│   └── readme_head.md / readme_tail.md   # README 的非论文部分（build_readme.py 的输入）
└── scripts/
    ├── build_slides.sh             # 编译幻灯片
    ├── build_readme.py             # 从 head/tail 与 papers_curated.md 拼装 README
    ├── build_full_report.py        # 合订全文报告（pandoc + Chrome headless）
    ├── make_figures.py             # 生成图 1 / 图 2
    └── translate_papers.sh         # super_translate 翻译流程
```

## [9. Translation & Build Pipeline](#content)

- **翻译**：[super_translate](https://github.com/asimfish/super_translate) `paper-translate` skill，DeepSeek 后端，`--preserve-graphics-text`（图表内文字与公式原样保留），译后 `inspect` 视觉 QA。三篇的 QA 报告随 PDF 存放在 `papers/zh/*.inspect.json`。已知瑕疵：VLAct 中文版 p29 公式内指示函数文字保留英文、p30 一行字号偏小；StarVLA-α 中文版 p16 附录目录 9 条保留英文（带 hyperref 引用被判为保护区）；StarVLA 报告中文版 p15 一处字号偏小。正文均已翻译。
- **合订报告与图**：`python3 scripts/make_figures.py && python3 scripts/build_full_report.py --pdf`（需要 pandoc 与 Google Chrome；公式由 MathJax 渲染，报告源文件统一用 `$...$` / `$$...$$` 定界以兼容 GitHub）。
- **幻灯片**：`bash scripts/build_slides.sh`，需要 XeLaTeX 与 Fandol 字体（MiKTeX / TeX Live 均可）。遵循 [beamer-skill](https://github.com/Noi1r/beamer-skill) 规范：16:9、10pt、无 overlay、每页 ≤2 个彩色框、参考文献页 + 备份页。
- **写作**：报告与 README 按 [anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing) 与 [shuorenhua](https://github.com/MrGeDiao/shuorenhua) 的规则写：直接陈述、不做防御性免责、不用"值得注意的是 / 综上所述"类套话、术语保留英文、数字必须有出处。
- **文献核验**：每条 arXiv 链接用 `curl "http://export.arxiv.org/api/query?id_list=<ID>"` 取回标题逐条比对；GitHub / 项目页链接经 HTTP 200 检查（2026-09-04）。

## [10. License & Credits](#content)

- 本仓库的报告、README、幻灯片与编目文字以 [CC BY 4.0](LICENSE) 发布；`scripts/` 下的脚本以 MIT 发布。
- `papers/en/` 三篇论文均由作者以 CC BY 4.0 授权发布于 arXiv（[2604.05014](https://arxiv.org/abs/2604.05014)、[2604.11757](https://arxiv.org/abs/2604.11757)、[2608.27550](https://arxiv.org/abs/2608.27550)）；`papers/zh/` 是它们的翻译衍生作品，同样遵循 CC BY 4.0 并保留原作者署名。版权归原作者所有。
- StarVLA 代码库本身以 MIT 发布于 [starVLA/starVLA](https://github.com/starVLA/starVLA)，本仓库只分析、不分发其代码。
- 报告中的所有数字均注明来源（论文表号 / 页码、代码文件与行号、官方网址）。如发现错误，请开 issue。
- 引用本仓库：

```bibtex
@misc{awesome_starvla2026,
  title  = {Awesome StarVLA Resources: Papers, Code Analysis and Roadmap for Representation-Centric VLA Continued Pre-training},
  author = {asimfish},
  year   = {2026},
  url    = {https://github.com/asimfish/awesome_starvla}
}
```
