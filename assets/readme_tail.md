
## [5. Cheat Sheets](#contents)

### 5.1 StarVLA Codebase

数字来自本地快照（分支 `starVLA_dev`，HEAD `d81fc66`，2026-09-04），细节与行号见 [02 · 代码库解析](reports/02_starvla_codebase_analysis.md)。

| 层 | 关键文件 | 一句话 |
|---|---|---|
| 框架 | `starVLA/model/framework/{VLM4A,WM4A,VM4A}/*.py` | 28 个注册框架；`build_framework(cfg)` 查 `FRAMEWORK_REGISTRY`；每个文件末尾带 `__main__` smoke test |
| 契约 | `starVLA/model/framework/base_framework.py` | `forward(examples) → {"action_loss"}`；`predict_action(examples) → {"normalized_actions"}`；两者都直接吃原始样本 dict |
| 动作头 | `starVLA/model/modules/action_model/` | 11 个文件：MLP（OFT，masked L1）、FAST（2048 个 action token 占用词表区间）、Layerwise-FM（PI，36 层 cross-DiT，4 步 Euler）、FM DiT-B（GR00T，16 层，状态前置） |
| VLM 接口 | `starVLA/model/modules/vlm/` | `get_vlm_model` 分派 9 个分支：Qwen2.5-VL / Qwen3-VL / Qwen3.5 / Gemma-4 / Molmo2 / MiniCPM-V / Florence-2 / Cosmos-Reason2 / VILA |
| 数据 | `starVLA/dataloader/` | GR00T LeRobot 管线移植（v2.0/v3.0，q99 / mean_std / min_max 归一化，`DATASET_NAMED_MIXTURES` 加权混合）；VLM LLaVA-json；UMI 适配 |
| 训练 | `starVLA/training/` | `train_starvla.py`（SFT）、`train_starvla_cotrain.py`（VLA + VLM 双 loader，`loss_scale.vlm`）、`train_starvlm.py`、`train_starvln.py`；`freeze_modules` 精确点路径；分模块 lr；只存 state_dict |
| 部署 | `deployment/model_server/` | WebSocket / ZMQ 策略服务器；基准侧 12 个 `model2*_interface.py` 适配器 |
| 生态 | `examples/` | 13 仿真基准 / 5 真机 / 6 模型扩展 / 1 UMI 人类数据 |

VLAct 六项配方在 StarVLA 中的状态：(b) caption 共训与 (f) 丢头重训**已有**；(a) 浅层冻结与 (d) 20 维部分统一布局**部分**（冻结需列 18 条精确路径，mask 只有 OFT 头消费）；(c) 多头共监督与 (e) wrap-aware loss **缺失**——后四项由 [`code/vlact_ext`](code/vlact_ext/) 补齐。

### 5.2 Benchmarks

面向"VLA 持续预训练 / 表示学习"研究的基准选择，依据见 [06 · 基准生态](reports/06_benchmarks_landscape.md) 第 5 章：

| 优先级 | 基准 | 考察 | 要点 |
|---|---|---|---|
| 1 | LIBERO-plus | 零样本鲁棒性（Camera / Robot / Language / Light / Background / Noise / Layout） | 训练集固定为标准 LIBERO，提升只能来自骨干；10,030 实例 |
| 2 | RoboTwin 2.0 **Base** | 少样本（50 clean / 任务）+ clean→random | clean 与 random 必须同时报；Data Scaling 只作补充曲线 |
| 3 | RoboCasa-GR1 数据比例曲线 | 未见人形本体的样本效率 | 10% / 20% / 50% / 100%，基线也要跑同样比例点 |
| 4 | VLA-Arena | L0→L1/L2 结构外推 + 安全代价 | 官方 30 episodes / 任务，StarVLA 默认 10 |
| 5 | RoboDojo | 第三方裁判 + Generalization / Precision / Long-Horizon / Memory / Open 五维 | 适合最终报告；Memory 维度几乎人人接近零 |
| 记忆 | RoboTwin-MeM / RMBench | 需记忆的关键帧数 n = 1–5 可控 / 持久布局记忆 | 协议细节（100 集、unseen 指令、LargeView、步数上限）见 [11 · 代码审计](reports/11_eventvla_code_audit.md) §4 |
| 不建议 | LIBERO 标准版、SimplerEnv、MetaWorld、CALVIN D→D、BEHAVIOR-1K | | 饱和（95–98）、方差大、或评测成本极高 |

报告协议：固定下游预算并写明 checkpoint 规则；≥ 3 seeds；区分见过 / held-out 本体（VLAct 只有 GR1 与 ARX X5 是真 held-out）；附表示层诊断（跨头迁移矩阵）。

### 5.3 Roadmap

完整版见 [07 · 研究路线图](reports/07_research_roadmap.md)，执行计划见 [10 · 改进方案](reports/10_improvement_plan.md)。六类 18 个方向：

| 类别 | 方向 | 出发点 |
|---|---|---|
| A 表征诊断 | A1 骨干可复用性探针套件；A2 自动化"哪层该冻" | decoder lock-in 只能间接观察；冻结只有 3 档消融 |
| B 预训练目标 | B1 头多样性推广（FAST / 未来帧 / 空间 QA 头）；B2 辅助数据调度；B3 潜动作 vs 多头；B4 世界模型辅助头 | 三头同头 +1.6～4.3；全混合被稀释；"广义 VLA 视角" |
| C 动作空间 | C1 可学习的部分统一布局；C2 几何一致参数化（SO(3)）；C3 零样本本体迁移 | 20 维手工表；wrap loss +5；GR1 20% 才 49.5 |
| D 能力短板 | D1 记忆与长程（EventVLA 路线）；D2 语言泛化；D3 低数据下生成式头落后 | Memory 0.66；Language −5.5；Random 下 OFT 41.5 vs GR00T 22.9 |
| E 系统 scaling | E1 规模曲线；E2 三头开销与降本；E3 数据 × 配方 Pareto | 只有 4B；开销已量化为 1.54×（§3.1） |
| F 评测方法 | F1 骨干基准协议；F2 generalist × 持续预训练；F3 真机统计功效 | "只换骨干"未标准化；两条线未合并；n = 10 |

先做（高影响、低成本）：A1 诊断套件、F1 骨干基准脚本、E2 三头开销测量（已完成第一组数字）。六个月计划以 16 GPU 为前提，每月一个交付物，最后一个月整合最优组合、全基准评测并向 StarVLA 提 PR。

## [6. Repository Layout & Build](#contents)

```
awesome_starvla/
├── README.md                       # 由 scripts/build_readme.py 生成：assets/readme_head.md + papers_curated.md + readme_tail.md
├── CONTRIBUTING.md                 # 条目格式、核验要求、报告写作规范
├── LICENSE                         # CC BY 4.0（报告、README、幻灯片）
├── papers/
│   ├── en/                         # 7 篇论文英文原版 PDF（arXiv，均 CC BY 4.0）
│   └── zh/                         # 保版式中文翻译 PDF + 翻译缓存 + QA 报告
├── reports/                        # 11 份中文深度报告（01–11）
├── report/
│   ├── awesome_starvla_slides.tex / .pdf        # 29 页 Beamer（XeLaTeX + ctex，16:9）
│   ├── awesome_starvla_slides.pptx              # 18 页原生 PPTX（ppt-master 生成；SVG 源页在 pptx_src/）
│   ├── action_heads_lecture_slides.tex / .pdf   # 20 页动作头讲解幻灯片（配 reports/08）
│   └── awesome_starvla_full_report.html / .pdf  # 62 页合订全文报告
├── code/
│   ├── vlact_ext/                  # VLAct 配方扩展包（多头框架、wrap loss、统一布局、冻结规则）+ 61 个测试
│   ├── starvla_lab/                # 改进方案研究包（probes / schedules / heads / data / train / bench / configs）+ 125 个测试
│   └── EventVLA/                   # git 子模块：EventVLA 模型 + RoboTwin-MeM 基准
├── experiments/
│   ├── README.md                   # 运行清单与结果 JSON 约定
│   ├── run_matrix*.csv             # 主矩阵（92 次）与跨头矩阵（各 8 次）
│   ├── budget.md                   # 全量 GPU 小时预算（脚本生成）
│   └── results/                    # wp6_overhead/、f0_libero_goal_smoke/：每次运行一个 JSON + README
├── assets/
│   ├── papers_curated.md           # 120 条文献编目（第 4 节的源）
│   ├── starvla_code_facts.md       # 代码库硬事实卡片（数字、路径、接口签名）
│   ├── fig1_timeline.svg / fig2_taxonomy.svg    # 图 1 / 图 2（make_figures.py 生成）
│   └── readme_head.md / readme_tail.md          # README 的非论文部分
└── scripts/
    ├── build_readme.py             # 拼装 README
    ├── build_full_report.py        # 合订全文报告（pandoc + Chrome headless）
    ├── build_slides.sh             # 编译幻灯片（XeLaTeX + Fandol 字体）
    ├── build_run_matrix.py         # 从 starvla_lab/configs 生成 experiments/run_matrix*.csv 与 budget.md
    ├── make_figures.py             # 生成图 1 / 图 2
    ├── translate_papers.sh         # super_translate 翻译流程
    ├── setup_cpu_env.sh / smoke_starvla_integration.py   # §3.3 的 CPU 集成环境与冒烟
    ├── gpu_overhead_bench.py / analyze_f0.py / probe_diagnostics.py   # §3.1 的 GPU 实验与分析
    └── cluster/                    # sync_to_node / setup_gpu_env / run_overhead_bench / run_f0_smoke
```

构建与流程：

- **README**：改 `assets/readme_head.md`、`assets/readme_tail.md` 或 `assets/papers_curated.md`，然后 `python3 scripts/build_readme.py`；不要直接编辑 README.md。
- **翻译**：[super_translate](https://github.com/asimfish/super_translate) `paper-translate` skill，DeepSeek 后端，`--preserve-graphics-text`（图表内文字与公式原样保留），译后 `inspect` 视觉 QA，报告随 PDF 存放在 `papers/zh/*.inspect.json`。已知瑕疵：VLAct 中文版 p29 公式内指示函数文字保留英文、p30 一行字号偏小；StarVLA-α 中文版 p16 附录目录 9 条保留英文（带 hyperref 引用被判为保护区）；StarVLA 报告中文版 p15 一处字号偏小；GR00T N1 中文版 p22 / p25 五处图表内保留文字的渲染墨迹密度与原文略有差异（非漏译）；EventVLA 中文版 p4 绕图排版的窄栏中文溢出到图 2 边缘。正文均已翻译。
- **合订报告与图**：`python3 scripts/make_figures.py && python3 scripts/build_full_report.py --pdf`（需要 pandoc 与 Google Chrome；公式由 MathJax 渲染，报告源文件统一用 `$...$` / `$$...$$` 定界以兼容 GitHub）。
- **幻灯片**：`bash scripts/build_slides.sh`，需要 XeLaTeX 与 Fandol 字体，遵循 [beamer-skill](https://github.com/Noi1r/beamer-skill) 规范（16:9、10pt、无 overlay、每页 ≤ 2 个彩色框）。PPTX 由 [ppt-master](https://github.com/hugohe3/ppt-master) v6.2 Quick Generate 从 Beamer 内容重排为 18 页，可用其 `svg_to_pptx.py --quick-generate --native-charts-and-tables` 重新导出；字体 Microsoft YaHei / Arial，LibreOffice 预览缺字体时会假性折行，PowerPoint 中正常。
- **写作**：报告与 README 按 [anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing) 与 [shuorenhua](https://github.com/MrGeDiao/shuorenhua) 的规则：直接陈述、不做防御性免责、不用"值得注意的是 / 综上所述"类套话、术语保留英文、数字必须有出处。
- **文献核验**：每条 arXiv 链接用 `curl "http://export.arxiv.org/api/query?id_list=<ID>"` 取回标题逐条比对；GitHub / 项目页链接经 HTTP 200 检查（2026-09-04）。

## [7. License & Citation](#contents)

- 本仓库的报告、README、幻灯片与编目文字以 [CC BY 4.0](LICENSE) 发布；`scripts/` 与 `code/vlact_ext`、`code/starvla_lab` 以 MIT 发布。
- `papers/en/` 七篇论文均由作者以 CC BY 4.0 授权发布于 arXiv（[2604.05014](https://arxiv.org/abs/2604.05014)、[2604.11757](https://arxiv.org/abs/2604.11757)、[2608.27550](https://arxiv.org/abs/2608.27550)、[2501.09747](https://arxiv.org/abs/2501.09747)、[2502.19645](https://arxiv.org/abs/2502.19645)、[2503.14734](https://arxiv.org/abs/2503.14734)、[2606.20092](https://arxiv.org/abs/2606.20092)）；`papers/zh/` 是它们的翻译衍生作品，同样遵循 CC BY 4.0 并保留原作者署名，版权归原作者所有。π0（[2410.24164](https://arxiv.org/abs/2410.24164)）为 arXiv 非独占许可，只提供链接。
- StarVLA 代码库以 MIT 发布于 [starVLA/starVLA](https://github.com/starVLA/starVLA)，本仓库只分析、不分发其代码；EventVLA（MIT）以子模块形式引用 [asimfish/EventVLA](https://github.com/asimfish/EventVLA)，仓库本身不含其代码副本。
- 报告中的所有数字均注明来源（论文表号 / 页码、代码文件与行号、官方网址）。如发现错误，请开 issue。

```bibtex
@misc{awesome_starvla2026,
  title  = {Awesome StarVLA Resources: Papers, Code Analysis and Roadmap for Representation-Centric VLA Continued Pre-training},
  author = {asimfish},
  year   = {2026},
  url    = {https://github.com/asimfish/awesome_starvla}
}
```
