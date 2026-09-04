# 04 · StarVLA 技术报告解读：一个乐高式 VLA 代码库的设计契约

| 项目 | 内容 |
|---|---|
| 论文 | StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing |
| arXiv | [2604.05014](https://arxiv.org/abs/2604.05014)，2026-04（持续更新的技术报告） |
| 作者 | StarVLA Community & Von Neumann Institute, HKUST |
| 代码 | https://github.com/starVLA/starVLA（MIT，~3.6k stars，源自 InternVLA-M1 fork） |
| 本仓库文件 | [英文 PDF](../papers/en/2604.05014_StarVLA_codebase.pdf) · [中文 PDF](../papers/zh/2604.05014_StarVLA_codebase_zh.pdf) |

## 0. 一句话

StarVLA 把 VLA 系统拆成"骨干 – 动作头"两个独立可换的部件，用两条契约（外层：原始观测 → 动作；内层：多模态输入 → hidden state → 动作）把 FAST / OFT / π / GR00T 四种解码范式、VLM 与视频世界模型两类骨干、SFT / 多模态共训 / 跨本体共训三种训练模式，以及 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1、BEHAVIOR-1K 等基准接进同一套训练与评测管线。它的目标不是提出新模型，而是让"只改一个变量"的受控实验成为默认工作方式。

## 1. 问题：三个层面的碎片化

- **架构层**：自回归 token、并行回归、diffusion、flow matching 各成一派，跨范式比较困难。
- **系统层**：方法发布时把模型、数据处理、训练管线耦死，部件无法复用。
- **评测层**：各自报告不同基准子集、协议不一致，公平比较不可行。

作者把根因归为"缺少统一抽象"，现有代码库（OpenPI、Isaac-GR00T、OpenVLA-OFT、Dexbotic、X-VLA）都不同时支持可换动作头、可换 VLM、世界模型骨干、混合数据加载、开源多模态/跨本体共训、多基准联训（表 1）。StarVLA 是唯一全勾选的一行，集成基准数 7。

## 2. 两条契约（第 2.2 节）

**统一 I/O 接口。** 所有 framework 继承同一基类，暴露两个方法：

- `forward({raw images, str, ...}) → loss dict`：训练入口，吃原始多视角 RGB + 指令 + 动作块。
- `predict_action({raw images, str, ...}) → {normalized_actions, ...}`：推理入口，吃同样格式（去掉 GT 动作）。

训练输入与部署观测同构，消除了 VLA 系统里常见的"dataloader 预处理张量 vs 真机原始流"的静默分布错配。作者把这条约定称为"部署时不变量"：无论骨干怎么预训练、怎么切图、用什么 tokenizer，推理时都必须吃机器人给的原始传感流。

**组合式框架。** 内部再拆成 VL 骨干（吃原始观测、按统一输出规格给 hidden state）和动作头（按统一输入规格读 hidden state、产生动作），两步组装（先加载骨干，再挂头），全部由 YAML 声明式配置。两条边界都受契约约束，骨干和头可以互不影响地独立替换。

## 3. 四种实例（第 2.3 节）

| 变体 | 做法 | 对应文献 |
|---|---|---|
| StarVLA-FAST | 接 FAST tokenizer，在 LLM 自己的词表空间里自回归生成离散动作 token | π0-FAST |
| StarVLA-OFT | 轻量 MLP 读预定义 action token 的 hidden state，并行回归连续动作，L1 loss | OpenVLA-OFT |
| StarVLA-π | 逐层 cross-DiT 流匹配动作专家，通过 cross-attention 条件于多层 VL hidden state，迭代去噪 | π0 |
| StarVLA-GR00T | 双系统：VL 骨干为 System 2（慢推理），DiT 流匹配模块为 System 1（快生成） | GR00T N1.5 |

四者共享骨干、基类和 forward/predict_action 契约，只在"如何从骨干表征里取动作"上不同。新范式只需实现并注册一个动作头。

## 4. 训练管线（第 3 节）

三种训练模式对应三个入口脚本，全部是显式 PyTorch 循环 + Accelerate + DeepSpeed：

| 模式 | 脚本 | 要点 |
|---|---|---|
| 行为克隆 SFT | `train_starvla.py` | 目标是 forward 返回的 `action_loss`；支持全参微调或 `trainer.freeze_modules` 按模块路径冻结；`trainer.learning_rate` 按模块分组 lr；bf16、梯度累积、梯度裁剪、带下限的 cosine |
| 多目标共训 | `train_starvla_cotrain.py` | 双 dataloader（VLA + VLM），每步两次前向/反向：一次 framework.forward 得 action_loss，一次 `qwen_vl_interface` 得 LM loss，后者乘 `trainer.loss_scale.vlm` |
| 跨本体共训 | 配置项 `datasets.vla_data.data_mix` | 一个命名 mixture 映射到 (数据集, 采样权重, 机器人类型) 列表，运行时物化为 `LeRobotMixtureDataset`，按权重采样并按机器人类型打本体标签。跨本体预训练是"配置选择"而不是专用脚本 |
| RL 微调 | 与 RLinf 合作 | 报告撰写时仍在集成中 |

## 5. 评测与部署（第 3.2 节）

薄 server–client 抽象：checkpoint 由 `baseframework.from_pretrained()` 加载，作为轻量 WebSocket 策略服务器运行在 StarVLA 环境；基准评测器在自己的 conda 环境（各模拟器依赖不同）里通过小客户端封装访问，payload 用 msgpack 序列化，包含 `image`、`lang`、可选 `state` 等字段，返回 `normalized_actions`。

基准差异被隔离在 `model2libero_interface.py`、`model2simpler_interface.py`、`model2robotwin_interface.py` 这类适配器里：缩放图像到训练分辨率、读 checkpoint 目录下的 `dataset_statistics.json` 反归一化、把归一化的 chunk 转成可执行动作、动作 ensembling、sticky gripper、delta/relative → absolute 转换。

真机部署复用同一契约：机器人控制器扮演客户端，采图、组同样的字典、查询远端策略服务器、执行返回动作。控制回路、安全逻辑、厂商 SDK / ROS 全在模型运行时之外。同一 checkpoint 可以不改代码从仿真搬到真机（RoboChallenge 就是这样跑的）。

## 6. 基准集成与基线结果（第 4–5 节）

每个基准的接入由三件对齐的东西组成：checkpoint 包（`config.yaml` + `dataset_statistics.json`）、可运行训练入口（`examples/<BENCH>/train_files/`）、可运行评测流程（`examples/<BENCH>/eval_files/`）。

**LIBERO（表 2）**——一个策略跑四个套件，8×A100，只训 30K 步（约 9.5 epoch）：

| 模型 | 步数 | Epochs | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 175K | 223 | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| GR00T-N1.5 | 20K | 203 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 |
| StarVLA-FAST（Qwen3-VL-4B） | 30K | 9.54 | 97.3 | 97.4 | 96.3 | 90.6 | 95.4 |
| StarVLA-OFT（Qwen3-VL-4B） | 30K | 9.54 | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| StarVLA-π（Qwen3-VL-4B） | 30K | 9.54 | 98.8 | 99.6 | 95.8 | 88.4 | 95.7 |
| StarVLA-GR00T（Qwen3-VL-4B） | 30K | 9.54 | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |
| StarVLA-OFT（Cosmos-Predict2-2B） | 30K | 9.54 | 98.6 | 97.6 | 95.0 | 91.8 | 95.8 |

两个信息：OpenVLA-OFT 用 6 倍步数、23 倍 epoch 才多 0.5 个点；把骨干从 Qwen3-VL-4B 换成视频世界模型 Cosmos-Predict2-2B，三种头平均仍 ≥95.2，"骨干可换"不是口号。

**SimplerEnv**：16×A100，Bridge + Fractal 混合训练，每个设定跑 5 次完整官方评测取均值。WidowX VM 最高 65.3%（Qwen3-VL-4B）、61.6%（Cosmos-Predict2-2B）。

## 7. 多模态共训案例（第 6 节，引用 ST4VLA）

动作-only 微调会在数千步内让 VLM"忘掉"预训练能力：RefCOCO-g IoU@0.5 在 20K 步内掉到接近随机。共训的机制是维持感知相关通路上的梯度流。ST4VLA（基于 StarVLA 的空间引导共训研究）表 8：

| 策略 | MME | RefCOCO-g IoU@0.5 | RoboRefIt Acc@0.5 | Google VM |
|---|---|---|---|---|
| Vanilla VLA | – | – | – | 66.1 |
| + 共训 | 1106 | 47.1 | 66.7 | 70.2 |
| + 空间引导 | 1374 | 68.1 | 72.5 | 78.8 |
| + 空间预训练 | 1411 | 71.2 | 74.3 | 84.6 |

共训不只是"保住多模态能力"，它同时把操作成功率从 66.1 拉到 84.6。VLAct 的 caption 混训与这条线一脉相承，只是选择了 caption 作为最强单源锚点。

## 8. 计算效率（第 8 节，数据来自 issue #158）

测量对象：StarVLA-GR00T + Qwen3-VL-4B，RoboCasa-GR1 数据，A100 80GB。

**单节点（8×A100）**：每 GPU batch 2→24，步延迟 0.703→2.404 s，样本吞吐 22.7→79.9 samples/s，GPU 利用率最高 96%。每 GPU batch 8 是延迟与利用率的平衡点。

**多节点（每 GPU batch 8）**：

| GPU 数 | 全局 batch | 秒/步 | samples/s | 扩展效率 |
|---|---|---|---|---|
| 8 | 64 | 0.735 | 87.0 | 100% |
| 32 | 256 | 0.899 | 284.7 | 81.9% |
| 64 | 512 | 0.925 | 553.8 | 79.6% |
| 256 | 2048 | 0.931 | 2200.0 | 79.1% |

跨节点通信带来一次性 ~0.2 s/步的开销，之后到 256 GPU 都平台化，扩展效率稳在 79–80%。实践指南：固定步数的训练不会因为加 GPU 变快；数据量驱动的训练可以放心扩到数百卡。结合 StarVLA-α 表 11 "batch size 是 generalist 训练最重要的优化因素"，大全局 batch 的价值在两篇论文里被同时确认。

## 9. "广义 VLA 视角"

作者从工程统一中提炼出一个观点：VLM 基方法与世界模型基方法不是两种范式，而是同一结构框架下的变体，差别主要在**辅助学习信号**的形式——语言对齐的推理，还是未来观测预测。这个观点的价值在于它给出了一个可操作的研究纲领：把"辅助信号"当作第三个可换部件，与骨干、动作头并列。VLAct 的 caption 共训、WM4A 的未来帧预测、ST4VLA 的空间 grounding，都是这个第三轴上的点。

## 10. 评价

### 优点

1. 两条契约的设计在报告里讲得很清楚，且代码确实这么做了（见 [02 · 代码库解析](02_starvla_codebase_analysis.md)）。
2. 表 2 的 LIBERO 基线把"步数 / epoch"列出来，是少数把训练成本和成功率放在同一张表里的报告。
3. 效率章节的数据来自公开 issue，可追溯。

### 局限

1. 报告以"持续更新"为前提，第 4.2 节声明的五个基准与 README 里的 13 个不一致；读者应以代码库为准。
2. 除 LIBERO 与 SimplerEnv 外，RoboTwin、GR1、BEHAVIOR 的详细结果在报告正文中较薄，多引用 StarVLA-α。
3. 世界模型骨干（Cosmos-Predict2）只在 LIBERO 上展示，WM4A 在其他基准上的表现未给出。
4. 效率测量只覆盖 GR00T 头；FAST 的自回归解码和 π 的迭代去噪在推理侧的延迟差异没有量化，而这正是部署时选头的关键依据。

## 11. 与本仓库其他材料的关系

- 代码级细节（每个契约对应哪个类、哪一行）：[02 · 代码库解析](02_starvla_codebase_analysis.md)
- 用这套基础设施做的两项研究：[03 · StarVLA-α](03_starvla_alpha.md)、[01 · VLAct](01_vlact_deep_dive.md)
- 支持的基准逐个拆解：[06 · 基准生态](06_benchmarks_landscape.md)
