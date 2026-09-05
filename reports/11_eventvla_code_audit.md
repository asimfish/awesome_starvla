# 11 · EventVLA 代码审计：论文与实现的差异、复现坑与改进落点

| 项目 | 内容 |
|---|---|
| 审计对象 | [`code/EventVLA`](../code/EventVLA/) 子模块，commit `108a0f3`（上游 [InternRobotics/EventVLA](https://github.com/InternRobotics/EventVLA) main，2026-08-30），fork 于 [asimfish/EventVLA](https://github.com/asimfish/EventVLA) |
| 规模 | `EventVLA/` 105 个 `.py`（核心 `eventvla/model/framework/EventVLA.py` 2,104 行、`dataloader/gr00t_lerobot/datasets.py` 3,345 行、评测客户端 `examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py` 1,144 行）；`RoboTwin-Mem/` 644 个 `.py`（RoboTwin 2.0 fork + 8 个记忆任务） |
| 上游状态 | 10 个 commit（2026-06-17 → 08-30）；16 个 issue，9 个 open，其中 4 个是复现失败或配置缺失 |
| 已发布 | RoboTwin-MeM 训练 / 评测代码、`steps_100000` checkpoint、LeRobot 2.1 数据（带 oracle 关键帧）；RMBench checkpoint（无对应配置，#13 / #14）；真机推理客户端。**未发布**：论文 §3.3 的 Qwen3-VL 自动标注管线（#11）、真机训练配置与权重（README To Do）、Table 2 下半部分的消融变体 |
| 配套 | [09 · EventVLA 解读](09_eventvla.md) 讲论文；本篇只讲代码。行号以子模块 `108a0f3` 为准，路径相对 `code/EventVLA/` |

## 0. 一句话

EventVLA 的代码是 StarVLA 的一个早期 fork，主干做法与论文一致：MLP-L1 动作头旁边并联一个逐 action token 的关键帧 MLP 头，raised-cosine 软标签 BCE，阈值 → 1D NMS → 冷却 → FIFO 的写入链，teacher → student 课程。但有三类事实与论文不符或论文没写，会直接影响复现和我们下一步的改动：**(1) 超参数**——N_max = 4 而非 5、λ = 1.0 而非 0.1、BCE 带 pos_weight = 7、NMS 半径 20 / 冷却 20 而非 8 / 10、课程只在第 10k–40k 步之间过渡、发布的 checkpoint 训了 100k 步而非 80k；**(2) 标签**——关键帧真值是仿真脚本专家在采集时按手写规则打的 oracle，论文所称的 Qwen3-VL 自动标注管线不在仓库里；**(3) 评测协议**——每次推理最多提交 1 个事件、只在前 3 次写入后立即 replan（注释明说"第 4 个关键帧是最后一个 inspect 锚点"）、训练与评测都是 3 视角 × 4 锚点 = 12 张图 + ≤ 4 张记忆图。第三方按作者补充的配置复现，8 个任务里 7 个与论文基本一致，Pick Objects in Order 只到 56%（论文 90%，#4 open）。

## 1. 仓库结构：StarVLA 的 fork 加两块新东西

```
code/EventVLA/
├── EventVLA/                          # 模型与训练，目录与 StarVLA 同构（starVLA/ → eventvla/）
│   ├── eventvla/model/framework/
│   │   ├── EventVLA.py                # 本体：记忆输入拼接、关键帧头、事件选择、runtime 记忆库（2,104 行）
│   │   ├── pi05_keyframe_mixin.py     # KEM 抽成 mixin，接 π0.5（PaliGemma）做真机
│   │   ├── Pi05MEM.py                 # πMEM 基线复现（656 行）
│   │   └── QwenPI.py / QwenGR00T.py / QwenFast.py / LangForce.py / M1.py …   # StarVLA 原文件，未改
│   ├── eventvla/model/memory_ablation.py          # 只剩一种模式 pure_image_keyframe_memory（L6–13）
│   ├── eventvla/dataloader/sequence_sampler.py    # 按 episode 连续采样、保持 batch slot（519 行）
│   ├── eventvla/dataloader/gr00t_lerobot/datasets.py   # 软标签、teacher 记忆图、exact fetch（关键帧相关约 350 行）
│   ├── eventvla/training/train_eventvla.py        # 有状态训练循环（1,027 行）
│   ├── examples/RoboTwin-Mem/{train_files,eval_files,eval_batch}/
│   └── tests/                         # 2 个文件：采样器 slot、runtime 记忆
└── RoboTwin-Mem/                      # RoboTwin 2.0 fork
    ├── envs/{cover_blocks_hard,pick_the_unhidden_block,…}.py   # 8 个任务，含 oracle 关键帧打点
    ├── policy/{ACT,DP,DP3,RDT,pi0,pi05,openvla-oft,X-VLA,Mem-0,…}/   # 基线接入
    └── script/eval_policy.py          # 每任务 100 集，seed 起点 100000×(1+seed)（L167–169）
```

与 [02 · StarVLA 代码库解析](02_starvla_codebase_analysis.md) 的对应关系：`eventvla/model/framework/base_framework.py` 的两条契约（`forward → {"action_loss"}`、`predict_action → {"normalized_actions"}`）原样保留，EventVLA 只是在返回 dict 里多塞了 `total_loss`、`chunk_keyframe_prob`、`pred_event_offset`、`should_trigger_event` 等键（`EventVLA.py` L1253–1292、L1479–1493）。checkpoint 是 StarVLA 格式，但发布的 `config.yaml` 写着 `framework.name: QwenOFT`，当前 `FRAMEWORK_REGISTRY` 里没有这个名字，加载前要改成 `EventVLA`（#3 / #10，作者确认权重逐键匹配）。

## 2. 模型：论文公式在代码里的落点

### 2.1 输入序列

一次前向送进 Qwen3-VL 的内容（`QWen3.py` L64–78；`EventVLA.py` L1101–1106、L1502–1507）：

```
Temporal observation images: [o_0 ×3 视角] [o_{t-30} ×3] [o_{t-15} ×3] [o_t ×3]
Past keyframe images:        [k_1] … [k_≤4]                      （只有 cam_high）
<instruction> Please predict the next 50 robot actions and estimate which future
observation positions inside this chunk are key events: <action>🔍×50<action>.
```

- **顺序**：论文 Eq. (4) 写 `concatenate([A_t, E_{t-1}, o_t])`，按时间排成一列；代码是锚点（含当前帧）在前、记忆帧在后（`keyframe_image_position: after_anchor_images_before_action`，`eventvla_robotwin_mem.yaml` L53；`EventVLA.py` L1105–1106），也没有逐帧时间标注，只有两个组标签 `"Temporal observation images:"` / `"Past keyframe images:"`（`QWen3.py` L74）。模型只能靠位置推断每张图是什么时候的。
- **视角**：训练数据有 3 路视频（`train_files/modality.json`：`cam_high`、`cam_left_wrist`、`cam_right_wrist`），4 个锚点各带 3 视角（`_fallback_image_metas_for_sample` 按 4 帧 × 视角数展开，`EventVLA.py` L983–1015；评测端 `deploy_policy.yml` 的 `temporal_view_names` 也是 3 个）。记忆帧只取头部相机（yaml L128–131 `include_names: [cam_high, head, main]`、`exclude_name_patterns: [wrist]`、`strict_single_view: true`；评测端 L482–508）。所以每次前向最多 12 + 4 = 16 张 224×224 图，这是论文 Table 9 吞吐从 2.91 Hz 掉到 0.94 Hz 的直接原因，论文正文没写视角数。
- **动作 token**：50 个 `🔍`（`EventVLA.py` L59、L1503），与 StarVLA QwenOFT 相同；prompt 后缀比 StarVLA 多了一句"estimate which future observation positions … are key events"。

### 2.2 动作头与关键帧头

- 动作头：`from …MLP_ActionHeader import get_action_model`（L28）→ `L1RegressionActionHead`（`MLP_ActionHeader.py` L57–84：LayerNorm → Linear(2048, 4096) → 2 个 ResNet block → Linear(·, 14)），L1 回归（L189、L1248）。这就是论文说的 OFT 头。yaml 里 `action_model_type: DiT-B` 和随后 20 行 diffusion 参数（L21–46）在 `get_action_model` 中被忽略（`MLP_ActionHeader.py` L99–111 只读 `action_dim` 与窗口长度），是 StarVLA 模板残留，不影响运行。
- 关键帧头：`LayerNorm(2048) → Linear(2048, 2048) → GELU → Linear(2048, 1)`（L300–305），作用在 50 个 action token 的最后一层 hidden 上，得到 `[B, 50]` 的 logits（L1156），sigmoid 后即论文 Eq. (3) 的 $\hat{\mathbf p}_t$。与论文一致。

### 2.3 标签与损失

- 软标签（`datasets.py` L2376–2393）：对 chunk 内每个位置取到**最近一个关键帧**的距离 $d$，$d \le R = 8$ 时 $y = 0.5(1 + \cos(\pi d / 8))$，否则 0。与论文 A.1 一致；注意距离是对该轨迹**全部**关键帧算的，chunk 起点前 8 步内的关键帧会通过头部渗入当前 chunk 的标签。
- teacher 事件（L2400–2403）：对 offset ≥ 1 的软标签取 argmax，值 ≥ 0.55 才算 `teacher_should_commit`。
- 损失（`EventVLA.py` L1544–1553）：`F.binary_cross_entropy_with_logits(logits, y, pos_weight=7.0)`，逐元素平均。论文 A.1 的 Eq. (6) 是不带权的 BCE，`pos_weight = 7.0` 论文没写（yaml L58；代码默认值 L193）。
- 总损失（L1249–1251）：`L1 + keyframe_loss_weight × L_kem`，`keyframe_loss_weight = 1.0`（yaml L56；默认值 L192）。论文 Table 6 / 7 是 λ = 0.1。

### 2.4 事件选择

- **训练**：`_select_chunk_event` 对 offset ≥ `event_future_min_offset` = 1 的概率取 argmax，≥ `event_commit_threshold` = 0.55 才 commit（L1589–1602）。每个样本每个 chunk **最多登记 1 个**待写入事件（L908–966）。
- **推理**：`_select_chunk_event_candidates` 先按阈值筛，再按置信度降序做贪心 1D NMS，半径 `keyframe_nms_window` = 20（yaml L64；L1624–1650）；`_select_inference_chunk_event` 再对候选逐个检查与 `last_committed_step`、`pending_step` 的冷却（`keyframe_cooldown_steps` = 20，yaml L65），接受第一个就 `break`（L1725–1774）。所以论文 A.2 里的集合 $\mathcal K_t$ 在代码里**每次推理最多产出 1 个事件**；评测客户端也只保留一个 pending，新响应直接覆盖（`model2robotwin_mem_interface.py` L644–666）。一个 50 步 chunk 里若有两个真事件（Press Button Keyframe 连按两次），只能靠 commit 后立即 replan 补救（§4）。
- 论文 Table 6 / 7 写 NMS 半径 $w = 8$、冷却 $C = 10$；代码两处都是 20。

### 2.5 记忆 buffer

服务端 `_append_runtime_keyframe_entry`：按 step 去重、排序，只留最后 `max_keyframe_images` 张（L624–640）；评测客户端同样（L541–546）。`max_keyframe_images` = 4（yaml L54、L124；上游 README L282 `max_keyframes: 4`）。论文 Table 6 / 7 写 $N_{\max} = 5$。

### 2.6 训练时的 student 记忆：一个有状态的训练循环

论文 A.1 用一句"scheduled teacher-to-student curriculum"带过，代码里这是最复杂、最脆弱的部分：

1. **采样器**（`sequence_sampler.py`）：每条轨迹展开成锚点流 `[0, r, r+50, r+100, …]`，`r ∈ [1, 50]` 由 `(seed, epoch, dataset_index, trajectory_id)` 哈希决定（L95–102、L117–124）；`sampling_interval: 50`（yaml L111）。`preserve_episode_batch_slots: true`（yaml L107）让同一 episode 始终占同一个 batch slot，因为模型的 runtime 记忆按 slot 索引（`EventVLA.py` L295–298）。上游 PR #8（2026-07-24）修的就是 slot 错位（#7）。
2. **登记与取回**：`forward` 里模型按自己的预测登记 pending 写入 `{slot, trajectory_id, target_step = t + offset}`（L908–966）。下一个 batch 到来时，trainer 调 `collect_due_predict_exact_fetch_requests` 找出 `target_step ≤ 当前 step` 的请求（L877–906），在**主进程同步解码视频帧**（`train_eventvla.py` L736–756，`dataset.get_memory_image_at_step`），塞回 `examples[0]["runtime_memory_exact_fetches"]`，模型 `_consume_runtime_memory_exact_fetches` 入库（L671–734）。新 episode 进入 slot 时 `reset_memory_by_mask` 清空（`train_eventvla.py` L779–815）。
3. **teacher / student 混合**：每个样本按 `teacher_prob` 独立二选一（`keyframe_schedule_mix_granularity: sample`）：teacher = 数据集给的 GT 关键帧图（≤ 当前步、最近 4 张，`datasets.py` L2560–2621）；student = runtime 库（`EventVLA.py` L814–875）。`teacher_prob` 调度：前 10k 步 1.0，10k → 40k 线性降到 0，之后 40k 步纯 student（yaml L74–77；`EventVLA.py` L1802–1821；未配置时默认 12.5% / 37.5%，L1797 / L1799）。论文写"α 在整个训练过程中从 1 线性衰减到 0"。

后果：(a) 一个 GPU 的 8 个 slot 各跑一条 episode 的锚点流，step 级别不能 shuffle；(b) exact fetch 在主进程做，是吞吐瓶颈；(c) 每个 chunk 只能写 1 帧；(d) 一条 1,544 步的 episode（Cover Blocks Hard，论文 Table 4）只产生约 33 个训练样本，50 集 × 8 任务 ≈ 1.3 万样本，`max_train_steps: 80000` × 8（yaml L133、L147）≈ 50 个 epoch。

## 3. 关键帧标签：脚本 oracle，不是 VLM

- 数据侧 `datasets.py` L851–858 从 LeRobot episode 元数据读 `keyframe_steps`；`RoboTwin-Mem/policy/GO1/scripts/process_data.py` L10 / L63 / L102 把采集时 `episode_info.json` 里的 `keyframe_steps` 映射到处理后的帧号。
- 采集侧每个任务的脚本专家在 `play_once` 里显式打点。`envs/cover_blocks_hard.py` L110–138：在一次 inspect 段内逐帧追踪 cover 与 block 的 xy 距离，取距离最大的那一帧（盖子掀到最开的瞬间）；`press_button_keyframe.py` L88–94、`reproduce_route.py` L121–127、`rearrange_blocks_hard.py` L105–110：在脚本阶段切换点调 `_append_keyframe_step`。环境还暴露 `get_keyframe_oracle_info`（`cover_blocks_hard.py` L140）供评测期 oracle 使用。
- 论文 §3.3 与 A.3 说训练标签来自 Qwen3-VL-235B 的离线标注，并称仿真里 VLM 标签与引擎 GT 的平均误差 < 10 步。仓库里没有这条管线，issue #11 问了没有回复；HF 数据集 `ganlinyang/RoboTwin-MeM` 里带的就是脚本 oracle。论文"自动标注可扩展"的主张在开源版本中无法验证。对我们来说这反而是干净的上界：先用 oracle 复现，再单独研究标签噪声。

## 4. 评测协议里论文没写的部分

| 项 | 代码 | 出处 |
|---|---|---|
| 每任务 episode 数 | 100，seed 起点 `100000 × (1 + seed)` | `RoboTwin-Mem/script/eval_policy.py` L167–169 |
| 指令 | `instruction_type: unseen` | `deploy_policy.yml`；#3 作者回复 |
| 相机 | `camera=LargeView`（fovy 50） | #3 作者回复；第三方从 HDF5 反推内参一致 |
| 步数上限 | `task_config/_eval_step_limit.yml` 未随代码发布；作者给出 put_back_block_hard 1800、rearrange_blocks_hard 1200，pick_objects_in_order 用 1500 | #3、#4、#9、#10 |
| 归一化键 | `deploy_policy.yml` 写 `unnorm_key: robotwin_mem`，发布的 `dataset_statistics.json` 只有 `new_embodiment` | #10 |
| checkpoint 配置 | `framework.name: QwenOFT`，需改 `EventVLA` | #3、#10 |
| 首个 chunk | 第一次规划后在 `[1, 50]` 内随机一步强制 replan（seed 42），与训练锚点流的随机 `r` 对齐 | 客户端 L587–608；采样器 L117–124 |
| commit 后 replan | 写入第 1–3 个关键帧后立即 replan，第 4 个不 replan | 客户端 L710–714 |
| pending | 只保留一个，新响应覆盖旧的 | 客户端 L644–666 |
| 落盘 | 每次 commit 把原图存到 eval 目录 | 客户端 L704–709、L931 |
| 动作重排 | `[0..5, 12, 6..11, 13]`（数据集 14 维顺序 → RoboTwin 顺序） | 客户端 L383 |

"只在前 3 次 commit 后 replan"的注释原文：*"The fourth keyframe is the last inspect anchor, so let the current chunk finish and replan at the next natural chunk boundary instead of interrupting mid-retreat."*（L710–713）。这是把 Cover Blocks Hard（4 个盖子）的任务结构写进了通用评测客户端；对 n 可达 5 的 Press Button Keyframe，第 4 次之后的事件都要等 chunk 结束才进上下文。所有 baseline 都没有"commit 即 replan"的机制，比较时它是一个必须报告的变量。

第三方复现现状（上游 issues）：#3 修好资产索引冲突（RoboTwin 2.0 与 RoboTwin-MeM 物体编号重叠）、`assets/embodiments/aloha-agilex/curobo_*.yml` 里的绝对路径、framework 名、websocket `ping_interval`、`ACTION_MODE="'abs'"` 的引号之后，press_button_keyframe 58/97 ≈ 60%（论文 48%），put_back_block_hard 10/10；#4 pick_objects_in_order 28/50 = 56%（论文 90%），作者本地 5/9，open；#9 π0.5 基线配方未公开，第三方用 LoRA 复现 rearrange_blocks_hard 得 0/30（论文 20%）。

## 5. 论文 vs 代码对照表

| 项 | 论文 | 代码 | 出处 |
|---|---|---|---|
| 动作头 | OFT | `L1RegressionActionHead`（MLP + L1）；yaml 的 DiT-B 段无效 | `EventVLA.py` L28；`MLP_ActionHeader.py` L90–113 |
| $N_{\max}$ | 5（Table 6 / 7） | 4 | yaml L54、L124；README L282 |
| λ | 0.1（Table 6 / 7） | 1.0 | yaml L56 |
| BCE | 无权（Eq. 6） | `pos_weight = 7.0` | yaml L58；L1544–1553 |
| NMS 半径 $w$ / 冷却 $C$ | 8 / 10 | 20 / 20 | yaml L64–65 |
| $\tau_{\text{commit}}$ | 0.55 | 0.55 | yaml L62 |
| 软标签 | raised cosine，R = 8 | 一致 | yaml L112–113；`datasets.py` L2386–2391 |
| 课程 | α 全程线性 1 → 0 | 前 10k 恒 1，10k–40k 线性到 0，后 40k 恒 0；按样本抽签 | yaml L74–78 |
| 训练步数 / batch | 80k / 每卡 4 | yaml 80k / 每卡 8；发布 checkpoint 为 `steps_100000` | yaml L133、L147；README L48；#3 |
| 关键帧标签 | Qwen3-VL-235B 自动标注 | 仿真脚本 oracle；标注管线未发布 | §3；#11 |
| 每 chunk 事件数 | 集合 $\mathcal K_t$ | 训练 1、推理 1、客户端 1 个 pending | L1598–1601、L1774；客户端 L644–666 |
| 图像顺序 | `[A_t, E_{t-1}, o_t]` 按时间 | `[o_0, o_{-30}, o_{-15}, o_t]` 再接记忆帧；只有组标签 | yaml L53；L1105–1106；`QWen3.py` L74 |
| 视角 | 未写 | 锚点 3 视角（12 张）+ 记忆帧单视角 | `modality.json`；yaml L128–131 |
| replan | 未写 | 首 chunk 随机 replan；前 3 次 commit 后立即 replan | 客户端 L587–608、L710–714 |
| VLM 共训 | 未写 | yaml 有 `vlm_data` 与 `loss_scale.vlm: 0.1`，但训练循环把 `vlm_iter` 注释掉了，不生效 | yaml L84–95、L162–164；`train_eventvla.py` L425 |
| 评测 | 100 集 | 100 集、unseen 指令、LargeView；步数上限文件缺失 | `eval_policy.py` L169；#9 / #10 |
| 真机 | π0.5 骨干 | `pi05_keyframe_mixin.py` 存在；训练配置与权重未发布 | README To Do |

## 6. 代码质量观察

- **死配置**：DiT-B 段（yaml L21–46）、`vlm_data` 段与 `loss_scale.vlm`（yaml L84–95、L162–164）都不生效；`use_keyframe_predict_head: auto` 会在没有标注的数据上自动关掉关键帧头（L501–515），这是给 RMBench 这类无关键帧数据准备的。
- **兼容层堆叠**：同一参数有三个别名（`keyframe_inference_nms_window` / `keyframe_nms_window` / `keyframe_cluster_timestep_window`，L213–235），`_cfg_value` 同时支持 dict 与属性两种取法（L70–82），说明配置 schema 改过多轮；写新配置时以 yaml 里出现的键名为准。
- **测试**：2 个文件，只覆盖采样器 slot 与 runtime 记忆；没有 CI；没有对事件选择（NMS / 冷却）与软标签的单测。
- **服务端状态**：cooldown 与 pending 存在 slot 0 上（L1662–1672），一个 policy server 同时服务两个任务会串状态，作者在 #3 确认要"一任务一服务器"。
- **硬编码**：`< 4` 的 replan 规则（客户端 L714）、动作维度重排（L383）、curobo yml 绝对路径（#3）、发布 checkpoint 的 `QwenOFT` 名（#10）。

## 7. 改进落点

按优先级排，每条给代码位置。与 [10 · 改进方案](10_improvement_plan.md) 的 WP3（自实现关键帧头）和 WP7 / R9（EventVLA 换 VLAct 骨干）对接。

**P0 · 先把基线复现出来（需 GPU，1–2 周）**

1. 补齐评测配置：`RoboTwin-Mem/task_config/`（含 `_eval_step_limit.yml`，按 #3 / #4 的数值）、`unnorm_key` 别名、`QwenOFT → EventVLA` 的注册别名；打包成 PR 回上游。8 任务 × 100 集跑通，重点解释 Pick Objects in Order 56 vs 90。
2. 把客户端 `replan_after_keyframe_commit and committed_keyframe_count < 4`（L714）改成配置项，跑 {不 replan、全部 replan、< 4} 三组，量化这个规则的贡献；R9 的所有对比都应报告用的是哪一组。

**P1 · 协议与训练管线（对接 WP3 / WP7）**

3. 多事件 / chunk：去掉服务端 `break`（L1774）改为返回候选列表；客户端 pending 改成按 step 排序的队列（L644–714）；训练侧 `_select_chunk_event` 支持 top-k 登记（L1589–1602、L908–966）。预期受益任务：Press Button Keyframe（论文 48%）、Find Seal and Seal Stamp（63%）。
4. 显式时间编码：在 `_build_image_content`（`QWen3.py` L64–78）里给每张图加相对时间文本（"30 steps ago" / "now" / "memory from step 412"）或 learned time token，替代现在的两个组标签；顺序与计数类任务受益，代价是几十个 token。
5. exact fetch 异步化：`get_memory_image_at_step` 在主进程逐帧解码视频（`train_eventvla.py` L743–756）；改为每条 episode 预解码候选帧到共享内存，或让 dataloader worker 按 pending 列表预取。先用 `data_time` / `model_time`（`train_eventvla.py` L487–488）量化瓶颈。
6. 标签噪声研究：按论文 A.3 的 prompt 实现 Qwen3-VL 标注（prompt 全文在论文 A.3），与 oracle 标签对比训练；再试自监督替代（未来帧不可预测性、关键帧头对 GT 的 attention rollout）。这是把 EventVLA 从"benchmark 自带标签"推向开放任务的必经之路。

**P2 · 算法**

7. 淘汰策略：FIFO（L638–639）→ 按置信度 / 任务阶段 / 被 attention 读取的频次淘汰；或分层记忆：raw 图短期 + 压缩 token 长期，解决 $N_{\max}$ 饱和（论文 §6 自述局限）。
8. 计数与顺序的显式化：让 VLM 生成一行文本摘要（"已按左键 2 次"）写回 prompt，与原图记忆并存；针对 Press Button Keyframe 的 48%。
9. 解耦 chunk 消融：论文 Table 2 把 chunk 从 50 缩到 15 时同时改了动作块和前瞻窗口。做两组：固定动作块 50、前瞻窗口 15 / 30（关键帧头只在前 k 个 token 上算 loss）；固定前瞻 50、执行 15 / 30 步后 replan。
10. 跨骨干：`pi05_keyframe_mixin.py` 已把 KEM 抽成 mixin；移植到 StarVLA 主干的 `QwenPI` / `QwenGR00T`，验证 flow-matching 头上的 KEM（[09](09_eventvla.md) §5 局限 3）。WP3 的 `starvla_lab.heads.KeyframeHead` 应直接复用 §2.3–2.5 的数值（R = 8、τ = 0.55、pos_weight、NMS / 冷却），并把 §2.6 的有状态训练作为设计决策点：要么照搬 slot 机制，要么改成"离线预生成 student 记忆"的两阶段训练。

## 8. 与仓库其他材料的关系

- [09 · EventVLA 解读](09_eventvla.md) 讲论文，本篇讲代码。09 §2.3"关键帧真值来自 Qwen3-VL 自动标注"与 §2.2"按时间顺序排成一个图像序列"描述的是论文，开源实现里分别是脚本 oracle（§3）和"锚点在前、记忆在后"（§2.1）。
- [02 · StarVLA 代码库解析](02_starvla_codebase_analysis.md)：EventVLA 是 StarVLA 的早期 fork，`eventvla/` 与 `starVLA/` 同构，02 的目录、契约与训练循环描述可直接套用；差异只在 §1 列出的新增文件。
- [07 · 路线图](07_research_roadmap.md) D1、[10 · 改进方案](10_improvement_plan.md) WP3 / WP7：§7 的 P0–P1 是 R9 之前必须做的前置工作。
- [06 · 基准生态](06_benchmarks_landscape.md) §5.4：RoboTwin-MeM 的评测协议细节（100 集、unseen、LargeView、步数上限）以本篇 §4 为准。
