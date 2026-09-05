# F0 · 真实数据微调冒烟：LIBERO-goal，QwenOFT vs QwenMultiHead（1 卡，各 300 步；v2 诊断 + v3 修正后的探针）

**三句话结论。** (1) `starvla_lab.train.train_starvla_lab` + `vlact_ext.QwenMultiHead` 在真实 LeRobot 数据上端到端跑通，三头模型每步 2.70 s / 47 GB，单头 OFT 1.86 s / 30 GB（1.45×）。(2) 同一数据、同一超参、头学习率对齐到 1e-4 后，**三头模型里 OFT 头最后 50 步的 L1 损失 0.243，与单头 OFT 的 0.244 相同**——300 步内多加 GR00T 与 PI 两个 flow-matching 头对 OFT 头没有任何损害。(3) 训练中记录的"逐层表征漂移"曲线**不能直接用**：诊断表明它几乎全部来自 `embed_tokens` 里 42–46 行 prompt 词嵌入的更新（相对 Frobenius 变化仅 ~2e-5）被一个近退化的探针批放大；把预训练嵌入换回去后，OFT 微调的骨干几乎没动（可训练层 1−CKA = 0.0002），而三头模型的上层动了约 20 倍（0.0038，第 35 层 0.019）。这直接改写了方案 WP1 / M1 对漂移度量的定义。**v3（§3.5）把修正后的探针接进训练循环重跑**：冻结层漂移精确为 0，单头 OFT 300 步只在第 35 层动了 2e-4，三头模型第 35 层动到 0.05（约 240×）、第 34 层 8e-4、第 33 层以下 < 1e-4，曲线单调、约 225 次更新后饱和；冻结 `embed_tokens` 的消融损失不变（0.247 vs 0.251）、漂移读数与"探针时换回嵌入"一致。

## 1. 设置

| 项 | 值 |
|---|---|
| 数据 | `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`（LeRobot v2.1，428 段轨迹，52,042 帧，每样本主视角 + 腕部两张 224×224 图，动作块 8×7 delta EE，`robot_tag: franka`，无 state 字段），经 hf-mirror 下载 353 MB |
| 硬件 / 软件 | 1 × A100-80GB（`tianyiyun-30110-pub2` GPU 0，与一个 1 GB 小进程共卡）；torch 2.6.0+cu124，transformers 4.57.0，StarVLA `starVLA_dev@d81fc66`；单进程、`STARVLA_DISABLE_DEEPSPEED=1`，无 DeepSpeed |
| 配置 | [`code/starvla_lab/configs/f0_libero_goal_smoke.yaml`](../../../code/starvla_lab/configs/f0_libero_goal_smoke.yaml)：300 步，batch 8，无梯度累积，warmup 30，cosine 到 1e-6；lr 骨干 1e-5 / 头 1e-4；VLAct 式冻结 = 视觉编码器 + LLM 前 18 层（19 条精确路径，**`embed_tokens` 不在冻结集合内**）；梯度检查点开；探针 `calibrate_only`，v2 每 25 步、32 样本（训练 loader 单场景批，mean-pool）；v3 每 25 次更新、32 样本跨场景批（LIBERO-goal + LIBERO-spatial 按指令轮询，20 条指令），token 级 CKA 为主、mean-pool 为次，提取时换回预训练 `embed_tokens` |
| 运行 | A `QwenOFT`（StarVLA 原生，可训练 2.27B）；B `QwenMultiHead`（`vlact_ext`，OFT + GR00T + PI，`state_dim 0`，`--trainer.learning_rate.heads 1e-4 --trainer.learning_rate.project_layers 1e-4`，可训练 3.06B）；v3 另加 C `QwenOFT` + 冻结 `embed_tokens`（20 条冻结路径，可训练 1.88B） |
| 启动 | `scripts/cluster/run_f0_smoke.sh <framework> <run_id> [覆盖项]`；分析 `scripts/analyze_f0.py`；探针诊断 `scripts/probe_diagnostics.py` |

v1（`v1/`、`v1_raw/`）是第一遍：B 的头学习率误落在 `base` 组（2.5e-5，因为 StarVLA 的 `learning_rate` 键是模块路径，`heads.*` 不匹配 `action_model`），探针 prompt 两条不一致、池化在 bf16 上做、每 50 步 16 样本。v2 修正了这三点，§2–§3 的数字来自 v2；§3.5 是 v3（探针口径按 §3 的诊断重定义后重跑，训练配置与 v2 相同）；v1 保留供对照。

## 2. 训练：单头 vs 三头

| 运行 | 总损失 1–50 步 | 总损失最后 50 步 | 逐头损失（最后 50 步） | 训练内 MSE（末） | s/step | 峰值显存 |
|---|---:|---:|---|---:|---:|---:|
| A `QwenOFT` | 0.628 | 0.244 | oft 0.244 | 0.0168 | 1.86 | 29.8 GB |
| B `QwenMultiHead` | 2.493（三头之和） | 0.998 | **oft 0.243**，pi 0.323，gr00t 0.432 | 0.0169 | 2.70 | 47.1 GB |

- OFT 头的 L1 在两种设置下走势一致：最后 50 步 0.243 vs 0.244，逐点差的均值 −0.02（三头版略低，前 50 步 0.53 vs 0.63，差异来自头的随机初始化与 300 步的噪声范围），训练内 MSE（用 OFT 头 `predict_action`）也一致（`v2/f0_curves.png`）：**头多样性至少不以牺牲原头为代价**，这是 VLAct 配方 (c) 的前提。
- 三头每步 1.45×（WP6 合成数据基准上是 1.54×；这里两张图让骨干占比更大）；显存 47 GB 含 3.06B 参数的 AdamW 状态。
- PI 与 GR00T 的 flow-matching 速度损失量纲与 L1 不同，不能与 oft 直接比；两者 300 步内都单调下降。
- 300 步只是冒烟；没有做 LIBERO 仿真评测（节点无 LIBERO 环境），所有比较都是训练侧的。

## 3. 探针：训练中的"漂移"曲线量的是什么

训练中每 25 步对固定的 32 样本探针批取 36 层 mean-pool 隐状态，与训练前的参考做线性 CKA，漂移 = 1 − CKA（`v2/*_drift.csv`）。曲线锯齿状且**最大漂移几乎总在第 16/17 层——冻结块的最后两层**：

| 步 | A 冻结 0–17 层均值 | A 可训练 18–35 层均值 | B 冻结 | B 可训练 |
|---:|---:|---:|---:|---:|
| 25 | 0.0038 | 0.0228 | 0.0087 | 0.0587 |
| 50 | 0.0057 | 0.0353 | 0.0020 | 0.0134 |
| 125 | 0.0299 | 0.1824 | 0.0060 | 0.0407 |
| 200 | 0.0008 | 0.0050 | 0.0015 | 0.0099 |
| 250 | 0.0030 | 0.0183 | 0.0183 | 0.1237 |

冻结层的输出只可能经 `embed_tokens`（未冻结）改变，而冻结块均值与可训练块均值始终保持约 1:6 的固定比例——同一个扰动在整个网络里传播。`scripts/probe_diagnostics.py` 在两个最终模型上做了受控实验（同一 32 样本探针批：10 条不同指令、4096 个图像 token、608 个文本 token）：

| 测量（1 − CKA，均值 / 冻结 0–17 / 可训练 18–35 / 最大） | A `QwenOFT` | B `QwenMultiHead` |
|---|---|---|
| 噪声底：同一模型提取两次 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| `embed_tokens` 改变的行数；相对 Frobenius 变化 | 42 行；1.7e-5 | 46 行；2.1e-5 |
| 微调后，全 token 池化 | 0.0080 / 0.0022 / 0.0138 / 0.020 @L16 | 0.0097 / 0.0022 / 0.0171 / 0.020 @L17 |
| 微调后，只池化图像 token | 0.0082 / 0.0022 / 0.0142 / 0.020 @L16 | 0.0101 / 0.0022 / 0.0180 / 0.020 @L17 |
| 微调后，只池化文本 token | 0.0000 / 0.0000 / 0.0001 / 0.0006 | 0.0001 / 0.0000 / 0.0002 / 0.002 |
| 微调后，token 级 CKA（4096 个 token 为样本） | 0.0006 / 0.0001 / 0.0011 / 0.0012 | 0.0017 / 0.0001 / 0.0034 / **0.045 @L35** |
| **换回预训练 `embed_tokens`**，全 token 池化 | **0.0001** / 0.0000 / 0.0002 / 0.0008 | **0.0019** / 0.0000 / 0.0038 / **0.019 @L35** |
| 换回预训练 `embed_tokens`，token 级 | 0.0000 / 0.0000 / 0.0000 / 0.0003 | 0.0013 / 0.0000 / 0.0026 / 0.045 @L35 |

读法：

1. **测量本身是确定性的**（噪声底为 0），训练中的锯齿是嵌入每步更新的真实反映，不是数值噪声。
2. **几十行嵌入、2e-5 的相对变化，贡献了池化漂移的 98%**（0.0080 → 0.0001）。机制：残差流把嵌入直接带进每一层输出；探针批来自同一个 LIBERO 场景，图像近乎相同，样本间差异集中在 10 条指令的少数词上，中心化 CKA 对这几个词的嵌入更新极度敏感。冻结层"漂移"完全是这个效应。
3. **控制嵌入后，OFT 微调 300 步几乎没有动骨干**（可训练层 0.0002），符合"低 lr、短程、任务简单时 VLM 先验基本保留"的直觉；**三头模型的上层动了约 20 倍**（0.0038；第 30–34 层 0.007–0.010，第 35 层 0.019；token 级第 35 层 0.045）。PI 头对全部 36 层做 cross-attention、GR00T 头读末层，把梯度更强地推进了骨干顶部。这是 Q1（骨干可复用性能否不跑下游就度量）与 Q3（先验保护）的第一个真实数据点——但漂移不等于侵蚀，它也可能是表征被"写入"了动作信息；判别要靠 WP1 的跨头探针 + 下游迁移，而不是 CKA 本身。
4. token 级 CKA（几千个 token 为样本）比 mean-pool（32 个样本）对嵌入效应不敏感一个数量级，且在 B 的第 35 层给出更尖锐的信号；文本 token 池化几乎不动（文本 token 靠因果注意力看不到后面的内容，且其嵌入变化本身极小）。

**对方案的修改**（已写回 [10 · 改进方案](../../../reports/10_improvement_plan.md) §6 / §9）：WP1 的漂移度量改为 (a) 探针提取时把 `embed_tokens` 临时换回预训练权重（或在 VLAct 冻结集合里显式加入 `embed_tokens`——论文对此未说明，需作为消融）；(b) 主指标改用 token 级 CKA，mean-pool 只作辅助；(c) 探针批必须跨场景 / 跨任务（例如 LIBERO 四套 + RoboTwin 混合，≥ 64 样本），当前单场景批的 Gram 矩阵近退化；(d) M1 的阈值标定（`drift_high` / `drift_low`）用修正后的度量重做，v2 训练中记录的曲线作废。

## 3.5 v3：修正后的探针接进训练循环（三条运行，各 300 步）

按 §3 的结论改了 `starvla_lab` 的探针实现（`probes/qwen_extract.py::QwenBackboneProbe`，配置见 `trainer.lab.probes.*`）：(a) 每次提取前把预训练 `embed_tokens` 换回（快照在 CPU，~780 MB，提取后再换回训练中的权重）；(b) 主指标 token 级 CKA——32 个样本共 4,834 个有效 token（4,096 图像 + 738 文本），在首次提取时固定 4,096 个位置，之后每次都用同一组位置；mean-pool 作次指标写进 `drift_secondary`；(c) 探针批来自单独的 loader：LIBERO-goal + LIBERO-spatial 两套按指令轮询抽样，32 个样本覆盖全部 20 条指令（每条 ≤ 2 个），训练数据仍只有 LIBERO-goal（`probes.probe_data_mix`，运行时向 StarVLA 的混合注册表注入一个临时条目）；(d) 记录以"已完成的更新次数"为步号，第 0 步是参考对自身；(e) CKA 的 Gram 乘积在 GPU 上算（fp64，36 层 × [4096, 2560]）。每次探针约 12–13 s（两次前向 + 两次嵌入换入换出 + CKA），300 步多花约 3 分钟。

| 运行 | 总损失最后 50 步 | OFT 头损失 | s/step（中位数） | 可训练层 18–35 均值 @300 | 第 34 层 @300 | **第 35 层 @300** | mean-pool 均值 @300 | `embed_tokens` 变化行数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A `QwenOFT` | 0.251 | 0.251 | 2.14¹ | 1.4e-5 | 2.1e-5 | **2.2e-4** | 1.2e-4 | 42 |
| B `QwenMultiHead` | 0.981 | **0.242** | 2.71 | 2.9e-3 | 8.4e-4 | **5.2e-2** | 1.7e-3 | 46 |
| C `QwenOFT` + 冻结 `embed_tokens` | 0.247 | 0.247 | 1.84 | 1.8e-5 | 2.3e-5 | **2.9e-4** | 1.0e-4 | 0 |

¹ A 与另一用户的进程共卡（约 25% 占用），B、C 期间该进程已退出；v2 的 OFT 是 1.86。

漂移随更新次数（token 级，1 − CKA；完整逐层数据在 `v3/*_drift.csv`，曲线 `v3/f0_curves.png` 右图）：

| 更新次数 | A 可训练均值 | A 第 35 层 | B 可训练均值 | B 第 35 层 | C 可训练均值 | C 第 35 层 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 25 | 1e-5 | 9e-5 | 1e-5 | 1.2e-4 | 1e-5 | 1.0e-4 |
| 50 | 1e-5 | 1.9e-4 | 2.1e-4 | 3.8e-3 | <1e-5 | 8e-5 |
| 100 | 1e-5 | 2.3e-4 | 1.6e-3 | 2.8e-2 | 1e-5 | 1.3e-4 |
| 150 | 2e-5 | 3.0e-4 | 2.4e-3 | 4.2e-2 | 1e-5 | 1.8e-4 |
| 200 | 1e-5 | 1.8e-4 | 2.6e-3 | 4.5e-2 | 2e-5 | 2.7e-4 |
| 250 | 1e-5 | 2.1e-4 | 2.9e-3 | 5.1e-2 | 2e-5 | 2.9e-4 |
| 300 | 1e-5 | 2.2e-4 | 2.9e-3 | 5.2e-2 | 2e-5 | 2.9e-4 |

读法：

1. **度量现在是自洽的。** 第 0 步（参考对自身）token 级与 mean-pool 都精确为 0；三条运行里冻结层 0–17 在所有探针点上都精确为 0（冻结 + 换回嵌入 → 逐位相同的输出），而 v2 里这些层"漂移"到 0.005–0.03。曲线不再锯齿：A 在约 75 次更新后进入平台，B 单调上升、约 225 次更新后饱和。
2. **漂移几乎只发生在最顶部两层，且按层深指数式增长。** A 在第 300 步的逐层剖面（18→35）从 1e-9 涨到 2.2e-4：第 18–33 层每深一层约 ×1.5，第 34、35 层各再跳一个数量级；第 33 层以下全部 < 1e-5。B 同样形状，但第 35 层 5.2e-2（A 的 240×）、第 34 层 8.4e-4（40×）、第 33 层 6.3e-5；也就是说三个头对骨干的额外改写集中在末层。这与 §3 在 v2 最终 checkpoint 上的受控诊断（换回嵌入后 token 级第 35 层 A 2.7e-4 / B 4.5e-2）一致，单场景批与跨场景批给出同量级读数。
3. **B 的 OFT 头损失仍与 A 相同或略低**（0.242 vs 0.251；v2 是 0.243 vs 0.244），但骨干顶层被改写得多得多。两件事同时成立说明：对 OFT 头而言，它所需的表征在 300 步内几乎不需要改骨干；PI / GR00T 两个 flow-matching 头把梯度更强地推进了第 34–35 层。这是 Q1 / Q3 的第一个可信数据点，但"顶层被改写"仍不等于"先验被侵蚀"——判别要靠 WP1 跨头探针 + 下游迁移。
4. **冻结 `embed_tokens` 没有代价。** C 的损失 0.247（A 0.251，噪声范围内），可训练参数少 389M（= 151,936 × 2,560，正是嵌入矩阵），漂移曲线与 A 重合（第 35 层 2.9e-4 vs 2.2e-4）。两种做法——训练时冻结，或探针时换回——给出同样的骨干漂移读数，互相验证了 §3 的归因；VLAct 论文没说明嵌入是否冻结，后续实验建议直接冻结（更便宜，也让探针更简单）。
5. **对 M1 阈值的量级提示**：在这个度量下，单头 OFT 的逐层漂移全部 < 3e-4，三头模型只有第 34–35 层超过 1e-3。`LLRDConfig` / `AuxSchedulerConfig` 里 0.10 / 0.05 的默认阈值差了两个数量级；`matrix_R0_R9.yaml` 的 R5 已改为 `drift_high 1e-2 / drift_low 1e-3` 作为起点，R3 的标定曲线出来后再定。

## 4. 这轮实跑抓到并修掉的问题

| 问题 | 修法 |
|---|---|
| `train_starvla_lab.__main__` 从 `train_starvla_cotrain` 导入工具函数，而该模块 import 时就构造 `Accelerator(DeepSpeedPlugin())`，无 DeepSpeed 即崩 | 从 `share_tools.apply_config_compat` / `trainer_tools.normalize_dotlist_args` 导入 |
| StarVLA `_train_step` 直接调 `self.model.forward(...)`，`nn.Module` forward hook 不触发，逐头损失进不了日志 | 在实例上包装绑定的 `forward`，`lab/loss_oft|pi|gr00t` 写进 metrics |
| 探针对 `QwenOFT` 用 `lang`、对 `QwenMultiHead` 用训练 prompt（含 8 个可学习 🔍 查询 token），两条不可比，且查询 token 的嵌入更新直接进入冻结层输出 | 统一用不含框架专有 token 的纯 VLM prompt |
| mean-pool 在 bf16 上做 | 先转 fp32 再池化 |
| `QwenMultiHead` 的头落入 StarVLA `learning_rate.base` 组（键是模块路径，`heads.*` ≠ `action_model`） | 运行时加 `--trainer.learning_rate.heads --trainer.learning_rate.project_layers`；README 已注明 |
| LLRD 优化器读 `trainer.weight_decay`，StarVLA 用 `trainer.optimizer.weight_decay` | 优先读后者 |
| 探针批只取一个 loader batch（8 个样本）而非 `probe_batch_size` | 累积多个 batch |
| `run_f0_smoke.sh` 在 `nvidia-smi \| head -1` 处被 `pipefail` 静默终止 | `\|\| true` |
| 钩子步号是自增前的，日志 `Step N` 行带的是前一步钩子的陈旧 `lab/drift`（v1 / v2） | v3 已改：`completed_updates_after()` 按"已完成的更新次数"给探针记录编号，第 0 步为参考自身；StarVLA 在 `_train_step` 返回后才自增的约定被显式处理 |
| 探针批的 `probe_data_mix` 注册进 `gr00t_lerobot.mixtures.DATASET_NAMED_MIXTURES` 后训练里仍 `KeyError`（v3 首次启动） | StarVLA 的 `gr00t_lerobot/registry.py` 在导入时把该字典复制了一份，`lerobot_datasets` 用的是副本；现在两份都注册 |
| `analyze_f0.py` 的 s/step 用 `timing/model` 均值，探针步（每 50 步一次与日志重合）把 1.9 s 拉成 4.7 s | 改为中位数，探针开销单列 |

## 5. 文件

```
f0_libero_goal_smoke/
├── README.md                     本文
├── v3/  summary.md  f0_curves.png  f0v3_{oft,multihead,oft_embedfrozen}_{metrics,drift}.csv   修正后的探针（§3.5）
├── v3_raw/  f0v3_*.log  *_probes.jsonl（含 drift / drift_secondary / embed_tokens / probe_tokens）  *_config.yaml
├── v2/  summary.md  f0_curves.png  f0v2_{oft,multihead}_{metrics,drift}.csv  probe_diag_f0v2_{oft,multihead}.json
├── v2_raw/  f0v2_{oft,multihead}.log  *_probes.jsonl  *_config.yaml（StarVLA 保存的实际生效配置）
├── v1/  v1_raw/                 第一遍（头 lr 未对齐、探针定义旧），仅供对照
```

最终模型（各 8.6 / 12 GB）留在节点 `/home/dataset-assist-0/liyufeng/awesome_starvla_work/checkpoints/f0v{2,3}_{oft,multihead}/final_model/`（v3 另有 `f0v3_oft_embedfrozen`），可用于 WP1 跨头线性探针的第一次真实运行。
