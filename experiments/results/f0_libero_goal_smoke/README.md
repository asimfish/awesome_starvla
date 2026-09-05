# F0 · 真实数据微调冒烟：LIBERO-goal，QwenOFT vs QwenMultiHead（1 卡，各 300 步）

**三句话结论。** (1) `starvla_lab.train.train_starvla_lab` + `vlact_ext.QwenMultiHead` 在真实 LeRobot 数据上端到端跑通，三头模型每步 2.70 s / 47 GB，单头 OFT 1.86 s / 30 GB（1.45×）。(2) 同一数据、同一超参、头学习率对齐到 1e-4 后，**三头模型里 OFT 头最后 50 步的 L1 损失 0.243，与单头 OFT 的 0.244 相同**——300 步内多加 GR00T 与 PI 两个 flow-matching 头对 OFT 头没有任何损害。(3) 训练中记录的"逐层表征漂移"曲线**不能直接用**：诊断表明它几乎全部来自 `embed_tokens` 里 42–46 行 prompt 词嵌入的更新（相对 Frobenius 变化仅 ~2e-5）被一个近退化的探针批放大；把预训练嵌入换回去后，OFT 微调的骨干几乎没动（可训练层 1−CKA = 0.0002），而三头模型的上层动了约 20 倍（0.0038，第 35 层 0.019）。这直接改写了方案 WP1 / M1 对漂移度量的定义。

## 1. 设置

| 项 | 值 |
|---|---|
| 数据 | `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`（LeRobot v2.1，428 段轨迹，52,042 帧，每样本主视角 + 腕部两张 224×224 图，动作块 8×7 delta EE，`robot_tag: franka`，无 state 字段），经 hf-mirror 下载 353 MB |
| 硬件 / 软件 | 1 × A100-80GB（`tianyiyun-30110-pub2` GPU 0，与一个 1 GB 小进程共卡）；torch 2.6.0+cu124，transformers 4.57.0，StarVLA `starVLA_dev@d81fc66`；单进程、`STARVLA_DISABLE_DEEPSPEED=1`，无 DeepSpeed |
| 配置 | [`code/starvla_lab/configs/f0_libero_goal_smoke.yaml`](../../../code/starvla_lab/configs/f0_libero_goal_smoke.yaml)：300 步，batch 8，无梯度累积，warmup 30，cosine 到 1e-6；lr 骨干 1e-5 / 头 1e-4；VLAct 式冻结 = 视觉编码器 + LLM 前 18 层（19 条精确路径，**`embed_tokens` 不在冻结集合内**）；梯度检查点开；探针 `calibrate_only`，v2 每 25 步、32 样本 |
| 两条运行 | A `QwenOFT`（StarVLA 原生，可训练 2.27B）；B `QwenMultiHead`（`vlact_ext`，OFT + GR00T + PI，`state_dim 0`，`--trainer.learning_rate.heads 1e-4 --trainer.learning_rate.project_layers 1e-4`，可训练 3.06B） |
| 启动 | `scripts/cluster/run_f0_smoke.sh <framework> <run_id> [覆盖项]`；分析 `scripts/analyze_f0.py`；探针诊断 `scripts/probe_diagnostics.py` |

v1（`v1/`、`v1_raw/`）是第一遍：B 的头学习率误落在 `base` 组（2.5e-5，因为 StarVLA 的 `learning_rate` 键是模块路径，`heads.*` 不匹配 `action_model`），探针 prompt 两条不一致、池化在 bf16 上做、每 50 步 16 样本。v2 修正了这三点，下面所有数字来自 v2；v1 保留供对照。

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
| 钩子步号是自增前的，日志 `Step N` 行带的是前一步钩子的陈旧 `lab/drift` | 已知偏差，JSONL 为准；下一版改为"完成 N 次更新后探测" |

## 5. 文件

```
f0_libero_goal_smoke/
├── README.md                     本文
├── v2/  summary.md  f0_curves.png  f0v2_{oft,multihead}_{metrics,drift}.csv  probe_diag_f0v2_{oft,multihead}.json
├── v2_raw/  f0v2_{oft,multihead}.log  *_probes.jsonl  *_config.yaml（StarVLA 保存的实际生效配置）
├── v1/  v1_raw/                 第一遍（头 lr 未对齐、探针定义旧），仅供对照
```

最终模型（各 8.6 / 12 GB）留在节点 `/home/dataset-assist-0/liyufeng/awesome_starvla_work/checkpoints/f0v2_{oft,multihead}/final_model/`，可用于 WP1 跨头线性探针的第一次真实运行。
