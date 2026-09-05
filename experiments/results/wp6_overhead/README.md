# WP6 · 真实 Qwen3-VL-4B 上的多头训练开销（GPU 实测）

**结论**：在 Qwen3-VL-4B 上，三头共监督（OFT + GR00T + PI，VLAct 配方 (c)）一步的时间是 OFT 单头的 **1.54×**、是 PI 单头的 1.19×；峰值显存 27.2 GB（batch 8，不含优化器状态）。头 dropout（每步只激活一个头）把时间降到 **1.21×**、显存 25.0 GB。把 OFT 查询位从 FM 头的 cross-attention 里屏蔽掉（`mask_oft_queries_for_fm_heads`，R2+ 的设定）不增加成本。PI 头（36 层逐层 cross-DiT，538M 参数）是三头里最贵的一个：单头就比 OFT 慢 29%、多用 6.6 GB 显存。

这些数字把 [10 · 改进方案](../../../reports/10_improvement_plan.md) 表 2 里 WP6 的"数字 B"填上了，并直接影响预算：`experiments/budget.md` 假设三头预训练是单头的约 1.5×，与实测一致。

## 设置

| 项 | 值 |
|---|---|
| 硬件 | 1 × NVIDIA A100-SXM4-80GB（`tianyiyun-30110-pub2` GPU 0，与一个 1 GB / ~25% 利用率的小进程共卡，计时噪声约 ±3%） |
| 软件 | torch 2.6.0+cu124，transformers 4.57.0，StarVLA `starVLA_dev@d81fc66`，awesome_starvla `3e550d1`（+ 显存统计修正） |
| 模型 | `Qwen_MultiHead`（`code/vlact_ext`）经生产构造路径：`get_vlm_model` 加载 Qwen3-VL-4B-Instruct bf16，StarVLA 头工厂构造全尺寸头（OFT MLP 66M；GR00T DiT-B 16 层 161M；PI 逐层 DiT 36 层 × 1024 宽 538M） |
| 冻结 | VLAct 配方 (a)：视觉编码器 + LLM 前 18 层（`freeze_rules` 语法 `qwen_vl_interface.model.model.visual,llm_layers_below:18`，513 个张量），可训练参数 2.27B（OFT 单头）～3.06B（三头） |
| 数据 | 合成样本：224×224 随机图像、16×7 动作块、7 维 state 以离散文本注入；batch 8 |
| 测量 | 前向 + 反向（不做优化器 step），warmup 2 步后计 10 步；`torch.cuda.max_memory_allocated`；FM 头 `repeated_diffusion_steps = 4`（StarVLA 默认）；推理为单样本 `predict_action`，FM 头 4 步 Euler |
| 脚本 | `scripts/gpu_overhead_bench.py`，经 `scripts/cluster/run_overhead_bench.sh` 启动；原始输出见 `run2_bs8_20260905/` |

## 结果（run2，显存统计修正后）

| 配置 | s/step | samples/s | 峰值显存 GB | 权重 GB | 时间 vs OFT | 推理 s/样本 |
|---|---:|---:|---:|---:|---:|---|
| `oft`（≈ QwenOFT） | 1.269 | 6.30 | 15.4 | 8.5 | 1.00× | 0.39 |
| `gr00t`（≈ QwenGR00T，state 走文本） | 1.341 | 5.97 | 15.7 | 8.9 | 1.06× | 0.53 |
| `pi`（≈ QwenPI_v3） | 1.642 | 4.87 | 22.0 | 10.6 | 1.29× | 0.67 |
| `three`（三头，VLAct (c)） | 1.952 | 4.10 | 27.2 | 11.5 | 1.54× | 0.41 / 0.53 / 0.68 |
| `three_dropout`（每步一个头） | 1.535 | 5.21 | 25.0 | 11.5 | 1.21× | — |
| `three_masked`（屏蔽 OFT 查询位） | 1.854 | 4.32 | 27.2 | 11.5 | 1.46× | — |

三头的边际成本可加：OFT 1.27 + GR00T 边际 0.07 + PI 边际 0.37 ≈ 1.71，实测 1.95（多出的 0.24 s 来自 `output_hidden_states=True` 下 36 层隐状态的逐层投影与两套 FM 头 4× 重复采样）。头 dropout 的 1.535 s 接近三个单头的均值 1.42 s，符合"期望每步只算一个头"的预期。

## 两个复现细节

1. **显存统计的顺序伪影**（run1 → run2 的修正）：run1 里 `pi` 单头峰值报 42.3 GB、高于三头，是因为上一配置的 HF 模型带引用环、`del` 后没被回收，下一配置测量时仍驻留。修法是每次加载前 `gc.collect()` + `torch.cuda.empty_cache()` 并记录"加载前驻留显存"（run2 每行均为 0.0 GB）。run1 的时间数字有效、显存数字作废，保留在 `run1_bs8_20260905/` 供对照。
2. **bf16 骨干 × fp32 头**：StarVLA 靠 `torch.autocast("cuda", dtype=float32)` 让 bf16 隐状态进 fp32 头；CPU 上没有这层 autocast，`Qwen_MultiHead._encode` 在非 CUDA 设备显式转 fp32（GPU 行为不变）。这是用迷你随机 Qwen3-VL 做 CPU 试跑时发现的。

## 尚未测

- 优化器状态与 DeepSpeed ZeRO-2 分片后的每卡显存（真实预训练用 8–16 卡）；batch 16 与 `repeated_diffusion_steps` 的敏感性；flash-attention（StarVLA 对 Qwen3-VL 强制 sdpa）。
- `QwenMultiHeadLab` 的两个辅助头（featpred / keyframe）的额外开销（预计很小：两个 MLP）。
