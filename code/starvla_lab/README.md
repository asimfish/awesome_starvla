# starvla_lab：改进方案的研究包

对应 [`reports/10_improvement_plan.md`](../../reports/10_improvement_plan.md) 的阶段 A 交付。所有模块只依赖 torch / numpy / 标准库，通过依赖注入在没有 StarVLA 与 transformers 的环境里也能导入和测试；接到 StarVLA 时与 [`vlact_ext`](../vlact_ext/) 一样不改动 StarVLA 源码。

```bash
cd awesome_starvla
python3 -m pytest code/starvla_lab/tests -q        # 95 passed（CPU，约 3 s；首次 import torch 约 20 s）
python3 scripts/build_run_matrix.py --print-commands 2   # 生成 experiments/run_matrix.csv 并打印示例命令
```

## 子包与工作包对应

| 子包 | WP | 公共 API | 一句话 |
|---|---|---|---|
| `probes/` | WP1 | `fit_linear_probe` / `fit_mlp_probe` / `cross_head_probe_report`；`linear_cka` / `layerwise_cka`；`DriftTracker` / `drift_to_llrd_decay`；`ProbeSchedule` / `ProbeRunner` / `read_jsonl` | 不跑下游微调就度量骨干的"线性可读性"（探针 MAE / R²）与逐层表征漂移（1 − CKA），按步触发写 JSONL |
| `schedules/` | WP2 / WP4 | `layerwise_lr_decay_groups`（复用 `vlact_ext.freeze_rules` 的路径语法，冻结参数不进优化器）；`DriftDrivenLLRD`（漂移高 → 降 lr，有下限、带滞回；可与 `LambdaLR` 协同）；`AuxDataScheduler`（`fixed` / `linear` / `drift` 三策略，输出 VLM 采样概率与 `loss_scale.vlm`，`apply_to_cfg` 写回 StarVLA 配置） | 把 VLAct 的"硬冻结 + 固定 0.5 caption 权重"变成可连续调节、可由漂移信号驱动的策略 |
| `heads/` | WP3 | `FutureFeaturePredictionHead` + `targets_from_sequence`；`KeyframeHead` + `soft_keyframe_labels` + `keyframe_bce_loss` + `KeyframeWritePolicy`（阈值 → 1D NMS → 冷却 → FIFO）+ `EvidenceMemory` + `TeacherStudentCurriculum`；`Qwen_MultiHeadLab`（注册名 `QwenMultiHeadLab`） | 把"头多样性即正则化"推到非动作头：未来视觉特征预测（与世界模型路线汇合）与关键帧预测（EventVLA 的 KEM 思想，自行实现），作为可开关、可加权的辅助头挂在 `QwenMultiHead` 之上 |
| `bench/` | WP5 / WP6 | `Protocol` / `BackboneSpec` / `BenchmarkSpec` / `build_runs` / `render_commands` / `varying_keys` / `summarize_results` / `format_summary_table`；`measure_step_overhead` / `compare_configs` / `HeadDropoutSchedule` | "只换骨干"协议：从骨干列表生成 StarVLA 命令矩阵并审计只有骨干 / seed / run_id / 数据比例在变；三头预训练的秒/步、samples/s、峰值显存测量与头 dropout 降本选项 |
| `configs/` | §3 | `protocol_f1.yaml`（固定下游协议）、`matrix_R0_R9.yaml`（九个预训练变体 + R9 记忆叠加） | 实验矩阵的唯一事实来源，`scripts/build_run_matrix.py` 据此生成 `experiments/run_matrix.csv` |

## 接到 StarVLA 的方式

- **框架**：`QwenMultiHeadLab` 与 `vlact_ext` 的 `QwenMultiHead` 一样，拷进（或以一行 shim 导入到）`starVLA/model/framework/VLM4A/` 即被 `build_framework` 自动注册；yaml 写 `framework.name: QwenMultiHeadLab` 并在 `framework.aux_heads.{featpred,keyframe}` 下开关与加权。样本 dict 约定：`future_features: [len(offsets), d_feat]`（缺省则该样本 mask）、`keyframe_steps: list[int]`（缺省则 mask）。
- **LLRD**：训练脚本里用 `layerwise_lr_decay_groups(model, base_lr, decay, freeze_rules_spec=cfg.trainer.freeze_modules)` 替换 StarVLA 的 `build_param_lr_groups` 返回值；`layer_group_index(optimizer.param_groups)` 交给 `DriftDrivenLLRD`。
- **探针与漂移**：`DriftTracker(extract_fn, probe_batch, reference=frozen_initial_vlm)`，其中 `extract_fn(model, batch) -> list[Tensor]` 由你按骨干写（例如取各层 hidden state 的 mean-pool）；`ProbeRunner` 每 `every_n_steps` 调用一次，把 `tracker.summary()` 喂给 `DriftDrivenLLRD.step` 与 `AuxDataScheduler.step(step, drift=...)`。`install_hook_example()` 的 docstring 给出了插进 `train_starvla_cotrain.py` 循环的最小改法。
- **协议**：`experiments/run_matrix.csv` 的每一行是一次下游微调；`render_commands` 给出 `accelerate launch ... --framework.qwenvl.base_vlm <init>` 或 `--trainer.pretrained_checkpoint <ckpt> --trainer.reload_modules qwen_vl_interface` 两种初始化写法，其余覆盖项全部固定。

## 已验证与未验证

**CPU 上已验证（95 个测试）**：探针在合成数据上恢复已知线性映射；CKA 对同一表征为 1、对正交旋转与缩放不变；漂移→衰减系数单调有界；LLRD 分组层深单调、冻结层不进优化器、与 `LambdaLR` 协同后倍率保持；调度器三策略轨迹与饱和边界；辅助头损失在完美预测时为 0、mask 生效、全 mask 无 NaN；写入策略的阈值 / NMS / 冷却 / FIFO；`Qwen_MultiHeadLab` 用 mock 骨干前向返回全部 loss 键且可反传、关闭辅助头时与父类一致；协议矩阵只有允许的键在变、CSV 往返、结果聚合。

**需要 GPU**：真实 Qwen3-VL-4B 上的探针数值与下游成功率的相关性（决策门 G2）；三头 + 辅助头的显存与吞吐（WP6 数字）；辅助头与动作头的梯度冲突；R0–R9 全部实验。

## 目录

```
starvla_lab/
├── probes/      action_probe.py  cka.py  drift.py  hooks.py
├── schedules/   llrd.py  aux_scheduler.py
├── heads/       feature_prediction_head.py  keyframe_head.py  register.py
├── bench/       backbone_bench.py  overhead_bench.py
├── configs/     protocol_f1.yaml  matrix_R0_R9.yaml
├── tests/       test_{probes,schedules,heads,bench}_*.py（95 个）
└── pytest.ini
```
