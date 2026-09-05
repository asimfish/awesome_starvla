# starvla_lab：改进方案的研究包

对应 [`reports/10_improvement_plan.md`](../../reports/10_improvement_plan.md) 的阶段 A 交付。所有模块只依赖 torch / numpy / 标准库，通过依赖注入在没有 StarVLA 与 transformers 的环境里也能导入和测试；接到 StarVLA 时与 [`vlact_ext`](../vlact_ext/) 一样不改动 StarVLA 源码。

```bash
cd awesome_starvla
python3 -m pytest code/starvla_lab/tests -q        # 125 passed（系统 python3.9，CPU，约 4 s；首次 import torch 约 20 s）
python3 scripts/build_run_matrix.py --print-commands 2   # 生成 experiments/run_matrix*.csv + budget.md 并打印示例命令

# 与真实 StarVLA 一起（StarVLA 需 Python >= 3.10；starVLA_code/ 是放在本仓库旁边的 StarVLA checkout）
bash scripts/setup_cpu_env.sh                      # 一次：.venv-starvla（py3.12 + CPU torch + StarVLA 可编辑安装）
PYTHONPATH=code:../starVLA_code .venv-starvla/bin/python -m pytest code/vlact_ext/tests code/starvla_lab/tests -q   # 184 passed, 2 skipped
PYTHONPATH=code:../starVLA_code .venv-starvla/bin/python scripts/smoke_starvla_integration.py   # 真实三头 + 全部 lab 钩子，约 15 s
```

建议的阅读顺序：先跑上面三条命令，再读 `scripts/smoke_starvla_integration.py`（约 420 行，就是一个最小的"真实头 + `QwenMultiHead` + `LabHooks`"训练循环，每个断言对应下表一个模块），然后按下表进各子包。

## 子包与工作包对应

| 子包 | WP | 公共 API | 一句话 |
|---|---|---|---|
| `probes/` | WP1 | `fit_linear_probe` / `fit_mlp_probe` / `fit_ridge_probe_cv`（z-score + 内层 CV 选岭系数）/ `split_indices_by_group` / `cross_head_probe_report`；`linear_cka` / `layerwise_cka`（可指定 GPU 算 Gram）；`DriftTracker` / `drift_to_llrd_decay`；`QwenBackboneProbe`（Qwen-VL 骨干的逐层表征提取：纯 VLM prompt、token 级或 mean-pool、提取时换回预训练 `embed_tokens`）+ `stratified_probe_batch` / `gather_probe_batch`；`ProbeSchedule` / `ProbeRunner` / `read_jsonl` | 不跑下游微调就度量骨干的"线性可读性"（探针 MAE / R²）与逐层表征漂移（1 − CKA），按步触发写 JSONL |
| `schedules/` | WP2 / WP4 | `layerwise_lr_decay_groups`（复用 `vlact_ext.freeze_rules` 的路径语法，冻结参数不进优化器）；`DriftDrivenLLRD`（漂移高 → 降 lr，有下限、带滞回；可与 `LambdaLR` 协同）；`AuxDataScheduler`（`fixed` / `linear` / `drift` 三策略，输出 VLM 采样概率与 `loss_scale.vlm`，`apply_to_cfg` 写回 StarVLA 配置） | 把 VLAct 的"硬冻结 + 固定 0.5 caption 权重"变成可连续调节、可由漂移信号驱动的策略 |
| `heads/` | WP3 | `FutureFeaturePredictionHead` + `targets_from_sequence`；`KeyframeHead` + `soft_keyframe_labels` + `keyframe_bce_loss` + `KeyframeWritePolicy`（阈值 → 1D NMS → 冷却 → FIFO）+ `EvidenceMemory` + `TeacherStudentCurriculum`；`Qwen_MultiHeadLab`（注册名 `QwenMultiHeadLab`） | 把"头多样性即正则化"推到非动作头：未来视觉特征预测（与世界模型路线汇合）与关键帧预测（EventVLA 的 KEM 思想，自行实现），作为可开关、可加权的辅助头挂在 `QwenMultiHead` 之上 |
| `data/` | WP3b / F1 | `TrajectorySubset` / `install_fraction_hook`（按轨迹的确定性子采样，挂到 StarVLA 的 `make_LeRobotSingleDataset`）；`build_feature_cache` / `FeatureCache` / `FutureFeatureTransform`（冻结提取器逐轨迹缓存未来帧特征）；`heuristic_keyframe_steps` / `FunctionLabeler` / `KeyframeLabelTransform`（关键帧标注与块内相对步） | 数据比例曲线与两个辅助头所需的离线数据准备 |
| `train/` | WP9 | `LabConfig`（`trainer.lab.*`）、`build_optimizer_and_scheduler`（LLRD 参数组 + 注入的调度器工厂）、`LabHooks` / `attach_to_trainer`（每步：辅助数据调度写回 `trainer.loss_scale.vlm`、头 dropout 设 `active_heads`、每 N 步探针 → 漂移 → 驱动 LLRD / 调度器 → JSONL）、`train_starvla_lab.py`（镜像 StarVLA 单 / 双 loader 主流程的入口） | 把上面所有库真正接进 StarVLA 训练循环，不改 StarVLA 源码 |
| `bench/` | WP5 / WP6 | `Protocol` / `BackboneSpec` / `BenchmarkSpec` / `build_runs` / `render_commands` / `varying_keys` / `summarize_results` / `format_summary_table`；`measure_step_overhead` / `compare_configs` / `HeadDropoutSchedule` | "只换骨干"协议：从骨干列表生成 StarVLA 命令矩阵并审计只有骨干 / seed / run_id / 数据比例在变；三头预训练的秒/步、samples/s、峰值显存测量与头 dropout 降本选项 |
| `configs/` | §3 | `protocol_f1.yaml`（真实 StarVLA 训练 yaml 与评测命令模板，`{ckpt_file}` / `{seed}` / `{run_id}` 占位）、`matrix_R0_R9.yaml`（分级 core / extra、每变体 `train_script`、`trainer.lab.*` 键） | 实验矩阵的唯一事实来源，`scripts/build_run_matrix.py` 据此生成 `experiments/run_matrix*.csv` 与 `budget.md` |

## 接到 StarVLA 的方式

- **入口**：在 StarVLA 仓库根目录，把本仓库的 `code/` 放进 `PYTHONPATH`，以模块方式启动：

```bash
PYTHONPATH=<awesome_starvla>/code accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 16 -m starvla_lab.train.train_starvla_lab --config_yaml code/vlact_ext/configs/vlact_pretrain_example.yaml \
  --trainer.lab.mode cotrain --trainer.lab.llrd.enabled true --trainer.lab.llrd.decay 0.85 \
  --trainer.lab.probes.enabled true --trainer.lab.probes.every_n_steps 2000
```

`trainer.lab.mode` 为 `single` / `cotrain` / `auto`；`trainer.lab.{llrd, aux_scheduler, probes, head_dropout}` 的全部字段见 `train/lab_config.py`。所有 `trainer.lab.*` 键只被这个入口消费，StarVLA 原生脚本会忽略它们。

- **框架**：`QwenMultiHeadLab` 与 `vlact_ext` 的 `QwenMultiHead` 一样，拷进（或以一行 shim 导入到）`starVLA/model/framework/VLM4A/` 即被 `build_framework` 自动注册；yaml 写 `framework.name: QwenMultiHeadLab` 并在 `framework.aux_heads.{featpred,keyframe}` 下开关与加权。样本 dict 约定：`future_features: [len(offsets), d_feat]`（缺省则该样本 mask）、`keyframe_steps: list[int]`（缺省则 mask）。
- **LLRD**：训练脚本里用 `layerwise_lr_decay_groups(model, base_lr, decay, freeze_rules_spec=cfg.trainer.freeze_modules)` 替换 StarVLA 的 `build_param_lr_groups` 返回值；`layer_group_index(optimizer.param_groups)` 交给 `DriftDrivenLLRD`。
- **探针与漂移**：`DriftTracker(extract_fn, probe_batch, reference=frozen_initial_vlm, compute_device="cuda")`，其中 `extract_fn(model, batch) -> list[Tensor]` 对 Qwen-VL 家族直接用 `QwenBackboneProbe(representation="token"|"pooled", token_subset, max_tokens, restore_pretrained_embeddings)`，其他骨干自己写；`ProbeRunner` 每 `every_n_steps` 调用一次，把 `tracker.summary()` 喂给 `DriftDrivenLLRD.step` 与 `AuxDataScheduler.step(step, drift=...)`。`install_hook_example()` 的 docstring 给出了插进 `train_starvla_cotrain.py` 循环的最小改法。
  探针的口径由 F0 实测定下（[`experiments/results/f0_libero_goal_smoke/`](../../experiments/results/f0_libero_goal_smoke/README.md)）：提取时把预训练 `embed_tokens` 临时换回（否则几十行 prompt 词嵌入的更新会主导整条曲线）；主指标 token 级 CKA（每个有效 token 是一个样本，≤ `max_tokens` 个固定位置），mean-pool 作次指标写进 `drift_secondary`；探针批按指令轮询抽样，可用 `probes.probe_data_mix`（StarVLA 混合名或内联 `目录:机器人[,…]`，运行时注册）从训练集以外的套件取样；记录以"已完成的更新次数"为步号，第 0 步为参考自身（噪声底应为 0）。`trainer.lab.probes.*` 全部字段见 `train/lab_config.py`。
- **协议**：`experiments/run_matrix.csv` 的每一行是一次下游微调；`render_commands` 给出 `accelerate launch ... --framework.qwenvl.base_vlm <init>` 或 `--trainer.pretrained_checkpoint <ckpt> --trainer.reload_modules qwen_vl_interface` 两种初始化写法，其余覆盖项全部固定。

## 已验证与未验证

**CPU 上已验证（125 个测试）**：探针 / 调度器 / 头 dropout 在 mock 训练器里按步生效、`calibrate_only` 只记录不干预、数据比例钩子只替换命名工厂、特征缓存复用不重算、关键帧标注块外裁剪；探针在合成数据上恢复已知线性映射；CKA 对同一表征为 1、对正交旋转与缩放不变；漂移→衰减系数单调有界；LLRD 分组层深单调、冻结层不进优化器、与 `LambdaLR` 协同后倍率保持；调度器三策略轨迹与饱和边界；辅助头损失在完美预测时为 0、mask 生效、全 mask 无 NaN；写入策略的阈值 / NMS / 冷却 / FIFO；`Qwen_MultiHeadLab` 用 mock 骨干前向返回全部 loss 键且可反传、关闭辅助头时与父类一致；协议矩阵只有允许的键在变、CSV 往返、结果聚合。

**CPU 上与真实 StarVLA 一起验证（`scripts/smoke_starvla_integration.py`）**：StarVLA 真实的 OFT / GR00T / PI 头工厂构造的头注入 `QwenMultiHead` 后三头前向 / 反传 / 逐头推理；在模块树与 Qwen3-VL 一致的迷你骨干上，`llm_layers_below:1` 冻结 + `layerwise_lr_decay_groups`（冻结层不进优化器、lr 随层深单调）、`LabHooks` 的头 dropout 轮换、探针每 N 步写 JSONL 并驱动 `DriftDrivenLLRD`、`AuxDataScheduler` 把 `loss_scale.vlm` 写回配置，全部在同一个 8 步 mock 训练循环里生效。

**需要 GPU**：真实 Qwen3-VL-4B 上的探针数值与下游成功率的相关性（决策门 G2）；三头 + 辅助头的显存与吞吐（WP6 数字）；辅助头与动作头的梯度冲突；R0–R9 全部实验。

## 目录

```
starvla_lab/
├── probes/      action_probe.py  cka.py  drift.py  hooks.py  qwen_extract.py
├── schedules/   llrd.py  aux_scheduler.py
├── heads/       feature_prediction_head.py  keyframe_head.py  register.py
├── data/        subsample.py  future_features.py  keyframe_labels.py  mixtures.py
├── train/       lab_config.py  integration.py  train_starvla_lab.py
├── bench/       backbone_bench.py  overhead_bench.py
├── configs/     protocol_f1.yaml  matrix_R0_R9.yaml
├── tests/       test_{probes,schedules,heads,data,train,bench}_*.py（125 个）
└── pytest.ini
```
