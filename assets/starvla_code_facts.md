# StarVLA 代码库硬事实卡片（供 PPT 使用）

来源：本地快照 `starVLA_code`（分支 `starVLA_dev`，HEAD `d81fc66`，2026-09-04）；全部数字为实际统计，行号以该快照为准。

## 规模
- Python 文件 261 个，共 57,625 行；`starVLA/` 139 个（model 25,667 行、dataloader 8,688 行、training 3,428 行），`examples/` 98 个（16,810 行），`deployment/` 15 个，`tests/` 7 个（1,067 行）。
- Git：131 次提交（浅历史自 2026-02-21）、40 位作者；无 `.github/workflows` CI。
- 依赖钉死：transformers==4.57.0、accelerate==1.5.2、deepspeed==0.16.9、torchvision==0.21.0、numpy==1.26.4、diffusers（`requirements.txt`）。

## 模型层
- 注册框架 28 个（VLM4A 20 / WM4A 6 / VM4A 2），31 个注册键（别名 `QwenFM`、`Pi0`、`Pi05`）；注册表 `FRAMEWORK_REGISTRY`（`starVLA/model/tools.py:L136-164`），自动扫描 `starVLA/model/framework/*/`（`base_framework.py:L60-82`）。
- 四个核心框架：`QwenOFT`（`VLM4A/QwenOFT.py`）、`QwenFast`（`QwenFast.py`，注意大小写）、`QwenPI_v3`（`QwenPI_v3.py`）、`QwenGR00T`（`QwenGR00T.py`）。
- 动作头文件 11 个（`starVLA/model/modules/action_model/`）；VLM 接口文件 10 个，`get_vlm_model` 分派 9 个分支（Qwen2.5-VL、Qwen3-VL、Qwen3.5、Gemma-4、Molmo2、MiniCPM-V、Florence-2、Cosmos-Reason2、VILA/EgoVLA）；无 InternVL 接口。
- 接口签名：`forward(examples: List[dict]) -> {"action_loss": Tensor}`（`base_framework.py:L151`）；`predict_action(examples) -> {"normalized_actions": np.ndarray[B,T,D]}`（L168）；`from_pretrained(ckpt, config_overrides=[...])`（L254）。
- 样本 dict：`{"image": List[PIL], "lang": str, "action": np.ndarray[T,D] float16, "state"?: np.ndarray[1,S], "robot_tag": str}`（`gr00t_lerobot/datasets.py:L1413-1418`）。

## 四种动作头
- FAST：动作 → `physical-intelligence/fast` token → `<robot_action_k>` 字串放 assistant turn；loss = VLM next-token CE；推理 `generate(max_length=2048)` 后按 id 区间 `151669..153716`（Qwen3-VL-Action，2048 个）抽 token 解码（`QwenFast.py:L153-176, L209-222`）。
- OFT：指令尾拼 `"Please predict the next N robot actions: <action>🔍…🔍<action>."`，取 `hidden_states[-1]` 中 N 个 🔍 位置 → `MLPResNet(2 blocks, H→2H→D)`；loss = masked L1 `(|â−a|·m).sum()/m.sum()`（`QwenOFT.py:L40-51, L188-226`）；单次前向。
- PI（QwenPI_v3）：最后 36 层 hidden 各经 `LayerNorm+Linear(2560→1024)` 投影，逐层 cross-attn 到 36 层 DiT（全 cross，`interleave_self_attention=False`）；state 量化 256 桶写进文本 `[STATE] … [ACTION]`；loss = `E‖v_θ(a_t,t) − (a−ε)‖²`，`a_t=(1−t)ε+t·a`，`t=(0.999−Beta(1.5,1))/0.999`；推理 4 步 Euler；参数 4.44B + 538.7M + 94.6M（`QwenPI_v3.py:L33-42`；`LayerwiseFM_ActionHeader.py:L288-347, L349-408`）。
- GR00T：`hidden_states[-1]` 作 cross-attn K/V；DiT-B（768 维、12 头）16 层交错 self/cross；state 经 MLP encoder 拼在序列前；同一 flow-matching MSE；4 步 Euler；`repeated_diffusion_steps=8`（`GR00T_ActionHeader.py:L190-193, L312-363, L365-421`）。
- 训练精度：VLM `autocast(bf16)`，动作头 `autocast(float32)`。

## 数据层
- LeRobot v2.0/v3.0 读取；统计 `mean/std/min/max/q01/q99` 缓存于 `meta/stats_gr00t.json`（format v2）；归一化模式 q99 / mean_std / min_max / scale / binary（`transform/state_action.py:L98-213`）。
- 多数据集：`DATASET_NAMED_MIXTURES[name] = [(dir, weight, robot_type)]`；按 embodiment tag 合并统计，q01/q99 默认 `min_max` 混合；`__len__` = 主数据集 `len/weight` 最大值（`datasets.py:L2274-2325, L2478-2541`）。
- 16 个外部 `examples/**/train_files/data_registry/data_config.py` 由 `registry.py:L116-144` 自动发现；基础 10 个 robot_type。
- action chunk 长度 = `DataConfig.action_indices`；模型取最后 `action_horizon` 步；`action_mode ∈ {abs, delta, rel}`。
- VLM 共训数据：LLaVA json（`sharegpt4v_coco`），dataloader 内已 tokenize，labels 只监督 assistant 段（`vlm_datasets.py:L232-246`）。

## 训练层
- 4 个入口：`train_starvla.py`（单 VLA）、`train_starvla_cotrain.py`（VLA+VLM 双 loader 双 backward）、`train_starvlm.py`（纯 VLM）、`train_starvln.py`（HF Trainer，VLN）。
- 冻结：`trainer.freeze_modules="a.b,c.d"` 精确点路径（`trainer_tools.py:L192-234`，非正则）；分模块 LR：`trainer.learning_rate.{base, qwen_vl_interface, action_model}`（L92-148）；LIBERO 默认 2.5e-5 / 1e-5 / 1e-4，cosine_with_min_lr 1e-6，warmup 5000，AdamW β=(0.9,0.95)。
- checkpoint：仅 state_dict，`<run>/checkpoints/steps_N_pytorch_model.pt` + `config.yaml`（仅访问过的键）+ `config.full.yaml` + `dataset_statistics.json` + `summary.jsonl`；无 optimizer 状态；加载 `pretrained_checkpoint` + `reload_modules`（子模块 strict）。
- 无 EMA（仅 VM4A/DiffusionPolicy 内部）；梯度累积由 `ds_config.yaml`（=1）决定，yaml 的 `trainer.gradient_accumulation_steps` 不生效；ZeRO-2 默认、ZeRO-3 换 `deepspeed_zero3.yaml`。

## 部署
- `python deployment/model_server/server_policy.py --ckpt_path X.pt --port 10093 --use_bf16`；响应 `data["actions"]` 已反归一化；握手元数据含 `action_chunk_size / available_unnorm_keys / action_keys / state_keys`。
- GR00T N1.6 ZMQ 兼容服务器 `server_policy_gr00t_zmq.py`（端口 5555）；Docker 三镜像（server / train / robocasa）。

## 生态
- 13 个仿真基准、5 个真机、6 个模型扩展、1 个 UMI 人类数据；12 个 `model2*_interface.py`；31 个训练 yaml。
- LIBERO 平均成功率（README）：Qwen3-VL-OFT 96.6、GR00T 96.5、PI 95.7、FAST 95.4（30K 步，单策略 4 套件）。

## VLAct 配方在代码中的状态
| 配方 | 状态 | 落点 |
|---|---|---|
| (a) 冻结视觉 + LLM 下半层 | 部分 | `trainer.freeze_modules` 列 `...visual` + `...language_model.layers.0..17`；无正则/区间 |
| (b) caption 共训 `L_action + 0.5·L_VLM-CE` | 已有 | `train_starvla_cotrain.py` + `trainer.loss_scale.vlm: 0.5` + `dataset_use: sharegpt4v_coco` |
| (c) OFT+PI+GR00T 多头共监督 | 缺失 | 需新框架 `QwenMultiHead`（三头共用一次骨干前向，`action_loss = Σ`） |
| (d) 20 维部分统一布局 + mask | 部分 | mask 消费：`QwenOFT.masked_l1_loss`、`AML_ActionHeader`；生产：`umi_datasets`、`MiniCPMRobotManip._to_80d`；GR00T/PI 头无 mask；无槽位映射 transform |
| (e) wrap-aware L1 `((â−a)+π) mod 2π − π` | 缺失 | 无任何模 2π 逻辑；需 `wrap_aware_l1` + 数据侧 wrap |
| (f) 丢头重训 + 全参解冻 | 已有 | `pretrained_checkpoint` + `reload_modules: "qwen_vl_interface"` + `freeze_modules: ""` |

## 已核实的问题（可作"改进空间"页）
- `QWen3.py:L52-53` 强制 `attn_implementation="sdpa"`；`fast_ActionHeader.py:L85-89` 引用不存在属性；`ABot_M0.py:L135` `VGGT` 未 import；`data_config.py:L126` `GR00TTransform` 未 import；`QwenPI_v3.py:L315-317` `repeated_diffusion_steps` 读错节点。
- 文档漂移：`train_internvla.py`、`--framework.framework_py`、`QwenFAST`、"regex 冻结"、`WM4A_OFT.py`、`eval_protocol.md` 的 `normalized_actions`、缺失的 `integrate_your_dataset.md` / agent skill README。
- 三个训练脚本约 80% 重复；三个 flow-matching 头文件前 190 行重复；`base_framework.compute_loss` 为无调用者的死代码。
