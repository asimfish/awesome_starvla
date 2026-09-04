# vlact_ext：VLAct 配方在 StarVLA 上缺失/半支持组件的独立实现

对照 [`reports/02_starvla_codebase_analysis.md`](../../reports/02_starvla_codebase_analysis.md) 第 9 章，VLAct（arXiv 2608.27550）六项配方里 (a) 只有精确路径冻结、(c)(e) 完全缺失、(d) 只有零散 mask 基础设施。本目录把这四项实现为一个**不修改 StarVLA 源码**即可拷入使用的扩展包，并附 CPU 单元测试；(b)(f) StarVLA 已支持，只给配置写法。

| 配方 | 文件 | 内容 |
|---|---|---|
| (a) 浅层冻结 | `freeze_rules.py` | `re:<regex>`、`path.layers[lo:hi]`（可带 `.sub.path` 后缀、负索引）、`llm_layers_below:N` 语法糖；`expand_to_exact_paths` 生成原生 StarVLA 能解析的精确路径；`install_into_starvla` 一行 monkeypatch 让 yaml 直接用新语法和 `trainer.freeze_llm_layers_below` 键 |
| (c) 多头共监督 | `multihead_framework.py` | `Qwen_MultiHead(baseframework)`，注册名 `QwenMultiHead`；一次骨干前向，OFT + GR00T + PI 三头各自算 loss，`action_loss = Σ w_h·L_h`；每头可开关/加权；`predict_action(head=...)` 路由 |
| (d) 20 维统一布局 | `unified_action_layout.py` | dict 驱动的 `robot_tag -> EmbodimentLayout` 槽位映射，`to_unified` / `from_unified` / `masks`；`UnifiedActionTransform` 重写样本 `action` 并附加 `action_mask` / `periodic_mask`；`TransformedDataset` 代理与 DataConfig `make_dataset` 钩子 |
| (e) wrap-aware L1 | `wrap_aware_loss.py` | `wrap_to_pi`、`wrap_aware_residual`、`masked_wrap_aware_l1`（torch）与 `masked_wrap_aware_l1_np`（numpy）、`flow_matching_sample_estimate`（PI/GR00T 用的 x1_hat） |
| (b) caption 共训 | `configs/vlact_pretrain_example.yaml` | `train_starvla_cotrain.py` + `loss_scale.vlm: 0.5` + `dataset_use: sharegpt4v_coco` |
| (f) 丢头重训 | 同上（注释段） | `pretrained_checkpoint` + `reload_modules: "qwen_vl_interface"` + `freeze_modules: ""` |

```
code/vlact_ext/
├── README.md
├── __init__.py                 # 导入即注册 QwenMultiHead（StarVLA 可导入时）
├── wrap_aware_loss.py          # (e)
├── unified_action_layout.py    # (d)
├── freeze_rules.py             # (a)
├── multihead_framework.py      # (c)
├── configs/vlact_pretrain_example.yaml
├── pytest.ini                  # 关掉 .pytest_cache，避免在仓库根目录留文件
└── tests/                      # 61 个测试，全部 CPU、mock 骨干
```

## 1. 运行测试

```bash
cd /Users/liyufeng/Desktop/research/awesome_starvla
python3 -m pytest code/vlact_ext/tests -q          # 60 passed, 1 skipped（约 30 s，主要是 import torch）
VLACT_TEST_QWEN3VL=1 python3 -m pytest code/vlact_ext/tests/test_freeze_rules.py -q -k Qwen3VL
#   ↑ 可选：用随机初始化的迷你 Qwen3-VL（transformers 4.57）核对默认冻结路径，import transformers 需 ~1 min
# 没有 pytest 时：
python3 -m unittest discover -s code/vlact_ext/tests -t code
```

测试环境只需 `torch`、`numpy`（系统 python3.9 已有）。多头框架测试通过依赖注入（`Qwen_MultiHead(config, vlm=..., heads=..., project_layers=...)`）用 mock 骨干与极小头运行，不需要 transformers / omegaconf / StarVLA。

## 2. 怎么拷进 StarVLA

推荐布局（`<StarVLA>` 为仓库根目录）：

```bash
cp -r code/vlact_ext <StarVLA>/starVLA/vlact_ext          # 工具包，import 路径 starVLA.vlact_ext.*
cat > <StarVLA>/starVLA/model/framework/VLM4A/QwenMultiHead.py <<'EOF'
from starVLA.vlact_ext.multihead_framework import *  # noqa: F401,F403  注册 QwenMultiHead
EOF
```

`build_framework` 会自动 import `VLM4A/` 下所有模块，这个一行 shim 触发 `@FRAMEWORK_REGISTRY.register("QwenMultiHead")`。两种等价替代：

* 直接 `cp code/vlact_ext/multihead_framework.py <StarVLA>/starVLA/model/framework/VLM4A/QwenMultiHead.py`。文件内依次尝试 `.`、`starVLA.vlact_ext`、`vlact_ext` 三个前缀导入 `wrap_aware_loss` / `unified_action_layout`，所以工具包仍需放在上面任一位置。
* `cp -r code/vlact_ext <StarVLA>/starVLA/model/framework/VLM4A/vlact_ext`：整包放进 VLM4A，`__init__.py` 导入 `multihead_framework` 即完成注册（`tests/` 子目录不会被扫描）。

不需要改动 StarVLA 任何已有文件。`omegaconf`、`transformers`、`diffusers` 等由 StarVLA 环境提供。

## 3. 各组件在 StarVLA 中的接入点

### 3.1 (c) `QwenMultiHead`

```yaml
framework:
  name: QwenMultiHead
  action_model: {action_dim: 20, state_dim: 0, action_horizon: 16, ...}   # 三头共享
  heads:
    oft:   {enabled: true, loss_weight: 1.0}
    gr00t: {enabled: true, loss_weight: 1.0, action_model: {repeated_diffusion_steps: 4, diffusion_model_cfg: {...}}}
    pi:    {enabled: true, loss_weight: 1.0, action_model: {diffusion_model_cfg: {action_dit_hidden_dim: 1024, ...}}}
  predict_head: oft
```

* 头的构造完全复用 StarVLA 工厂：`MLP_ActionHeader.get_action_model`、`GR00T_ActionHeader.get_action_model`、`LayerwiseFM_ActionHeader.get_action_model` + `share_tools.populate_layerwise_dit_cfg`。每个头拿到的是"共享 `action_model` 深合并 `heads.<name>.action_model`"生成的独立 `framework.action_model` 视图，因此三个头可以各有自己的 `diffusion_model_cfg`，而 `action_dim` / `action_horizon` 保证一致。默认值与 `QwenGR00T` / `QwenPI_v3` 的 DefaultConfig 逐项相同，运行期同样把 `cross_attention_dim` / `action_hidden_dim` 对齐到 VLM hidden size。
* 子模块命名：`qwen_vl_interface`、`heads.oft`、`heads.gr00t`、`heads.pi`、`project_layers`（PI 的逐层 LayerNorm+Linear 投影，与 QwenPI_v3 一样挂在框架上）。`learning_rate` / `freeze_modules` / `reload_modules` 直接用这些路径。
* `forward(examples)` 返回 `{"action_loss", "loss_oft", "loss_pi", "loss_gr00t"}`；关闭的头对应 0。现有 `_train_step` 只读 `action_loss`，其余键若想记录到 wandb，在 `train_starvla*.py` 的 `_train_step` 返回值里加 `{k: v.item() for k, v in output_dict.items() if k != "action_loss"}` 即可（可选改动）。
* `predict_action(examples, head=None, robot_tag=None, **kwargs)`：`head` 缺省取 `framework.predict_head`；`robot_tag` 给定且启用 `unified_layout` 时用 `from_unified` 返回该本体的原生维度（部署侧 `PolicyNormProcessor` 反归一化需要原生维度；客户端把 `head` / `robot_tag` 放进请求 kwargs 即可，`server_policy.py` 会透传）。训练器 `eval_action_model` 传的 `use_ddim` 等 kwargs 被忽略。
* OFT 的 `<action>🔍…🔍<action>` 查询后缀只在 OFT 头启用时追加，训练与推理一致；`mask_oft_queries_for_fm_heads: true` 可让两个 FM 头的 cross-attention 看不到这些位置（报告 §9.3 (iii) 的建议，默认关）。
* state 统一以 π0.5 式离散文本注入（`share_tools.add_discretized_state_to_instruction`），三个头共用；GR00T 头因此以 `state_dim: 0` 构造（无 `state_encoder`）。这与 `QwenGR00T` 用 MLP 编码 state 不同，是为了让混合本体（state 维度不同）也能共训。
* 两个 FM 头的 loss 由 `flow_matching_loss` 用头自己的子模块（`sample_time`、`action_encoder`、`future_tokens`、`position_embedding`、`model`、`action_decoder`）重算，逐行对应 `FlowmatchingActionHead.forward` / `LayerwiseFlowmatchingActionHead.forward`（同一 Beta 时间采样、`x_t=(1-t)ε+t·a`、速度目标 `a-ε`、PI 走 `return_pre_output=True`），只多了 mask 归约与 wrap-aware 项。这样做是因为 StarVLA 的两个头 `forward` 不接受 mask，而任务要求不改 StarVLA 源码。构造时会检查头是否具备这些属性，接口漂移会直接报错。
* 下游微调（f）：`reload_modules: "qwen_vl_interface"` 只加载骨干，checkpoint 里 `heads.*` / `project_layers.*` 被丢弃，任务框架可以换回 `QwenOFT` / `QwenPI_v3` / `QwenGR00T` 任一单头。

### 3.2 (a) 冻结规则

规则语法（逗号分隔或 yaml 列表；正则内不能含逗号，需要时用列表形式）：

```
qwen_vl_interface.model.model.visual                          精确路径（原生行为）
re:^qwen_vl_interface\.model\.model\.visual\.                 正则，匹配 named_parameters() 的参数名
qwen_vl_interface.model.model.language_model.layers[0:18]     ModuleList 区间（Python 切片语义，支持 [:18]、[-2:]）
qwen_vl_interface.model.model.language_model.layers[0:18].mlp 区间 + 子路径
llm_layers_below:18                                           等价于 <llm_layers_path>[0:18]
```

默认 `llm_layers_path = qwen_vl_interface.model.model.language_model.layers`，已用 transformers 4.57 的 `Qwen3VLForConditionalGeneration` 结构核对（`model.visual`、`model.language_model.{embed_tokens,layers}`），Qwen2.5-VL 在 ≥4.52 版同样如此；其他骨干请 `print(model)` 后用 `llm_layers_path` 覆盖。

三种使用方式：

1. **不改 StarVLA、不打补丁**：用 `expand_to_exact_paths(model, "…visual,llm_layers_below:18")` 把规则展开为 19 条精确路径写进 yaml 的 `trainer.freeze_modules`（正则无法展开）。
2. **零改动打补丁**：在自己的启动脚本里
   ```python
   import starVLA.training.train_starvla_cotrain as train_mod
   from starVLA.vlact_ext.freeze_rules import install_into_starvla
   install_into_starvla(train_mod)   # 替换 TrainerUtils.freeze_backbones 与 train_mod.build_param_lr_groups
   # 之后照抄 train_starvla_cotrain.py 的 __main__（OmegaConf.load / normalize_dotlist_args / apply_config_compat / main(cfg)）
   ```
   之后 yaml 可直接写任意规则形式，并支持独立键 `trainer.freeze_llm_layers_below: 18`、`trainer.llm_layers_path`。补丁版 `build_param_lr_groups` 会把该键折叠进 `cfg.trainer.freeze_modules`，保证随后 `freeze_backbones(model, cfg.trainer.freeze_modules)` 冻结同一组参数——这解决了报告 §10.1 指出的"冻结字串在两处各解析一次"的耦合。
3. **永久集成**：把 `trainer_tools.py` 中 `freeze_backbones` 的循环体和 `build_param_lr_groups` L117-125 的循环替换为 `resolve_frozen_param_ids(model, freeze_modules)`。

### 3.3 (d) 20 维统一布局

槽位（0-based；论文按 1 计数）：0-5 左臂关节、6-11 右臂关节、12-17 单臂 delta EE、18 共享夹爪（Franka 夹爪 = AgileX 左夹爪）、19 右夹爪。内置两个本体：

```python
DEFAULT_LAYOUTS = {
    "franka": EmbodimentLayout(slots=(12,13,14,15,16,17,18)),                         # LIBERO 7 维
    "agilex": EmbodimentLayout(slots=(0,...,5, 6,...,11, 18, 19), periodic=range(12)),  # RoboTwin 14 维
}
```

加新本体只需一个 dict 条目（`slots[i]` 是原生第 i 维的目标槽位，`periodic` 是需要 wrap 的原生维）。`robot_tag` 键必须等于 StarVLA 样本里的 `robot_tag`（即 DataConfig 的 `embodiment_tag.value`）：LIBERO 是 `franka`，而 `AgilexDataConfig` 用的是 `new_embodiment`，所以示例 yaml 写 `new_embodiment: {preset: agilex}`。

三条接入路径（任选）：

* **框架内自动映射（零改动）**：yaml `framework.unified_layout.enabled: true`。样本 `action` 维度 ≠ `action_dim` 时，`Qwen_MultiHead._collect_targets` 按 `robot_tag` 调 `to_unified` 并生成 mask；batch 是 python list，所以 Franka 7 维与 AgileX 14 维可以同批。缺点：训练器 `eval_action_model` 会对原始 `example["action"]` 做 `np.array`，混合维度时报错，请把 `eval_interval` 设大或改用下一条。
* **数据侧 transform**：在 `data_registry/data_config.py` 的 DataConfig 上加
  ```python
  make_dataset = staticmethod(make_dataset_hook(UnifiedActionTransform(UnifiedActionLayout.from_config(...))))
  ```
  `lerobot_datasets.make_LeRobotSingleDataset` 检测到 `make_dataset` 就会用它；返回的 `LeRobotSingleDataset` 子类在 `_pack_sample` 里应用 transform，`LeRobotMixtureDataset.__getitem__` 直接调 `_pack_sample`，因此混合数据集也生效。样本从此带 `action[T,20]`、`action_mask[T,20]`、`periodic_mask[T,20]`（与 `QwenOFT.masked_l1_loss` 已消费的 `[T,D]` 约定一致）。
* **任意 map-style 数据集**：`TransformedDataset(dataset, UnifiedActionTransform(...))`，其余属性（`save_dataset_statistics` 等）透明转发。

归一化仍按本体在原生 key 上做（`dataset_statistics.json` 不变）；统一布局作用于归一化之后。`state` 不做统一（混合本体请 `include_state: false`，或自行统一）。`UnifiedActionTransform(wrap_period=...)` 可选地在数据侧对 periodic 维做 `wrap_to_pi`（配方 (e) 的式 (1)）。

### 3.4 (e) wrap-aware L1

```python
masked_wrap_aware_l1(pred, target, active_mask=None, periodic_mask=None, period=2*pi)
```

* `δ = ((â−a)+p/2) mod p − p/2` 只作用于 `periodic_mask` 为 True 的维，其余维普通 L1；对 `active_mask` 做掩码均值；`active_mask` 全 0 返回 0（不产生 NaN，且保留计算图）；`periodic_mask=None` 退化为 `QwenOFT.masked_l1_loss`。numpy 版 `masked_wrap_aware_l1_np` 同语义。
* **`period` 必须按动作所在空间给**：StarVLA 样本里的动作已归一化。关节若用固定界 `[-π,π]→[-1,1]` 的 min_max，周期为 2.0（框架默认）。`AgilexDataConfig` 用数据集统计的 min/max，此时每维周期为 `4π/(max_d−min_d)`，请给 `period` 传长度 20 的列表（`wrap_to_pi` / `masked_wrap_aware_l1` 都接受逐维周期），或改 DataConfig 让关节 key 不归一化（原生弧度，`period=2π`）。
* **PI/GR00T 上的用法**：速度目标 `a−ε` 不是周期量，wrap 必须作用于最终生成样本。训练期不跑 4 步采样，而是用一步估计 `x1_hat = x_t + (1−t)·v̂`（`flow_matching_sample_estimate`）加 `fm_sample_loss_weight · masked_wrap_aware_l1(x1_hat, a, active∧periodic, periodic, period)`；速度 MSE 仍在全部激活维上算。`QwenMultiHead` 已内置（`framework.wrap_aware`），单头框架可参照 `multihead_framework.flow_matching_loss` 接入。

### 3.5 (b)(f) 已支持项的写法

见 `configs/vlact_pretrain_example.yaml` 中 `datasets.vlm_data`、`trainer.loss_scale.vlm: 0.5`（需用 `train_starvla_cotrain.py`）以及注释掉的 `pretrained_checkpoint` / `reload_modules` / `freeze_modules: ""` 段。注意 `vlm_data.model_type` 要与骨干匹配（Qwen3-VL → `qwen3vl`）。

## 4. yaml 怎么写

`configs/vlact_pretrain_example.yaml` 是完整的持续预训练配置：19 条路径的浅层冻结（或 `llm_layers_below:18`）、`loss_scale.vlm 0.5`、双 DataLoader、三头开关与权重、20 维布局与 wrap-aware 参数。已在 CPU 上核对：yaml 可被 OmegaConf 加载；与 `QwenMultiHeadDefaultConfig` 深合并后，每个头的 `action_model` 视图包含真实构造函数读取的全部字段（OFT: `action_model_type/action_hidden_dim/action_dim/action_horizon`；GR00T: `DiT-B` + `diffusion_model_cfg` 全字段、`hidden_size`、`noise_*` 等；PI: `populate_layerwise_dit_cfg` 后 `num_layers=36 / input_embedding_dim=1024 / cross_attention_dim=1024 / num_attention_heads=16`）；用 StarVLA 的 `AccessTrackedConfig` 包装后，`save_accessed_config` 导出的 `config.yaml` 保留完整的 `heads` / `action_model` / `wrap_aware` / `unified_layout` 子树，`from_pretrained` 能重建同一结构。

`data_mix: vlact_franka_agilex` 需要你在任一 `examples/**/train_files/data_registry/` 下注册 `DATASET_NAMED_MIXTURES` 条目，本包不含数据注册。

## 5. 哪些行为只在 CPU/mock 上验证、需要 GPU 复核

* 用真实 `get_vlm_model`（Qwen3-VL-4B 权重）与真实三头构造 `Qwen_MultiHead`，以及 `forward`/`predict_action` 在 bf16 autocast 下的数值与显存（三头 + `repeated_diffusion_steps` 会显著增加 DiT 前向次数）。
* `flow_matching_loss` 与 `FlowmatchingActionHead.forward` / `LayerwiseFlowmatchingActionHead.forward` 的逐项等价（代码逐行对照过，但没有在真实 DiT 上做数值对比；目标 tensor 同样用 `last_hidden.dtype`，DeepSpeed bf16 下与头的权重 dtype 一致，但 loss 归约升到 fp32 计算）。
* `install_into_starvla` 在 Accelerate/DeepSpeed 启动流程中的顺序（`setup_optimizer_and_scheduler` 先于 `prepare_training` 里的冻结，与原实现一致，但未实跑）。
* `make_dataset_hook` 返回的 `LeRobotSingleDataset` 子类在 `LeRobotMixtureDataset` 中的行为（依赖真实 LeRobot 数据）。
* 部署链路：`server_policy.py` 透传 `head` / `robot_tag` → `from_unified` → `PolicyNormProcessor.unapply_actions`。
* wrap-aware 项对训练效果的影响、`fm_sample_loss_weight` / 各头 `loss_weight` 的取值均未调参。

## 6. 与报告第 9 章伪代码的差异

* §9.1：除 `re:` 与 `[lo:hi]` 外增加 `llm_layers_below:N` 规则 token 与 `[lo:hi].sub.path` 后缀；正则匹配参数名（与伪代码一致），因此 `expand_to_exact_paths` 明确拒绝正则；提供 monkeypatch 而不是直接改 `trainer_tools.py`。
* §9.3：头存放在 `nn.ModuleDict` `heads.{oft,gr00t,pi}` 而非 `oft_head`/`groot_head`/`pi_head` 三个属性；输出键按任务要求为 `loss_oft/loss_pi/loss_gr00t`；`repeated_diffusion_steps` 按头从 `framework.heads.<name>.action_model` 读（不再像 QwenPI_v3 那样从 `trainer` 读）；state 走文本而非 GR00T 的 `state_encoder`；两个 FM 头的 mask 支持在框架内实现（伪代码假定改头的 `forward` 加 `action_mask`）。
* §9.4：没有实现 `ModalityTransform` 级的 `UnifiedActionLayoutTransform`（那需要改 `_pack_sample`），改为作用于 `_pack_sample` 输出的样本级 transform + `make_dataset` 钩子 + 框架内兜底映射；`periodic_mask` 与 `action_mask` 一起由布局生成，不需要 DataConfig 声明 `periodic_action_keys`；部署侧 `PolicyNormProcessor` 未改，由 `predict_action(robot_tag=...)` 先还原原生维度。
* §9.5：没有新增 `wrap_pi` 归一化模式（要改 `Normalizer`），用 `UnifiedActionTransform(wrap_period=...)` 代替数据侧 wrap；`period` 支持逐维数组以适配 min_max 统计归一化。
