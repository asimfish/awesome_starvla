# experiments/：运行清单与结果台账

本目录只放**清单与结果**，不放代码；代码在 `code/starvla_lab/`，方案在 [`reports/10_improvement_plan.md`](../reports/10_improvement_plan.md)。

## 文件

| 文件 | 由谁生成 | 内容 |
|---|---|---|
| `run_matrix.csv` | `python3 scripts/build_run_matrix.py` | 主矩阵（OFT 头，92 次）：core 级变体（R0/R1/R3/R8）3 seeds + GR1 四个比例，extra 级 1 seed + 两个比例；每行含 `run_id`、`overrides`（JSON，StarVLA 点路径覆盖项）、`eval_cmd_template`、`est_gpu_hours`、`status` |
| `run_matrix_QwenPI_v3.csv`、`run_matrix_QwenGR00T.csv` | 同上 | 跨头矩阵：core 级变体 × LIBERO-plus + RoboTwin Base × 1 seed（各 8 次） |
| `budget.md` | 同上 | 预训练 + 下游的 GPU 小时合计（当前约 21,000；最小子集约 9,000） |
| `results/<run_id>.json` | 评测脚本（人工或 CI）写入 | 一次运行一个文件，字段见下 |
| `results/summary.md` | `python3 -c "from starvla_lab.bench import *; ..."`（见下） | 由 `summarize_results` 聚合成 mean ± std 表 |
| `results/f0_libero_goal_smoke/` | `scripts/cluster/run_f0_smoke.sh` → `scripts/analyze_f0.py` / `scripts/probe_diagnostics.py` | **已有数字**：LIBERO-goal 上 `QwenOFT` vs `QwenMultiHead` 各 300 步的损失曲线、逐头损失、探针漂移与探针方法学诊断（1×A100，2026-09-06），含 `README.md` 解读 |
| `results/wp6_overhead/` | `scripts/gpu_overhead_bench.py`（经 `scripts/cluster/run_overhead_bench.sh`） | **已有数字**：Qwen3-VL-4B 上单头 / 三头 / 头 dropout 的 s/step、samples/s、峰值显存与推理延迟（1×A100，2026-09-05），含 `overhead.csv`、`results.json`、`stdout.log` 与解读 `README.md` |

## 结果 JSON 约定

```json
{
  "run_id": "bb-R3__libero_plus__s0",
  "backbone": "R3",
  "benchmark": "libero_plus",
  "data_fraction": 1.0,
  "seed": 0,
  "success_rate": 82.6,
  "camera": 73.9, "robot": 68.4, "language": 81.5, "light": 96.7, "background": 96.7, "noise": 86.0, "layout": 83.3,
  "checkpoint": "playground/Checkpoints/backbone_bench/bb-R3__libero_plus__s0/checkpoints/steps_30000_pytorch_model.pt",
  "steps": 30000,
  "commit": "<starVLA git sha>",
  "notes": ""
}
```

必填：`run_id`、`backbone`、`benchmark`、`seed` 与至少一个指标；RoboTwin 必须同时给 `success_rate_clean` 与 `success_rate_random`；数据比例实验给 `data_fraction`。

## 汇总

```bash
cd awesome_starvla
python3 - <<'PY'
import sys; sys.path.insert(0, "code")
from starvla_lab.bench import summarize_results, format_summary_table
s = summarize_results("experiments/results", metric="success_rate")
open("experiments/results/summary.md", "w").write(format_summary_table(s))
print(format_summary_table(s))
PY
```

## 协议提醒（F1）

- 只换骨干：`run_matrix.csv` 里各行 `overrides` 的差异只能是 `framework.qwenvl.base_vlm` / `trainer.pretrained_checkpoint` / `trainer.reload_modules` / `seed` / `run_id` / `datasets.vla_data.data_fraction`。`scripts/build_run_matrix.py` 会打印实际变化的键，多出任何键都要解释。
- 训练入口统一是 `-m starvla_lab.train.train_starvla_lab`（`trainer.lab.mode: single`），它与 StarVLA 原生 `train_starvla.py` 的唯一差别是 `data_fraction < 1` 时按轨迹子采样；需要 `PYTHONPATH` 包含本仓库的 `code/`。
- 评测命令由 `protocol_f1.yaml` 的模板渲染：LIBERO-plus 用 `your_ckpt=... bash eval_libero.sh`（需设 `LIBERO_HOME`），RoboTwin 用 `start_eval.sh -m demo_clean` 与 `-m demo_randomized` 各跑一次，GR1 用 `batch_eval_args.sh <ckpt> 1 720 12`。
- checkpoint 规则固定为"最后一个"，不挑最优。
- 每个骨干同时报"同头"（OFT）与"跨头"（PI、GR00T）——跨头矩阵由脚本按 core 级自动生成（`run_matrix_QwenPI_v3.csv`、`run_matrix_QwenGR00T.csv`）。
- R9（EventVLA × VLAct 骨干）走 EventVLA 自己的协议，不在本清单内；结果同样以 JSON 放进 `results/`，`benchmark` 写 `robotwin_mem` / `rmbench`。
