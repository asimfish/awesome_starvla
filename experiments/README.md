# experiments/：运行清单与结果台账

本目录只放**清单与结果**，不放代码；代码在 `code/starvla_lab/`，方案在 [`reports/10_improvement_plan.md`](../reports/10_improvement_plan.md)。

## 文件

| 文件 | 由谁生成 | 内容 |
|---|---|---|
| `run_matrix.csv` | `python3 scripts/build_run_matrix.py` | R0–R8 九个骨干 × F1 协议的 6 个基准设定（LIBERO-plus、RoboTwin Base、RoboCasa-GR1 的 10/20/50/100%）× 3 seeds = 162 次下游微调；每行含 `run_id`、`overrides`（JSON，StarVLA 点路径覆盖项）、`status` |
| `results/<run_id>.json` | 评测脚本（人工或 CI）写入 | 一次运行一个文件，字段见下 |
| `results/summary.md` | `python3 -c "from starvla_lab.bench import *; ..."`（见下） | 由 `summarize_results` 聚合成 mean ± std 表 |

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
- checkpoint 规则固定为"最后一个"，不挑最优。
- 每个骨干同时报"同头"（OFT）与"跨头"（PI、GR00T）——跨头微调另建两份矩阵（把 `protocol.head` 改为 `QwenPI_v3` / `QwenGR00T` 重新生成）。
- R9（EventVLA × VLAct 骨干）走 EventVLA 自己的协议，不在本清单内；结果同样以 JSON 放进 `results/`，`benchmark` 写 `robotwin_mem` / `rmbench`。
