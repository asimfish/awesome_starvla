# F5 · 放大一档：LIBERO-goal + LIBERO-spatial 两套共训 2000 步（1 卡，fp32 主权重）

**状态：排队中（2026-09-12 重排）。** 配置 [`code/starvla_lab/configs/f5_libero_2suite_2k.yaml`](../../../code/starvla_lab/configs/f5_libero_2suite_2k.yaml)，
运行脚本 [`scripts/cluster/run_f5_scale.sh`](../../../scripts/cluster/run_f5_scale.sh)，经单卡队列
[`scripts/cluster/run_queue_20260907.sh`](../../../scripts/cluster/run_queue_20260907.sh)（F5 → F3 fp32 → F2 PI fp32）由 `wait_for_gpu_and_run.sh 62000 queue0912 2880` 在 pub2 等一张 ≥ 62 GB 空闲的 A100。

## 1. 目的

方案 [§10](../../../reports/10_improvement_plan.md) 的核心待检假设：**多头共监督写进骨干的动作表征，会随任务多样性变通用**。F0–F4 都是 1 套 × 300 步；这里
goal + spatial 两套 × 2000 步（warmup 100、探针每 100 次更新、探针批 48 样本 / 30 条指令来自 goal + spatial + **object**），`QwenOFT` vs `QwenMultiHead`，然后

1. 跨头探针（`scripts/cross_head_probe.py`，3072 样本，查询位第 33–35 层）在 goal / spatial / **object（完全未见）** 上测查询位可读性、pooled 可读性、保留度与跨套迁移；
2. F2 协议：冻结 F5 骨干（+ 预训练锚点）接全新 OFT 头在 object 上训 300 步（`run_f2_transfer.sh f5x object`）。

判据（写在方案 §10）：若三头骨干在**未见的 object** 上的查询位可读性 / 迁移损失优于 OFT 骨干且差距大于 F4 的 300 步 regime，H1/配方 (c) 进入 R3；若两者相同或反向，多头共监督的"通用表征"主张降级为"不伤单头"。

## 2. 首跑事故（2026-09-09）与修复

首跑 09-09 03:40 拿到 GPU 0，`f5_oft` 以 1.07 s/step 跑到 **374/2000 步**（03:50）后日志再无更新，进程存活、GPU 利用率 0%、占卡 42.9 GB 共 69 小时，队列后两段一直没轮到。诊断（`raw/hung_20260909/`）：

- 4 个 DataLoader worker 各有 **2059–3211 个线程**，其中 2048–3200 个是 `dav1d-worker`（libdav1d 的 AV1 解码线程）；一个 worker 卡在 `poll`，主进程在 `futex` 里无超时地等 worker 队列。
- 根因：StarVLA `gr00t_lerobot/video.py` 的 `torchvision_av` 后端每次解码开一个 PyAV 容器，`InputContainer ↔ Stream ↔ CodecContext` 是引用环，`container.close()` 释放不了解码器的线程池（128 核节点上每个解码器 16 个线程），要等分代 GC 的第 1/2 代回收；worker 堆大、这类回收稀少，线程越积越多直到死锁。节点实测（LIBERO AV1 视频，64 次解码）：不回收 +640 线程（峰值 +1400）；`gc.collect(0)` 无效（对象已升代）；解码后 `gc.collect(1)` 线程数持平，每次 2 ms（解码本身 20 ms）。
- 修复：`starvla_lab/data/decoder_gc.py` 在 fork worker 之前把 `datasets.py` 里绑定的 `get_frames_by_timestamps` / `get_all_frames` 包成"解码后 `gc.collect(1)`"（`trainer.lab.decoder_gc_every`，默认 1，0 关闭；`cross_head_probe.py` 同样开启）。补丁后经真实模块 96 次解码线程数 128 → 128。
- 防御：`scripts/cluster/run_with_stall_guard.sh` 监视日志 mtime，20 分钟无变化整组杀掉并返回 124；`run_f5_scale.sh` 对停滞的训练自动重试一次（旧日志与探针 JSONL 归档为 `.stalledN`）。
- F0–F4 各 ≤ 500 步、线程数未到死锁门槛，结果不受影响；泄漏只影响线程数，不改变训练语义。

### 前 374 步的探针记录（token 级 1−CKA，换回预训练嵌入；OFT，fp32）

| 更新次数 | 第 35 层 | 第 34 层 | 第 30 层 | 全层均值 | pooled 均值（次指标） |
|---|---|---|---|---|---|
| 0 | 0.0000 | 0.00000 | 0.000000 | 0.00000 | 0.0000 |
| 100 | 0.0032 | 0.00010 | 0.000006 | 0.00009 | 0.0002 |
| 200 | 0.0152 | 0.00107 | 0.000086 | 0.00047 | 0.0064 |
| 300 | 0.0655 | 0.00346 | 0.000269 | 0.00198 | 0.0226 |

与 F4（单套 goal，300 步：第 35 层 0.026）相比，两套共训 300 步时第 35 层已到 0.066（单条运行、仅供参考；训练分布更宽时顶层改写可能更快，待完整运行确认）；第 0–17 层严格为 0（冻结集合含 `embed_tokens`）。OFT 头 L1 在 300–374 步区间 0.19–0.28（未收敛，仅供 sanity）。

## 3. 文件

- `raw/hung_20260909/f5_oft_lab_probes_partial.jsonl`：首跑前 300 步的 4 条探针记录（步 0/100/200/300）。
- `raw/hung_20260909/f5_oft_log_tail.txt`：首跑日志尾部（去掉进度条），最后一条为 374 步。
- 完成后：`raw/f5_{oft,mh}.log`、`raw/f5_{oft,mh}_lab_probes.jsonl`、`cross_head_f5.json`、`raw/f5x_*`、`f5_curves.png`、`f5x_curves.png`、`summary.md`。
