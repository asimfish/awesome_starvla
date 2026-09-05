#!/usr/bin/env bash
# One-shot cluster job: fetch code, verify the environment, run the WP6 GPU overhead benchmark.
#
#   PY=/path/to/env/bin/python bash run_overhead_bench.sh [extra gpu_overhead_bench.py args]
#
# Environment (all optional except PY):
#   PY          python with torch + transformers 4.57 + diffusers/timm/accelerate/omegaconf (a StarVLA env)
#   WORK        scratch dir on the shared filesystem            (default /home/dataset-assist-0/liyufeng/awesome_starvla_work)
#   BASE_VLM    Qwen3-VL-4B-Instruct directory                  (default wangpeishuo/Pretrained_models/Qwen3-VL-4B-Instruct)
#   STARVLA_REF StarVLA commit to pin (branch starVLA_dev)      (default d81fc66 = snapshot analysed in reports/02)
#   SKIP_SMOKE  set to 1 to skip scripts/smoke_starvla_integration.py
#
# Designed to be re-runnable: clones are reused, awesome_starvla is fast-forwarded, results get a new
# timestamped directory under $WORK/results/. Uses exactly the GPU(s) in CUDA_VISIBLE_DEVICES.
set -euo pipefail

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to a python interpreter that has torch + transformers 4.57}"
BASE_VLM="${BASE_VLM:-/home/dataset-assist-0/wangpeishuo/Pretrained_models/Qwen3-VL-4B-Instruct}"
STARVLA_REF="${STARVLA_REF:-d81fc66}"

echo "[job] host=$(hostname) user=$(whoami) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} PY=$PY"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
mkdir -p "$WORK" && cd "$WORK"

# Code can arrive either by git (nodes with GitHub access) or by rsync from the workstation
# (scripts/cluster/sync_to_node.sh); both layouts are accepted here.
if [ ! -d starVLA ]; then
  git clone -q --single-branch -b starVLA_dev https://github.com/starVLA/starVLA starVLA
fi
if [ -d starVLA/.git ]; then
  git -C starVLA fetch -q origin starVLA_dev 2>/dev/null || true
  git -C starVLA checkout -q "$STARVLA_REF" 2>/dev/null || true
  echo "[job] starVLA @ $(git -C starVLA rev-parse --short HEAD)"
fi

if [ ! -d awesome_starvla ]; then
  git clone -q https://github.com/asimfish/awesome_starvla awesome_starvla   # no submodules needed
fi
if [ -d awesome_starvla/.git ]; then
  git -C awesome_starvla pull -q --ff-only 2>/dev/null || true
  echo "[job] awesome_starvla @ $(git -C awesome_starvla rev-parse --short HEAD)"
else
  echo "[job] awesome_starvla @ $(cat awesome_starvla/.git_head 2>/dev/null || echo unknown) (rsync copy)"
fi

export PYTHONPATH="$WORK/awesome_starvla/code:$WORK/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
cd awesome_starvla

"$PY" - <<'EOF'
import sys, torch, transformers
print(f"[env] python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} transformers {transformers.__version__}")
for mod in ("accelerate", "diffusers", "timm", "omegaconf", "einops", "qwen_vl_utils"):
    try:
        m = __import__(mod); print(f"[env] {mod} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"[env] MISSING {mod}: {e}")
import starVLA; print("[env] starVLA ->", starVLA.__file__)
print("[env] gpu:", torch.cuda.get_device_name(0), f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
EOF

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  "$PY" scripts/smoke_starvla_integration.py
fi

OUT="$WORK/results/overhead_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/gpu_overhead_bench.py --base_vlm "$BASE_VLM" --out "$OUT" "$@"
echo "[job] results in $OUT"
