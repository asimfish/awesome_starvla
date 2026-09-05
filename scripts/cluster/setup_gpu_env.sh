#!/usr/bin/env bash
# Self-contained GPU environment for StarVLA + vlact_ext + starvla_lab on a cluster node (Linux, NVIDIA).
#
#   bash setup_gpu_env.sh [VENV_DIR] [STARVLA_DIR]
#
# Defaults: VENV_DIR=/home/dataset-assist-0/liyufeng/envs/starvla, STARVLA_DIR=$WORK/starVLA where
# WORK=/home/dataset-assist-0/liyufeng/awesome_starvla_work. Versions follow StarVLA's requirements.txt
# (torch 2.6.0+cu124, transformers 4.57.0, accelerate 1.5.2, numpy 1.26.4). Caches go next to the venv so
# the (small) root filesystem is not touched. Needs `uv` on PATH or python3.10+.
set -euo pipefail

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
VENV="${1:-/home/dataset-assist-0/liyufeng/envs/starvla}"
STARVLA="${2:-$WORK/starVLA}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$VENV")/.uv_cache}"
export TMPDIR="${TMPDIR:-$(dirname "$VENV")/.tmp}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

[ -f "$STARVLA/pyproject.toml" ] || { echo "StarVLA not found at $STARVLA (run sync_to_node.sh first)" >&2; exit 1; }

PKGS=("torch==2.6.0" "torchvision==0.21.0" "transformers==4.57.0" "accelerate==1.5.2" tiktoken einops scipy pillow
      tensorboard matplotlib "pydantic==2.10.6" "numpydantic==1.6.9" pyarrow fastparquet omegaconf "numpy==1.26.4"
      rich diffusers timm tyro websockets pyzmq qwen-vl-utils "wandb<0.19" "protobuf<5" pytest)

# INDEX_URL: PyPI mirror. On the tianyiyun nodes files.pythonhosted.org is throttled to ~3 KB/s while the
# Tsinghua mirror delivers ~15 MB/s, so the mirror is the default there.
INDEX_URL="${INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
INDEX_ARGS=()
[ -n "$INDEX_URL" ] && INDEX_ARGS=(--index-url "$INDEX_URL")

UV="$(command -v uv || true)"
if [ -n "$UV" ]; then
  [ -x "$VENV/bin/python" ] || "$UV" venv --python 3.11 "$VENV"
  "$UV" pip install --python "$VENV/bin/python" "${INDEX_ARGS[@]}" -e "$STARVLA" "${PKGS[@]}"
else
  PY="$(command -v python3.11 || command -v python3.10 || command -v python3)"
  [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install -q "${INDEX_ARGS[@]}" --upgrade pip
  "$VENV/bin/pip" install -q "${INDEX_ARGS[@]}" -e "$STARVLA" "${PKGS[@]}"
fi

"$VENV/bin/python" - <<'EOF'
import sys, torch, transformers, starVLA
print(f"ok: python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} "
      f"available={torch.cuda.is_available()} transformers {transformers.__version__}")
print("    starVLA ->", starVLA.__file__)
EOF
echo "PY=$VENV/bin/python"
