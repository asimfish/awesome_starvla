#!/usr/bin/env bash
# Create a CPU-only Python 3.12 environment that can import StarVLA + vlact_ext + starvla_lab.
#
#   bash scripts/setup_cpu_env.sh [VENV_DIR] [STARVLA_DIR]
#
# Defaults: VENV_DIR=.venv-starvla (inside this repo, git-ignored), STARVLA_DIR=../starVLA_code
# (a checkout of https://github.com/starVLA/starVLA next to this repo). Needs `uv`
# (https://docs.astral.sh/uv/) or python3.10+ on PATH.
#
# StarVLA requires Python >= 3.10 (it uses `str | None` annotations); the macOS system python3.9 can
# only run the mock-backed unit tests, not the StarVLA integration smoke test.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO/.venv-starvla}"
STARVLA="${2:-$REPO/../starVLA_code}"

if [ ! -f "$STARVLA/pyproject.toml" ]; then
  echo "StarVLA checkout not found at $STARVLA" >&2
  echo "  git clone https://github.com/starVLA/starVLA \"$STARVLA\"" >&2
  exit 1
fi

# CPU wheels. transformers==4.57.0 is what StarVLA's requirements.txt pins (PyPI marks it yanked but it
# installs and is the version the Qwen3-VL code paths were written against). wandb<0.19 / protobuf<5 avoid a
# wandb import error.
PKGS=(torch torchvision "transformers==4.57.0" accelerate omegaconf einops timm diffusers qwen-vl-utils
      tyro rich pydantic numpydantic scipy websockets pyzmq tiktoken pyyaml pillow matplotlib tensorboard
      pytest "wandb<0.19" "protobuf<5" pyarrow fastparquet)

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" -e "$STARVLA" "${PKGS[@]}"
else
  PY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)"
  [ -n "$PY" ] || { echo "need uv or python3.10+ on PATH" >&2; exit 1; }
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -e "$STARVLA" "${PKGS[@]}"
fi

"$VENV/bin/python" - <<'EOF'
import sys, torch, transformers, starVLA
print(f"ok: python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}")
print(f"    starVLA -> {starVLA.__file__}")
EOF

rel() { python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$REPO"; }
VENV_REL="$(rel "$VENV")"; STARVLA_REL="$(rel "$STARVLA")"
cat <<EOF

Next (from $REPO):
  PYTHONPATH=code:$STARVLA_REL $VENV_REL/bin/python -m pytest code/vlact_ext/tests code/starvla_lab/tests -q
  PYTHONPATH=code:$STARVLA_REL $VENV_REL/bin/python scripts/smoke_starvla_integration.py
EOF
