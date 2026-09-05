#!/usr/bin/env bash
# rsync this repo's code (no PDFs / slides / submodule / venv) and the StarVLA checkout to a cluster node
# that cannot reach GitHub. Companion of run_overhead_bench.sh.
#
#   bash scripts/cluster/sync_to_node.sh <ssh-host> [WORK_DIR] [STARVLA_DIR]
#
# Defaults: WORK_DIR=/home/dataset-assist-0/liyufeng/awesome_starvla_work, STARVLA_DIR=../starVLA_code
set -euo pipefail

HOST="${1:?ssh host alias}"
WORK="${2:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA="${3:-$REPO/../starVLA_code}"

git -C "$REPO" rev-parse --short HEAD > "$REPO/.git_head"
ssh -o ConnectTimeout=25 "$HOST" "mkdir -p '$WORK/awesome_starvla' '$WORK/starVLA'"

rsync -az --delete -e "ssh -o ConnectTimeout=25" \
  --exclude '.git' --exclude '.venv-starvla' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'code/EventVLA' --exclude 'papers' --exclude 'report' --exclude 'assets/*.pdf' --exclude '*.pptx' --exclude '*.pdf' \
  "$REPO/" "$HOST:$WORK/awesome_starvla/"

# StarVLA with its .git so the commit can be pinned/recorded on the node; skip caches and checkpoints.
rsync -az --delete -e "ssh -o ConnectTimeout=25" \
  --exclude '__pycache__' --exclude '*.egg-info' --exclude 'playground' --exclude 'results' --exclude 'checkpoints' --exclude 'wandb' \
  "$STARVLA/" "$HOST:$WORK/starVLA/"

ssh -o ConnectTimeout=25 "$HOST" "du -sh '$WORK/awesome_starvla' '$WORK/starVLA' && cat '$WORK/awesome_starvla/.git_head' && git -C '$WORK/starVLA' rev-parse --short HEAD"
