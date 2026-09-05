#!/usr/bin/env bash
# F3: WP2 learning-rate control on the three-head model (one GPU, 2 x ~15 min). Same data / steps / probe as F0 v3,
# so the runs compare directly with f0v3_multihead (hard freeze of visual + LLM layers 0-17, uniform 1e-5 above):
#
#   a) f3_mh_llrd085     R4-style: no hard freeze, static layer-wise decay 0.85 (layer i lr = 1e-5 * 0.85^(36-i);
#                        vision encoder and embeddings at 1e-5 * 0.85^37), probes record only.
#   b) f3_mh_driftllrd   R5-style: F0 freeze + drift-driven LLRD (decay 1.0, drift_high 1e-2 / drift_low 1e-3 from the
#                        F0 v3 scale, down 0.5 / up 1.1 / min 0.05) acting at every probe (25 updates).
#
#   PY=<env python> bash run_f3_llrd.sh [extra --dotlist overrides...]
set -euo pipefail

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
RUNNER="$WORK/awesome_starvla/scripts/cluster/run_f0_smoke.sh"
COMMON=(--framework.action_model.state_dim 0 --trainer.lab.llrd.enabled true)

echo "[f3] === f3_mh_llrd085 ==="
PY="$PY" WORK="$WORK" bash "$RUNNER" QwenMultiHead f3_mh_llrd085 "${COMMON[@]}" \
  --trainer.freeze_modules "" --trainer.lab.llrd.decay 0.85 \
  "$@" > "$WORK/logs/f3_mh_llrd085.log" 2>&1 || echo "[f3] FAILED: f3_mh_llrd085"
echo "[f3] done: f3_mh_llrd085"

echo "[f3] === f3_mh_driftllrd ==="
PY="$PY" WORK="$WORK" bash "$RUNNER" QwenMultiHead f3_mh_driftllrd "${COMMON[@]}" \
  --trainer.lab.llrd.decay 1.0 --trainer.lab.llrd.drift_driven true \
  --trainer.lab.llrd.drift_high 1.0e-2 --trainer.lab.llrd.drift_low 1.0e-3 \
  --trainer.lab.llrd.down_factor 0.5 --trainer.lab.llrd.up_factor 1.1 --trainer.lab.llrd.min_scale 0.05 \
  --trainer.lab.probes.calibrate_only false \
  "$@" > "$WORK/logs/f3_mh_driftllrd.log" 2>&1 || echo "[f3] FAILED: f3_mh_driftllrd"
echo "[f3] done: f3_mh_driftllrd"
echo "[f3] ALL_DONE"
