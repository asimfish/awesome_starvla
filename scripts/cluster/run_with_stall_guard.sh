#!/usr/bin/env bash
# Run a command with its output in <log_file> and kill the whole process tree if that log stops changing for
# <stall_min> minutes. Exit code: the command's own, or 124 when it was killed for stalling.
#
#   bash run_with_stall_guard.sh <log_file> <stall_min> -- <command...>
#
# Why: the 2026-09-09 F5 run deadlocked in a DataLoader worker (PyAV/dav1d thread leak, see
# starvla_lab/data/decoder_gc.py) and sat at 0 % GPU for 69 h holding the card and the queue. The trainer's tqdm bar
# touches the log every step and the probes finish within a few minutes, so a 20-minute silence means a hang.
set -uo pipefail
LOG="${1:?log file}"; STALL_MIN="${2:?stall minutes}"; shift 2
[ "${1:-}" = "--" ] && shift
[ $# -ge 1 ] || { echo "no command given"; exit 2; }

: > "$LOG"
setsid "$@" >> "$LOG" 2>&1 &      # own session/process group so the tree can be killed as a unit
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  sleep 60
  kill -0 "$pid" 2>/dev/null || break
  mtime=$(stat -c %Y "$LOG" 2>/dev/null || date +%s)
  age=$(( $(date +%s) - mtime ))
  if [ "$age" -ge $(( STALL_MIN * 60 )) ]; then
    echo "[stall_guard] $(date '+%F %T') $LOG unchanged for $(( age / 60 )) min; killing process group $pid" | tee -a "$LOG"
    kill -TERM -- "-$pid" 2>/dev/null; sleep 15
    kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit 124
  fi
done
wait "$pid"
exit $?
