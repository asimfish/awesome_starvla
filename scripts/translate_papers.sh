#!/usr/bin/env bash
# Translate papers/en/*.pdf into papers/zh/*_zh.pdf with super_translate (paper-translate skill, DeepSeek backend).
# Requires: a clone of https://github.com/asimfish/super_translate with `uv sync` done, and
#           PAPER_CHINA_DEEPSEEK_API_KEY (or DEEPSEEK_API_KEY) exported.
set -euo pipefail
SUPER_TRANSLATE_HOME="${SUPER_TRANSLATE_HOME:-$HOME/Desktop/research/super_translate}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SUPER_TRANSLATE_HOME"
for pdf in "$ROOT"/papers/en/*.pdf; do
  name="$(basename "${pdf%.pdf}")"
  out="$ROOT/papers/zh/${name}_zh.pdf"
  [ -f "$out" ] && { echo "skip $name (exists)"; continue; }
  echo "== translating $name"
  bash skills/paper-translate/scripts/translate_one.sh "$pdf" "$out" || echo "QA reported issues for $name; see ${out%.pdf}.inspect.json"
done
