#!/usr/bin/env bash
# Compile both Beamer decks in report/ with XeLaTeX (2 passes) and report overfull boxes > 10pt.
set -euo pipefail
cd "$(dirname "$0")/../report"
for deck in awesome_starvla_slides action_heads_lecture_slides; do
  xelatex -interaction=nonstopmode -halt-on-error "$deck.tex" > build.log 2>&1
  xelatex -interaction=nonstopmode -halt-on-error "$deck.tex" > build.log 2>&1
  echo "$deck: $(pdfinfo "$deck.pdf" | awk '/Pages/{print $2}') pages; overfull > 10pt:"
  grep -E 'Overfull \\hbox \([0-9.]+pt' build.log | sed -E 's/.*\(([0-9.]+)pt.*/\1/' | awk '$1 > 10' | sort -rn | uniq -c || true
  rm -f "$deck".{aux,log,nav,out,snm,toc,vrb} build.log
done
