#!/usr/bin/env bash
# Compile report/awesome_starvla_slides.tex with XeLaTeX (2 passes) and report overfull boxes > 10pt.
set -euo pipefail
cd "$(dirname "$0")/../report"
xelatex -interaction=nonstopmode -halt-on-error awesome_starvla_slides.tex > build.log 2>&1
xelatex -interaction=nonstopmode -halt-on-error awesome_starvla_slides.tex > build.log 2>&1
echo "pages: $(pdfinfo awesome_starvla_slides.pdf | awk '/Pages/{print $2}')"
echo "overfull > 10pt:"
grep -E 'Overfull \\hbox \([0-9.]+pt' build.log | sed -E 's/.*\(([0-9.]+)pt.*/\1/' | awk '$1 > 10' | sort -rn | uniq -c || true
rm -f awesome_starvla_slides.{aux,log,nav,out,snm,toc,vrb} build.log
