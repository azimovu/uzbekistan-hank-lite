#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode preprint_uzbekistan_hank_lite.tex
bibtex preprint_uzbekistan_hank_lite
pdflatex -interaction=nonstopmode preprint_uzbekistan_hank_lite.tex
pdflatex -interaction=nonstopmode preprint_uzbekistan_hank_lite.tex
pdflatex -interaction=nonstopmode preprint_uzbekistan_hank_lite.tex

echo "Built paper/preprint_uzbekistan_hank_lite.pdf cleanly."
