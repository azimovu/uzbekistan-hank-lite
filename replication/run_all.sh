#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"

echo "=== 01 build national series, CPI deflation, SEs, survey forensics, attrition ==="
"$PYTHON_BIN" 01_build_series.py
echo "=== 02 transitions and subperiod regimes (clustered SEs) ==="
"$PYTHON_BIN" 02_transitions_and_regimes.py
echo "=== 03 income distribution, equivalence checks, and lag-12/24/48 PPT ==="
"$PYTHON_BIN" 03_income_process.py
echo "=== 04 feedback, Round/HH-FE job-loss exposure, exact permutation checks ==="
"$PYTHON_BIN" 04_feedback_and_exposure.py
echo "=== 07 audience, weighted HMM misclassification, M1 validation, lagged exposure, event study, remittances ==="
"$PYTHON_BIN" 07_diagnostics.py
echo "=== 05 IBC-consistent 3-state Markov-TANK CRN microsimulation ==="
"$PYTHON_BIN" 05_model_microsim.py

FIG_PYTHON="${FIG_PYTHON:-$PYTHON_BIN}"
if "$FIG_PYTHON" -c "import matplotlib" 2>/dev/null; then
  echo "=== 06 generate publication figures ==="
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}" "$FIG_PYTHON" 06_make_figures.py
else
  echo "WARNING: matplotlib not found in $FIG_PYTHON; set FIG_PYTHON to a Python with matplotlib to build figures." >&2
fi

echo "=== ALL REPLICATION STEPS COMPLETE ==="
ls -lh ../outputs/
