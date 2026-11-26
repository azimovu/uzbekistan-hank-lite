# Hand-to-Mouth Households in Uzbekistan: Survey Forensics, Income Risk, and a Three-State Markov-TANK Calibration, 2018–2025

**Author:** Utkirbek Azimov (PhD Candidate, Termez State University, Termez, Uzbekistan)  
**ORCID:** [0000-0002-8088-324X](https://orcid.org/0000-0002-8088-324X) | **Contact:** `azimov2398@gmail.com`  
**Version:** `1.0.0` (November 2026)

This repository contains the working-paper preprint, derived aggregate outputs, publication figures, and complete open-source Python replication package for *Hand-to-Mouth Households in Uzbekistan: Survey Forensics, Income Risk, and a Three-State Markov-TANK Calibration, 2018–2025*.

## Preprint

- Compiled PDF: [`paper/preprint_uzbekistan_hank_lite.pdf`](paper/preprint_uzbekistan_hank_lite.pdf)
- LaTeX source & bibliography: [`paper/preprint_uzbekistan_hank_lite.tex`](paper/preprint_uzbekistan_hank_lite.tex), [`paper/refs_ecop.bib`](paper/refs_ecop.bib)

## Repository Layout

- `data/` — Open official monthly CPI files (`sdmx_cpi_1285.csv`, `sdmx_cpi_1286.csv` from `siat.stat.uz`) and [`data/README.md`](data/README.md) with download instructions and SHA-256 checksum for the World Bank L2CU v03 microdata (`UZB_2018-2025_L2CU_v03_M_CSV.zip`, Catalog 6412, not redistributed publicly).
- `replication/` — Self-contained modular Python pipeline (`common.py`, `01`–`07`, and one-command `run_all.sh`). Scripts `01`–`05` and `07` use only the Python 3.12+ standard library; `06_make_figures.py` requires `matplotlib`.
- `outputs/` — All generated aggregate CSV/JSON moment files (zero household-level PII) and five 300-DPI figures (`fig1`–`fig5`).
- `paper/` — Preprint LaTeX source, BibTeX bibliography, build script (`build_preprint.sh`), and compiled PDF.

## Full One-Command Reproduction

1. Download `UZB_2018-2025_L2CU_v03_M_CSV.zip` from the [World Bank Microdata Library (Catalog 6412)](https://microdata.worldbank.org/index.php/catalog/6412) and place it in `data/` (see [`data/README.md`](data/README.md)).
2. Run the full replication pipeline and rebuild the preprint PDF:

```bash
./replication/run_all.sh
./paper/build_preprint.sh
```

If `matplotlib` is installed in a separate virtual environment, pass `FIG_PYTHON=/path/to/python ./replication/run_all.sh`.

## Pipeline Overview

- `01_build_series.py` — 82-round national HtM series (`outputs/national_series.csv`), CPI deflation (`2018-09 = 100`), Kish and household-clustered SEs, forensic audit of the round-44 food-sufficiency anomaly, rounds 58–65 and 78–79 savings-drawdown breaks, early 2018–2019 drawdown regime, and panel attrition (`outputs/pooled_and_subperiods.json`).
- `02_transitions_and_regimes.py` — Weighted and unweighted three-state transition matrices, household-clustered SEs, and subperiod stationary distributions across all five HtM definitions (`outputs/transitions.json`).
- `03_income_process.py` — Within-round real log-income gaps, weighting/equivalence/99%-winsorized robustness, income sources, within-round quintiles, extensive-margin transitions, clustered AR(1) regressions, and weighted lag-12/24/48 and agricultural-split PPT decompositions with 200-draw household block-bootstrap SEs (`outputs/income_process.json`).
- `04_feedback_and_exposure.py` — Weighted Round-FE, unweighted Round-FE, and Round+HH-FE job-loss regressions by lagged state (intensive + positive-to-zero extensive margins, pre-trends, exact permutation remittance tests), bank-card ownership, cash-at-home saving audit, and staple regressions (`outputs/exposure_validation.json`).
- `07_diagnostics.py` — Weighted demographic profile, round-44 spillover persistence, round-76 refresh decomposition, job-loss boundary audit, population-weighted 3-state Baum–Welch HMMs (post-r29 core, post-trough r36, break-window-imputed, Q10-excluded) with 50-draw household block-bootstrap SEs and feasible Bayesian fiscal targeting, $M_1$ validation and difference tests, lagged $t-2/t-3$ job-loss exposure, event study, earner shares, and remittance-inclusive income checks (`outputs/diagnostics.json`).
- `05_model_microsim.py` — Partial-equilibrium IBC-consistent 3-state Markov-TANK CRN microsimulation (`N = 40,000`, seed `12345`, `H = 120` months), smoothed buffer rebuilding vs. no-borrowing clawback, HMM latent kernel $A^*$, forward-looking iMPC and fixed-status PIH saver rules, five subperiod regimes, 60-cell $(m_S, m_M)$ sensitivity grid (`m_M` in `[0.50, 1.00]`), and exact equal-aggregate-loss discrete job-loss counterfactuals (`-10%`, earner-capped `-10%`, HMM-kernel `-10%`, and uncapped `-5%`) (`outputs/microsim_results.json`, `outputs/sensitivity_grid.csv`).
- `06_make_figures.py` — Generates all five 300-DPI figures (`fig1`–`fig5`).

## Citation & License

- Code is licensed under **MIT**; preprint, documentation, and derived figures are licensed under **CC-BY-4.0** (see [`LICENSE`](LICENSE)).
- Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).
