"""Step 06: Generate 5 publication-grade figures (300 DPI) for the blinded submission.

Addresses every visual & editorial check:
- Zero raw column names in legends/titles (plain English throughout).
- Unified m_S notation.
- Distinguishes Window-Only Break-Imputed vs. All-Round Imputed cleanly in Figs 1 & 3.
- Includes steady-state AggMPC line in Fig 3(a) legend.
- Plots clean Smoothed Buffer-Rebuilding IBC paths in Fig 4(a) and labels 'Strict-HtM cohort (H)' accurately.
- Exact half-up rounding & '%' units in Fig 5(b) matching Table 8 to 2 decimal places.
"""
import csv
import datetime as dt
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from common import OUT_DIR

plt.rcParams.update({
    'font.size': 9,
    'axes.linewidth': 0.75,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
})


def fmt2_halfup(val):
    return f'{math.floor(val * 100.0 + 0.5 + 1e-12) / 100.0:.2f}'


with open(os.path.join(OUT_DIR, 'national_series.csv')) as f:
    ser = list(csv.DictReader(f))
with open(os.path.join(OUT_DIR, 'sensitivity_grid.csv')) as f:
    grid = list(csv.DictReader(f))
with open(os.path.join(OUT_DIR, 'exposure_validation.json')) as f:
    exp = json.load(f)
with open(os.path.join(OUT_DIR, 'microsim_results.json')) as f:
    msim = json.load(f)

dates = [dt.datetime.strptime(r['date'], '%Y-%m-%d') for r in ser]
rds = [int(r['round']) for r in ser]
date_of = dict(zip(rds, dates))

broad = [float(r['broad']) for r in ser]
se_b = [float(r['se_broad']) for r in ser]
strict = [float(r['strict']) for r in ser]
se_s = [float(r['se_strict']) for r in ser]
broad_win = [float(r['broad_win']) for r in ser]
broad_unadj = [float(r['broad_unadj']) for r in ser]
took = [float(r['took_sav']) for r in ser]
p_fd_sn = [float(r['p_foodno_given_savno']) for r in ser]
p_fd_ns = [float(r['p_foodno_given_nosav']) for r in ser]
fies = [float(r['fies_any']) if r['fies_any'] else float('nan') for r in ser]
inc_real = [float(r['mean_income_real']) / 1e6 for r in ser]
inc_w99 = [float(r['mean_income_real_wins99']) / 1e6 for r in ser]
finw = [float(r['fin_worse']) for r in ser]
loan = [float(r['loan']) for r in ser]
agg_pop_t = [float(r['aggmpc_pop']) for r in ser]
agg_inc_t = [float(r['aggmpc_inc']) for r in ser]
agg_inc_win_t = [float(r['aggmpc_inc_win']) for r in ser]

EVENTS = [
    (20, 'COVID-19\n(Apr 2020)'),
    (29, 'r29 food-cut\nshift'),
    (43, 'Price shock\n(Mar 2022)'),
    (58, 'r58\u201365\nbreak'),
    (78, 'r78\u201379\nbreak'),
]


def fmt_date_ax(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


# Figure 1: National HtM shares (Core + Window-Imputed + All-Round Imputed + Unadjusted) & Income
fig, ax = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True)

ax[0].fill_between(
    dates,
    [b - 1.96 * s for b, s in zip(broad, se_b)],
    [b + 1.96 * s for b, s in zip(broad, se_b)],
    color='#1b2631', alpha=0.14, lw=0, zorder=1,
)
ax[0].plot(dates, broad, color='#1b2631', lw=1.4, label='Broad HtM (core harmonized)', zorder=4)
ax[0].plot(dates, broad_win, color='#2471a3', lw=1.2, ls=':', label='Broad HtM (break-window imputed)', zorder=3)
ax[0].plot(dates, broad_unadj, color='#95a5a6', lw=0.95, ls='--', label='Broad HtM (unadjusted food item)', zorder=2)

ax[0].fill_between(
    dates,
    [v - 1.96 * s for v, s in zip(strict, se_s)],
    [v + 1.96 * s for v, s in zip(strict, se_s)],
    color='#b03a2e', alpha=0.15, lw=0, zorder=1,
)
ax[0].plot(dates, strict, color='#b03a2e', lw=1.4, label='Strict HtM (no savings and coping)', zorder=5)

for rd_ev, lab in EVENTS:
    d_ev = date_of[rd_ev]
    ax[0].axvline(d_ev, color='#bdc3c7', lw=0.75, ls=':', zorder=0)
    ax[0].text(d_ev + dt.timedelta(days=18), 1.07, lab, fontsize=6.9, color='#566573', va='top')

ax[0].set_ylabel('Weighted household share')
ax[0].set_ylim(0.0, 1.12)
ax[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[0].set_title('(a) National hand-to-mouth shares, Sep 2018\u2013Jun 2025 (95% Kish CIs)', loc='left', fontsize=9.4)
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=7.6)

ax[1].plot(dates, inc_real, color='#1b2631', lw=1.35, label='Real mean income (M Sep-2018 som)')
ax[1].plot(dates, inc_w99, color='#7f8c8d', lw=0.95, ls='-.', label='Real mean income (99% winsorized)')
ax[1].set_ylabel('Real income (M Sep-2018 som)')
ax[1].set_ylim(1.15, 2.95)
ax2 = ax[1].twinx()
ax2.spines['top'].set_visible(False)
l1, = ax2.plot(dates, finw, color='#b03a2e', lw=1.1, ls='--', label='Finances worse (past 30d)')
l2, = ax2.plot(dates, loan, color='#2471a3', lw=1.1, ls=':', label='Took loan (past 30d)')
ax2.set_ylabel('Distress share')
ax2.set_ylim(0.0, 0.30)
for rd_ev, _ in EVENTS:
    ax[1].axvline(date_of[rd_ev], color='#bdc3c7', lw=0.75, ls=':', zorder=0)
ax[1].set_title('(b) CPI-deflated household income and financial-distress indicators', loc='left', fontsize=9.4)
fmt_date_ax(ax[1])
ax[1].set_xlabel('Survey date')
h1, lb1 = ax[1].get_legend_handles_labels()
ax[1].legend(h1 + [l1, l2], lb1 + ['Finances worse (past 30d)', 'Took loan (past 30d)'],
             loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.6)

fig.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.15, hspace=0.52)
fig.savefig(os.path.join(OUT_DIR, 'fig1_htm_dynamics.png'), dpi=300)
plt.close(fig)

# Figure 2: Forensic Survey-Break Audit & FIES Co-movement
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.5))

ax[0].plot(dates, p_fd_sn, color='#b03a2e', lw=1.35, label='Food-shortage rate | untouched savings')
ax[0].plot(dates, p_fd_ns, color='#1b2631', lw=1.2, ls='--', label='Food-shortage rate | no savings')
ax[0].plot(dates, took, color='#2471a3', lw=1.15, ls=':', label='Reported savings drawdown')
ax[0].axvspan(date_of[58], date_of[65], color='#d6eaf8', alpha=0.55, zorder=0)
ax[0].axvspan(date_of[78], date_of[79], color='#d6eaf8', alpha=0.55, zorder=0)
ax[0].axvline(date_of[44], color='#b03a2e', lw=0.75, ls=':', alpha=0.7)
ax[0].text(date_of[44] - dt.timedelta(days=28), 0.56, 'r44 food\nanomaly', fontsize=6.9, color='#922b21', va='top', ha='right')
ax[0].set_ylabel('Weighted share (cond. food / uncond. drawdown)')
ax[0].set_xlabel('Survey date')
ax[0].set_ylim(0.0, 0.85)
ax[0].set_title('(a) Forensic audit of survey discontinuities', loc='left', fontsize=9.1)
fmt_date_ax(ax[0])
ax[0].legend(loc='upper left', fontsize=6.9)

f_idx = [i for i, v in enumerate(fies) if v == v]
f_dates = [dates[i] for i in f_idx]
ax[1].plot(f_dates, [broad[i] for i in f_idx], color='#1b2631', lw=1.35, label='Broad HtM (core harmonized)')
ax[1].plot(f_dates, [broad_win[i] for i in f_idx], color='#2471a3', lw=1.15, ls=':', label='Broad HtM (break-window imputed)')
ax[1].plot(f_dates, [broad_unadj[i] for i in f_idx], color='#95a5a6', lw=1.0, ls='--', label='Broad HtM (unadjusted food)')
ax[1].plot(f_dates, [fies[i] for i in f_idx], color='#b03a2e', lw=1.3, ls='-.', label='FIES food insecurity (any Yes)')
ax[1].axvspan(date_of[58], date_of[65], color='#d6eaf8', alpha=0.55, zorder=0)
ax[1].axvspan(date_of[78], date_of[79], color='#d6eaf8', alpha=0.55, zorder=0)
ax[1].set_ylabel('Weighted household share')
ax[1].set_xlabel('Survey date')
ax[1].set_ylim(0.0, 0.96)
ax[1].set_title('(b) Broad HtM versus FIES food insecurity (r49\u201382)', loc='left', fontsize=9.1)
fmt_date_ax(ax[1])
ax[1].legend(loc='center right', bbox_to_anchor=(0.99, 0.44), fontsize=6.8)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_forensic_breaks.png'), dpi=300)
plt.close(fig)

# Figure 3: Time-varying AggMPC (Core & Break-Window Imputed + Steady State) & Sensitivity Grid
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.5))

cal = msim['calibration']
ax[0].axvspan(date_of[58], date_of[65], color='#eaecee', alpha=0.8, zorder=0, label='Drawdown break windows')
ax[0].axvspan(date_of[78], date_of[79], color='#eaecee', alpha=0.8, zorder=0)
ax[0].plot(dates, agg_pop_t, color='#1b2631', lw=1.25, label='Population-weighted (core)')
ax[0].plot(dates, agg_inc_t, color='#b03a2e', lw=1.25, ls='--', label='Income-weighted (core)')
ax[0].plot(dates, agg_inc_win_t, color='#2471a3', lw=1.1, ls=':', label='Income-weighted (break-window imp.)')
ax[0].axhline(cal['aggmpc_inc_base'], color='#7f8c8d', lw=0.85, ls='-.', alpha=0.8,
              label=f"Steady-state AggMPC$_\\mathrm{{inc}}$ ({cal['aggmpc_inc_base']:.3f})")
ax[0].set_ylabel('Aggregate MPC ($m_S = 0.07$)')
ax[0].set_xlabel('Survey date')
ax[0].set_ylim(0.22, 0.84)
ax[0].set_title('(a) Time-varying aggregate MPC across rounds', loc='left', fontsize=9.1)
fmt_date_ax(ax[0])
ax[0].legend(loc='lower left', fontsize=6.0)

ms_vals = sorted(set(float(r['saver_mpc']) for r in grid))
ys_pop_1 = [float(r['AggMPC_pop']) for r in grid
            if float(r['M_mpc']) == 1.00 and float(r['shock']) == -0.10]
ax[1].plot(ms_vals, ys_pop_1, color='#7f8c8d', lw=1.1, ls='-.', marker='d', ms=3.5,
           label='Pop-wtd ($m_M=1.00$)')
for mM, col, ls, mk in (
    (1.00, '#1b2631', '-', 'o'),
    (0.85, '#2471a3', '--', 's'),
    (0.70, '#b03a2e', ':', '^'),
    (0.50, '#7d3c98', '-.', 'v'),
):
    ys_inc = [float(r['AggMPC_inc']) for r in grid
              if float(r['M_mpc']) == mM and float(r['shock']) == -0.10]
    ax[1].plot(ms_vals, ys_inc, color=col, lw=1.3, ls=ls, marker=mk, ms=4,
               label=f'Income-wtd ($m_M={mM:.2f}$)')
ax[1].set_xlabel('Saver MPC ($m_S$)')
ax[1].set_ylabel('Steady-state AggMPC')
ax[1].set_ylim(0.28, 0.78)
ax[1].set_title('(b) Sensitivity to $m_S$ and non-strict HtM $m_M$', loc='left', fontsize=9.1)
ax[1].legend(loc='upper left', ncol=2, fontsize=6.4, columnspacing=0.7)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_aggmpc_and_sensitivity.png'), dpi=300)
plt.close(fig)

# Figure 4: Smoothed IBC Dynamic IRFs (Buffer-Stock vs FL-iMPC vs Fixed-S PIH) + All 5 Core Regimes
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.55))

uni = msim['uniform_10pct_smoothed_ibc']
uni_fl = msim['uniform_10pct_fl_impc']
uni_pih = msim['uniform_10pct_pih_fixed_S']
h_grid = uni['horizons']

ax[0].plot(h_grid, [100 * uni['dC'][h] for h in h_grid], color='#1b2631', lw=1.55,
           label=f"Agg. smoothed IBC ({100*uni['dC'][0]:.2f}%)")
ax[0].plot(h_grid, [100 * uni_fl['dC'][h] for h in h_grid], color='#d35400', lw=1.25, ls='--',
           label=f"Agg. forward-looking iMPC ({100*uni_fl['dC'][0]:.2f}%)")
ax[0].plot(h_grid, [100 * uni_pih['dC'][h] for h in h_grid], color='#7f8c8d', lw=1.15, ls=':',
           label=f"Agg. PIH saver, fixed S ({100*uni_pih['dC'][0]:.2f}%)")
ax[0].plot(h_grid, [100 * uni['dC_by_state']['H'][h] for h in h_grid], color='#b03a2e', lw=1.1, ls='-.',
           label="Strict-HtM cohort H (impact -10.0%)")
ax[0].plot(h_grid, [100 * uni['dC_by_state']['S'][h] for h in h_grid], color='#2471a3', lw=1.2, ls='-', alpha=0.88,
           label="Saver cohort S (smoothed IBC)")
ax[0].set_xlabel(f"Months after $-10\\%$ persistent shock ($\\rho_z = {cal['rho_z']:.3f}$)")
ax[0].set_ylabel('Consumption response (%)')
ax[0].set_ylim(-11.2, 0.8)
ax[0].set_title('(a) IBC-consistent CRN IRFs across saver rules', loc='left', fontsize=9.1)
ax[0].legend(loc='lower right', fontsize=6.5)

reg = msim['regime_irfs']
for code, lab, col, ls in (
    ('COVID', 'COVID (r20\u201329)', '#b03a2e', '--'),
    ('late', 'Late period (r56\u201382)', '#d35400', '-.'),
    ('pooled', 'Pooled core baseline', '#1b2631', '-'),
    ('recovery', '2021 recovery (r30\u201342)', '#2471a3', ':'),
    ('war', '2022\u201323 shock (r43\u201355)', '#8e44ad', (0, (4, 1.5))),
    ('preCOVID', 'Pre-COVID (r2\u201319)', '#27ae60', (0, (3, 1, 1, 1))),
):
    if code == 'pooled':
        dC = uni['dC']
        am = cal['aggmpc_inc_base']
    else:
        dC = reg[code]['sim']['dC']
        am = reg[code]['aggmpc_inc']
    ax[1].plot(h_grid, [100 * dC[h] for h in h_grid], color=col, lw=1.2, ls=ls,
               label=f'{lab} ({am:.3f})')
ax[1].set_xlabel('Months after $-10\\%$ persistent shock')
ax[1].set_ylabel('Aggregate consumption response (%)')
ax[1].set_ylim(-8.4, 0.5)
ax[1].set_title('(b) All five core subperiod regimes (by AggMPC$_\\mathrm{inc}$)', loc='left', fontsize=9.1)
ax[1].legend(loc='lower right', fontsize=6.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_ibc_irfs_and_regimes.png'), dpi=300)
plt.close(fig)

# Figure 5: Round-FE Job-Loss Effects (95% CIs) & Mean-Preserving Discrete Amplification (% units!)
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.5))

wl = exp['work_loss_by_lagged_state']
states = ['S', 'M', 'H']
x_pos = [0, 1, 2]
y_drops = [-100 * wl[s]['primary_round_fe_weighted']['implied_pct_Y_drop_total_incl_zero'] for s in states]
y_errs = [1.96 * 100 * (wl[s]['primary_round_fe_weighted']['dlogY_within_hh']['se_cl'] ** 2 +
                        wl[s]['primary_round_fe_weighted']['dP_pos_to_zero']['se_cl'] ** 2) ** 0.5
          for s in states]
str_jumps = [100 * wl[s]['primary_round_fe_weighted']['dP_strict']['beta'] for s in states]
str_errs = [1.96 * 100 * wl[s]['primary_round_fe_weighted']['dP_strict']['se_cl'] for s in states]

w = 0.34
b1 = ax[0].bar([x - w / 2 for x in x_pos], y_drops, yerr=y_errs, capsize=3, error_kw={'lw': 0.9},
               width=w, color='#1b2631', alpha=0.88, label='Total income drop, % (incl. zeros)')
b2 = ax[0].bar([x + w / 2 for x in x_pos], str_jumps, yerr=str_errs, capsize=3, error_kw={'lw': 0.9},
               width=w, color='#b03a2e', alpha=0.85, label='Strict-HtM entry / persistence jump, pp (Rd FE)')
for bar, err in zip(b1, y_errs):
    h = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width() / 2, h + err + 0.8, f'{h:.1f}%', ha='center', fontsize=7.0)
for bar, err in zip(b2, str_errs):
    h = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width() / 2 + 0.06, h + err + 0.8, f'+{h:.1f} pp', ha='center', fontsize=6.4)
ax[0].set_xticks(x_pos)
ax[0].set_xticklabels(['Lag Saver\n($S_{t-1}$)', 'Lag Non-strict\nHtM ($M_{t-1}$)', 'Lag Strict\nHtM ($H_{t-1}$)'])
ax[0].set_ylabel('Effect size (% drop or pp jump)')
ax[0].set_ylim(0, 44)
ax[0].set_title('(a) Round-FE job-loss effects (95% clustered CIs)', loc='left', fontsize=9.1)
ax[0].legend(loc='upper left', fontsize=6.8)

td = msim['targeted_discrete_10pct']
u_end = msim['uniform_10pct_endog_switch']
vals = [
    -100 * uni['dC'][0],
    -100 * u_end['dC'][0],
    -100 * td['exogenous_P']['dC'][0],
    -100 * td['endogenous_P']['dC'][0],
]
labs = ['Uniform\n(exog. $P$)', 'Uniform\n+ switching', 'Targeted\n(exog. $P$)', 'Targeted\n+ switching']
cols = ['#1b2631', '#566573', '#2471a3', '#b03a2e']
bars = ax[1].bar(range(4), vals, color=cols, alpha=0.88, width=0.58)
for bar, v in zip(bars, vals):
    ax[1].text(bar.get_x() + bar.get_width() / 2, v + 0.18, f'{fmt2_halfup(v)}%', ha='center', fontsize=7.5)
ax[1].set_xticks(range(4))
ax[1].set_xticklabels(labs, fontsize=7.5)
ax[1].set_ylabel('Impact drop in aggregate consumption (%)')
ax[1].set_ylim(0, max(vals) * 1.24)
ax[1].set_title('(b) Equal-aggregate-loss ($-10.0\\%$) shock', loc='left', fontsize=9.1)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_jobloss_and_amplification.png'), dpi=300)
plt.close(fig)

print('Generated all 5 submission figures in', OUT_DIR)
