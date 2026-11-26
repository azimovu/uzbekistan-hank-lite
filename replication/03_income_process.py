"""Step 03: Income distribution, within-round gaps, weighting/equivalence/winsorization
robustness, quintiles, extensive margin, clustered AR(1) (unweighted & weighted),
and Weighted PPT decomposition across lag-12, lag-24, lag-48, and net-of-mu_s with
uniform 200-draw household block-bootstrap SEs.
"""
import json
import math
import os
import random
import statistics
from collections import defaultdict

from common import (
    INCOME_FIELDS,
    OUT_DIR,
    clean,
    ensure_out_dir,
    htm_flags,
    iter_household_rows,
    load_monthly_cpi,
    parse_float,
    totinc,
    weighted_quantile,
)

ensure_out_dir()
cpi_map = load_monthly_cpi(base_ym='2018-09')

per_round = defaultdict(lambda: {
    'ly_w': [[], [], []],
    'ly_hhw': [[], [], []],
    'ly_pc_w': [[], [], []],
    'ly_eq_w': [[], [], []],
    'all_y_st': [],
})

src_w = {s: defaultdict(float) for s in (0, 1, 2)}
src_tot_y = {s: 0.0 for s in (0, 1, 2)}
zero_w = {s: [0.0, 0.0] for s in (0, 1, 2)}
zero_uw = {s: [0, 0] for s in (0, 1, 2)}
inc_share_w = [0.0, 0.0, 0.0]
pop_share_w = [0.0, 0.0, 0.0]
inc_share_hhw = [0.0, 0.0, 0.0]
pop_share_hhw = [0.0, 0.0, 0.0]
inc_share_pc_w = [0.0, 0.0, 0.0]
hhsize_sum_w = [0.0, 0.0, 0.0]

outlier_gt_50m = 0
max_y_nom = 0.0

hh = defaultdict(list)
hh_ever_ag = defaultdict(bool)

for row in iter_household_rows():
    rd = int(row['round'])
    hid = row['hhid']
    w = float(row['popw'] or 0.0)
    hsz = max(1.0, float(clean(row.get('hhsize', '')) or 1.0))
    w_hh = w / hsz
    ym = clean(row['date_start'])[:7]
    cpi = cpi_map[ym]
    scale = 100.0 / cpi
    y_nom = totinc(row)
    if parse_float(row['aginc_amount']) > 0.0:
        hh_ever_ag[hid] = True
    if y_nom > 5e7:
        outlier_gt_50m += 1
    if y_nom > max_y_nom:
        max_y_nom = y_nom
    y_real = y_nom * scale
    fl = htm_flags(row)
    st = fl['st_core']

    pop_share_w[st] += w
    inc_share_w[st] += w * y_real
    pop_share_hhw[st] += w_hh
    inc_share_hhw[st] += w_hh * y_real
    inc_share_pc_w[st] += w * (y_real / hsz)
    hhsize_sum_w[st] += w * hsz

    zero_w[st][1] += w
    zero_uw[st][1] += 1
    if y_real == 0.0:
        zero_w[st][0] += w
        zero_uw[st][0] += 1
    else:
        ly = math.log(y_real)
        ly_pc = math.log(y_real / hsz)
        ly_eq = math.log(y_real / math.sqrt(hsz))
        per_round[rd]['ly_w'][st].append((ly, w))
        per_round[rd]['ly_hhw'][st].append((ly, w_hh))
        per_round[rd]['ly_pc_w'][st].append((ly_pc, w))
        per_round[rd]['ly_eq_w'][st].append((ly_eq, w))

    per_round[rd]['all_y_st'].append((y_real, st, w, w_hh, hsz))

    for k in INCOME_FIELDS:
        v_real = parse_float(row[k]) * scale
        src_w[st][k] += w * v_real
    src_tot_y[st] += w * y_real

    hh[hid].append((rd, st, y_real, math.log(y_real) if y_real > 0 else None, w))

gaps_med_w = {'S-H': [], 'S-M': [], 'M-H': []}
gaps_mean_w = {'S-H': [], 'S-M': [], 'M-H': []}
gaps_mean_pc = {'S-H': [], 'S-M': [], 'M-H': []}
gaps_mean_eq = {'S-H': [], 'S-M': [], 'M-H': []}
gaps_med_hhw = {'S-H': [], 'S-M': [], 'M-H': []}

q_htm_w = [[0.0, 0.0] for _ in range(5)]
q_str_w = [[0.0, 0.0] for _ in range(5)]
rd_mean_ly_w = {}
inc_share_wins99_w = [0.0, 0.0, 0.0]

for rd in sorted(per_round):
    pr = per_round[rd]
    med_w = [weighted_quantile(pr['ly_w'][s], 0.5) for s in (0, 1, 2)]
    med_hhw = [weighted_quantile(pr['ly_hhw'][s], 0.5) for s in (0, 1, 2)]
    mn_w = [sum(v * w for v, w in pr['ly_w'][s]) / sum(w for _, w in pr['ly_w'][s]) for s in (0, 1, 2)]
    mn_pc = [sum(v * w for v, w in pr['ly_pc_w'][s]) / sum(w for _, w in pr['ly_pc_w'][s]) for s in (0, 1, 2)]
    mn_eq = [sum(v * w for v, w in pr['ly_eq_w'][s]) / sum(w for _, w in pr['ly_eq_w'][s]) for s in (0, 1, 2)]

    for d_map, vals in [
        (gaps_med_w, med_w), (gaps_med_hhw, med_hhw),
        (gaps_mean_w, mn_w), (gaps_mean_pc, mn_pc), (gaps_mean_eq, mn_eq),
    ]:
        d_map['S-H'].append(vals[0] - vals[2])
        d_map['S-M'].append(vals[0] - vals[1])
        d_map['M-H'].append(vals[1] - vals[2])

    all_pos = [x for s in (0, 1, 2) for x in pr['ly_w'][s]]
    rd_mean_ly_w[rd] = sum(v * w for v, w in all_pos) / sum(w for _, w in all_pos)

    pos_yw = [(x[0], x[2]) for x in pr['all_y_st'] if x[0] > 0.0]
    cap99 = weighted_quantile(pos_yw, 0.99) if pos_yw else 1e12
    for yr, st, w, _, _ in pr['all_y_st']:
        inc_share_wins99_w[st] += w * min(yr, cap99)

    ys = sorted(pr['all_y_st'], key=lambda x: x[0])
    W_r = sum(x[2] for x in ys)
    cum = 0.0
    for y_r, st, w, _, _ in ys:
        cum += w
        q = min(4, int(5.0 * (cum - 0.5 * w) / W_r))
        q_htm_w[q][1] += w
        q_str_w[q][1] += w
        if st >= 1:
            q_htm_w[q][0] += w
        if st == 2:
            q_str_w[q][0] += w


def summarize_gap(vals):
    m = statistics.mean(vals)
    return {
        'mean': round(m, 4),
        'min': round(min(vals), 4),
        'max': round(max(vals), 4),
        'implied_level_ratio': round(math.exp(-m), 4),
    }


mean_ly_S_w = sum(v * w for rd in per_round for v, w in per_round[rd]['ly_w'][0]) / sum(
    w for rd in per_round for _, w in per_round[rd]['ly_w'][0]
)
mu_S = mean_ly_S_w
mu_M = mu_S - statistics.mean(gaps_mean_w['S-M'])
mu_H = mu_S - statistics.mean(gaps_mean_w['S-H'])
mu_offset = (0.0, mu_M - mu_S, mu_H - mu_S)


def ar1_clustered(pairs, weighted=False):
    n = len(pairs)
    ws = [p[3] if weighted else 1.0 for p in pairs]
    W = sum(ws)
    mx = sum(w * p[0] for w, p in zip(ws, pairs)) / W
    my = sum(w * p[1] for w, p in zip(ws, pairs)) / W
    Sxx = sum(w * (p[0] - mx) ** 2 for w, p in zip(ws, pairs))
    Sxy = sum(w * (p[0] - mx) * (p[1] - my) for w, p in zip(ws, pairs))
    rho = Sxy / Sxx
    resid = [(p[1] - my) - rho * (p[0] - mx) for p in pairs]
    sc = defaultdict(float)
    for w, (x, _, h, _), e in zip(ws, pairs, resid):
        sc[h] += w * (x - mx) * e
    G = len(sc)
    se_cl = math.sqrt((G / (G - 1)) * sum(v * v for v in sc.values()) / (Sxx * Sxx))
    sig = math.sqrt(sum(w * e * e for w, e in zip(ws, resid)) / W)
    return {'n': n, 'n_hh': G, 'rho': round(rho, 4), 'se_cl': round(se_cl, 4), 'sig': round(sig, 4)}


pairs_real, pairs_rd_fe, pairs_hh_fe = [], [], []
pairs_by_lag_st = {0: [], 1: [], 2: []}
hh_r = {}
hh_r_netmu = {}
ext_trans = defaultdict(int)
n_pos_obs = 0

rd_net_num = defaultdict(float)
rd_net_den = defaultdict(float)
for hid, v in hh.items():
    for rd, st, yr, ly, w in v:
        if ly is not None:
            rd_net_num[rd] += w * (ly - mu_offset[st])
            rd_net_den[rd] += w
rd_net_mean = {r: rd_net_num[r] / rd_net_den[r] for r in rd_net_num}

for hid, v in hh.items():
    v.sort(key=lambda x: x[0])
    rs = [(rd, ly - rd_mean_ly_w[rd], ly - mu_offset[st] - rd_net_mean[rd], w)
          for rd, st, yr, ly, w in v if ly is not None]
    if rs:
        n_pos_obs += len(rs)
        m_i = sum(x[1] for x in rs) / len(rs)
        hh_r[hid] = {rd: (rval, w) for rd, rval, _, w in rs}
        hh_r_netmu[hid] = {rd: (rnet, w) for rd, _, rnet, w in rs}
        hh_u_map = {rd: rval - m_i for rd, rval, _, _ in rs}
    else:
        hh_u_map = {}

    for i in range(1, len(v)):
        r0, st0, y0, ly0, w0 = v[i - 1]
        r1, st1, y1, ly1, w1 = v[i]
        if r1 == r0 + 1:
            ext_trans[(0 if y0 > 0 else 1, 0 if y1 > 0 else 1)] += 1
            if ly0 is not None and ly1 is not None:
                pairs_real.append((ly0, ly1, hid, w1))
                r_0 = ly0 - rd_mean_ly_w[r0]
                r_1 = ly1 - rd_mean_ly_w[r1]
                pairs_rd_fe.append((r_0, r_1, hid, w1))
                pairs_by_lag_st[st0].append((r_0, r_1, hid, w1))
                pairs_hh_fe.append((hh_u_map[r0], hh_u_map[r1], hid, w1))

LAGS_ALL = [0, 1, 2, 3, 4, 6, 9, 11, 12, 13, 15, 18, 24, 30, 36, 42, 48]
FIT_LAGS_12 = [1, 2, 3, 4, 6, 9, 12]
FIT_LAGS_24 = [1, 2, 3, 4, 6, 9, 12, 15, 18, 24]
FIT_LAGS_48 = [1, 2, 3, 4, 6, 9, 12, 15, 18, 24, 30, 36, 42, 48]


def hh_lag_contributions(series_dict, lags):
    hids = sorted(series_dict.keys())
    per_hh = []
    for hid in hids:
        d = series_dict[hid]
        num = {}
        den = {}
        for k in lags:
            s_xy = 0.0
            s_w = 0.0
            for rd, (val, w) in d.items():
                nxt = d.get(rd + k)
                if nxt is not None:
                    wm = 0.5 * (w + nxt[1])
                    s_xy += wm * val * nxt[0]
                    s_w += wm
            num[k] = s_xy
            den[k] = s_w
        per_hh.append((num, den))
    return per_hh


def fit_ppt_from_cov(cov_map, fit_lags):
    best = None
    for ri in range(600, 996):
        rho = ri / 1000.0
        xs = [rho ** k for k in fit_lags]
        ys = [cov_map[k] for k in fit_lags]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vz = sxy / sxx
        va = my - vz * mx
        ve = cov_map[0] - va - vz
        if va > 0 and vz > 0 and ve > 0:
            sse = sum((y - (va + vz * x)) ** 2 for x, y in zip(xs, ys))
            if best is None or sse < best[0]:
                best = (sse, rho, va, vz, ve)
    sse, rho, va, vz, ve = best
    veta = vz * (1.0 - rho * rho)
    return {
        'rho_z': rho,
        'half_life_months': math.log(0.5) / math.log(rho),
        'sig_alpha': math.sqrt(va),
        'sig_z_stationary': math.sqrt(vz),
        'sig_eta': math.sqrt(veta),
        'sig_eps': math.sqrt(ve),
        'share_alpha': va / cov_map[0],
        'share_z': vz / cov_map[0],
        'share_eps': ve / cov_map[0],
        'rmse': math.sqrt(sse / len(fit_lags)),
    }


def estimate_ppt_with_bootstrap(series_dict, fit_lags, n_boot=200, seed=2026):
    per_hh = hh_lag_contributions(series_dict, LAGS_ALL)
    G = len(per_hh)
    cov_exact = {k: sum(h[0][k] for h in per_hh) / sum(h[1][k] for h in per_hh) for k in LAGS_ALL}
    pt = fit_ppt_from_cov(cov_exact, fit_lags)

    rng = random.Random(seed)
    b_vals = defaultdict(list)
    for _ in range(n_boot):
        idx = [rng.randrange(G) for _ in range(G)]
        bcov = {}
        for k in ([0] + fit_lags):
            num = sum(per_hh[i][0][k] for i in idx)
            den = sum(per_hh[i][1][k] for i in idx)
            bcov[k] = num / den
        bfit = fit_ppt_from_cov(bcov, fit_lags)
        for key in ('rho_z', 'half_life_months', 'sig_alpha', 'sig_z_stationary', 'sig_eta', 'sig_eps'):
            b_vals[key].append(bfit[key])

    se_boot = {key: round(statistics.stdev(vals), 4) for key, vals in b_vals.items()}
    return {
        'n_hh': G,
        'n_boot': n_boot,
        'rho_z': round(pt['rho_z'], 4),
        'half_life_months': round(pt['half_life_months'], 2),
        'sig_alpha': round(pt['sig_alpha'], 4),
        'sig_z_stationary': round(pt['sig_z_stationary'], 4),
        'sig_eta': round(pt['sig_eta'], 4),
        'sig_eps': round(pt['sig_eps'], 4),
        'boot_se': se_boot,
        'var_shares': {
            'permanent_alpha': round(pt['share_alpha'], 4),
            'persistent_z': round(pt['share_z'], 4),
            'transitory_eps': round(pt['share_eps'], 4),
        },
        'rmse': round(pt['rmse'], 5),
        'empirical_cov_w': {str(k): round(cov_exact[k], 5) for k in LAGS_ALL},
    }


# Uniform n_boot=200 across all PPT specifications (including agricultural seasonality split)
ppt_24 = estimate_ppt_with_bootstrap(hh_r, FIT_LAGS_24, n_boot=200, seed=2026)
ppt_12 = estimate_ppt_with_bootstrap(hh_r, FIT_LAGS_12, n_boot=200, seed=2027)
ppt_48 = estimate_ppt_with_bootstrap(hh_r, FIT_LAGS_48, n_boot=200, seed=2029)
ppt_24_netmu = estimate_ppt_with_bootstrap(hh_r_netmu, FIT_LAGS_24, n_boot=200, seed=2028)
hh_r_never_ag = {h: d for h, d in hh_r.items() if not hh_ever_ag[h]}
hh_r_ever_ag = {h: d for h, d in hh_r.items() if hh_ever_ag[h]}
ppt_24_never_ag = estimate_ppt_with_bootstrap(hh_r_never_ag, FIT_LAGS_24, n_boot=200, seed=2030)
ppt_24_ever_ag = estimate_ppt_with_bootstrap(hh_r_ever_ag, FIT_LAGS_24, n_boot=200, seed=2031)

W_all = sum(pop_share_w)
Y_all = sum(inc_share_w)
W_hh_all = sum(pop_share_hhw)
Y_hh_all = sum(inc_share_hhw)
Y_pc_all = sum(inc_share_pc_w)
Y_w99_all = sum(inc_share_wins99_w)

t_pos = ext_trans[(0, 0)] + ext_trans[(0, 1)]
t_zero = ext_trans[(1, 0)] + ext_trans[(1, 1)]


def agg_from_s_share(s_sh, ms=0.07):
    return round(s_sh * ms + (1.0 - s_sh) * 1.0, 4)


out = {
    'sample_counts': {
        'n_pos_obs': n_pos_obs,
        'n_hh_ppt': len(hh_r),
        'n_ar1_pairs': len(pairs_rd_fe),
        'n_hh_ar1': ar1_clustered(pairs_rd_fe)['n_hh'],
    },
    'within_round_median_log_gaps_weighted': {k: summarize_gap(v) for k, v in gaps_med_w.items()},
    'within_round_median_log_gaps_hhweight': {k: summarize_gap(v) for k, v in gaps_med_hhw.items()},
    'within_round_mean_log_gaps_weighted': {k: summarize_gap(v) for k, v in gaps_mean_w.items()},
    'within_round_mean_log_gaps_per_capita': {k: summarize_gap(v) for k, v in gaps_mean_pc.items()},
    'within_round_mean_log_gaps_sqrt_equiv': {k: summarize_gap(v) for k, v in gaps_mean_eq.items()},
    'microsim_mu_within_round': {'S': round(mu_S, 4), 'M': round(mu_M, 4), 'H': round(mu_H, 4)},
    'pooled_real_income_shares': {
        'pop_share_w': {nm: round(pop_share_w[i] / W_all, 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'inc_share_w': {nm: round(inc_share_w[i] / Y_all, 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'mean_real_y_som_2018m09': {nm: round(inc_share_w[i] / pop_share_w[i]) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'mean_hhsize_w': {nm: round(hhsize_sum_w[i] / pop_share_w[i], 2) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
    },
    'weighting_and_size_robustness': {
        'outlier_rows_gt_50m_nom': outlier_gt_50m,
        'max_monthly_nom_income': round(max_y_nom),
        'baseline_popw_total_Y': {
            'S_inc_share': round(inc_share_w[0] / Y_all, 4),
            'AggMPC_inc_ms07': agg_from_s_share(inc_share_w[0] / Y_all),
            'S_H_mean_log_gap': round(statistics.mean(gaps_mean_w['S-H']), 4),
        },
        'hh_weights_popw_div_hhsize': {
            'S_pop_share': round(pop_share_hhw[0] / W_hh_all, 4),
            'S_inc_share': round(inc_share_hhw[0] / Y_hh_all, 4),
            'AggMPC_inc_ms07': agg_from_s_share(inc_share_hhw[0] / Y_hh_all),
            'S_H_med_log_gap': round(statistics.mean(gaps_med_hhw['S-H']), 4),
        },
        'per_capita_income_Y_div_hhsize': {
            'S_inc_share': round(inc_share_pc_w[0] / Y_pc_all, 4),
            'AggMPC_inc_ms07': agg_from_s_share(inc_share_pc_w[0] / Y_pc_all),
            'S_H_mean_log_gap': round(statistics.mean(gaps_mean_pc['S-H']), 4),
        },
        'sqrt_equivalized_income': {
            'S_H_mean_log_gap': round(statistics.mean(gaps_mean_eq['S-H']), 4),
        },
        'winsorized_99pct_within_round': {
            'S_inc_share': round(inc_share_wins99_w[0] / Y_w99_all, 4),
            'AggMPC_inc_ms07': agg_from_s_share(inc_share_wins99_w[0] / Y_w99_all),
        },
    },
    'income_sources_weighted': {
        nm: {k: round(src_w[i][k] / src_tot_y[i], 4) for k in INCOME_FIELDS}
        for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
    },
    'zero_income_share': {
        'weighted': {nm: round(zero_w[i][0] / zero_w[i][1], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'unweighted': {nm: round(zero_uw[i][0] / zero_uw[i][1], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
    },
    'extensive_margin_transitions': {
        'P_pos_to_zero': round(ext_trans[(0, 1)] / t_pos, 4),
        'P_zero_to_pos': round(ext_trans[(1, 0)] / t_zero, 4),
        'n_pos': t_pos, 'n_zero': t_zero,
    },
    'within_round_quintiles_weighted': {
        f'Q{q+1}': {
            'broad': round(q_htm_w[q][0] / q_htm_w[q][1], 4),
            'strict': round(q_str_w[q][0] / q_str_w[q][1], 4),
        } for q in range(5)
    },
    'ar1_real_log_income': {
        'pooled_real_uw': ar1_clustered(pairs_real, weighted=False),
        'round_fe_uw': ar1_clustered(pairs_rd_fe, weighted=False),
        'round_fe_w': ar1_clustered(pairs_rd_fe, weighted=True),
        'round_plus_hh_fe_uw': ar1_clustered(pairs_hh_fe, weighted=False),
        'by_lag_state_round_fe_uw': {
            nm: ar1_clustered(pairs_by_lag_st[i], weighted=False) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
        },
    },
    'ppt_lag24_weighted': ppt_24,
    'ppt_lag12_weighted': ppt_12,
    'ppt_lag48_weighted': ppt_48,
    'ppt_lag24_net_mu_for_microsim': ppt_24_netmu,
    'ppt_lag24_never_ag_weighted': ppt_24_never_ag,
    'ppt_lag24_ever_ag_weighted': ppt_24_ever_ag,
}

out_path = os.path.join(OUT_DIR, 'income_process.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"wrote {out_path} (PPT HHs={len(hh_r)}, AR1 HHs={out['sample_counts']['n_hh_ar1']})")
for k in ('ppt_lag12_weighted', 'ppt_lag24_weighted', 'ppt_lag48_weighted', 'ppt_lag24_net_mu_for_microsim'):
    p = out[k]
    print(f"  {k:30s}: rho={p['rho_z']:.3f}({p['boot_se']['rho_z']:.3f}), HL={p['half_life_months']:.2f}m({p['boot_se']['half_life_months']:.2f}), "
          f"sig_a={p['sig_alpha']:.3f}({p['boot_se']['sig_alpha']:.3f}), sig_eta={p['sig_eta']:.3f}({p['boot_se']['sig_eta']:.3f}), "
          f"sig_e={p['sig_eps']:.3f}({p['boot_se']['sig_eps']:.3f}), shares={p['var_shares']}")
