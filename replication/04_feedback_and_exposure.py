"""Step 04: Income-to-HtM feedback, Round-FE + Household-FE + Pre-Trend job-loss
regressions with extensive-margin exposure, subperiod split (r1--29 vs r30--82),
exact small-cell remittance counts & permutation p-values, remittance use,
banking-card audit, cash-saving audit, and clustered staple regressions.
"""
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict

from common import (
    OUT_DIR,
    clean,
    ensure_out_dir,
    htm_flags,
    iter_household_rows,
    iter_individual_rows,
    load_monthly_cpi,
    parse_float,
    totinc,
)

ensure_out_dir()
cpi_map = load_monthly_cpi(base_ym='2018-09')

ind_exp = {}
rem_use = Counter()
rem_amt_nonzero = 0
for row in iter_individual_rows():
    k = (int(row['round']), row['hhid'])
    rec = ind_exp.get(k)
    if rec is None:
        rec = ind_exp[k] = [0, 0]
    if clean(row['work_lost_hh']) == 'Yes':
        rec[0] = 1
    if clean(row['mig_living_remittance']) == 'Yes':
        rec[1] = 1
    u = clean(row.get('mig_living_remittance_use', ''))
    if u:
        rem_use[u] += 1
    amt = parse_float(row.get('mig_living_remittance_amount', ''))
    if amt > 0:
        rem_amt_nonzero += 1

hh = defaultdict(list)
rd_ly = defaultdict(lambda: [0.0, 0.0])
linked_n = missing_n = 0
wl_by_period = Counter()

dep_past_by_st = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
sav_next_by_st = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
dep_amt_ratios = []
dep_forms = Counter()
dep_amt_states = Counter()
bank_by_st = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}

flour_rep_st = Counter()
all_st = Counter()
flour_ly, all_ly = [], []

for row in iter_household_rows():
    rd = int(row['round'])
    hid = row['hhid']
    w = float(row['popw'] or 0.0)
    ym = clean(row['date_start'])[:7]
    cpi = cpi_map[ym]
    scale = 100.0 / cpi
    y_nom = totinc(row)
    y_real = y_nom * scale
    ly = math.log(y_real) if y_real > 0 else None
    fl = htm_flags(row)
    st = fl['st_core']

    if ly is not None:
        rd_ly[rd][0] += w * ly
        rd_ly[rd][1] += w
        all_ly.append(ly)
    all_st[st] += 1

    bc = clean(row.get('bank_card_hh', ''))
    if bc in ('Yes', 'No'):
        bank_by_st[st][1] += w
        if bc == 'Yes':
            bank_by_st[st][0] += w

    dep_past_by_st[st][1] += 1
    if clean(row['deposit_past30d']) == 'Yes':
        dep_past_by_st[st][0] += 1
    sav_next_by_st[st][1] += 1
    if clean(row['saving_next30d']) == 'Yes':
        sav_next_by_st[st][0] += 1
        dep_amt_states[st] += 1
        amt = parse_float(row['deposit_amount'])
        if amt > 0 and y_nom > 0:
            dep_amt_ratios.append(amt / y_nom)
            dep_forms[clean(row['deposit_form'])] += 1

    hb = clean(row['bread_past7d_cost'])
    hf = clean(row['flour_past7d_cost'])
    b_real = parse_float(hb) * scale if hb else None
    f_real = parse_float(hf) * scale if hf else None
    if f_real is not None and f_real > 0:
        flour_rep_st[st] += 1
        if ly is not None:
            flour_ly.append(ly)

    exp = ind_exp.get((rd, hid))
    if exp is None:
        missing_n += 1
        wl, rem = None, None
    else:
        linked_n += 1
        wl, rem = exp[0], exp[1]
        p_grp = 'r1_29' if rd <= 29 else ('r30_42' if rd <= 42 else 'r43_82')
        if wl == 1:
            wl_by_period[p_grp] += 1

    hh[hid].append((rd, st, y_real, ly, w, wl, rem, 1 if fl['fin_worse'] else 0, b_real, f_real))

rd_mean_ly = {r: v[0] / v[1] for r, v in rd_ly.items()}


def wls_fe_clustered(obs, use_round_fe=True, use_hh_fe=False, weighted=True):
    n = len(obs)
    if n < 5:
        return None
    ws = [o[4] if weighted else 1.0 for o in obs]
    xs = [o[2] for o in obs]
    ys = [o[3] for o in obs]
    rds = [o[0] for o in obs]
    hids = [o[1] for o in obs]

    if use_round_fe and not use_hh_fe:
        rx = defaultdict(float); ry = defaultdict(float); rw = defaultdict(float)
        for rd, w, x, y in zip(rds, ws, xs, ys):
            rx[rd] += w * x; ry[rd] += w * y; rw[rd] += w
        dx = [x - rx[rd] / rw[rd] for rd, x in zip(rds, xs)]
        dy = [y - ry[rd] / rw[rd] for rd, y in zip(rds, ys)]
    elif use_round_fe and use_hh_fe:
        dx = xs[:]
        dy = ys[:]
        for _ in range(25):
            rx = defaultdict(float); ry = defaultdict(float); rw = defaultdict(float)
            for rd, w, x, y in zip(rds, ws, dx, dy):
                rx[rd] += w * x; ry[rd] += w * y; rw[rd] += w
            dx = [x - rx[rd] / rw[rd] for rd, x in zip(rds, dx)]
            dy = [y - ry[rd] / rw[rd] for rd, y in zip(rds, dy)]
            hx = defaultdict(float); hy = defaultdict(float); hw = defaultdict(float)
            for h, w, x, y in zip(hids, ws, dx, dy):
                hx[h] += w * x; hy[h] += w * y; hw[h] += w
            dx = [x - hx[h] / hw[h] for h, x in zip(hids, dx)]
            dy = [y - hy[h] / hw[h] for h, y in zip(hids, dy)]
    else:
        W = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / W
        my = sum(w * y for w, y in zip(ws, ys)) / W
        dx = [x - mx for x in xs]
        dy = [y - my for y in ys]

    Sxx = sum(w * x * x for w, x in zip(ws, dx))
    if Sxx <= 1e-12:
        return None
    Sxy = sum(w * x * y for w, x, y in zip(ws, dx, dy))
    b = Sxy / Sxx
    resid = [y - b * x for x, y in zip(dx, dy)]
    sc = defaultdict(float)
    for h, w, x, e in zip(hids, ws, dx, resid):
        sc[h] += w * x * e
    G = len(sc)
    se_cl = math.sqrt((G / (G - 1)) * sum(v * v for v in sc.values()) / (Sxx * Sxx)) if G > 1 else 0.0
    return {'n': n, 'n_hh': G, 'beta': round(b, 4), 'se_cl': round(se_cl, 4),
            't_cl': round(b / se_cl, 2) if se_cl > 0 else None}


def within_round_perm_pvalue(obs, n_perm=2000, seed=42):
    """Exact within-round Monte Carlo permutation p-value for small treated cells.
    obs: list of (rd, hhid, wl, y, w).
    """
    est = wls_fe_clustered(obs, True, False, True)
    if est is None:
        return None
    b_obs = est['beta']
    by_rd = defaultdict(list)
    for o in obs:
        by_rd[o[0]].append(o)
    # Keep only rounds with variation in wl
    active_rds = {rd: rows for rd, rows in by_rd.items() if 0 < sum(1 for r in rows if r[2] == 1.0) < len(rows)}
    if not active_rds:
        return {'beta': b_obs, 'perm_p_two_sided': 1.0, 'n_treated': 0}

    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        sxx = sxy = 0.0
        for rd, rows in active_rds.items():
            k = sum(1 for r in rows if r[2] == 1.0)
            chosen = set(rng.sample(range(len(rows)), k))
            W = sum(r[4] for r in rows)
            mx = sum(rows[i][4] for i in chosen) / W
            my = sum(r[4] * r[3] for r in rows) / W
            for idx, r in enumerate(rows):
                x = 1.0 if idx in chosen else 0.0
                dx = x - mx
                dy = r[3] - my
                sxx += r[4] * dx * dx
                sxy += r[4] * dx * dy
        b_p = sxy / sxx if sxx > 1e-12 else 0.0
        if abs(b_p) >= abs(b_obs) - 1e-12:
            ge += 1
    return {
        'beta': b_obs,
        'perm_p_two_sided': round((ge + 1) / (n_perm + 1), 4),
        'n_perm': n_perm,
    }


fb_S_to_broad, fb_H_to_strict = [], []
wl_obs_by_lag = {0: [], 1: [], 2: []}
wl_pretrend_by_lag = {0: [], 1: [], 2: []}
wl_inc_lag = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}
wl_inc_r1_29 = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}
wl_inc_r30_82 = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}
bread_obs, flour_obs = [], []

for hid, v in hh.items():
    v.sort(key=lambda x: x[0])
    for i in range(1, len(v)):
        r0, st0, y0, ly0, w0, wl0, rem0, fw0, b0, f0 = v[i - 1]
        r1, st1, y1, ly1, w1, wl1, rem1, fw1, b1, f1 = v[i]
        if r1 != r0 + 1:
            continue
        brd1 = 1.0 if st1 >= 1 else 0.0
        str1 = 1.0 if st1 == 2 else 0.0
        m1 = 1.0 if st1 == 1 else 0.0
        dy_rd = ((ly1 - rd_mean_ly[r1]) - (ly0 - rd_mean_ly[r0])) if (ly0 is not None and ly1 is not None) else None

        if dy_rd is not None:
            if st0 == 0:
                fb_S_to_broad.append((r1, hid, dy_rd, brd1, w1))
            if st0 == 2:
                fb_H_to_strict.append((r1, hid, dy_rd, str1, w1))
            if b0 and b1:
                bread_obs.append((r1, hid, dy_rd, math.log(b1) - math.log(b0), w1))
            if f0 and f1:
                flour_obs.append((r1, hid, dy_rd, math.log(f1) - math.log(f0), w1))

        if wl1 is not None:
            wl_inc_lag[st0][1] += w1
            if wl1 == 1:
                wl_inc_lag[st0][0] += w1
            tgt_p = wl_inc_r1_29 if r1 <= 29 else wl_inc_r30_82
            tgt_p[st0][1] += w1
            if wl1 == 1:
                tgt_p[st0][0] += w1

            to_zero = (1.0 if (y0 > 0 and y1 == 0) else 0.0) if y0 > 0 else None
            wl_obs_by_lag[st0].append((r1, hid, float(wl1), dy_rd, brd1, str1, m1, float(fw1), to_zero, w1, rem1))

            if i >= 2 and v[i - 2][0] == r0 - 1 and v[i - 2][3] is not None and ly0 is not None:
                dy_prev = (ly0 - rd_mean_ly[r0]) - (v[i - 2][3] - rd_mean_ly[r0 - 1])
                wl_pretrend_by_lag[st0].append((r1, hid, float(wl1), dy_prev, w1))

wl_summary = {}
for st, nm in ((0, 'S'), (1, 'M'), (2, 'H')):
    rows_s = wl_obs_by_lag[st]
    dy_obs = [(r, h, wl, dy, w) for r, h, wl, dy, _, _, _, _, _, w, _ in rows_s if dy is not None]
    brd_obs = [(r, h, wl, brd, w) for r, h, wl, _, brd, _, _, _, _, w, _ in rows_s]
    str_obs = [(r, h, wl, stv, w) for r, h, wl, _, _, stv, _, _, _, w, _ in rows_s]
    m_obs = [(r, h, wl, mv, w) for r, h, wl, _, _, _, mv, _, _, w, _ in rows_s]
    fw_obs = [(r, h, wl, fw, w) for r, h, wl, _, _, _, _, fw, _, w, _ in rows_s]
    z_obs = [(r, h, wl, z, w) for r, h, wl, _, _, _, _, _, z, w, _ in rows_s if z is not None]

    dy_rdfe_w = wls_fe_clustered(dy_obs, True, False, True)
    brd_rdfe_w = wls_fe_clustered(brd_obs, True, False, True)
    str_rdfe_w = wls_fe_clustered(str_obs, True, False, True)
    m_rdfe_w = wls_fe_clustered(m_obs, True, False, True)
    fw_rdfe_w = wls_fe_clustered(fw_obs, True, False, True)
    z_rdfe_w = wls_fe_clustered(z_obs, True, False, True)

    brd_rdfe_uw = wls_fe_clustered(brd_obs, True, False, False)
    str_rdfe_uw = wls_fe_clustered(str_obs, True, False, False)
    dy_hhfe_w = wls_fe_clustered(dy_obs, True, True, True)
    brd_hhfe_w = wls_fe_clustered(brd_obs, True, True, True)
    str_hhfe_w = wls_fe_clustered(str_obs, True, True, True)
    pretrend_w = wls_fe_clustered(wl_pretrend_by_lag[st], True, False, True)

    # Subperiod split: r1--29 vs r30--82
    str_r1_29 = wls_fe_clustered([o for o in str_obs if o[0] <= 29], True, False, True)
    str_r30_82 = wls_fe_clustered([o for o in str_obs if o[0] >= 30], True, False, True)
    brd_r30_82 = wls_fe_clustered([o for o in brd_obs if o[0] >= 30], True, False, True)
    dy_r30_82 = wls_fe_clustered([o for o in dy_obs if o[0] >= 30], True, False, True)

    # Remittance split with explicit treated cell counts + exact permutation p-values!
    rem0_obs = [(r, h, wl, stv, w) for r, h, wl, _, _, stv, _, _, _, w, rem in rows_s if rem == 0]
    rem1_obs = [(r, h, wl, stv, w) for r, h, wl, _, _, stv, _, _, _, w, rem in rows_s if rem == 1]
    str_rem0 = wls_fe_clustered(rem0_obs, True, False, True)
    rem1_treated = [o for o in rem1_obs if o[2] == 1.0]
    rem1_strict_hits = sum(1 for o in rem1_treated if o[3] == 1.0)
    rem1_perm = within_round_perm_pvalue(rem1_obs, n_perm=2000, seed=42 + st)
    rem1_perm['n_total'] = len(rem1_obs)
    rem1_perm['n_treated_wl1'] = len(rem1_treated)
    rem1_perm['strict_entries_among_treated'] = rem1_strict_hits
    rem1_perm['raw_strict_rate_treated'] = round(rem1_strict_hits / len(rem1_treated), 4) if rem1_treated else None

    pct_cont = math.exp(dy_rdfe_w['beta']) - 1.0
    dz = max(0.0, z_rdfe_w['beta'])
    pct_tot = (1.0 - dz) * pct_cont - dz

    wl_summary[nm] = {
        'work_loss_rate_w': round(wl_inc_lag[st][0] / wl_inc_lag[st][1], 4),
        'work_loss_rate_r1_29_w': round(wl_inc_r1_29[st][0] / wl_inc_r1_29[st][1], 4),
        'work_loss_rate_r30_82_w': round(wl_inc_r30_82[st][0] / wl_inc_r30_82[st][1], 4),
        'n_wl1': sum(1 for r in rows_s if r[2] == 1.0),
        'n_wl1_r30_82': sum(1 for r in rows_s if r[2] == 1.0 and r[0] >= 30),
        'n_total': len(rows_s),
        'primary_round_fe_weighted': {
            'dlogY_within_hh': dy_rdfe_w,
            'implied_pct_Y_drop_intensive': round(pct_cont, 4),
            'dP_pos_to_zero': z_rdfe_w,
            'implied_pct_Y_drop_total_incl_zero': round(pct_tot, 4),
            'dP_broad': brd_rdfe_w,
            'dP_M': m_rdfe_w,
            'dP_strict': str_rdfe_w,
            'dP_finworse': fw_rdfe_w,
        },
        'robustness': {
            'dP_broad_rdfe_uw': brd_rdfe_uw,
            'dP_strict_rdfe_uw': str_rdfe_uw,
            'dlogY_round_plus_hh_fe_w': dy_hhfe_w,
            'dP_broad_round_plus_hh_fe_w': brd_hhfe_w,
            'dP_strict_round_plus_hh_fe_w': str_hhfe_w,
            'pretrend_dy_lag1_on_future_wl': pretrend_w,
            'dP_strict_r1_29': str_r1_29,
            'dP_strict_r30_82': str_r30_82,
            'dP_broad_r30_82': brd_r30_82,
            'dlogY_r30_82': dy_r30_82,
            'dP_strict_no_remit': str_rem0,
            'dP_strict_with_remit_exact': rem1_perm,
        },
    }

rs_dep = sorted(dep_amt_ratios)
n_dep_tot = sum(dep_amt_states.values())
n_rem_use_tot = sum(rem_use.values())

out = {
    'linkage': {
        'linked_hh_rounds': linked_n,
        'missing_hh_rounds': missing_n,
        'link_rate': round(linked_n / (linked_n + missing_n), 4),
        'mig_living_remittance_yes': sum(v[1] for v in ind_exp.values()),
        'mig_living_remittance_share': round(sum(v[1] for v in ind_exp.values()) / len(ind_exp), 4),
        'mig_remittance_nonzero_amount_rows': rem_amt_nonzero,
        'work_loss_by_period_counts': dict(wl_by_period),
        'work_loss_share_in_r1_29': round(wl_by_period['r1_29'] / sum(wl_by_period.values()), 4),
    },
    'remittance_primary_use': {
        'n_reports': n_rem_use_tot,
        'top_uses': {k: {'n': v, 'share': round(v / n_rem_use_tot, 4)} for k, v in rem_use.most_common(8)},
    },
    'bank_card_ownership_weighted': {
        nm: round(bank_by_st[i][0] / bank_by_st[i][1], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
    },
    'feedback_noisy_dlogy_round_fe_w': {
        'S_to_broad': wls_fe_clustered(fb_S_to_broad, True, False, True),
        'H_to_strict': wls_fe_clustered(fb_H_to_strict, True, False, True),
    },
    'work_loss_by_lagged_state': wl_summary,
    'deposit_and_cash_saving': {
        'deposit_past30d_rate_by_state': {
            nm: round(dep_past_by_st[i][0] / dep_past_by_st[i][1], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
        },
        'saving_next30d_rate_by_state': {
            nm: round(sav_next_by_st[i][0] / sav_next_by_st[i][1], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
        },
        'saving_next30d_total_n': n_dep_tot,
        'saving_next30d_state_composition': {
            nm: {'n': dep_amt_states[i], 'share': round(dep_amt_states[i] / n_dep_tot, 4)}
            for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
        },
        'deposit_amount_n_pos_inc': len(rs_dep),
        'amt_to_income_median': round(statistics.median(rs_dep), 3),
        'amt_to_income_p10': round(rs_dep[len(rs_dep) // 10], 3),
        'amt_to_income_p90': round(rs_dep[9 * len(rs_dep) // 10], 3),
        'cash_at_home_share': round(dep_forms['In cash at home'] / sum(dep_forms.values()), 4),
    },
    'staples_validation': {
        'bread_7d_round_fe_cl_uw': wls_fe_clustered(bread_obs, True, False, False),
        'flour_7d_round_fe_cl_uw': wls_fe_clustered(flour_obs, True, False, False),
        'bread_7d_round_fe_cl_w': wls_fe_clustered(bread_obs, True, False, True),
        'flour_7d_round_fe_cl_w': wls_fe_clustered(flour_obs, True, False, True),
        'reporter_balance': {
            'population_state_shares': {nm: round(all_st[i] / sum(all_st.values()), 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
            'flour_reporter_state_shares': {nm: round(flour_rep_st[i] / sum(flour_rep_st.values()), 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
            'mean_real_logY_pop': round(statistics.mean(all_ly), 4),
            'mean_real_logY_flour': round(statistics.mean(flour_ly), 4),
        },
    },
}

out_path = os.path.join(OUT_DIR, 'exposure_validation.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"wrote {out_path}")
for nm in ('S', 'M', 'H'):
    r = wl_summary[nm]
    rem = r['robustness']['dP_strict_with_remit_exact']
    r30 = r['robustness']['dP_strict_r30_82']
    print(f"  {nm}: rate_all={r['work_loss_rate_w']:.4f} | r1_29={r['work_loss_rate_r1_29_w']:.4f} | "
          f"r30_82={r['work_loss_rate_r30_82_w']:.4f} (n_wl1={r['n_wl1_r30_82']}, dP_str={r30['beta']:+.3f} SE {r30['se_cl']:.3f}) | "
          f"remit_treated={rem['strict_entries_among_treated']}/{rem['n_treated_wl1']} (beta={rem['beta']:+.3f}, perm_p={rem['perm_p_two_sided']})")
