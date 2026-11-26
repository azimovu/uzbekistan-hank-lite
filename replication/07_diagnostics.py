"""Step 07: L2CU Audience Profile, Forensic Checks (r29 & r44), Weighted 3-State HMM
Misclassification Correction (Post-r29 & Q10-Excluded with Bootstrap SEs and Feasible
Bayesian Transfer Targeting), M1 Validation, Lagged t-2/t-3 Job-Loss Exposure & Event
Study, Patterson Cyclicality, and Remittance-Inclusive Robustness.

Pure Python 3.12 standard library. Writes outputs/diagnostics.json.
"""
import json
import math
import os
import random
from collections import Counter, defaultdict

from common import (
    BREAK_ROUNDS,
    OUT_DIR,
    SUBPERIODS,
    clean,
    clustered_mean_se,
    ensure_out_dir,
    htm_flags,
    iter_household_rows,
    iter_individual_rows,
    load_monthly_cpi,
    parse_float,
    subperiod_of,
    totinc,
    weighted_quantile,
)

ensure_out_dir()
cpi = load_monthly_cpi()

FX_USD = {'2018': 8100.0, '2019': 8850.0, '2020': 10050.0, '2021': 10610.0,
          '2022': 11050.0, '2023': 11740.0, '2024': 12650.0, '2025': 12900.0}
FX_RUB = {'2018': 128.0, '2019': 137.0, '2020': 139.0, '2021': 144.0,
          '2022': 162.0, '2023': 138.0, '2024': 137.0, '2025': 142.0}
FX_KZT = {'2018': 23.0, '2019': 23.0, '2020': 24.0, '2021': 25.0,
          '2022': 24.0, '2023': 26.0, '2024': 27.0, '2025': 25.0}


def rd_fe_ols(obs):
    by_rd = defaultdict(lambda: [0.0, 0.0, 0.0])
    for hid, rd, w, y, x in obs:
        by_rd[rd][0] += w
        by_rd[rd][1] += w * y
        by_rd[rd][2] += w * x
    ym = {rd: v[1] / v[0] for rd, v in by_rd.items() if v[0] > 0}
    xm = {rd: v[2] / v[0] for rd, v in by_rd.items() if v[0] > 0}
    num = den = 0.0
    for hid, rd, w, y, x in obs:
        dx = x - xm[rd]
        dy = y - ym[rd]
        num += w * dx * dy
        den += w * dx * dx
    if den <= 1e-12:
        return {'n': len(obs), 'n_hh': 0, 'beta': 0.0, 'se_cl': 0.0, 't_cl': 0.0}
    b = num / den
    scores = defaultdict(float)
    for hid, rd, w, y, x in obs:
        dx = x - xm[rd]
        dy = y - ym[rd]
        scores[hid] += w * dx * (dy - b * dx)
    G = len(scores)
    var = (G / max(G - 1, 1)) * sum(s * s for s in scores.values()) / (den * den)
    se = math.sqrt(max(var, 0.0))
    return {
        'n': len(obs),
        'n_hh': G,
        'beta': round(b, 4),
        'se_cl': round(se, 4),
        't_cl': round(b / se, 2) if se > 0 else 0.0,
    }


def mat_mul(A, B):
    K = len(A)
    return [[sum(A[i][m] * B[m][j] for m in range(K)) for j in range(K)] for i in range(K)]


def mat_pow(A, p):
    K = len(A)
    res = [[1.0 if i == j else 0.0 for j in range(K)] for i in range(K)]
    for _ in range(p):
        res = mat_mul(res, A)
    return res


def fit_weighted_hmm(runs, n_iter=60, init_params=None):
    """Population-weighted 3-state Baum-Welch HMM on contiguous runs (obs, wt, hid)."""
    K = 3
    if init_params is not None:
        pi = init_params[0][:]
        A = [r[:] for r in init_params[1]]
        E = [r[:] for r in init_params[2]]
    else:
        pi = [0.40, 0.45, 0.15]
        A = [[0.92, 0.06, 0.02], [0.05, 0.90, 0.05], [0.03, 0.12, 0.85]]
        E = [[0.85, 0.13, 0.02], [0.10, 0.85, 0.05], [0.03, 0.15, 0.82]]
    mean_w = sum(rw for _, rw, _ in runs) / len(runs)
    ll = 0.0
    for _ in range(n_iter):
        pi_acc = [0.0] * K
        A_acc = [[0.0] * K for _ in range(K)]
        E_acc = [[0.0] * K for _ in range(K)]
        ll = 0.0
        for obs, rw, _ in runs:
            wt = rw / mean_w
            T_len = len(obs)
            alpha = [[0.0] * K for _ in range(T_len)]
            c_sc = [0.0] * T_len
            o0 = obs[0]
            s = 0.0
            for i in range(K):
                alpha[0][i] = pi[i] * E[i][o0]
                s += alpha[0][i]
            c_sc[0] = s
            ll += wt * math.log(s)
            for i in range(K):
                alpha[0][i] /= s
            for t_i in range(1, T_len):
                ot = obs[t_i]
                s = 0.0
                ap = alpha[t_i - 1]
                for j in range(K):
                    val = (ap[0] * A[0][j] + ap[1] * A[1][j] + ap[2] * A[2][j]) * E[j][ot]
                    alpha[t_i][j] = val
                    s += val
                c_sc[t_i] = s
                ll += wt * math.log(s)
                for j in range(K):
                    alpha[t_i][j] /= s
            beta = [[1.0] * K for _ in range(T_len)]
            for t_i in range(T_len - 2, -1, -1):
                ot1 = obs[t_i + 1]
                sc = c_sc[t_i + 1]
                bn = beta[t_i + 1]
                for i in range(K):
                    beta[t_i][i] = (
                        A[i][0] * E[0][ot1] * bn[0]
                        + A[i][1] * E[1][ot1] * bn[1]
                        + A[i][2] * E[2][ot1] * bn[2]
                    ) / sc
            for i in range(K):
                pi_acc[i] += wt * alpha[0][i] * beta[0][i]
            for t_i in range(T_len):
                ot = obs[t_i]
                for i in range(K):
                    E_acc[i][ot] += wt * alpha[t_i][i] * beta[t_i][i]
            for t_i in range(T_len - 1):
                ot1 = obs[t_i + 1]
                sc = c_sc[t_i + 1]
                at = alpha[t_i]
                bn = beta[t_i + 1]
                for i in range(K):
                    for j in range(K):
                        A_acc[i][j] += wt * at[i] * A[i][j] * E[j][ot1] * bn[j] / sc
        s_pi = sum(pi_acc)
        pi = [x / s_pi for x in pi_acc]
        for i in range(K):
            sa = sum(A_acc[i])
            se = sum(E_acc[i])
            A[i] = [x / sa for x in A_acc[i]]
            E[i] = [x / se for x in E_acc[i]]

    # Stationary distribution of A
    v = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    for _ in range(500):
        v = [sum(v[i] * A[i][j] for i in range(K)) for j in range(K)]

    # Asymptotic observed shares (pi* E) and finite-horizon length-weighted predicted observed shares
    pred_asymp = [sum(v[i] * E[i][j] for i in range(K)) for j in range(K)]
    max_T = max(len(obs) for obs, _, _ in runs)
    # State distribution at step t starting from pi0
    dist_t = [pi[:]]
    for _ in range(1, max_T):
        prev = dist_t[-1]
        dist_t.append([sum(prev[i] * A[i][j] for i in range(K)) for j in range(K)])
    pred_fin = [0.0, 0.0, 0.0]
    tot_wt_len = 0.0
    for obs, rw, _ in runs:
        T_len = len(obs)
        tot_wt_len += rw * T_len
        for t_i in range(T_len):
            d = dist_t[t_i]
            for j in range(K):
                pred_fin[j] += rw * (d[0] * E[0][j] + d[1] * E[1][j] + d[2] * E[2][j])
    pred_fin = [x / tot_wt_len for x in pred_fin]
    return pi, A, E, v, pred_asymp, pred_fin, ll


def _boot_worker(args):
    by_hh, hids, draw_seed, init_params = args
    G = len(hids)
    rng = random.Random(draw_seed)
    sample_runs = []
    for _ in range(G):
        h = hids[rng.randrange(G)]
        sample_runs.extend(by_hh[h])
    _, Ab, Eb, vb, _, _, _ = fit_weighted_hmm(sample_runs, n_iter=35, init_params=init_params)
    return Ab, Eb, vb


def bootstrap_hmm(runs, init_params, n_boot=50, seed=12345):
    """Parallel household block-bootstrap SEs (warm-started from MLE) for weighted 3-state HMM."""
    import multiprocessing

    by_hh = defaultdict(list)
    for rec in runs:
        by_hh[rec[2]].append(rec)
    by_hh = dict(by_hh)
    hids = sorted(by_hh.keys())
    tasks = [(by_hh, hids, seed + b * 97, init_params) for b in range(n_boot)]
    n_workers = min(os.cpu_count() or 1, 8)
    if n_workers > 1 and hasattr(os, 'fork'):
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(processes=n_workers) as pool:
            res = pool.map(_boot_worker, tasks)
    else:
        res = [_boot_worker(t) for t in tasks]

    A_draws = [r[0] for r in res]
    E_draws = [r[1] for r in res]
    pi_draws = [r[2] for r in res]

    def sd(vals):
        mu = sum(vals) / len(vals)
        return math.sqrt(sum((x - mu) ** 2 for x in vals) / max(len(vals) - 1, 1))

    A_se = [[round(sd([d[i][j] for d in A_draws]), 4) for j in range(3)] for i in range(3)]
    E_se = [[round(sd([d[i][j] for d in E_draws]), 4) for j in range(3)] for i in range(3)]
    pi_se = [round(sd([d[j] for d in pi_draws]), 4) for j in range(3)]
    broad_se = round(sd([d[1] + d[2] for d in pi_draws]), 4)
    return {'A_se': A_se, 'E_se': E_se, 'pi_se': pi_se, 'broad_se': broad_se, 'n_boot': n_boot}


def summarize_hmm(runs, obs_w_counts, n_boot=50, seed=12345):
    pi0, A, E, v, pred_asymp, pred_fin, ll = fit_weighted_hmm(runs, n_iter=60)
    bse = bootstrap_hmm(runs, (pi0, A, E), n_boot=n_boot, seed=seed)
    tot_o = sum(obs_w_counts)
    obs_sh = [x / tot_o for x in obs_w_counts]

    # Feasible Bayesian targeting from observed H_{t-lag} using consistent HMM universal denominator
    den_H = sum(v[k] * E[k][2] for k in range(3))
    post_H = [v[i] * E[i][2] / den_H for i in range(3)]
    A1 = mat_pow(A, 1)
    A3 = mat_pow(A, 3)
    dist_lag1 = [sum(post_H[i] * A1[i][j] for i in range(3)) for j in range(3)]
    dist_lag3 = [sum(post_H[i] * A3[i][j] for i in range(3)) for j in range(3)]

    targeting = {}
    for mM, tag in ((1.0, 'mM100'), (0.85, 'mM85'), (0.50, 'mM50')):
        m = [0.07, mM, 1.0]
        univ = sum(v[j] * m[j] for j in range(3))
        b1 = sum(dist_lag1[j] * m[j] for j in range(3))
        b3 = sum(dist_lag3[j] * m[j] for j in range(3))
        o3 = sum(A3[2][j] * m[j] for j in range(3))
        targeting[tag] = {
            'hmm_universal_mpc': round(univ, 4),
            'bayes_lag1_mpc': round(b1, 4),
            'bayes_lag1_lift_pct': round(100.0 * (b1 / univ - 1.0), 1),
            'bayes_lag3_mpc': round(b3, 4),
            'bayes_lag3_lift_pct': round(100.0 * (b3 / univ - 1.0), 1),
            'oracle_lag3_mpc': round(o3, 4),
            'oracle_lag3_lift_pct': round(100.0 * (o3 / univ - 1.0), 1),
        }

    return {
        'n_contiguous_runs': len(runs),
        'log_likelihood_weighted': round(ll, 1),
        'observed_weighted_shares': [round(x, 4) for x in obs_sh],
        'predicted_finite_horizon_shares': [round(x, 4) for x in pred_fin],
        'predicted_asymptotic_piE': [round(x, 4) for x in pred_asymp],
        'initial_pi0': [round(x, 4) for x in pi0],
        'latent_stationary_pi': [round(x, 4) for x in v],
        'latent_broad_htm': round(v[1] + v[2], 4),
        'latent_A': [[round(x, 4) for x in row] for row in A],
        'latent_expected_spells_mo': {
            nm: round(1.0 / (1.0 - A[i][i]), 1) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
        },
        'emission_E': [[round(x, 4) for x in row] for row in E],
        'bootstrap_se': bse,
        'posterior_given_obs_H': [round(x, 4) for x in post_H],
        'dist_3m_after_obs_H': [round(x, 4) for x in dist_lag3],
        'latent_strict_retention_3m': round(A3[2][2], 4),
        'bayesian_feasible_targeting': targeting,
    }


# ---------------------------------------------------------------------------
# Pass 1: Read household panel once
# ---------------------------------------------------------------------------
hh = defaultdict(dict)
hh_yr = {}
hh_person_weight = {}

dep_form = Counter()
bank_use = {col: [0.0, 0.0] for col in ('bank_receive_past30d', 'bank_purchase_past30d', 'bank_atm_past30d')}
bank_card_by_r = defaultdict(lambda: [0.0, 0.0])
sav_next_yes_by_r = Counter()
sav_next_n_by_r = Counter()
cig_yes_w = cig_tot_w = 0.0
cig_yes_n = cig_tot_n = 0
hhsize_by_r = defaultdict(lambda: [0.0, 0.0])
r82_pos_y = []
r82_wages = []
r82_ben_w = r82_pen_w = r82_wage_w = r82_tot_w = 0.0

nocons_sub = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
nocons_r20 = [0.0, 0.0, 0.0, 0.0, 0.0]
nocons_all = [0.0, 0.0, 0.0, 0.0, 0.0]

ext_markers = {grp: defaultdict(lambda: defaultdict(list)) for grp in ('S', 'M1', 'M2', 'H')}
rd_mean_lny = defaultdict(lambda: [0.0, 0.0])
rd_real_y_all = defaultdict(list)

obs_w_core_r30 = [0.0, 0.0, 0.0]
obs_w_win_r30 = [0.0, 0.0, 0.0]
obs_w_nc_all = [0.0, 0.0, 0.0]
earner_w_by_st = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}

for row in iter_household_rows():
    rd = int(float(row['round']))
    hid = row['hhid']
    w = parse_float(row.get('popw', ''))
    if w <= 0:
        w = 1.0
    ym = clean(row.get('date_start', ''))[:7]
    yr = ym[:4]
    hh_yr[(hid, rd)] = yr
    p = cpi.get(ym, 100.0)
    sz_s = clean(row.get('hhsize', ''))
    hsz = max(1.0, float(sz_s)) if sz_s else 1.0
    if rd in (1, 82):
        hh_person_weight[(rd, hid)] = w / hsz
    if sz_s:
        hhsize_by_r[rd][0] += hsz * w
        hhsize_by_r[rd][1] += w

    f = htm_flags(row)
    y_nom = totinc(row)
    y_real = y_nom * 100.0 / p
    lny = math.log(y_real) if y_real > 0 else None

    bread = parse_float(row.get('bread_past7d_cost', ''))
    flour = parse_float(row.get('flour_past7d_cost', ''))
    bread_r = bread * 100.0 / p if bread > 0 else None
    flour_r = flour * 100.0 / p if flour > 0 else None
    staple_r = (bread_r or 0.0) + (flour_r or 0.0)
    if (bread_r is None and flour_r is None) or staple_r <= 0:
        staple_r = None

    med = int(clean(row.get('medical_past30d', '')) == 'Yes')
    econ_bad = int(clean(row.get('economic_condition', '')) in ('Bad', 'Very bad'))
    save_past = int(clean(row.get('saving_next30d', '')) == 'Yes')

    strict_nc = f['nosav'] and (f['loan'] or f['assets'] or f['foodno'])
    broad_nc = f['nosav'] or f['loan'] or f['assets']
    st_nc = 2 if strict_nc else (1 if broad_nc else 0)

    hh[hid][rd] = {
        'w': w, 'hsz': hsz, 'p': p, 'st': f['st_core'], 'st_nc': st_nc, 'sv': f['sv'],
        'nosav': f['nosav'], 'took_sav': f['took_sav'], 'coping_hard': f['coping_hard'],
        'foodno': f['foodno'],
        'is_m1': f['is_m1'], 'is_m2': f['is_m2'],
        'y_real': y_real, 'lny': lny,
        'loan': int(f['loan']), 'assets': int(f['assets']), 'cons': int(f['cons']),
        'util': int(f['util']), 'fin_worse': int(f['fin_worse']),
        'fies': (int(f['fies_any']) if f['fies_any'] is not None else None),
        'med': med, 'econ_bad': econ_bad, 'save_past': save_past,
        'bread_r': bread_r, 'flour_r': flour_r, 'staple_r': staple_r,
    }
    has_earn = (parse_float(row.get('wage_amount', '')) +
                parse_float(row.get('selfempinc_amount', '')) +
                parse_float(row.get('aginc_amount', ''))) > 0.0
    earner_w_by_st[f['st_core']][1] += w
    if has_earn:
        earner_w_by_st[f['st_core']][0] += w
    if lny is not None:
        rd_mean_lny[rd][0] += w * lny
        rd_mean_lny[rd][1] += w
    rd_real_y_all[rd].append((y_real, w, f['st_core'], hsz))

    obs_w_nc_all[st_nc] += w
    if rd >= 30:
        obs_w_core_r30[f['st_core']] += w

    sp = subperiod_of(rd)
    for arr in (nocons_sub[sp], nocons_all):
        arr[0] += w
        arr[1] += w * float(f['strict_core'])
        arr[2] += w * float(strict_nc)
        arr[3] += w * float(f['broad_core'])
        arr[4] += w * float(broad_nc)
    if rd == 20:
        nocons_r20[0] += w
        nocons_r20[1] += w * float(f['strict_core'])
        nocons_r20[2] += w * float(strict_nc)
        nocons_r20[3] += w * float(f['broad_core'])
        nocons_r20[4] += w * float(broad_nc)

    grp = 'S' if f['st_core'] == 0 else ('M1' if f['is_m1'] else ('M2' if f['is_m2'] else 'H'))
    for nm, val in (
        ('med', float(med)),
        ('util', float(f['util'])),
        ('econ_bad', float(econ_bad)),
        ('fin_worse', float(f['fin_worse'])),
        ('save_past30d', float(save_past)),
    ):
        ext_markers[grp][nm][hid].append((val, w))
    if f['fies_any'] is not None:
        ext_markers[grp]['fies_r49_82'][hid].append((float(f['fies_any']), w))

    df = clean(row.get('deposit_form', ''))
    if df:
        dep_form[df] += 1
    sn = clean(row.get('saving_next30d', ''))
    if sn:
        sav_next_n_by_r[rd] += 1
        if sn == 'Yes':
            sav_next_yes_by_r[rd] += 1
    bc = clean(row.get('bank_card_hh', ''))
    if bc in ('Yes', 'No'):
        bank_card_by_r[rd][1] += w
        if bc == 'Yes':
            bank_card_by_r[rd][0] += w
    for col in bank_use:
        v = clean(row.get(col, ''))
        if v in ('Yes', 'No'):
            bank_use[col][1] += w
            if v == 'Yes':
                bank_use[col][0] += w
    cg = clean(row.get('cigarette', ''))
    if cg in ('Yes', 'No'):
        cig_tot_w += w
        cig_tot_n += 1
        if cg == 'Yes':
            cig_yes_w += w
            cig_yes_n += 1
    if rd == 82:
        r82_tot_w += w
        if y_nom > 0:
            r82_pos_y.append((y_nom, w))
        wg = parse_float(row.get('wage_amount', ''))
        if wg > 0:
            r82_wages.append((wg, w))
            r82_wage_w += w
        if parse_float(row.get('pension_amount', '')) > 0:
            r82_pen_w += w
        if parse_float(row.get('benefit_amount', '')) > 0:
            r82_ben_w += w

rd_lny_bar = {rd: v[0] / v[1] for rd, v in rd_mean_lny.items()}

# ---------------------------------------------------------------------------
# Pass 2: Read individual file once
# ---------------------------------------------------------------------------
age_r = {1: [], 82: []}
edu_r_w = {1: defaultdict(float), 82: defaultdict(float)}
sex_r_w = {1: defaultdict(float), 82: defaultdict(float)}
work_type_r82_w = defaultdict(float)

wl_map = defaultdict(int)
ind_cov = set()
rem_som = defaultdict(float)

for row in iter_individual_rows():
    rd = int(float(row['round']))
    hid = row['hhid']
    key = (hid, rd)
    ind_cov.add(key)
    if clean(row.get('work_lost_hh', '')) == 'Yes':
        wl_map[key] = 1

    yr = hh_yr.get(key, '2021')
    for acol, ccol in (
        ('mig_living_remittance_amount', 'mig_living_remittance_currency'),
        ('remittance_hh_amount', 'remittance_hh_currency'),
    ):
        amt = parse_float(row.get(acol, ''))
        if amt > 0:
            cur = clean(row.get(ccol, ''))
            if cur == 'Dollar':
                val = amt * FX_USD.get(yr, 11000.0)
            elif cur == 'Uzbek soum':
                val = amt
            elif cur == 'Ruble':
                val = amt * FX_RUB.get(yr, 140.0)
            elif cur == 'Tenge':
                val = amt * FX_KZT.get(yr, 25.0)
            else:
                val = 0.0
            if val < 5e8:
                rem_som[key] += val

    if rd in (1, 82):
        pw = hh_person_weight.get((rd, hid), 0.0)
        if pw > 0.0:
            ag_s = clean(row.get('age', ''))
            ag = None
            if ag_s:
                try:
                    ag = float(ag_s)
                    age_r[rd].append((ag, pw))
                except ValueError:
                    pass
            sx = clean(row.get('gender', ''))
            if sx:
                sex_r_w[rd][sx] += pw
            ed = clean(row.get('education', ''))
            if ag is not None and ag >= 25 and ed:
                edu_r_w[rd][ed] += pw
            if rd == 82:
                wt = clean(row.get('work_type_new', ''))
                if wt:
                    work_type_r82_w[wt] += pw

for hid, rds in hh.items():
    for rd, cur in rds.items():
        cur['wl'] = wl_map.get((hid, rd), 0)
        cur['in_ind'] = (hid, rd) in ind_cov

# ---------------------------------------------------------------------------
# 1. Audience profile
# ---------------------------------------------------------------------------
ind_stats = {}
for rd in (1, 82):
    ap = age_r[rd]
    W_a = sum(w for _, w in ap)
    ed_tot = sum(edu_r_w[rd].values())
    hi_ed = edu_r_w[rd]['Higher education/bachelor'] + edu_r_w[rd]['Graduate school/aspirantura']
    sec_sp = edu_r_w[rd]['Secondary special'] + edu_r_w[rd]['Secondary technical']
    sec_gn = edu_r_w[rd]['Secondary general']
    ind_stats[f'r{rd:02d}'] = {
        'n_persons': len(ap),
        'mean_age_w': round(sum(a * w for a, w in ap) / W_a, 1),
        'share_under18_w': round(sum(w for a, w in ap if a < 18) / W_a, 4),
        'share_60plus_w': round(sum(w for a, w in ap if a >= 60) / W_a, 4),
        'male_share_w': round(sex_r_w[rd]['Male'] / sum(sex_r_w[rd].values()), 4),
        'adult25_higher_ed_w': round(hi_ed / ed_tot, 4),
        'adult25_sec_special_tech_w': round(sec_sp / ed_tot, 4),
        'adult25_sec_general_w': round(sec_gn / ed_tot, 4),
        'weighted_hhsize': round(hhsize_by_r[rd][0] / hhsize_by_r[rd][1], 2),
    }

dep_tot = sum(dep_form.values())
bank_dep_n = dep_form['Time deposit at the bank'] + dep_form['Saving account at the bank']
wt_tot_w = sum(work_type_r82_w.values())

audience = {
    'demographics_weighted': ind_stats,
    'r82_work_type_shares_weighted': {
        k: round(v / wt_tot_w, 4) for k, v in sorted(work_type_r82_w.items(), key=lambda x: -x[1])
    },
    'r82_income_profile_weighted': {
        'pos_total_Y_mean': round(sum(x * w for x, w in r82_pos_y) / sum(w for _, w in r82_pos_y)),
        'pos_total_Y_median': round(weighted_quantile(r82_pos_y, 0.50)),
        'pos_total_Y_p90': round(weighted_quantile(r82_pos_y, 0.90)),
        'wage_earner_mean': round(sum(x * w for x, w in r82_wages) / sum(w for _, w in r82_wages)),
        'wage_earner_median': round(weighted_quantile(r82_wages, 0.50)),
        'hh_share_with_wage': round(r82_wage_w / r82_tot_w, 4),
        'hh_share_with_pension': round(r82_pen_w / r82_tot_w, 4),
        'hh_share_with_benefit': round(r82_ben_w / r82_tot_w, 4),
    },
    'banking_rails_weighted': {
        'bank_card_r40': round(bank_card_by_r[40][0] / bank_card_by_r[40][1], 4),
        'bank_card_r82': round(bank_card_by_r[82][0] / bank_card_by_r[82][1], 4),
        'card_receive_past30d': round(bank_use['bank_receive_past30d'][0] / bank_use['bank_receive_past30d'][1], 4),
        'card_purchase_past30d': round(bank_use['bank_purchase_past30d'][0] / bank_use['bank_purchase_past30d'][1], 4),
        'card_atm_past30d': round(bank_use['bank_atm_past30d'][0] / bank_use['bank_atm_past30d'][1], 4),
    },
    'deposit_form_breakdown': {
        'total_n': dep_tot,
        'cash_at_home_share': round(dep_form['In cash at home'] / dep_tot, 4),
        'formal_bank_deposit_share': round(bank_dep_n / dep_tot, 4),
        'rosca_share': round(dep_form['With group savings/rosca'] / dep_tot, 4),
        'counts': dict(dep_form.most_common()),
    },
    'saving_past30d_fatigue': {
        'r01_yes_n': sav_next_yes_by_r[1],
        'r01_yes_share': round(sav_next_yes_by_r[1] / sav_next_n_by_r[1], 4),
        'r82_yes_n': sav_next_yes_by_r[82],
        'r82_yes_share': round(sav_next_yes_by_r[82] / sav_next_n_by_r[82], 4),
        'r75_82_avg_yes_share': round(
            sum(sav_next_yes_by_r[r] for r in range(75, 83)) / sum(sav_next_n_by_r[r] for r in range(75, 83)), 4
        ),
    },
    'cigarette_purchase_7d': {
        'weighted_yes_share': round(cig_yes_w / cig_tot_w, 4),
        'unweighted_yes_n': cig_yes_n,
        'unweighted_tot_n': cig_tot_n,
    },
}

# ---------------------------------------------------------------------------
# 2. Forensic additions: r43->r44 migration & Q10-excluded robustness
# ---------------------------------------------------------------------------
r43_44_matched = 0
r44_cell_src = Counter()
r43_44_nosav_to_no = 0
r42_43_cell_src = Counter()
r42_43_nosav_to_no = 0
r43_44_spillover_hids = []
for hid, rds in hh.items():
    if 43 in rds and 44 in rds:
        r43_44_matched += 1
        sv43 = rds[43]['sv']
        sv44 = rds[44]['sv']
        if sv43 == 'Do not have any savings' and sv44 == 'No':
            r43_44_nosav_to_no += 1
        if sv44 == 'No' and rds[44]['foodno']:
            r44_cell_src[sv43] += 1
            if sv43 == 'Do not have any savings':
                r43_44_spillover_hids.append(hid)
    if 42 in rds and 43 in rds:
        sv42 = rds[42]['sv']
        sv43 = rds[43]['sv']
        if sv42 == 'Do not have any savings' and sv43 == 'No':
            r42_43_nosav_to_no += 1
        if sv43 == 'No' and rds[43]['foodno']:
            r42_43_cell_src[sv42] += 1

# Persistence of the 75 r43->r44 spillover households in rounds 45-47
r44_spill_persist = {}
for f_rd in (45, 46, 47):
    n_obs = n_anom = n_nosav = 0
    for hid in r43_44_spillover_hids:
        if f_rd in hh[hid]:
            n_obs += 1
            if hh[hid][f_rd]['sv'] == 'No' and hh[hid][f_rd]['foodno']:
                n_anom += 1
            if hh[hid][f_rd]['nosav']:
                n_nosav += 1
    r44_spill_persist[f'r{f_rd}'] = {
        'n_observed': n_obs,
        'still_untouched_foodno': n_anom,
        'share_still_anom': round(n_anom / max(1, n_obs), 4),
        'reverted_to_nosav': n_nosav,
        'share_reverted_nosav': round(n_nosav / max(1, n_obs), 4),
    }

# Round-75 -> Round-76 refresh decomposition (continuing vs new entrants)
r75_cont_w = r75_cont_b = 0.0
r76_cont_w = r76_cont_b = 0.0
r76_new_w = r76_new_b = 0.0
r76_cont_n = r76_new_n = 0
for hid, rds in hh.items():
    in75 = 75 in rds
    in76 = 76 in rds
    if in75 and in76:
        r76_cont_n += 1
        w75, w76 = rds[75]['w'], rds[76]['w']
        r75_cont_w += w75
        if rds[75]['st'] >= 1:
            r75_cont_b += w75
        r76_cont_w += w76
        if rds[76]['st'] >= 1:
            r76_cont_b += w76
    elif in76 and not in75:
        r76_new_n += 1
        w76 = rds[76]['w']
        r76_new_w += w76
        if rds[76]['st'] >= 1:
            r76_new_b += w76

# Job-loss monthly rate audit around round 29 & across subperiods
wl_round_w = defaultdict(lambda: [0.0, 0.0, 0])
wl_sub_w = defaultdict(lambda: [0.0, 0.0])
for hid, rds in hh.items():
    for rd, cur in rds.items():
        if not cur['in_ind']:
            continue
        w = cur['w']
        wl_v = float(cur['wl'])
        wl_round_w[rd][0] += w * wl_v
        wl_round_w[rd][1] += w
        wl_round_w[rd][2] += int(cur['wl'])
        grp = 'r1_19' if rd <= 19 else ('r20_29' if rd <= 29 else 'r30_82')
        wl_sub_w[grp][0] += w * wl_v
        wl_sub_w[grp][1] += w

forensic_ext = {
    'r43_to_r44_migration': {
        'matched_n': r43_44_matched,
        'r44_untouched_foodno_total': sum(r44_cell_src.values()),
        'r44_untouched_foodno_from_r43_nosav': r44_cell_src['Do not have any savings'],
        'r44_untouched_foodno_share_from_nosav': round(
            r44_cell_src['Do not have any savings'] / max(1, sum(r44_cell_src.values())), 4
        ),
        'r43_44_total_nosav_to_untouched': r43_44_nosav_to_no,
        'r42_43_untouched_foodno_total': sum(r42_43_cell_src.values()),
        'r42_43_untouched_foodno_from_r42_nosav': r42_43_cell_src['Do not have any savings'],
        'r42_43_total_nosav_to_untouched': r42_43_nosav_to_no,
        'spillover_persistence_r45_47': r44_spill_persist,
    },
    'r75_to_r76_refresh_decomp': {
        'n_continuing': r76_cont_n,
        'n_new_entrants': r76_new_n,
        'continuing_broad_r75': round(r75_cont_b / r75_cont_w, 4),
        'continuing_broad_r76': round(r76_cont_b / r76_cont_w, 4),
        'continuing_delta_pp': round(100.0 * (r76_cont_b / r76_cont_w - r75_cont_b / r75_cont_w), 2),
        'new_entrants_broad_r76': round(r76_new_b / r76_new_w, 4),
    },
    'work_loss_rate_audit': {
        'subperiods_weighted': {
            k: round(v[0] / v[1], 4) for k, v in wl_sub_w.items()
        },
        'boundary_rounds_r26_32': {
            f'r{r}': {'rate_w': round(wl_round_w[r][0] / wl_round_w[r][1], 4), 'n_events': wl_round_w[r][2]}
            for r in range(26, 33)
        },
    },
    'no_cons_robustness': {
        'pooled': {
            'strict_core': round(nocons_all[1] / nocons_all[0], 4),
            'strict_noCons': round(nocons_all[2] / nocons_all[0], 4),
            'broad_core': round(nocons_all[3] / nocons_all[0], 4),
            'broad_noCons': round(nocons_all[4] / nocons_all[0], 4),
        },
        'r20_april2020': {
            'strict_core': round(nocons_r20[1] / nocons_r20[0], 4),
            'strict_noCons': round(nocons_r20[2] / nocons_r20[0], 4),
            'broad_core': round(nocons_r20[3] / nocons_r20[0], 4),
            'broad_noCons': round(nocons_r20[4] / nocons_r20[0], 4),
        },
        'subperiods': {
            code: {
                'strict_core': round(nocons_sub[code][1] / nocons_sub[code][0], 4),
                'strict_noCons': round(nocons_sub[code][2] / nocons_sub[code][0], 4),
                'broad_core': round(nocons_sub[code][3] / nocons_sub[code][0], 4),
                'broad_noCons': round(nocons_sub[code][4] / nocons_sub[code][0], 4),
            }
            for code, _, _, _ in SUBPERIODS
        },
    },
}

# ---------------------------------------------------------------------------
# 3. Weighted 3-State HMM Misclassification Correction (Post-r29 & Q10-Excluded)
# ---------------------------------------------------------------------------
trips = Counter()
runs_r30 = []
runs_r30_win = []
runs_nc = []

for hid, rds in hh.items():
    sr = sorted(rds.keys())
    # Recursive break-window imputation for r58-65 and r78-79
    prev_rd_imp = None
    prev_win_nosav = False
    for rd in sr:
        cur = rds[rd]
        wn = cur['nosav'] or ((rd in BREAK_ROUNDS) and cur['took_sav'] and prev_rd_imp == rd - 1 and prev_win_nosav)
        cw = cur['coping_hard'] or (cur['foodno'] and wn)
        cur['st_win'] = 2 if (wn and cw) else (1 if (wn or cw) else 0)
        if rd >= 30:
            obs_w_win_r30[cur['st_win']] += cur['w']
        prev_rd_imp = rd
        prev_win_nosav = wn

    for rd in sr:
        if (rd - 1) in rds and (rd + 1) in rds:
            trips[(rds[rd - 1]['st'], rds[rd]['st'], rds[rd + 1]['st'])] += 1

    # Contiguous runs for post-r29 core (r30-82) and break-window-imputed (r30-82)
    for target_list, st_key in ((runs_r30, 'st'), (runs_r30_win, 'st_win')):
        run_s, run_w, prev = [], [], None
        for rd in sr:
            if rd < 30:
                continue
            st = rds[rd][st_key]
            w = rds[rd]['w']
            if prev is not None and rd != prev + 1:
                if len(run_s) >= 3:
                    target_list.append((run_s, sum(run_w) / len(run_w), hid))
                run_s, run_w = [st], [w]
            else:
                run_s.append(st)
                run_w.append(w)
            prev = rd
        if len(run_s) >= 3:
            target_list.append((run_s, sum(run_w) / len(run_w), hid))

    # Contiguous runs for Q10-excluded full panel (r1-82)
    run_s, run_w, prev = [], [], None
    for rd in sr:
        st = rds[rd]['st_nc']
        w = rds[rd]['w']
        if prev is not None and rd != prev + 1:
            if len(run_s) >= 3:
                runs_nc.append((run_s, sum(run_w) / len(run_w), hid))
            run_s, run_w = [st], [w]
        else:
            run_s.append(st)
            run_w.append(w)
        prev = rd
    if len(run_s) >= 3:
        runs_nc.append((run_s, sum(run_w) / len(run_w), hid))

bounce = {}
for a, b, nm in ((0, 0, 'S_S_to_S'), (0, 1, 'S_M_to_S'), (1, 0, 'M_S_to_M'), (1, 1, 'M_M_to_M'), (2, 2, 'H_H_to_H')):
    tot = sum(trips[(a, b, k)] for k in (0, 1, 2))
    bounce[nm] = {'n': tot, 'prob': round(trips[(a, b, a)] / tot, 4)}

hmm_r30 = summarize_hmm(runs_r30, obs_w_core_r30, n_boot=50, seed=12345)
hmm_r30_win = summarize_hmm(runs_r30_win, obs_w_win_r30, n_boot=50, seed=67890)
hmm_nc = summarize_hmm(runs_nc, obs_w_nc_all, n_boot=50, seed=54321)

# Post-trough HMM start-date sensitivity (r36-82, well after Feb 2021 trough; point MLE)
runs_r36 = []
obs_w_r36 = [0.0, 0.0, 0.0]
for hid, rds in hh.items():
    sr = sorted(rds.keys())
    run_s, run_w, prev = [], [], None
    for rd in sr:
        if rd < 36:
            continue
        st = rds[rd]['st']
        w = rds[rd]['w']
        obs_w_r36[st] += w
        if prev is not None and rd != prev + 1:
            if len(run_s) >= 3:
                runs_r36.append((run_s, sum(run_w) / len(run_w), hid))
            run_s, run_w = [st], [w]
        else:
            run_s.append(st)
            run_w.append(w)
        prev = rd
    if len(run_s) >= 3:
        runs_r36.append((run_s, sum(run_w) / len(run_w), hid))
pi0_36, A_36, E_36, v_36, pa_36, pf_36, ll_36 = fit_weighted_hmm(runs_r36, n_iter=60)
tot_o36 = sum(obs_w_r36)
hmm_r36_summary = {
    'n_contiguous_runs': len(runs_r36),
    'observed_weighted_shares': [round(x / tot_o36, 4) for x in obs_w_r36],
    'predicted_finite_horizon_shares': [round(x, 4) for x in pf_36],
    'predicted_asymptotic_piE': [round(x, 4) for x in pa_36],
    'initial_pi0': [round(x, 4) for x in pi0_36],
    'latent_stationary_pi': [round(x, 4) for x in v_36],
    'latent_broad_htm': round(v_36[1] + v_36[2], 4),
    'latent_expected_spells_mo': {nm: round(1.0 / (1.0 - A_36[i][i]), 1) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
}

# Raw-P targeting with matched raw universal denominator (read directly from transitions.json)
with open(os.path.join(OUT_DIR, 'transitions.json')) as f:
    trans_core = json.load(f)['core_harmonized']['pooled']
P_raw = trans_core['P_w']
PI_raw = trans_core['steady_w']
with open(os.path.join(OUT_DIR, 'income_process.json')) as f:
    q1_shares = json.load(f)['within_round_quintiles_weighted']['Q1']

q1_s = 1.0 - q1_shares['broad']
q1_m = q1_shares['broad'] - q1_shares['strict']
q1_h = q1_shares['strict']
P_raw_3 = mat_pow(P_raw, 3)

raw_targeting = {}
for mM, tag in ((1.0, 'mM100'), (0.85, 'mM85'), (0.50, 'mM50')):
    m = [0.07, mM, 1.0]
    univ = PI_raw[0] * 0.07 + PI_raw[1] * mM + PI_raw[2] * 1.0
    q1_mpc = q1_s * 0.07 + q1_m * mM + q1_h * 1.0
    lag3_mpc = sum(P_raw_3[2][j] * m[j] for j in range(3))
    raw_targeting[tag] = {
        'raw_universal_mpc': round(univ, 4),
        'q1_income_mpc': round(q1_mpc, 4),
        'q1_lift_pct': round(100.0 * (q1_mpc / univ - 1.0), 1),
        'raw_P_lag3_mpc': round(lag3_mpc, 4),
        'raw_P_lag3_lift_pct': round(100.0 * (lag3_mpc / univ - 1.0), 1),
        'raw_strict_retention_3m': round(P_raw_3[2][2], 4),
    }

hmm_out = {
    'bounce_back_triplets': bounce,
    'raw_expected_spells_mo': {nm: round(1.0 / (1.0 - P_raw[i][i]), 1) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
    'post_r29_weighted': hmm_r30,
    'post_r29_win_imputed_weighted': hmm_r30_win,
    'post_r36_weighted_point': hmm_r36_summary,
    'q10_excluded_weighted': hmm_nc,
    'raw_targeting_matched_denom': raw_targeting,
}

# ---------------------------------------------------------------------------
# 4. M1 External & Dynamic Validation
# ---------------------------------------------------------------------------
ext_summary = {}
for grp in ('S', 'M1', 'M2', 'H'):
    ext_summary[grp] = {}
    for nm in ('fies_r49_82', 'util', 'med', 'econ_bad', 'fin_worse', 'save_past30d'):
        m, se = clustered_mean_se(ext_markers[grp][nm])
        ext_summary[grp][nm] = {'mean': round(m, 4), 'se_cl': round(se, 4), 'n_hh': len(ext_markers[grp][nm])}

dyn_wl = {}
for outcome in ('loan', 'util', 'fin_worse', 'fies'):
    dyn_wl[outcome] = {}
    for lbl, cond in (('S', lambda p: p['st'] == 0), ('M1', lambda p: p['is_m1']),
                      ('M', lambda p: p['st'] == 1), ('H', lambda p: p['st'] == 2)):
        obs = []
        for hid, rds in hh.items():
            for rd, cur in rds.items():
                if not cur['in_ind'] or (rd - 1) not in rds:
                    continue
                prev = rds[rd - 1]
                if not cond(prev):
                    continue
                y = cur[outcome]
                if y is None:
                    continue
                obs.append((hid, rd, cur['w'], float(y), float(cur['wl'])))
        dyn_wl[outcome][lbl] = rd_fe_ols(obs)

staple_by_st = {}
for outcome, key in (('bread', 'bread_r'), ('flour', 'flour_r'), ('staple', 'staple_r')):
    staple_by_st[outcome] = {}
    for lbl, cond in (('S', lambda p: p['st'] == 0), ('M1', lambda p: p['is_m1']),
                      ('M', lambda p: p['st'] == 1), ('H', lambda p: p['st'] == 2)):
        obs = []
        for hid, rds in hh.items():
            for rd, cur in rds.items():
                if (rd - 1) not in rds:
                    continue
                prev = rds[rd - 1]
                if not cond(prev):
                    continue
                if cur[key] and prev[key] and cur['lny'] is not None and prev['lny'] is not None:
                    obs.append((hid, rd, cur['w'],
                                math.log(cur[key]) - math.log(prev[key]),
                                cur['lny'] - prev['lny']))
        staple_by_st[outcome][lbl] = rd_fe_ols(obs)

m1_vs_s_diff = {}
for prefix, block in (('jobloss', dyn_wl), ('staple', staple_by_st)):
    for outc, sub in block.items():
        bS, seS = sub['S']['beta'], sub['S']['se_cl']
        bM, seM = sub['M1']['beta'], sub['M1']['se_cl']
        diff = bM - bS
        se_d = math.hypot(seM, seS)
        m1_vs_s_diff[f'{prefix}_{outc}'] = {
            'diff_M1_minus_S': round(diff, 4),
            'se_diff': round(se_d, 4),
            't_diff': round(diff / se_d, 2) if se_d > 0 else 0.0,
        }

m1_validation = {
    'cross_sectional_weighted_clustered': ext_summary,
    'job_loss_response_round_fe': dyn_wl,
    'staple_elasticity_round_fe': staple_by_st,
    'm1_vs_s_diff_tests': m1_vs_s_diff,
}

# ---------------------------------------------------------------------------
# 5. Job-Loss Exposure (t-1..t-3), Event Study, Cyclicality, April 2020
# ---------------------------------------------------------------------------
lag_exposure = {}
for lag in (1, 2, 3):
    for tag, min_rd in (('all_rounds', 1), ('post_covid_r30_82', 30)):
        rates = defaultdict(lambda: [0.0, 0.0, 0])
        obs_M = []
        obs_H = []
        for hid, rds in hh.items():
            for rd, cur in rds.items():
                if rd < min_rd or not cur['in_ind'] or (rd - lag) not in rds:
                    continue
                st = rds[rd - lag]['st']
                w = cur['w']
                wl = float(cur['wl'])
                rates[st][0] += w * wl
                rates[st][1] += w
                rates[st][2] += 1
                if st in (0, 1):
                    obs_M.append((hid, rd, w, wl, 1.0 if st == 1 else 0.0))
                if st in (0, 2):
                    obs_H.append((hid, rd, w, wl, 1.0 if st == 2 else 0.0))
        rS = rates[0][0] / rates[0][1]
        rM = rates[1][0] / rates[1][1]
        rH = rates[2][0] / rates[2][1]
        lag_exposure[f'lag{lag}_{tag}'] = {
            'rate_S': round(rS, 4),
            'rate_M': round(rM, 4),
            'rate_H': round(rH, 4),
            'ratio_M_to_S': round(rM / rS, 2),
            'ratio_H_to_S': round(rH / rS, 2),
            'rdfe_M_minus_S': rd_fe_ols(obs_M),
            'rdfe_H_minus_S': rd_fe_ols(obs_H),
        }

event_study = {}
for h in (-3, -1, 0, 1, 2, 3, 6, 12):
    obs = []
    for hid, rds in hh.items():
        for rd, cur in rds.items():
            if not cur['in_ind']:
                continue
            if (rd - 2) in rds and (rd + h) in rds:
                y0 = rds[rd - 2]['lny']
                yh = rds[rd + h]['lny']
                if y0 is not None and yh is not None:
                    obs.append((hid, rd, cur['w'], yh - y0, float(cur['wl'])))
    event_study[f'h{h:+d}'] = rd_fe_ols(obs)

cyclicality = {}
for lag in (1, 2):
    for tag, excl_cov in (('all_rounds', False), ('excl_r20_21', True)):
        rec = {}
        for st, nm in ((0, 'S'), (1, 'M'), (2, 'H')):
            xs, ys, ws, hids = [], [], [], []
            for hid, rds in hh.items():
                for rd, cur in rds.items():
                    if excl_cov and rd in (20, 21):
                        continue
                    if (rd - 1) in rds and (rd - lag) in rds and rd in rd_lny_bar and (rd - 1) in rd_lny_bar:
                        if rds[rd - lag]['st'] != st:
                            continue
                        if cur['lny'] is not None and rds[rd - 1]['lny'] is not None:
                            xs.append(rd_lny_bar[rd] - rd_lny_bar[rd - 1])
                            ys.append(cur['lny'] - rds[rd - 1]['lny'])
                            ws.append(cur['w'])
                            hids.append(hid)
            W = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs)) / W
            my = sum(w * y for w, y in zip(ws, ys)) / W
            num = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
            den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
            b = num / den
            scores = defaultdict(float)
            for hid, w, x, y in zip(hids, ws, xs, ys):
                scores[hid] += w * (x - mx) * ((y - my) - b * (x - mx))
            G = len(scores)
            se = math.sqrt((G / max(G - 1, 1)) * sum(s * s for s in scores.values()) / (den * den))
            rec[nm] = {'beta': round(b, 3), 'se_cl': round(se, 3), 'n': len(xs)}
        cyclicality[f'lag{lag}_{tag}'] = rec

april2020 = {}
for lag_rd in (19, 18):
    by_st = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for hid, rds in hh.items():
        if lag_rd in rds and 19 in rds and 20 in rds:
            st = rds[lag_rd]['st']
            w = rds[20]['w']
            by_st[st][0] += w * rds[19]['y_real']
            by_st[st][1] += w * rds[20]['y_real']
            by_st[st][2] += w
            by_st[st][3] += 1
    april2020[f'state_at_r{lag_rd}'] = {
        nm: {
            'n': by_st[st][3],
            'pct_change_real': round(100.0 * (by_st[st][1] / by_st[st][0] - 1.0), 2),
        }
        for st, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
    }

job_loss_ext = {
    'lagged_exposure': lag_exposure,
    'event_study_rel_t_minus_2': event_study,
    'patterson_cyclicality': cyclicality,
    'april2020_realized_drop': april2020,
    'earner_share_by_state': {
        nm: round(earner_w_by_st[st][0] / earner_w_by_st[st][1], 4)
        for st, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
    },
}

# ---------------------------------------------------------------------------
# 6. Remittance-Inclusive Income & Size/Winsorization Robustness
# ---------------------------------------------------------------------------
inc_base = [0.0, 0.0, 0.0]
inc_plus = [0.0, 0.0, 0.0]
rem_by_st = [0.0, 0.0, 0.0]
inc_sqrt = [0.0, 0.0, 0.0]
rem_gaps_sh = []
rem_gaps_sm = []
wins_gaps_sh = []
wins_gaps_sm = []

for rd, rows_rd in rd_real_y_all.items():
    pos_y = [(y, w) for y, w, _, _ in rows_rd if y > 0]
    p99 = weighted_quantile(pos_y, 0.99) if pos_y else 1e12
    by_st_w = defaultdict(lambda: [0.0, 0.0])
    for y, w, st, hsz in rows_rd:
        yw = min(y, p99)
        if yw > 0:
            by_st_w[st][0] += w * math.log(yw)
            by_st_w[st][1] += w
        inc_sqrt[st] += w * (y / math.sqrt(hsz))
    if all(by_st_w[s][1] > 0 for s in (0, 1, 2)):
        m0 = by_st_w[0][0] / by_st_w[0][1]
        m1 = by_st_w[1][0] / by_st_w[1][1]
        m2 = by_st_w[2][0] / by_st_w[2][1]
        wins_gaps_sh.append(m0 - m2)
        wins_gaps_sm.append(m0 - m1)

by_rd_rem = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
for hid, rds in hh.items():
    for rd, cur in rds.items():
        w = cur['w']
        st = cur['st']
        yr = cur['y_real']
        rm_r = rem_som.get((hid, rd), 0.0) * 100.0 / cur['p']
        inc_base[st] += w * yr
        inc_plus[st] += w * (yr + rm_r)
        rem_by_st[st] += w * rm_r
        if yr + rm_r > 0:
            by_rd_rem[rd][st][0] += w * math.log(yr + rm_r)
            by_rd_rem[rd][st][1] += w

for rd, st_d in by_rd_rem.items():
    if all(st_d[s][1] > 0 for s in (0, 1, 2)):
        m0 = st_d[0][0] / st_d[0][1]
        m1 = st_d[1][0] / st_d[1][1]
        m2 = st_d[2][0] / st_d[2][1]
        rem_gaps_sh.append(m0 - m2)
        rem_gaps_sm.append(m0 - m1)

sb = sum(inc_base)
sp = sum(inc_plus)
ssq = sum(inc_sqrt)
om_base = [x / sb for x in inc_base]
om_plus = [x / sp for x in inc_plus]
om_sqrt = [x / ssq for x in inc_sqrt]

remittance_and_size = {
    'omega_base': [round(x, 4) for x in om_base],
    'omega_plus_remittances': [round(x, 4) for x in om_plus],
    'remit_share_of_agg_Y': round(sum(rem_by_st) / sp, 4),
    'remit_share_by_state': {nm: round(rem_by_st[i] / inc_plus[i], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
    'emp_aggmpc_inc_base': round(om_base[0] * 0.07 + om_base[1] + om_base[2], 4),
    'emp_aggmpc_inc_plus_remit': round(om_plus[0] * 0.07 + om_plus[1] + om_plus[2], 4),
    'mean_log_gap_plus_remit_SH': round(sum(rem_gaps_sh) / len(rem_gaps_sh), 3),
    'mean_log_gap_plus_remit_SM': round(sum(rem_gaps_sm) / len(rem_gaps_sm), 3),
    'sqrt_equiv_saver_share': round(om_sqrt[0], 4),
    'sqrt_equiv_emp_aggmpc_inc': round(om_sqrt[0] * 0.07 + om_sqrt[1] + om_sqrt[2], 4),
    'wins99_mean_log_gap_SH': round(sum(wins_gaps_sh) / len(wins_gaps_sh), 3),
    'wins99_mean_log_gap_SM': round(sum(wins_gaps_sm) / len(wins_gaps_sm), 3),
}

out = {
    'audience_profile': audience,
    'forensic_ext': forensic_ext,
    'hmm_misclassification': hmm_out,
    'm1_validation': m1_validation,
    'job_loss_ext': job_loss_ext,
    'remittance_and_size': remittance_and_size,
}

out_path = os.path.join(OUT_DIR, 'diagnostics.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print('Wrote', out_path)
print('Post-r29 weighted HMM:', hmm_r30['latent_stationary_pi'], 'SE:', hmm_r30['bootstrap_se']['pi_se'])
print('  Obs vs FinPred vs Asymp:', hmm_r30['observed_weighted_shares'], hmm_r30['predicted_finite_horizon_shares'], hmm_r30['predicted_asymptotic_piE'])
print('  Bayes targeting:', hmm_r30['bayesian_feasible_targeting'])
print('Q10-excl weighted HMM:', hmm_nc['latent_stationary_pi'], 'SE:', hmm_nc['bootstrap_se']['pi_se'])
print('  Obs vs FinPred vs Asymp:', hmm_nc['observed_weighted_shares'], hmm_nc['predicted_finite_horizon_shares'], hmm_nc['predicted_asymptotic_piE'])
print('  Bayes targeting:', hmm_nc['bayesian_feasible_targeting'])
print('Raw targeting:', raw_targeting)
