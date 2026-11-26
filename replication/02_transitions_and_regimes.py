"""Step 02: Estimate 3-state HtM transition matrices with household-clustered weighted SEs,
exact analytical steady states, and subperiod regimes across Core Harmonized,
Window-Only Break-Imputed, All-Round Imputed, Unadjusted-Food, and Inclusive definitions.
"""
import json
import math
import os
from collections import defaultdict

from common import (
    BREAK_ROUNDS,
    OUT_DIR,
    SUBPERIODS,
    ensure_out_dir,
    htm_flags,
    iter_household_rows,
    subperiod_of,
)

ensure_out_dir()

hist = defaultdict(list)
for row in iter_household_rows():
    rd = int(row['round'])
    w = float(row['popw'] or 0.0)
    fl = htm_flags(row)
    hist[row['hhid']].append((rd, fl['st_core'], fl['st_unadj'], fl['st_wt'],
                              fl['nosav'], fl['took_sav'], fl['coping_hard'], fl['foodno'], w))

hist_full = {}
for hid, v in hist.items():
    v.sort(key=lambda x: x[0])
    out_v = []
    prev_rd = None
    prev_win_nosav = False
    prev_all_nosav = False
    for rd, st_core, st_unadj, st_wt, nosav, took_sav, cop_hard, foodno, w in v:
        win_nosav = nosav or ((rd in BREAK_ROUNDS) and took_sav and prev_rd == rd - 1 and prev_win_nosav)
        cop_win = cop_hard or (foodno and win_nosav)
        st_win = 2 if (win_nosav and cop_win) else (1 if (win_nosav or cop_win) else 0)

        all_nosav = nosav or (took_sav and prev_rd == rd - 1 and prev_all_nosav)
        cop_all = cop_hard or (foodno and all_nosav)
        st_all = 2 if (all_nosav and cop_all) else (1 if (all_nosav or cop_all) else 0)

        out_v.append((rd, st_core, st_win, st_all, st_unadj, st_wt, w))
        prev_rd = rd
        prev_win_nosav = win_nosav
        prev_all_nosav = all_nosav
    hist_full[hid] = out_v


def steady_state(P):
    pi = [1.0 / 3.0] * 3
    for _ in range(3000):
        pi = [sum(pi[r] * P[r][c] for r in range(3)) for c in range(3)]
    assert abs(sum(pi) - 1.0) < 1e-10
    return pi


def summarize_transitions(pair_records):
    C_uw = [[0] * 3 for _ in range(3)]
    C_w = [[0.0] * 3 for _ in range(3)]
    by_row_hh = [defaultdict(list) for _ in range(3)]

    for hid, s0, s1, w in pair_records:
        C_uw[s0][s1] += 1
        C_w[s0][s1] += w
        by_row_hh[s0][hid].append((s1, w))

    row_n = [sum(r) for r in C_uw]
    P_uw, P_w, SE_binom, SE_cl_w = [], [], [], []
    for i in range(3):
        nu = row_n[i]
        nw = sum(C_w[i])
        pu = [c / nu for c in C_uw[i]]
        pw = [c / nw for c in C_w[i]]
        se_b = [math.sqrt(p * (1.0 - p) / nu) for p in pu]

        G = len(by_row_hh[i])
        se_c = []
        for j in range(3):
            p_ij = pw[j]
            meat = 0.0
            for obs in by_row_hh[i].values():
                score_g = sum(w * ((1.0 if s1 == j else 0.0) - p_ij) for s1, w in obs)
                meat += score_g * score_g
            var_cl = (G / (G - 1)) * meat / (nw * nw) if G > 1 else 0.0
            se_c.append(math.sqrt(var_cl))

        P_uw.append(pu)
        P_w.append(pw)
        SE_binom.append(se_b)
        SE_cl_w.append(se_c)

    max_diff = max(abs(P_uw[i][j] - P_w[i][j]) for i in range(3) for j in range(3))
    return {
        'n_pairs': sum(row_n),
        'row_n': row_n,
        'row_n_hh': [len(by_row_hh[i]) for i in range(3)],
        'counts_uw': C_uw,
        'P_uw': [[round(x, 6) for x in r] for r in P_uw],
        'P_w': [[round(x, 6) for x in r] for r in P_w],
        'SE_uw_binom': [[round(x, 6) for x in r] for r in SE_binom],
        'SE_w_clustered': [[round(x, 6) for x in r] for r in SE_cl_w],
        'max_abs_diff_uw_vs_w': round(max_diff, 4),
        'steady_uw': [round(x, 6) for x in steady_state(P_uw)],
        'steady_w': [round(x, 6) for x in steady_state(P_w)],
    }


def compute_suite(state_idx):
    pooled_recs = []
    sub_recs = {code: [] for code, _, _, _ in SUBPERIODS}
    for hid, v in hist_full.items():
        for i in range(1, len(v)):
            if v[i][0] == v[i - 1][0] + 1:
                rec = (hid, v[i - 1][state_idx], v[i][state_idx], v[i][6])
                pooled_recs.append(rec)
                sub_recs[subperiod_of(v[i][0])].append(rec)

    res = {'pooled': summarize_transitions(pooled_recs), 'regimes': {}}
    for code, _, _, _ in SUBPERIODS:
        res['regimes'][code] = summarize_transitions(sub_recs[code])
    return res


out = {
    'core_harmonized': compute_suite(1),
    'window_break_imputed': compute_suite(2),
    'all_round_imputed': compute_suite(3),
    'unadj_food': compute_suite(4),
    'inclusive_withtook': compute_suite(5),
}

out_path = os.path.join(OUT_DIR, 'transitions.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"wrote {out_path}")
for nm in ('core_harmonized', 'window_break_imputed', 'all_round_imputed'):
    print(f"  {nm:22s}: steady_w={[round(x,4) for x in out[nm]['pooled']['steady_w']]}")
