"""Step 01: Build 82-round national HtM series (Core Harmonized + Window-Only Break-Imputed
+ All-Round Impossible-Transition Imputed + Robustness), CPI-deflated incomes, Kish &
HH-clustered SEs, r44--82 food_past30d forensic audit, r1--19 early drawdown audit,
r57->r58 & r77->r78 break audits, and attrition diagnostics.
"""
import csv
import json
import math
import os
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
    load_monthly_cpi,
    subperiod_of,
    totinc,
    weighted_quantile,
)

MS_BASE = 0.07
ensure_out_dir()
cpi_map = load_monthly_cpi(base_ym='2018-09')

hh_rows = defaultdict(list)
for row in iter_household_rows():
    rd = int(row['round'])
    hid = row['hhid']
    w = float(row['popw'] or 0.0)
    hsz = max(1.0, float(clean(row.get('hhsize', '')) or 1.0))
    date_str = clean(row['date_start'])
    ym = date_str[:7]
    cpi = cpi_map[ym]
    y_nom = totinc(row)
    y_real = y_nom * (100.0 / cpi)
    fl = htm_flags(row)
    hh_rows[hid].append((rd, date_str, ym, cpi, w, hsz, y_nom, y_real, fl))

acc = {}
hh_obs = {k: defaultdict(list) for k in ('broad', 'strict', 'broad_win', 'broad_allimp',
                                         'broad_unadj', 'broad_withtook', 'nosav', 'm1', 'm2')}

pooled_w = defaultdict(float)
pooled_u = defaultdict(int)
W_tot = 0.0
N_tot = 0

sub_w = {code: defaultdict(float) for code, _, _, _ in SUBPERIODS}
sub_W = {code: 0.0 for code, _, _, _ in SUBPERIODS}

by_round_ids = defaultdict(set)
hh_states = defaultdict(dict)

# Forensic audit of (savings_past30d, food_past30d) cells in r44--82 (exact onset at r44!)
food_cells_r44 = defaultdict(lambda: {'n': 0, 'W': 0.0, 'wy_real': 0.0, 'w_fw': 0.0,
                                      'w_loan': 0.0, 'w_cons': 0.0, 'w_fies': 0.0, 'W_fies': 0.0})

# Early-panel r1--19 drawdown transition audit
early_drawdown_from_nosav = 0
early_drawdown_total = 0

for hid, v in hh_rows.items():
    v.sort(key=lambda x: x[0])
    prev_rd = None
    prev_sv = None
    prev_win_nosav = False
    prev_all_nosav = False

    for rd, date_str, ym, cpi, w, hsz, y_nom, y_real, fl in v:
        N_tot += 1
        W_tot += w
        by_round_ids[rd].add(hid)
        hh_states[hid][rd] = (fl['st_core'], fl['sv'], fl['foodno'])

        # Early-panel r2--19 drawdown origin audit
        if prev_rd == rd - 1 and rd <= 19 and fl['took_sav']:
            early_drawdown_total += 1
            if prev_sv == 'Do not have any savings':
                early_drawdown_from_nosav += 1

        # 1) Window-only break-imputed series (applied strictly inside r58--65 and r78--79)
        win_nosav = fl['nosav'] or (
            (rd in BREAK_ROUNDS) and fl['took_sav'] and prev_rd == rd - 1 and prev_win_nosav
        )
        cop_win = fl['coping_hard'] or (fl['foodno'] and win_nosav)
        broad_win = win_nosav or cop_win
        strict_win = win_nosav and cop_win
        st_win = 2 if strict_win else (1 if broad_win else 0)

        # 2) All-round recursive impossible-transition imputed series (all 82 rounds)
        all_nosav = fl['nosav'] or (fl['took_sav'] and prev_rd == rd - 1 and prev_all_nosav)
        cop_all = fl['coping_hard'] or (fl['foodno'] and all_nosav)
        broad_allimp = all_nosav or cop_all
        strict_all = all_nosav and cop_all
        st_all = 2 if strict_all else (1 if broad_allimp else 0)

        prev_rd = rd
        prev_sv = fl['sv']
        prev_win_nosav = win_nosav
        prev_all_nosav = all_nosav

        a = acc.get(rd)
        if a is None:
            a = acc[rd] = {
                'date': date_str, 'ym': ym, 'cpi': cpi,
                'n': 0, 'W': 0.0, 'W2': 0.0,
                'wy_nom': 0.0, 'wy_real': 0.0,
                'wy_real_s': [0.0, 0.0, 0.0],
                'wy_real_win_s': [0.0, 0.0, 0.0],
                'wy_real_all_s': [0.0, 0.0, 0.0],
                'broad': 0.0, 'strict': 0.0, 'broad_win': 0.0, 'broad_allimp': 0.0,
                'broad_unadj': 0.0, 'broad_withtook': 0.0,
                'nosav': 0.0, 'coping_core': 0.0, 'coping_hard': 0.0,
                'm1': 0.0, 'm2': 0.0, 'took_sav': 0.0, 'sav_no': 0.0,
                'savno_foodno': 0.0, 'nosav_foodno': 0.0,
                'loan': 0.0, 'assets': 0.0, 'cons': 0.0, 'foodno': 0.0,
                'util': 0.0, 'fin_worse': 0.0, 'fin_better_next': 0.0,
                'q5_agree': 0.0, 'wfies': 0.0, 'wfies_w': 0.0,
                'y_real_list': [],
            }

        a['n'] += 1
        a['W'] += w
        a['W2'] += w * w
        a['wy_nom'] += w * y_nom
        a['wy_real'] += w * y_real
        a['wy_real_s'][fl['st_core']] += w * y_real
        a['wy_real_win_s'][st_win] += w * y_real
        a['wy_real_all_s'][st_all] += w * y_real
        a['y_real_list'].append((y_real, w, fl['st_core']))

        flags_map = {
            'broad': fl['broad_core'],
            'strict': fl['strict_core'],
            'broad_win': broad_win,
            'broad_allimp': broad_allimp,
            'broad_unadj': fl['broad_unadj'],
            'broad_withtook': fl['broad_wt'],
            'nosav': fl['nosav'],
            'coping_core': fl['coping_core'],
            'coping_hard': fl['coping_hard'],
            'm1': fl['is_m1'],
            'm2': fl['is_m2'],
            'took_sav': fl['took_sav'],
            'sav_no': fl['sav_no'],
            'savno_foodno': (fl['sav_no'] and fl['foodno']),
            'nosav_foodno': (fl['nosav'] and fl['foodno']),
            'loan': fl['loan'],
            'assets': fl['assets'],
            'cons': fl['cons'],
            'foodno': fl['foodno'],
            'util': fl['util'],
            'fin_worse': fl['fin_worse'],
            'fin_better_next': fl['fin_better_next'],
            'q5_agree': fl['q5_agree'],
        }
        for col, val in flags_map.items():
            if val:
                a[col] += w
                pooled_w[col] += w
                pooled_u[col] += 1

        if fl['fies_any'] is not None:
            a['wfies_w'] += w
            if fl['fies_any']:
                a['wfies'] += w

        sp = subperiod_of(rd)
        sub_W[sp] += w
        for col in ('broad', 'strict', 'broad_win', 'broad_allimp', 'broad_unadj', 'broad_withtook',
                    'nosav', 'coping_core', 'coping_hard', 'm1', 'm2',
                    'took_sav', 'cons', 'foodno', 'loan', 'savno_foodno'):
            if flags_map[col]:
                sub_w[sp][col] += w

        if rd >= 44 and fl['sv'] in ('No', 'Do not have any savings'):
            ckey = f"{fl['sv']} | food={'No' if fl['foodno'] else 'Yes'}"
            fc = food_cells_r44[ckey]
            fc['n'] += 1
            fc['W'] += w
            fc['wy_real'] += w * y_real
            if fl['fin_worse']:
                fc['w_fw'] += w
            if fl['loan']:
                fc['w_loan'] += w
            if fl['cons']:
                fc['w_cons'] += w
            if fl['fies_any'] is not None:
                fc['W_fies'] += w
                if fl['fies_any']:
                    fc['w_fies'] += w

        for key in hh_obs:
            hh_obs[key][hid].append((1.0 if flags_map[key] else 0.0, w))

csv_path = os.path.join(OUT_DIR, 'national_series.csv')
fieldnames = [
    'round', 'date', 'ym', 'cpi_2018m09_100', 'n', 'neff',
    'broad', 'se_broad', 'strict', 'se_strict',
    'broad_win', 'broad_allimp', 'broad_unadj', 'broad_withtook',
    'nosav', 'coping_core', 'coping_hard', 'm1_nosav_nocop', 'm2_sav_hardcop',
    'took_sav', 'p_foodno_given_savno', 'p_foodno_given_nosav',
    'loan', 'assets', 'cons_cut', 'food_no', 'util_nopay',
    'fin_worse', 'fin_better_next', 'q5_agree', 'fies_any',
    'mean_income_nom', 'mean_income_real', 'mean_income_real_wins99',
    'inc_share_S', 'inc_share_M', 'inc_share_H',
    'aggmpc_pop', 'aggmpc_inc', 'aggmpc_inc_win', 'aggmpc_inc_allimp', 'is_break_round',
]

series_rows = []
with open(csv_path, 'w', newline='') as f:
    wtr = csv.DictWriter(f, fieldnames=fieldnames)
    wtr.writeheader()
    for rd in sorted(acc):
        a = acc[rd]
        W = a['W']
        neff = (W * W / a['W2']) if a['W2'] > 0 else float(a['n'])
        pb = a['broad'] / W
        ps = a['strict'] / W
        seb = math.sqrt(pb * (1.0 - pb) / neff)
        ses = math.sqrt(ps * (1.0 - ps) / neff)
        fies = round(a['wfies'] / a['wfies_w'], 4) if a['wfies_w'] > 0 else ''
        y_tot = sum(a['wy_real_s'])
        inc_S = a['wy_real_s'][0] / y_tot
        inc_M = a['wy_real_s'][1] / y_tot
        inc_H = a['wy_real_s'][2] / y_tot
        inc_S_win = a['wy_real_win_s'][0] / sum(a['wy_real_win_s'])
        inc_S_all = a['wy_real_all_s'][0] / sum(a['wy_real_all_s'])
        agg_pop = (1.0 - pb) * MS_BASE + pb * 1.0
        agg_inc = inc_S * MS_BASE + (inc_M + inc_H) * 1.0
        agg_inc_win = inc_S_win * MS_BASE + (1.0 - inc_S_win) * 1.0
        agg_inc_all = inc_S_all * MS_BASE + (1.0 - inc_S_all) * 1.0

        pos_yw = [(y_i, w_i) for y_i, w_i, _ in a['y_real_list'] if y_i > 0.0]
        cap99 = weighted_quantile(pos_yw, 0.99) if pos_yw else 1e12
        wy_w99 = sum(w_i * min(y_i, cap99) for y_i, w_i, _ in a['y_real_list']) / W

        p_fd_sn = a['savno_foodno'] / a['sav_no'] if a['sav_no'] > 0 else 0.0
        p_fd_ns = a['nosav_foodno'] / a['nosav'] if a['nosav'] > 0 else 0.0

        row_out = {
            'round': rd, 'date': a['date'], 'ym': a['ym'],
            'cpi_2018m09_100': round(a['cpi'], 2),
            'n': a['n'], 'neff': round(neff),
            'broad': round(pb, 4), 'se_broad': round(seb, 4),
            'strict': round(ps, 4), 'se_strict': round(ses, 4),
            'broad_win': round(a['broad_win'] / W, 4),
            'broad_allimp': round(a['broad_allimp'] / W, 4),
            'broad_unadj': round(a['broad_unadj'] / W, 4),
            'broad_withtook': round(a['broad_withtook'] / W, 4),
            'nosav': round(a['nosav'] / W, 4),
            'coping_core': round(a['coping_core'] / W, 4),
            'coping_hard': round(a['coping_hard'] / W, 4),
            'm1_nosav_nocop': round(a['m1'] / W, 4),
            'm2_sav_hardcop': round(a['m2'] / W, 4),
            'took_sav': round(a['took_sav'] / W, 4),
            'p_foodno_given_savno': round(p_fd_sn, 4),
            'p_foodno_given_nosav': round(p_fd_ns, 4),
            'loan': round(a['loan'] / W, 4),
            'assets': round(a['assets'] / W, 4),
            'cons_cut': round(a['cons'] / W, 4),
            'food_no': round(a['foodno'] / W, 4),
            'util_nopay': round(a['util'] / W, 4),
            'fin_worse': round(a['fin_worse'] / W, 4),
            'fin_better_next': round(a['fin_better_next'] / W, 4),
            'q5_agree': round(a['q5_agree'] / W, 4),
            'fies_any': fies,
            'mean_income_nom': round(a['wy_nom'] / W),
            'mean_income_real': round(a['wy_real'] / W),
            'mean_income_real_wins99': round(wy_w99),
            'inc_share_S': round(inc_S, 4),
            'inc_share_M': round(inc_M, 4),
            'inc_share_H': round(inc_H, 4),
            'aggmpc_pop': round(agg_pop, 4),
            'aggmpc_inc': round(agg_inc, 4),
            'aggmpc_inc_win': round(agg_inc_win, 4),
            'aggmpc_inc_allimp': round(agg_inc_all, 4),
            'is_break_round': 1 if rd in BREAK_ROUNDS else 0,
        }
        wtr.writerow(row_out)
        series_rows.append(row_out)

cl_stats = {}
for nm, hmap in hh_obs.items():
    mu, se = clustered_mean_se(hmap)
    cl_stats[nm] = {'mean': round(mu, 4), 'se_cl': round(se, 4)}


def audit_sav_break(r0, r1):
    c = Counter()
    for hid, rmap in hh_states.items():
        if r0 in rmap and r1 in rmap:
            c[f"{rmap[r0][1]} -> {rmap[r1][1]}"] += 1
    tot_to_yes = sum(v for k, v in c.items() if k.endswith('-> Yes'))
    from_nosav = c.get('Do not have any savings -> Yes', 0)
    return {
        'transitions': dict(sorted(c.items())),
        'n_matched': sum(c.values()),
        'total_to_Yes': tot_to_yes,
        'from_nosav_to_Yes': from_nosav,
        'share_from_nosav': round(from_nosav / tot_to_yes, 4) if tot_to_yes else 0.0,
    }


rounds = sorted(by_round_ids)
cont_by_round = []
below_97 = []
for i in range(1, len(rounds)):
    r_prev, r_cur = rounds[i - 1], rounds[i]
    prev_s, cur_s = by_round_ids[r_prev], by_round_ids[r_cur]
    cont = len(prev_s & cur_s)
    rec = {
        'from': r_prev, 'to': r_cur, 'n_prev': len(prev_s), 'n_cur': len(cur_s),
        'continuing': cont, 'exits': len(prev_s - cur_s), 'new_entries': len(cur_s - prev_s),
        'retention_rate': round(cont / len(prev_s), 4),
        'cont_share_of_cur': round(cont / len(cur_s), 4),
    }
    cont_by_round.append(rec)
    if rec['retention_rate'] < 0.97 or rec['cont_share_of_cur'] < 0.97:
        below_97.append(rec)

attr_all = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
attr_58_74 = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
for hid, rmap in hh_states.items():
    for rd, (st, _, _) in rmap.items():
        if rd < 82:
            attr_all[st][0] += 1
            if (rd + 1) not in rmap:
                attr_all[st][1] += 1
            if 58 <= rd <= 74:
                attr_58_74[st][0] += 1
                if (rd + 1) not in rmap:
                    attr_58_74[st][1] += 1

clean_rows = [r for r in series_rows if not r['is_break_round']]


def extrema_of(rows_sub, key):
    lo = min(rows_sub, key=lambda x: x[key])
    hi = max(rows_sub, key=lambda x: x[key])
    return {'min': lo[key], 'min_round': lo['round'], 'min_ym': lo['ym'],
            'max': hi[key], 'max_round': hi['round'], 'max_ym': hi['ym']}


food_audit_out = {}
for k, fc in sorted(food_cells_r44.items()):
    food_audit_out[k] = {
        'n': fc['n'],
        'mean_real_y': round(fc['wy_real'] / fc['W']),
        'fin_worse': round(fc['w_fw'] / fc['W'], 4),
        'loan': round(fc['w_loan'] / fc['W'], 4),
        'cons_cut': round(fc['w_cons'] / fc['W'], 4),
        'fies_any_r49_82': round(fc['w_fies'] / fc['W_fies'], 4) if fc['W_fies'] > 0 else None,
    }

early_start_rounds = sum(1 for r in series_rows if int(r['date'].split('-')[2]) <= 10)

summary = {
    'sample': {
        'hh_rounds': N_tot,
        'unique_hh': len(hh_states),
        'rounds': len(rounds),
        'date_start': series_rows[0]['date'],
        'date_end': series_rows[-1]['date'],
        'cpi_start': series_rows[0]['cpi_2018m09_100'],
        'cpi_end': series_rows[-1]['cpi_2018m09_100'],
        'early_month_start_rounds_day_le_10': early_start_rounds,
    },
    'pooled_weighted': {k: round(pooled_w[k] / W_tot, 4) for k in sorted(pooled_w)},
    'pooled_unweighted': {k: round(pooled_u[k] / N_tot, 4) for k in sorted(pooled_u)},
    'pooled_mean_income': {
        'real_sep2018_som': round(sum(a['wy_real'] for a in acc.values()) / W_tot),
        'nominal_som': round(sum(a['wy_nom'] for a in acc.values()) / W_tot),
    },
    'pooled_hh_clustered_se': cl_stats,
    'subperiods_weighted': {
        code: {k: round(sub_w[code][k] / sub_W[code], 4) for k in sorted(sub_w[code])}
        for code, _, _, _ in SUBPERIODS
    },
    'extrema_all_82_rounds': {
        'broad': extrema_of(series_rows, 'broad'),
        'broad_win': extrema_of(series_rows, 'broad_win'),
        'broad_allimp': extrema_of(series_rows, 'broad_allimp'),
        'aggmpc_pop': extrema_of(series_rows, 'aggmpc_pop'),
        'aggmpc_inc': extrema_of(series_rows, 'aggmpc_inc'),
        'aggmpc_inc_win': extrema_of(series_rows, 'aggmpc_inc_win'),
        'aggmpc_inc_allimp': extrema_of(series_rows, 'aggmpc_inc_allimp'),
    },
    'extrema_excl_break_windows': {
        'n_clean_rounds': len(clean_rows),
        'broad': extrema_of(clean_rows, 'broad'),
        'aggmpc_pop': extrema_of(clean_rows, 'aggmpc_pop'),
        'aggmpc_inc': extrema_of(clean_rows, 'aggmpc_inc'),
    },
    'forensic_food_past30d_r44_82': food_audit_out,
    'early_r1_19_drawdown_audit': {
        'r01_took_sav_w': series_rows[0]['took_sav'],
        'r1_19_avg_took_sav_w': round(sub_w['preCOVID']['took_sav'] / sub_W['preCOVID'], 4),
        'r30_42_avg_took_sav_w': round(sub_w['recovery']['took_sav'] / sub_W['recovery'], 4),
        'r2_19_drawdown_transitions': early_drawdown_total,
        'r2_19_from_nosav_transitions': early_drawdown_from_nosav,
        'r2_19_share_from_nosav': round(early_drawdown_from_nosav / early_drawdown_total, 4),
    },
    'r57_to_r58_savings_break': audit_sav_break(57, 58),
    'r77_to_r78_savings_break': audit_sav_break(77, 78),
    'attrition': {
        'transitions_below_97pct': below_97,
        'covid_r19_to_r20': [x for x in cont_by_round if x['from'] == 19][0],
        'refresh_r75_to_r76': [x for x in cont_by_round if x['from'] == 75][0],
        'one_month_exit_rate_all': {
            nm: {'rate': round(attr_all[i][1] / attr_all[i][0], 4), 'exits': attr_all[i][1], 'n': attr_all[i][0]}
            for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
        },
        'one_month_exit_rate_r58_74': {
            nm: {'rate': round(attr_58_74[i][1] / attr_58_74[i][0], 4), 'exits': attr_58_74[i][1], 'n': attr_58_74[i][0]}
            for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
        },
    },
}

json_path = os.path.join(OUT_DIR, 'pooled_and_subperiods.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"wrote {csv_path} ({len(series_rows)} rounds) and {json_path}")
print(f"Core={cl_stats['broad']['mean']:.4f} (SE {cl_stats['broad']['se_cl']:.4f}) | "
      f"Window-Imputed={cl_stats['broad_win']['mean']:.4f} (SE {cl_stats['broad_win']['se_cl']:.4f}) | "
      f"All-Round-Imputed={cl_stats['broad_allimp']['mean']:.4f} (SE {cl_stats['broad_allimp']['se_cl']:.4f})")
print("r44-82 food forensic:", food_audit_out)
print("Early r1-19 drawdown audit:", summary['early_r1_19_drawdown_audit'])
