"""Shared data loaders, CPI deflator, and harmonized HtM definitions.

Self-contained module with no external user-specific path dependencies.
Set DATA_DIR environment variable to override default source-data lookup.
"""
import csv
import io
import math
import os
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(SUBMISSION_DIR, 'outputs')


def _resolve_data_dir():
    env_dir = os.environ.get('DATA_DIR')
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    for cand in (
        os.path.join(SUBMISSION_DIR, 'data'),
        os.path.join(os.path.dirname(SUBMISSION_DIR), 'data'),
    ):
        if os.path.isdir(cand):
            return cand
    return os.path.join(SUBMISSION_DIR, 'data')


DATA_DIR = _resolve_data_dir()
ZIP_PATH = os.path.join(DATA_DIR, 'UZB_2018-2025_L2CU_v03_M_CSV.zip')
CPI_1285 = os.path.join(DATA_DIR, 'sdmx_cpi_1285.csv')
CPI_1286 = os.path.join(DATA_DIR, 'sdmx_cpi_1286.csv')

INCOME_FIELDS = [
    'wage_amount',
    'aginc_amount',
    'selfempinc_amount',
    'otherinc_amount',
    'pension_amount',
    'benefit_amount',
]

SUBPERIODS = [
    ('preCOVID', 'Pre-COVID (r1--19)', 1, 19),
    ('COVID', 'COVID (r20--29)', 20, 29),
    ('recovery', '2021 recovery (r30--42)', 30, 42),
    ('war', '2022--23 price shock (r43--55)', 43, 55),
    ('late', 'Late period (r56--82)', 56, 82),
]

# Flagged savings-drawdown recording-break windows (r58--65 and r78--79)
BREAK_ROUNDS = set(range(58, 66)) | {78, 79}


def clean(val):
    return (val or '').strip()


def parse_float(val):
    a = clean(val).replace(',', '')
    if not a:
        return 0.0
    try:
        return float(a)
    except ValueError:
        return 0.0


def totinc(row):
    return sum(parse_float(row[k]) for k in INCOME_FIELDS)


def load_monthly_cpi(base_ym='2018-09'):
    """Chain official Uzbekistan StatAgency monthly MoM CPI (Code == '1') into level index."""
    mom = {}
    for path in (CPI_1285, CPI_1286):
        with open(path, encoding='utf-8-sig', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if clean(row.get('Code')) == '1':
                    for col, val in row.items():
                        c_norm = col.replace('\u041c', 'M').replace('\u043c', 'M')
                        if len(c_norm) == 8 and c_norm[4:6] == '-M':
                            ym = c_norm[:4] + '-' + c_norm[6:8]
                            v = clean(val)
                            if v:
                                mom[ym] = float(v)
                    break
    months = sorted(mom.keys())
    idx = {}
    lvl = 100.0
    for ym in months:
        lvl = lvl * (mom[ym] / 100.0)
        idx[ym] = lvl
    base_val = idx[base_ym]
    return {ym: 100.0 * v / base_val for ym, v in idx.items()}


def htm_flags(row):
    """Return all HtM component flags and harmonized/robustness states."""
    sv = clean(row['savings_past30d'])
    nosav = (sv == 'Do not have any savings')
    sav_no = (sv == 'No')
    took_sav = (sv == 'Yes')

    loan = (clean(row['loan_past30d']) == 'Yes')
    assets = (clean(row['assets_past30d']) == 'Yes')
    cons = (clean(row['consumption_past30d']) == 'Yes')
    foodno = (clean(row['food_past30d']) == 'No')
    util = (clean(row['utility_past30d']) == 'Yes')

    coping_hard = loan or assets or cons
    coping_core = coping_hard or (nosav and foodno)
    coping_unadj = coping_hard or foodno
    coping_wt = coping_unadj or took_sav

    broad_core = nosav or coping_hard
    strict_core = nosav and (coping_hard or foodno)
    st_core = 2 if strict_core else (1 if broad_core else 0)

    broad_unadj = nosav or coping_unadj
    st_unadj = 2 if strict_core else (1 if broad_unadj else 0)

    broad_wt = nosav or coping_wt
    st_wt = 2 if strict_core else (1 if broad_wt else 0)

    fin_ago = clean(row['finance_30d_ago']).lower()
    fin_next = clean(row['finance_30d_next']).lower()
    fin_worse = ('worse' in fin_ago)
    fin_better_next = ('better' in fin_next)
    q5_agree = (clean(row['views_q5']) in ('Agree', 'Completely agree'))

    fies_vals = [clean(row[f'ln_{k}']) for k in range(1, 9)]
    if any(v in ('Yes', 'No') for v in fies_vals):
        fies_any = any(v == 'Yes' for v in fies_vals)
    else:
        fies_any = None

    return {
        'sv': sv,
        'nosav': nosav,
        'sav_no': sav_no,
        'took_sav': took_sav,
        'loan': loan,
        'assets': assets,
        'cons': cons,
        'foodno': foodno,
        'util': util,
        'coping_hard': coping_hard,
        'coping_core': coping_core,
        'coping_unadj': coping_unadj,
        'coping_wt': coping_wt,
        'broad_core': broad_core,
        'strict_core': strict_core,
        'st_core': st_core,
        'is_m1': (nosav and not coping_core),
        'is_m2': ((not nosav) and coping_hard),
        'broad_unadj': broad_unadj,
        'st_unadj': st_unadj,
        'broad_wt': broad_wt,
        'st_wt': st_wt,
        'fin_worse': fin_worse,
        'fin_better_next': fin_better_next,
        'q5_agree': q5_agree,
        'fies_any': fies_any,
    }


def subperiod_of(rd):
    for code, _, lo, hi in SUBPERIODS:
        if lo <= rd <= hi:
            return code
    raise ValueError(f'Unknown round: {rd}')


def iter_household_rows():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        hh_name = [n for n in zf.namelist() if 'household' in n.lower()][0]
        with zf.open(hh_name) as f:
            rdr = csv.DictReader(io.TextIOWrapper(f, 'utf-8-sig'))
            for row in rdr:
                yield row


def iter_individual_rows():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        ind_name = [n for n in zf.namelist() if 'individual' in n.lower()][0]
        with zf.open(ind_name) as f:
            rdr = csv.DictReader(io.TextIOWrapper(f, 'utf-8-sig'))
            for row in rdr:
                yield row


def weighted_quantile(vals_weights, q):
    arr = sorted(vals_weights, key=lambda x: x[0])
    tot = sum(w for _, w in arr)
    if tot <= 0:
        return float('nan')
    target = q * tot
    cum = 0.0
    for v, w in arr:
        cum += w
        if cum >= target:
            return v
    return arr[-1][0]


def clustered_mean_se(hh_obs_map):
    G = len(hh_obs_map)
    W_tot = sum(w for obs in hh_obs_map.values() for _, w in obs)
    mu = sum(w * y for obs in hh_obs_map.values() for y, w in obs) / W_tot
    ssq = 0.0
    for obs in hh_obs_map.values():
        g_score = sum(w * (y - mu) for y, w in obs)
        ssq += g_score * g_score
    var = (G / (G - 1)) * ssq / (W_tot * W_tot)
    return mu, math.sqrt(var)


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
