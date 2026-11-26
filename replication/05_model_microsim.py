"""Step 05: IBC-Consistent 3-State Markov-TANK Microsimulation (CRN).

Features:
1. Baseline Smoothed Buffer-Rebuilding IBC rule:
     dc_t = m_{s_t} * dy_t + delta_a * (1+r) * da_{t-1},  da_t = (1+r)*da_{t-1} + dy_t - dc_t
   with delta_a = m_S across states -> exact 120-month PV(dC)/PV(dY) = 1.000, ZERO
   zero-consumption floor hits, and no artificial single-month callable-debt crash.
2. Explicit Callable-Debt Clawback sensitivity variant (delta_a = m_{s_t} = 1 on HtM switch)
   reporting floor-hit counts and hump comparison transparently.
3. Companion Forward-Looking Buffer-Stock iMPC rule (smoothed IBC) and Fixed-Status
   Infinite-Horizon PIH benchmark (P = I; zero floor hits; exact 120m & asymptotic PV).
4. Monthly m_S = 0.03 check (quarterly saver MPC 0.087) alongside m_S = 0.07 baseline,
   m_M = 0.85, r = 0 cash-at-home, window-only break-imputed, and all-round imputed runs.
5. Exact finite-N discrete job-loss targeting for both -10.00% (p_H capped at 100%) and
   uncapped -5.00% (p_H ~ 58%), with split S->M (+2.6pp) / S->H (+20.3pp) and M->H (+16.0pp)
   endogenous switching.
"""
import csv
import json
import math
import os
import random
import time

from common import OUT_DIR, SUBPERIODS, ensure_out_dir

ensure_out_dir()

SEED = 12345
N = 40000
BURN = 30
SHOCK_T = BURN + 10   # t* = 40
H_TRACK = 120         # 120 post-shock months to verify exact IBC closure
T = SHOCK_T + H_TRACK + 1
R_MONTH = 0.0025      # 3% annualized real rate benchmark

MS_BASE = 0.07
MM_BASE = 1.00
MH_BASE = 1.00

with open(os.path.join(OUT_DIR, 'transitions.json')) as f:
    trans_data = json.load(f)
with open(os.path.join(OUT_DIR, 'income_process.json')) as f:
    inc_data = json.load(f)
with open(os.path.join(OUT_DIR, 'exposure_validation.json')) as f:
    exp_data = json.load(f)
with open(os.path.join(OUT_DIR, 'diagnostics.json')) as f:
    diag_data = json.load(f)

P_BASE = trans_data['core_harmonized']['pooled']['P_w']
PI_BASE = trans_data['core_harmonized']['pooled']['steady_w']
P_HMM = diag_data['hmm_misclassification']['post_r29_weighted']['latent_A']
PI_HMM = diag_data['hmm_misclassification']['post_r29_weighted']['latent_stationary_pi']

ppt_net = inc_data['ppt_lag24_net_mu_for_microsim']
RHO_Z = ppt_net['rho_z']
SIG_ETA = ppt_net['sig_eta']
SIG_Z = ppt_net['sig_z_stationary']
SIG_ALPHA = ppt_net['sig_alpha']

mu_dict = inc_data['microsim_mu_within_round']
MU = [mu_dict['S'], mu_dict['M'], mu_dict['H']]


def income_weights_from_pi(pi):
    raw = [pi[s] * math.exp(MU[s]) for s in range(3)]
    tot = sum(raw)
    return [r / tot for r in raw]


def agg_mpc_pop(pi, ms=MS_BASE, mM=MM_BASE, mH=MH_BASE):
    return pi[0] * ms + pi[1] * mM + pi[2] * mH


def agg_mpc_inc(pi, ms=MS_BASE, mM=MM_BASE, mH=MH_BASE):
    w = income_weights_from_pi(pi)
    return w[0] * ms + w[1] * mM + w[2] * mH


def make_cum(P):
    return [[P[r][0], P[r][0] + P[r][1], 1.0] for r in range(3)]


# Pre-draw Common Random Numbers (CRN)
t0 = time.time()
rng = random.Random(SEED)
u_init = [rng.random() for _ in range(N)]
alpha = [rng.gauss(0.0, SIG_ALPHA) for _ in range(N)]
z0 = [rng.gauss(0.0, SIG_Z) for _ in range(N)]
E = [[rng.gauss(0.0, SIG_ETA) for _ in range(N)] for _ in range(T)]
U = [[rng.random() for _ in range(N)] for _ in range(T)]
U_job = [rng.random() for _ in range(N)]
U_sw = [rng.random() for _ in range(N)]


def init_states(pi):
    c0, c1 = pi[0], pi[0] + pi[1]
    return [0 if r < c0 else (1 if r < c1 else 2) for r in u_init]


def run_baseline_snapshot(P, pi):
    cum = make_cum(P)
    st = init_states(pi)
    z = z0[:]
    for t in range(SHOCK_T + 1):
        Et = E[t]
        Ut = U[t]
        if t == SHOCK_T:
            y_star = [0.0] * N
            st_star = st[:]
            for i in range(N):
                zi = RHO_Z * z[i] + Et[i]
                y_star[i] = math.exp(MU[st[i]] + alpha[i] + zi)
            return st_star, y_star
        for i in range(N):
            z[i] = RHO_Z * z[i] + Et[i]
            r = Ut[i]
            c = cum[st[i]]
            st[i] = 0 if r < c[0] else (1 if r < c[1] else 2)


def sim_ibc(P, pi, m_vec=(MS_BASE, MM_BASE, MH_BASE), r_rate=R_MONTH,
            uniform_shocks=None, lump_sum_frac=0.0,
            discrete_job_probs=None, discrete_job_log_hits=None,
            sw_S_to_M=0.0, sw_S_to_H=0.0, sw_M_to_H=0.0,
            saver_mode='smoothed_ibc', clawback_on_htm=False,
            fixed_state_after_shock=False):
    """IBC-consistent CRN microsimulation.

    saver_mode:
      - 'smoothed_ibc': dc_t = m_{s_t}*dy_t + delta_a*(1+r)*da_{t-1}, where delta_a = m_S
        (or m_{s_t} if clawback_on_htm=True). Exact 120m PV(dC)/PV(dY) = 1.000.
      - 'fl_impc': forward-looking buffer-stock iMPC for Savers (m0_fl*dy + m_S*(1+r)*da),
        smoothed rebuilding on switch -> exact 120m PV = 1.000.
      - 'pih_fixed': infinite-horizon PIH for Savers with fixed post-shock state (P = I)
        so permanent-income evaluation is internally consistent -> 0 floor hits.
    """
    cum = make_cum(P)
    st_b = init_states(pi)
    st_s = st_b[:]
    zb = z0[:]
    zs = z0[:]
    da = [0.0] * N
    m0, m1, m2 = m_vec
    one_r = 1.0 + r_rate

    m0_fl = m0 / (1.0 - (1.0 - m0) * RHO_Z / one_r)
    m0_pih = r_rate / (one_r - RHO_Z) if r_rate > 0 else 0.0

    H_len = H_TRACK + 1
    Yb_h = [0.0] * H_len
    Ys_h = [0.0] * H_len
    Cb_h = [0.0] * H_len
    Cs_h = [0.0] * H_len
    Cb_st_h = [[0.0] * H_len for _ in range(3)]
    Cs_st_h = [[0.0] * H_len for _ in range(3)]
    Yb_st_h = [[0.0] * H_len for _ in range(3)]
    Ys_st_h = [[0.0] * H_len for _ in range(3)]

    cohort_st = None
    floor_hits = 0

    for t in range(T):
        Et = E[t]
        Ut = U[t]
        h = t - SHOCK_T
        u_sh = uniform_shocks.get(t) if uniform_shocks else None

        if t == SHOCK_T:
            cohort_st = st_b[:]

        ls_som = 0.0
        if lump_sum_frac != 0.0 and t == SHOCK_T:
            yt_tmp = 0.0
            for i in range(N):
                sb = st_b[i]
                yt_tmp += math.exp(MU[sb] + alpha[i] + RHO_Z * zb[i] + Et[i])
            ls_som = lump_sum_frac * (yt_tmp / N)

        yb_sum = ys_sum = cb_sum = cs_sum = 0.0
        for i in range(N):
            sb = st_b[i]
            ss = st_s[i]

            zbi = RHO_Z * zb[i] + Et[i]
            zb[i] = zbi

            if t < SHOCK_T:
                zs[i] = zbi
                r = Ut[i]
                cb_row = cum[sb]
                ns = 0 if r < cb_row[0] else (1 if r < cb_row[1] else 2)
                st_b[i] = ns
                st_s[i] = ns
                continue

            sh_i = 0.0
            hit_job_i = False
            if u_sh is not None:
                sh_i = u_sh[sb]
            elif discrete_job_probs is not None and t == SHOCK_T:
                if U_job[i] < discrete_job_probs[sb]:
                    sh_i = discrete_job_log_hits[sb]
                    hit_job_i = True

            # Empirical endogenous switching strictly on job-losing households (split S->M/H & M->H)
            if t == SHOCK_T and hit_job_i:
                if ss == 0 and (sw_S_to_H > 0.0 or sw_S_to_M > 0.0):
                    u_s = U_sw[i]
                    if u_s < sw_S_to_H:
                        ss = 2
                        st_s[i] = 2
                    elif u_s < sw_S_to_H + sw_S_to_M:
                        ss = 1
                        st_s[i] = 1
                elif ss == 1 and sw_M_to_H > 0.0:
                    if U_sw[i] < sw_M_to_H:
                        ss = 2
                        st_s[i] = 2

            zsi = RHO_Z * zs[i] + Et[i] + sh_i
            zs[i] = zsi

            yb = math.exp(MU[sb] + alpha[i] + zbi)
            ys = math.exp(MU[sb] + alpha[i] + zsi) + ls_som
            dy = ys - yb

            ms_i = m0 if ss == 0 else (m1 if ss == 1 else m2)
            a_prev = one_r * da[i]
            coh = a_prev + dy

            if ss == 0:
                if saver_mode == 'smoothed_ibc':
                    dc = m0 * dy + m0 * a_prev
                elif saver_mode == 'fl_impc':
                    dc = (m0_fl * dy + m0 * a_prev) if lump_sum_frac == 0.0 else (m0 * coh)
                elif saver_mode == 'pih_fixed':
                    pv_future = 0.0 if lump_sum_frac != 0.0 else (dy * (RHO_Z / (one_r - RHO_Z)))
                    dc = (r_rate / one_r) * (a_prev + dy + pv_future) if r_rate > 0 else 0.0
                else:
                    raise ValueError(saver_mode)
            else:
                if clawback_on_htm:
                    dc = ms_i * coh
                else:
                    # Smoothed buffer-rebuilding IBC: HtM spends ms_i of current dy
                    # and smoothly rebuilds any prior buffer deviation at rate delta_a = m0.
                    dc = ms_i * dy + m0 * a_prev

            if yb + dc < 0.0:
                dc = -yb
                floor_hits += 1
            da[i] = coh - dc

            cb = yb
            cs = yb + dc

            yb_sum += yb
            ys_sum += ys
            cb_sum += cb
            cs_sum += cs

            cs0 = cohort_st[i]
            Yb_st_h[cs0][h] += yb
            Ys_st_h[cs0][h] += ys
            Cb_st_h[cs0][h] += cb
            Cs_st_h[cs0][h] += cs

            if not fixed_state_after_shock:
                r = Ut[i]
                cb_row = cum[sb]
                st_b[i] = 0 if r < cb_row[0] else (1 if r < cb_row[1] else 2)
                cs_row = cum[ss]
                st_s[i] = 0 if r < cs_row[0] else (1 if r < cs_row[1] else 2)

        Yb_h[h] = yb_sum / N
        Ys_h[h] = ys_sum / N
        Cb_h[h] = cb_sum / N
        Cs_h[h] = cs_sum / N

    pv_dy = sum((Ys_h[h] - Yb_h[h]) / (one_r ** h) for h in range(H_len))
    pv_dc = sum((Cs_h[h] - Cb_h[h]) / (one_r ** h) for h in range(H_len))
    ibc_ratio = pv_dc / pv_dy if abs(pv_dy) > 1e-9 else None

    pv_dy_S = sum((Ys_st_h[0][h] - Yb_st_h[0][h]) / (one_r ** h) for h in range(H_len))
    pv_dc_S = sum((Cs_st_h[0][h] - Cb_st_h[0][h]) / (one_r ** h) for h in range(H_len))
    ibc_ratio_S = pv_dc_S / pv_dy_S if abs(pv_dy_S) > 1e-9 else None

    keep = list(range(25))
    dY = [Ys_h[h] / Yb_h[h] - 1.0 for h in keep]
    dC = [Cs_h[h] / Cb_h[h] - 1.0 for h in keep]
    dC_by_st = {
        nm: [Cs_st_h[s][h] / Cb_st_h[s][h] - 1.0 for h in keep]
        for s, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
    }
    dY_by_st = {
        nm: [Ys_st_h[s][h] / Yb_st_h[s][h] - 1.0 for h in keep]
        for s, nm in ((0, 'S'), (1, 'M'), (2, 'H'))
    }

    trough_h = min(range(len(dC)), key=lambda idx: dC[idx])

    return {
        'horizons': keep,
        'dY': [round(x, 6) for x in dY],
        'dC': [round(x, 6) for x in dC],
        'dY_by_state': {k: [round(x, 6) for x in v] for k, v in dY_by_st.items()},
        'dC_by_state': {k: [round(x, 6) for x in v] for k, v in dC_by_st.items()},
        'impact_dC_over_dY': round(dC[0] / dY[0], 4) if abs(dY[0]) > 1e-9 else None,
        'trough_h': trough_h,
        'trough_dC': round(dC[trough_h], 6),
        'ibc_pv_ratio_120m_agg': round(ibc_ratio, 4) if ibc_ratio is not None else None,
        'ibc_pv_ratio_120m_S': round(ibc_ratio_S, 4) if ibc_ratio_S is not None else None,
        'cum_dC_over_cum_dY_12m': round(
            sum(Cs_h[h] - Cb_h[h] for h in range(12)) / sum(Ys_h[h] - Yb_h[h] for h in range(12)), 4
        ),
        'zero_cons_floor_hits': floor_hits,
        'm0_fl_implied': round(m0_fl, 4),
        'm0_pih_implied': round(m0_pih, 4),
    }


# 1. Baseline Uniform -10% Shocks (Smoothed IBC, Clawback Sensitivity, FL-iMPC, PIH Fixed-State, m_S=0.03, r=0, m_M=0.85)
DELTA_10 = math.log(0.90)
uni_10 = sim_ibc(P_BASE, PI_BASE, uniform_shocks={SHOCK_T: [DELTA_10] * 3}, saver_mode='smoothed_ibc')
uni_10_claw = sim_ibc(P_BASE, PI_BASE, uniform_shocks={SHOCK_T: [DELTA_10] * 3}, saver_mode='smoothed_ibc', clawback_on_htm=True)
uni_10_fl = sim_ibc(P_BASE, PI_BASE, uniform_shocks={SHOCK_T: [DELTA_10] * 3}, saver_mode='fl_impc')
uni_10_pih = sim_ibc(P_BASE, PI_BASE, uniform_shocks={SHOCK_T: [DELTA_10] * 3}, saver_mode='pih_fixed', fixed_state_after_shock=True)
uni_10_ms03 = sim_ibc(P_BASE, PI_BASE, m_vec=(0.03, MM_BASE, MH_BASE), uniform_shocks={SHOCK_T: [DELTA_10] * 3})
uni_10_r0 = sim_ibc(P_BASE, PI_BASE, r_rate=0.0, uniform_shocks={SHOCK_T: [DELTA_10] * 3})
uni_10_mM85 = sim_ibc(P_BASE, PI_BASE, m_vec=(MS_BASE, 0.85, MH_BASE), uniform_shocks={SHOCK_T: [DELTA_10] * 3})
uni_10_mM50 = sim_ibc(P_BASE, PI_BASE, m_vec=(MS_BASE, 0.50, MH_BASE), uniform_shocks={SHOCK_T: [DELTA_10] * 3})
rank_10 = sim_ibc(P_BASE, PI_BASE, m_vec=(MS_BASE, MS_BASE, MS_BASE), uniform_shocks={SHOCK_T: [DELTA_10] * 3})

# Window-only break-imputed AND all-round imputed pooled runs
P_WIN = trans_data['window_break_imputed']['pooled']['P_w']
PI_WIN = trans_data['window_break_imputed']['pooled']['steady_w']
uni_10_win = sim_ibc(P_WIN, PI_WIN, uniform_shocks={SHOCK_T: [DELTA_10] * 3})

P_ALL = trans_data['all_round_imputed']['pooled']['P_w']
PI_ALL = trans_data['all_round_imputed']['pooled']['steady_w']
uni_10_all = sim_ibc(P_ALL, PI_ALL, uniform_shocks={SHOCK_T: [DELTA_10] * 3})

# HMM misclassification-corrected latent transition matrix A* (both smoothed IBC and clawback/no-borrowing)
uni_10_hmm = sim_ibc(P_HMM, PI_HMM, uniform_shocks={SHOCK_T: [DELTA_10] * 3}, saver_mode='smoothed_ibc')
uni_10_hmm_claw = sim_ibc(P_HMM, PI_HMM, uniform_shocks={SHOCK_T: [DELTA_10] * 3}, saver_mode='smoothed_ibc', clawback_on_htm=True)

# 2. Equal Lump-Sum Transfer (+1% of aggregate Y)
lump_1pct = sim_ibc(P_BASE, PI_BASE, lump_sum_frac=0.01, saver_mode='smoothed_ibc')

# 3. All 5 Core Subperiod Regimes
regime_irfs = {}
for code, _, _, _ in SUBPERIODS:
    rg = trans_data['core_harmonized']['regimes'][code]
    P_r = rg['P_w']
    pi_r = rg['steady_w']
    sim_r = sim_ibc(P_r, pi_r, uniform_shocks={SHOCK_T: [DELTA_10] * 3})
    regime_irfs[code] = {
        'steady_w': [round(x, 4) for x in pi_r],
        'inc_weights': [round(x, 4) for x in income_weights_from_pi(pi_r)],
        'aggmpc_pop': round(agg_mpc_pop(pi_r), 4),
        'aggmpc_inc': round(agg_mpc_inc(pi_r), 4),
        'sim': sim_r,
    }

# 5. Exact Mean-Preserving Discrete Job-Loss Counterfactuals (-10% AND Uncapped -5%!)
st_star, y_star = run_baseline_snapshot(P_BASE, PI_BASE)
Y_star_tot = sum(y_star)

wl = exp_data['work_loss_by_lagged_state']
wl_rates = [wl[nm]['work_loss_rate_w'] for nm in ('S', 'M', 'H')]
tot_drops = [abs(wl[nm]['primary_round_fe_weighted']['implied_pct_Y_drop_total_incl_zero']) for nm in ('S', 'M', 'H')]
job_log_hits = [math.log(1.0 - d) for d in tot_drops]
sw_S_M = max(0.0, wl['S']['primary_round_fe_weighted']['dP_M']['beta'])
sw_S_H = max(0.0, wl['S']['primary_round_fe_weighted']['dP_strict']['beta'])
sw_M_H = max(0.0, wl['M']['primary_round_fe_weighted']['dP_strict']['beta'])


def solve_discrete_targeting(target_drop, st_snap=None, y_snap=None, max_probs=(1.0, 1.0, 1.0)):
    st_use = st_star if st_snap is None else st_snap
    y_use = y_star if y_snap is None else y_snap
    Y_tot_use = sum(y_use)

    def realized_agg_dY(kappa_val):
        probs = [min(max_probs[s], kappa_val * wl_rates[s]) for s in range(3)]
        dy_sum = 0.0
        for i in range(N):
            s = st_use[i]
            if U_job[i] < probs[s]:
                dy_sum += y_use[i] * (-tot_drops[s])
        return dy_sum / Y_tot_use

    lo_k, hi_k = 0.0, 80.0
    for _ in range(60):
        mid = 0.5 * (lo_k + hi_k)
        if realized_agg_dY(mid) > target_drop:
            lo_k = mid
        else:
            hi_k = mid
    kap = 0.5 * (lo_k + hi_k)
    return kap, [min(max_probs[s], kap * wl_rates[s]) for s in range(3)]


def solve_discrete_uniform(target_drop, st_snap=None, y_snap=None):
    st_use = st_star if st_snap is None else st_snap
    y_use = y_star if y_snap is None else y_snap
    probs = []
    for s in range(3):
        Y_s_tot = sum(y_use[i] for i in range(N) if st_use[i] == s)
        lo_p, hi_p = 0.0, 1.0
        for _ in range(50):
            mp = 0.5 * (lo_p + hi_p)
            dy_s = sum(-y_use[i] * tot_drops[s] for i in range(N) if st_use[i] == s and U_job[i] < mp)
            if dy_s / Y_s_tot > target_drop:
                lo_p = mp
            else:
                hi_p = mp
        probs.append(0.5 * (lo_p + hi_p))
    return probs


# HH+Round-FE switching parameters (conservative robustness)
sw_S_H_hhfe = max(0.0, wl['S']['robustness']['dP_broad_round_plus_hh_fe_w']['beta'])
sw_S_M_hhfe = 0.0
sw_M_H_hhfe = max(0.0, wl['M']['robustness']['dP_strict_round_plus_hh_fe_w']['beta'])

# -10.00% targeted & uniform discrete
kappa_10, targ_probs_10 = solve_discrete_targeting(-0.10)
uni_disc_probs_10 = solve_discrete_uniform(-0.10)

targ_exog_10 = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=targ_probs_10, discrete_job_log_hits=job_log_hits)
targ_endog_10 = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=targ_probs_10, discrete_job_log_hits=job_log_hits,
                        sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)
uni_endog_10 = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=uni_disc_probs_10, discrete_job_log_hits=job_log_hits,
                       sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)

# HH+Round-FE switching robustness on -10%
targ_endog_10_hhfe = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=targ_probs_10, discrete_job_log_hits=job_log_hits,
                             sw_S_to_M=sw_S_M_hhfe, sw_S_to_H=sw_S_H_hhfe, sw_M_to_H=sw_M_H_hhfe)
uni_endog_10_hhfe = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=uni_disc_probs_10, discrete_job_log_hits=job_log_hits,
                            sw_S_to_M=sw_S_M_hhfe, sw_S_to_H=sw_S_H_hhfe, sw_M_to_H=sw_M_H_hhfe)

targ_exog_mM85 = sim_ibc(P_BASE, PI_BASE, m_vec=(MS_BASE, 0.85, MH_BASE),
                         discrete_job_probs=targ_probs_10, discrete_job_log_hits=job_log_hits)
targ_endog_mM85 = sim_ibc(P_BASE, PI_BASE, m_vec=(MS_BASE, 0.85, MH_BASE),
                          discrete_job_probs=targ_probs_10, discrete_job_log_hits=job_log_hits,
                          sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)
targ_exog_mM50 = sim_ibc(P_BASE, PI_BASE, m_vec=(MS_BASE, 0.50, MH_BASE),
                         discrete_job_probs=targ_probs_10, discrete_job_log_hits=job_log_hits)
targ_endog_mM50 = sim_ibc(P_BASE, PI_BASE, m_vec=(MS_BASE, 0.50, MH_BASE),
                          discrete_job_probs=targ_probs_10, discrete_job_log_hits=job_log_hits,
                          sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)

# Earner-capped -10.00% check (capping job-loss prob at empirical positive-earner share per state)
earner_sh = diag_data['job_loss_ext']['earner_share_by_state']
earner_caps = (earner_sh['S'], earner_sh['M'], earner_sh['H'])
kappa_10_ecap, targ_probs_10_ecap = solve_discrete_targeting(-0.10, max_probs=earner_caps)
targ_exog_10_ecap = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=targ_probs_10_ecap, discrete_job_log_hits=job_log_hits)
targ_endog_10_ecap = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=targ_probs_10_ecap, discrete_job_log_hits=job_log_hits,
                             sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)

# Targeted -10.00% under persistent post-r29 HMM latent kernel A*
st_star_hmm, y_star_hmm = run_baseline_snapshot(P_HMM, PI_HMM)
kappa_10_hmm, targ_probs_10_hmm = solve_discrete_targeting(-0.10, st_snap=st_star_hmm, y_snap=y_star_hmm)
uni_disc_probs_10_hmm = solve_discrete_uniform(-0.10, st_snap=st_star_hmm, y_snap=y_star_hmm)
targ_exog_10_hmm = sim_ibc(P_HMM, PI_HMM, discrete_job_probs=targ_probs_10_hmm, discrete_job_log_hits=job_log_hits)
targ_endog_10_hmm = sim_ibc(P_HMM, PI_HMM, discrete_job_probs=targ_probs_10_hmm, discrete_job_log_hits=job_log_hits,
                            sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)
uni_endog_10_hmm = sim_ibc(P_HMM, PI_HMM, discrete_job_probs=uni_disc_probs_10_hmm, discrete_job_log_hits=job_log_hits,
                           sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)

# Uncapped -5.00% targeted & uniform discrete (p_H ~ 58% < 74.1% earner share)
DELTA_5 = math.log(0.95)
uni_5 = sim_ibc(P_BASE, PI_BASE, uniform_shocks={SHOCK_T: [DELTA_5] * 3})
kappa_5, targ_probs_5 = solve_discrete_targeting(-0.05)
uni_disc_probs_5 = solve_discrete_uniform(-0.05)
targ_exog_5 = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=targ_probs_5, discrete_job_log_hits=job_log_hits)
targ_endog_5 = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=targ_probs_5, discrete_job_log_hits=job_log_hits,
                       sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)
uni_endog_5 = sim_ibc(P_BASE, PI_BASE, discrete_job_probs=uni_disc_probs_5, discrete_job_log_hits=job_log_hits,
                      sw_S_to_M=sw_S_M, sw_S_to_H=sw_S_H, sw_M_to_H=sw_M_H)

# 6. 60-Cell Sensitivity Grid (5 m_S x 4 m_M x 3 shocks)
grid_path = os.path.join(OUT_DIR, 'sensitivity_grid.csv')
with open(grid_path, 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['saver_mpc', 'M_mpc', 'shock', 'AggMPC_pop', 'AggMPC_inc',
                  'dC_lumpsum_impact', 'dC_prop_impact'])
    for ms in (0.03, 0.05, 0.07, 0.10, 0.15):
        for mM in (0.50, 0.70, 0.85, 1.00):
            ap = agg_mpc_pop(PI_BASE, ms=ms, mM=mM)
            ai = agg_mpc_inc(PI_BASE, ms=ms, mM=mM)
            for sh in (-0.05, -0.10, -0.15):
                wtr.writerow([ms, mM, sh, round(ap, 4), round(ai, 4),
                              round(ap * sh, 4), round(ai * sh, 4)])

w_inc = income_weights_from_pi(PI_BASE)
pi_wt = trans_data['inclusive_withtook']['pooled']['steady_w']
pi_unadj = trans_data['unadj_food']['pooled']['steady_w']

match_extra = abs(targ_exog_10['dC'][0]) - abs(uni_10['dC'][0])
sw_extra_targ = abs(targ_endog_10['dC'][0]) - abs(targ_exog_10['dC'][0])
sw_extra_uni = abs(uni_endog_10['dC'][0]) - abs(uni_10['dC'][0])

results = {
    'calibration': {
        'N': N, 'seed': SEED, 'r_month': R_MONTH,
        'rho_z': RHO_Z, 'sig_eta': SIG_ETA, 'sig_alpha': SIG_ALPHA, 'MU': MU,
        'steady_w': [round(x, 4) for x in PI_BASE],
        'inc_weights': [round(x, 4) for x in w_inc],
        'ms_base': MS_BASE, 'mM_base': MM_BASE, 'mH_base': MH_BASE,
        'ms_quarterly_implied': round(1.0 - (1.0 - MS_BASE) ** 3, 4),
        'ms_annual_implied': round(1.0 - (1.0 - MS_BASE) ** 12, 4),
        'ms03_quarterly_implied': round(1.0 - (1.0 - 0.03) ** 3, 4),
        'aggmpc_pop_base': round(agg_mpc_pop(PI_BASE), 4),
        'aggmpc_inc_base': round(agg_mpc_inc(PI_BASE), 4),
        'aggmpc_inc_ms03': round(agg_mpc_inc(PI_BASE, ms=0.03), 4),
        'aggmpc_inc_mM85': round(agg_mpc_inc(PI_BASE, mM=0.85), 4),
        'aggmpc_inc_mM70': round(agg_mpc_inc(PI_BASE, mM=0.70), 4),
        'aggmpc_inc_mM50': round(agg_mpc_inc(PI_BASE, mM=0.50), 4),
        'aggmpc_pop_mM50': round(agg_mpc_pop(PI_BASE, mM=0.50), 4),
        'win_imputed_steady_w': [round(x, 4) for x in PI_WIN],
        'win_imputed_aggmpc_pop': round(agg_mpc_pop(PI_WIN), 4),
        'win_imputed_aggmpc_inc': round(agg_mpc_inc(PI_WIN), 4),
        'all_imputed_steady_w': [round(x, 4) for x in PI_ALL],
        'all_imputed_aggmpc_pop': round(agg_mpc_pop(PI_ALL), 4),
        'all_imputed_aggmpc_inc': round(agg_mpc_inc(PI_ALL), 4),
        'hmm_latent_steady_pi': [round(x, 4) for x in PI_HMM],
        'hmm_latent_aggmpc_pop': round(agg_mpc_pop(PI_HMM), 4),
        'hmm_latent_aggmpc_inc': round(agg_mpc_inc(PI_HMM), 4),
        'unadj_food_aggmpc_inc': round(agg_mpc_inc(pi_unadj), 4),
        'inclusive_withtook_aggmpc_inc': round(agg_mpc_inc(pi_wt), 4),
    },
    'uniform_10pct_smoothed_ibc': uni_10,
    'uniform_10pct_clawback_ibc': uni_10_claw,
    'uniform_10pct_hmm_latent': uni_10_hmm,
    'uniform_10pct_hmm_latent_clawback': uni_10_hmm_claw,
    'uniform_10pct_fl_impc': uni_10_fl,
    'uniform_10pct_pih_fixed_S': uni_10_pih,
    'uniform_10pct_ms03': uni_10_ms03,
    'uniform_10pct_r0_cash': uni_10_r0,
    'uniform_10pct_mM85': uni_10_mM85,
    'uniform_10pct_mM50': uni_10_mM50,
    'uniform_10pct_win_imputed': uni_10_win,
    'uniform_10pct_all_imputed': uni_10_all,
    'uniform_10pct_endog_switch': uni_endog_10,
    'uniform_10pct_endog_switch_hhfe': uni_endog_10_hhfe,
    'rank_10pct': rank_10,
    'lumpsum_1pct': lump_1pct,
    'regime_irfs': regime_irfs,
    'targeted_discrete_10pct': {
        'kappa': round(kappa_10, 4),
        'wl_rates_empirical': {nm: round(wl_rates[i], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'job_loss_total_pct_drop': {nm: round(tot_drops[i], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'scaled_job_loss_probs': {nm: round(targ_probs_10[i], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'switch_probs': {'S_to_M': round(sw_S_M, 4), 'S_to_H': round(sw_S_H, 4), 'M_to_H': round(sw_M_H, 4)},
        'switch_probs_hhfe': {'S_to_M': round(sw_S_M_hhfe, 4), 'S_to_H': round(sw_S_H_hhfe, 4), 'M_to_H': round(sw_M_H_hhfe, 4)},
        'exogenous_P': targ_exog_10,
        'endogenous_P': targ_endog_10,
        'endogenous_P_hhfe': targ_endog_10_hhfe,
        'exogenous_P_mM85': targ_exog_mM85,
        'endogenous_P_mM85': targ_endog_mM85,
        'exogenous_P_mM50': targ_exog_mM50,
        'endogenous_P_mM50': targ_endog_mM50,
        'matching_multiplier_ratio': round(targ_exog_10['impact_dC_over_dY'] / uni_10['impact_dC_over_dY'], 4),
        'targeting_holding_switching_fixed_ratio': round(targ_endog_10['impact_dC_over_dY'] / uni_endog_10['impact_dC_over_dY'], 4),
        'total_amplification_ratio': round(targ_endog_10['impact_dC_over_dY'] / uni_10['impact_dC_over_dY'], 4),
        'targeting_holding_switching_fixed_hhfe': round(targ_endog_10_hhfe['impact_dC_over_dY'] / uni_endog_10_hhfe['impact_dC_over_dY'], 4),
        'total_amplification_ratio_hhfe': round(targ_endog_10_hhfe['impact_dC_over_dY'] / uni_10['impact_dC_over_dY'], 4),
        'matching_extra_pct': round(100 * match_extra, 2),
        'switching_extra_targ_pct': round(100 * sw_extra_targ, 2),
        'switching_extra_uni_pct': round(100 * sw_extra_uni, 2),
        'matching_to_switching_ratio': round(match_extra / sw_extra_targ, 2) if sw_extra_targ > 0 else None,
        'matching_multiplier_ratio_mM85': round(targ_exog_mM85['impact_dC_over_dY'] / uni_10_mM85['impact_dC_over_dY'], 4),
        'total_amplification_ratio_mM85': round(targ_endog_mM85['impact_dC_over_dY'] / uni_10_mM85['impact_dC_over_dY'], 4),
        'matching_multiplier_ratio_mM50': round(targ_exog_mM50['impact_dC_over_dY'] / uni_10_mM50['impact_dC_over_dY'], 4),
        'total_amplification_ratio_mM50': round(targ_endog_mM50['impact_dC_over_dY'] / uni_10_mM50['impact_dC_over_dY'], 4),
    },
    'targeted_discrete_10pct_earner_capped': {
        'earner_caps': {'S': earner_caps[0], 'M': earner_caps[1], 'H': earner_caps[2]},
        'kappa': round(kappa_10_ecap, 4),
        'scaled_job_loss_probs': {nm: round(targ_probs_10_ecap[i], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'exogenous_P': targ_exog_10_ecap,
        'endogenous_P': targ_endog_10_ecap,
        'matching_multiplier_ratio': round(targ_exog_10_ecap['impact_dC_over_dY'] / uni_10['impact_dC_over_dY'], 4),
        'targeting_holding_switching_fixed_ratio': round(targ_endog_10_ecap['impact_dC_over_dY'] / uni_endog_10['impact_dC_over_dY'], 4),
        'total_amplification_ratio': round(targ_endog_10_ecap['impact_dC_over_dY'] / uni_10['impact_dC_over_dY'], 4),
    },
    'targeted_discrete_10pct_hmm_kernel': {
        'kappa': round(kappa_10_hmm, 4),
        'scaled_job_loss_probs': {nm: round(targ_probs_10_hmm[i], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'uniform_10pct_hmm': uni_10_hmm,
        'uniform_10pct_hmm_endog_switch': uni_endog_10_hmm,
        'exogenous_A': targ_exog_10_hmm,
        'endogenous_A': targ_endog_10_hmm,
        'matching_multiplier_ratio': round(targ_exog_10_hmm['impact_dC_over_dY'] / uni_10_hmm['impact_dC_over_dY'], 4),
        'targeting_holding_switching_fixed_ratio': round(targ_endog_10_hmm['impact_dC_over_dY'] / uni_endog_10_hmm['impact_dC_over_dY'], 4),
        'total_amplification_ratio': round(targ_endog_10_hmm['impact_dC_over_dY'] / uni_10_hmm['impact_dC_over_dY'], 4),
    },
    'targeted_discrete_5pct_uncapped': {
        'kappa': round(kappa_5, 4),
        'scaled_job_loss_probs': {nm: round(targ_probs_5[i], 4) for i, nm in ((0, 'S'), (1, 'M'), (2, 'H'))},
        'uniform_5pct': uni_5,
        'uniform_5pct_endog_switch': uni_endog_5,
        'exogenous_P': targ_exog_5,
        'endogenous_P': targ_endog_5,
        'matching_multiplier_ratio': round(targ_exog_5['impact_dC_over_dY'] / uni_5['impact_dC_over_dY'], 4),
        'targeting_holding_switching_fixed_ratio': round(targ_endog_5['impact_dC_over_dY'] / uni_endog_5['impact_dC_over_dY'], 4),
        'total_amplification_ratio': round(targ_endog_5['impact_dC_over_dY'] / uni_5['impact_dC_over_dY'], 4),
    },
    'runtime_sec': round(time.time() - t0, 1),
}

out_path = os.path.join(OUT_DIR, 'microsim_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"wrote {out_path} in {results['runtime_sec']}s")
print(f"Smoothed IBC: dC[0,3,12]={uni_10['dC'][0]}, {uni_10['dC'][3]}, {uni_10['dC'][12]} | trough=h{uni_10['trough_h']}({uni_10['trough_dC']}) | IBC={uni_10['ibc_pv_ratio_120m_agg']} | floor_hits={uni_10['zero_cons_floor_hits']}")
print(f"  S cohort [0,1,3,6,12]={[uni_10['dC_by_state']['S'][h] for h in (0,1,3,6,12)]}")
print(f"  H cohort [0,1,3,6,12]={[uni_10['dC_by_state']['H'][h] for h in (0,1,3,6,12)]}")
print(f"Clawback IBC: dC[0,3,12]={uni_10_claw['dC'][0]}, {uni_10_claw['dC'][3]}, {uni_10_claw['dC'][12]} | floor_hits={uni_10_claw['zero_cons_floor_hits']}")
print(f"FL-iMPC     : dC[0,3,12]={uni_10_fl['dC'][0]}, {uni_10_fl['dC'][3]}, {uni_10_fl['dC'][12]} | IBC={uni_10_fl['ibc_pv_ratio_120m_agg']} | floor_hits={uni_10_fl['zero_cons_floor_hits']}")
print(f"PIH fixed-S : dC[0,3,12]={uni_10_pih['dC'][0]}, {uni_10_pih['dC'][3]}, {uni_10_pih['dC'][12]} | IBC_120m={uni_10_pih['ibc_pv_ratio_120m_agg']} | floor_hits={uni_10_pih['zero_cons_floor_hits']}")
print(f"Targeted -10%: exog={targ_exog_10['dC'][0]} ({results['targeted_discrete_10pct']['matching_multiplier_ratio']}x), endog={targ_endog_10['dC'][0]} ({results['targeted_discrete_10pct']['total_amplification_ratio']}x), mM85 endog={targ_endog_mM85['dC'][0]} ({results['targeted_discrete_10pct']['total_amplification_ratio_mM85']}x)")
print(f"Targeted -10% earner-cap: probs={results['targeted_discrete_10pct_earner_capped']['scaled_job_loss_probs']}, exog={results['targeted_discrete_10pct_earner_capped']['matching_multiplier_ratio']}x, endog={results['targeted_discrete_10pct_earner_capped']['total_amplification_ratio']}x")
print(f"Targeted -10% HMM kernel: impact exog={targ_exog_10_hmm['dC'][0]} ({results['targeted_discrete_10pct_hmm_kernel']['matching_multiplier_ratio']}x), endog={targ_endog_10_hmm['dC'][0]} ({results['targeted_discrete_10pct_hmm_kernel']['total_amplification_ratio']}x), 12m cum ratio={targ_endog_10_hmm['cum_dC_over_cum_dY_12m']} vs uni {uni_10_hmm['cum_dC_over_cum_dY_12m']}")
print(f"Targeted -5% uncapped: probs={results['targeted_discrete_5pct_uncapped']['scaled_job_loss_probs']}, exog={results['targeted_discrete_5pct_uncapped']['matching_multiplier_ratio']}x, endog={results['targeted_discrete_5pct_uncapped']['total_amplification_ratio']}x")
