# -*- coding: utf-8 -*-
"""
CF/DR 渠道归因赛马（Paper 2，2026-08-17）
==========================================
动机：把预测目标从收益换成 Campbell (1991) 分解的现金流新闻（CF news）与折现率新闻
（DR news）。理论预测（Atmaz：分歧是关于现金流信念的）：d̃+CSR 的优势应加载在 CF 渠道；
raw+CSR 的 pre-2020 优势（如存在）应加载在 DR 渠道（lv 是折现率对象）。交叉形态 = 渠道归因证据。

分解口径 = Deng (2024) 附录 IA.2 逐式设定：
  x_t = log(D_t/P_t)；ρ̂ = 1/(1+exp(x̄))（x̄ 全样本均值）
  VAR(1)：v_t = [r_t, x_t, z_t']'（各元素去均值），v_{t+1} = A v_t + u_{t+1}，OLS
  ε^r_{t+1} = e1'u_{t+1}；η_DR = e1'ρA(I−ρA)^{-1}u_{t+1}；η_CF = ε^r + η_DR；E_t r_{t+1} = e1'Av_t
  r = log 全收益 log1p(CRSP_SPvw)。主线全样本 OLS 估 VAR（Deng IA.2 末段口径，对所有预测器对称）；
  稳健变体 = expanding-VAR（每步只用 ≤t−1 数据估 A/ρ，burn-in 120 月）。

数据与对齐：
  r 与 x 取自 Goyal PredictorData2024.xlsx（Monthly，1984-01~2024-12，与面板 492 行同网格）；
  ev.y[t]（date=t）= 月 t+1 的 log 超额收益（CRSP 源，与 Goyal log 全收益−log1p(Rfree) 逐月
  相关 0.9996，差为 RF vintage）。行 t 的新闻目标 = 月 t+1 实现的新闻 u_{t+1}（与 y[t] 同月）。
  Goyal 止于 2024-12 → 行 491（目标月 2025-01）新闻缺失；赛马三目标统一在共同有效行
  （h=1: 行 N_TRAIN..490）上评估，严格配对；收益腿锚点另在完整 OOS 行上单独断言。
  z 控制（固定先验，不事后挑选）：tbl、DFY=BAA−AAA、TMS=lty−tbl（Goyal 原始变量，非 u_ 族）。

赛马：目标 ∈ {收益(参照), CF, DR} × 预测器 ∈ {raw, d̃_lv}（--panel9 加 d̃_pair），
N_TRAIN 由 --start 给出（默认 1991 主线 = 84；--start 2001 复现 2001 口径），
R²OOS 对同目标 expanding 历史均值（ha_h 口径：仅用已完成目标）。h=1 主线 + h=3/h=12 累积
（前向求和；训练截断 s ≤ i−h，同 run_frequency 口径）。

输出：results/cfdr_news_series{PTAG}.csv（新闻序列与起点无关）、results/cfdr_race[_1991]{PTAG}.csv
（2001 起点保持旧名，1991 起点加 _1991）

--panel9（2026-08-19，T3）：9 面板绑定版——集成走幂集 CSR；赛马臂加 d~_pair
（RACE_VARIANTS）；锚点 gate 对照 preds_{start}_p9.csv；新闻序列面板无关
（仅 Goyal + VAR），写 cfdr_news_series_p9.csv 备查；输出 cfdr_race[_1991]_p9.csv。
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # noqa: E402

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
STAG = '' if START == 2001 else f'_{START}'  # cfdr_race 表：2001 起点沿用旧名（回归兼容）

R2OOS = ev.R2OOS
RES = HERE / 'results'
N_TRAIN = ev.STARTS[START]
BURN_VAR = 120          # expanding-VAR burn-in（月）
HORIZONS = [1, 3, 12]
GOYAL = HERE.parents[1] / '1-数据文件' / 'PredictorData2024.xlsx'

y, dates, D = ev.y, ev.dates, ev.D
DLV = ev.variants['d~_lv']
n = len(y)
PTAG = '_p9' if ev.PANEL9 else ''
# 赛马臂：21 面板 = {raw, d~_lv}（锚点口径）；--panel9 插入 d~_pair 臂（T3：pair 口径赛马）
RACE_VARIANTS = [('raw', D), ('d~_lv', DLV)]
if ev.PANEL9:
    RACE_VARIANTS.insert(1, ('d~_pair', ev.variants['d~_pair']))

# ---------------- 1. Goyal 面板与对齐 ----------------
g = pd.read_excel(GOYAL, sheet_name='Monthly')
g = g[(g['yyyymm'] >= 198401) & (g['yyyymm'] <= 202412)].reset_index(drop=True)
r_log = np.log1p(g['CRSP_SPvw'].astype(float)).values          # log 全收益
x_dp = (np.log(g['D12'].astype(float)) - np.log(g['Index'].astype(float))).values  # log D/P
z_ctl = np.column_stack([
    g['tbl'].astype(float).values,
    (g['BAA'].astype(float) - g['AAA'].astype(float)).values,   # DFY
    (g['lty'].astype(float) - g['tbl'].astype(float)).values,   # TMS
])
assert len(g) == n == 492 and not np.isnan(np.column_stack([r_log, x_dp, z_ctl])).any()
gidx = {int(ym): i for i, ym in enumerate(g['yyyymm'])}

# 行 t → 新闻月（t+1 月）在 Goyal 网格中的位置；行 491（2025-01）缺失 → -1
news_pos = np.full(n, -1, dtype=int)
for t, d in enumerate(dates):
    nd = d + pd.offsets.MonthBegin(1)
    news_pos[t] = gidx.get(nd.year * 100 + nd.month, -1)

# 对齐 gate：ev.y[t] vs Goyal log 超额（月 t+1）
xs_goyal = r_log - np.log1p(pd.read_excel(GOYAL, sheet_name='Monthly')
                            .pipe(lambda df: df[(df['yyyymm'] >= 198401) & (df['yyyymm'] <= 202412)])
                            ['Rfree'].astype(float).values)
m = news_pos >= 0
corr_y = np.corrcoef(y[m], xs_goyal[news_pos[m]])[0, 1]
mad_y = np.mean(np.abs(y[m] - xs_goyal[news_pos[m]]))
print('===== A. 数据与对齐 =====')
print(f'ev.y vs Goyal log 超额（月 t+1）: n={m.sum()} corr={corr_y:.6f} mean|diff|={mad_y:.6f}')
assert corr_y > 0.999, '收益源对齐失败'

# ---------------- 2. Campbell 分解（Deng IA.2 逐式） ----------------
def campbell_full(V, rho_override=None):
    """全样本 VAR(1)，V = (T, p) 原始状态（内部去均值）。返回 dict(rho, A, u, lam, Er_fitted)。"""
    T, p = V.shape
    Vc = V - V.mean(axis=0)
    xbar = V[:, 1].mean()                       # x̄ 用未去均值序列的样本均值
    rho = 1.0 / (1.0 + np.exp(xbar)) if rho_override is None else rho_override
    At = np.linalg.lstsq(Vc[:-1], Vc[1:], rcond=None)[0]   # V1 = V0 @ At，A = At.T
    A = At.T
    U = Vc[1:] - Vc[:-1] @ At                   # u_{t+1}，行 j ↔ 网格月 j+1
    e1 = np.zeros(p); e1[0] = 1.0
    lam = e1 @ (rho * A) @ np.linalg.inv(np.eye(p) - rho * A)
    return {'rho': rho, 'A': A, 'U': U, 'lam': lam}


def campbell_expanding(V, burn=BURN_VAR):
    """expanding-VAR：新闻月 j 的 A/ρ/均值只用 ≤ j−1 数据；burn 前 NaN。"""
    T, p = V.shape
    U = np.full((T, p), np.nan)     # U[j] = 月 j 的创新（j ≥ 1）
    lam_ts = np.full((T, p), np.nan)
    for j in range(1, T):
        if j - 1 < burn:
            continue
        Vw = V[:j]                            # 数据 ≤ j−1
        Vc = Vw - Vw.mean(axis=0)
        rho = 1.0 / (1.0 + np.exp(Vw[:, 1].mean()))
        At = np.linalg.lstsq(Vc[:-1], Vc[1:], rcond=None)[0]
        A = At.T
        e1 = np.zeros(p); e1[0] = 1.0
        lam_ts[j] = e1 @ (rho * A) @ np.linalg.inv(np.eye(p) - rho * A)
        U[j] = V[j] - Vw.mean(axis=0) - A @ Vc[-1]
    return U, lam_ts


def news_from(U, lam_or_ts, p):
    """u → (eps_r, cf, dr)。lam_or_ts：全长 (T,) 或逐行 (T,p)。"""
    T = U.shape[0]
    e1 = np.zeros(p); e1[0] = 1.0
    eps = U @ e1
    if lam_or_ts.ndim == 1:
        dr = U @ lam_or_ts
    else:
        dr = np.nansum(U * lam_or_ts, axis=1)
        dr[np.isnan(U).any(axis=1)] = np.nan
    return eps, eps + dr, dr


def build_news(V):
    """返回 dict(full=(eps,cf,dr,rho,A,U,lam), exp=(eps,cf,dr))，长度 = 网格月数；月 0 无创新为 NaN。"""
    p = V.shape[1]
    f = campbell_full(V)
    U_full = np.row_stack([np.full(p, np.nan), f['U']])
    eps, cf, dr = news_from(U_full, f['lam'], p)
    Ue, lam_ts = campbell_expanding(V)
    eps_e, cf_e, dr_e = news_from(Ue, lam_ts, p)
    return {'rho': f['rho'], 'A': f['A'], 'lam': f['lam'],
            'full': (eps, cf, dr), 'exp': (eps_e, cf_e, dr_e), 'U_full': U_full}


V_base = np.column_stack([r_log, x_dp])
V_z = np.column_stack([r_log, x_dp, z_ctl])
NB, NZ = build_news(V_base), build_news(V_z)

# ρ 频率口径敏感性：Deng 字面公式 ρ̂=1/(1+e^{x̄}) 作用在 Goyal D12（年度股利）上得 0.9794，
# 为年度口径折现系数；月度一致口径 ρ_m = ρ̂^(1/12) = 1/(1+e^{x̄}/12) ≈ 0.998（VAR 去均值后不受
# x 平移影响，仅 ρ 变）。主线沿字面口径，本变体作敏感性。
RHO_M = NB['rho'] ** (1.0 / 12.0)
f_rm = campbell_full(V_base, rho_override=RHO_M)
U_rm = np.row_stack([np.full(2, np.nan), f_rm['U']])
eps_rm, cf_rm, dr_rm = news_from(U_rm, f_rm['lam'], 2)


def var_shares(eps, cf, dr):
    ve, vc, vd = np.nanvar(eps), np.nanvar(cf), np.nanvar(dr)
    cov = np.nanmean((cf - np.nanmean(cf)) * (dr - np.nanmean(dr)))
    return (vc - cov) / ve * 100, (vd - cov) / ve * 100


s_rm = var_shares(eps_rm, cf_rm, dr_rm)
print(f'[ρ 频率口径] ρ̂(D12 字面)={NB["rho"]:.4f} → 月度一致 ρ_m={RHO_M:.4f}；'
      f'ρ_m 下基线份额 CF={s_rm[0]:.1f}% DR={s_rm[1]:.1f}%')

print('\n===== B. VAR 估计与 sanity gates =====')
for tag, Ns, V in (('基线 [r, x]', NB, V_base), ('+z [r, x, tbl, DFY, TMS]', NZ, V_z)):
    eps, cf, dr = Ns['full']
    U = Ns['U_full']
    vv = ~np.isnan(eps)
    # (i) 恒等式 ε = CF − DR
    ident = np.nanmax(np.abs(eps[vv] - (cf[vv] - dr[vv])))
    # (iv) E_t r = e1'Av：ε = r − e1'Av（去均值口径，构造恒等，数值核验）
    Vc = V - V.mean(axis=0)
    Er = Vc[:-1] @ Ns['A'].T @ np.array([1.0] + [0.0] * (V.shape[1] - 1))
    recon = np.nanmax(np.abs(eps[vv] - (Vc[1:, 0] - Er)))
    # (ii) 方差分解
    ve, vc, vd = np.nanvar(eps), np.nanvar(cf), np.nanvar(dr)
    cov = np.nanmean((cf - np.nanmean(cf)) * (dr - np.nanmean(dr)))
    s_cf = (vc - cov) / ve
    s_dr = (vd - cov) / ve
    print(f'[{tag}] ρ̂={Ns["rho"]:.4f} | (i) max|ε−(CF−DR)|={ident:.2e} | (iv) max|ε−(r−E_tr)|={recon:.2e}')
    print(f'  Var: ε={ve:.6f} CF={vc:.6f} DR={vd:.6f} corr(CF,DR)={np.corrcoef(cf[vv],dr[vv])[0,1]:.3f}')
    print(f'  份额（协方差归属）: CF={s_cf*100:.1f}% DR={s_dr*100:.1f}%；Var比: CF/ε={vc/ve:.3f} DR/ε={vd/ve:.3f} −2Cov/ε={-2*cov/ve:.3f}')
    assert ident < 1e-10 and recon < 1e-10
    assert s_cf < 0.90, 'CF 份额 >90%：实现错误信号'
print(f'expanding-VAR：burn-in {BURN_VAR} 月，首个有效新闻月 = 网格第 {BURN_VAR+1} 月')

# 行 t 的新闻序列（目标月 = t+1）；行 491 → NaN
def to_rows(arr):
    out = np.full(n, np.nan)
    mm = news_pos >= 0
    out[mm] = arr[news_pos[mm]]
    return out

SER = {}
for pre, Ns in (('', NB), ('z_', NZ)):
    for kind, key in (('full', ''), ('exp', 'exp_')):
        eps, cf, dr = Ns[kind if kind == 'full' else 'exp']
        SER[f'{pre}{key}eps'] = to_rows(eps)
        SER[f'{pre}{key}cf'] = to_rows(cf)
        SER[f'{pre}{key}dr'] = to_rows(dr)
SER['rm_eps'], SER['rm_cf'], SER['rm_dr'] = to_rows(eps_rm), to_rows(cf_rm), to_rows(dr_rm)

news_df = pd.DataFrame({'date': dates,
                        'eps_r': SER['eps'], 'cf_news': SER['cf'], 'dr_news': SER['dr'],
                        'eps_r_z': SER['z_eps'], 'cf_news_z': SER['z_cf'], 'dr_news_z': SER['z_dr'],
                        'eps_r_exp': SER['exp_eps'], 'cf_news_exp': SER['exp_cf'], 'dr_news_exp': SER['exp_dr'],
                        'eps_r_z_exp': SER['z_exp_eps'], 'cf_news_z_exp': SER['z_exp_cf'],
                        'dr_news_z_exp': SER['z_exp_dr'],
                        'eps_r_rhom': SER['rm_eps'], 'cf_news_rhom': SER['rm_cf'],
                        'dr_news_rhom': SER['rm_dr']})
news_df.round(8).to_csv(RES / f'cfdr_news_series{PTAG}.csv', index=False)
print(f'\n落盘 results/cfdr_news_series{PTAG}.csv（{len(news_df)} 行；行 491 新闻 NaN，expanding 列前 ~121 行 NaN）')

# ---------------- 3. 赛马 machinery（同子集流、horizon 截断、NaN 感知训练） ----------------
def expanding_preds_h(X_in, y_in, n_train, h, i_max):
    """预测行 i ∈ [n_train, i_max]，训练只用行 s ≤ i−h 且目标非 NaN（h=1 无 NaN 时 ≡ 主线）。"""
    X_in = np.atleast_2d(X_in).T if X_in.ndim == 1 else X_in
    Xc = np.column_stack([np.ones(len(y_in)), X_in])
    preds = np.empty(i_max - n_train + 1)
    for j, i in enumerate(range(n_train, i_max + 1)):
        tr = slice(0, i - h + 1)
        ok = ~np.isnan(y_in[tr])
        beta, *_ = np.linalg.lstsq(Xc[tr][ok], y_in[tr][ok], rcond=None)
        preds[j] = Xc[i] @ beta
    return preds


def subset_ensemble_t(M, y_in, n_train, h, i_max, k=ev.K, b=ev.B, seeds=ev.SEEDS):
    """ev.subset_ensemble 的 target 泛化版。主线口径（2026-08-17 起）：C(p,k) ≤ ev.ENUM_CAP
    时全枚举 CSR（确定性，跨目标一致性自动成立）；超上限保留 S=b×3 抽样
    （抽样路径 RNG 流只依赖 seed/列数/b ⇒ 跨目标逐位相同）。
    --panel9（2026-08-19）：幂集 CSR（ev.powerset_csr_preds，确定性同）。"""
    if ev.PANEL9:
        return [ev.powerset_csr_preds(M, y_in, n_train, h=h, i_max=i_max, nan_aware=True)] * 3
    kk = min(k, M.shape[1])
    if ev.comb(M.shape[1], kk) <= ev.ENUM_CAP:
        pr = ev.csr_enum_preds(M, y_in, n_train, kk, h=h, i_max=i_max, nan_aware=True)
        return [pr, pr, pr]
    prs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ss = [sorted(rng.choice(M.shape[1], kk, replace=False).tolist()) for _ in range(b)]
        prs.append(np.column_stack([expanding_preds_h(M[:, s], y_in, n_train, h, i_max) for s in ss]).mean(axis=1))
    return prs


def fwd_sum(arr, h):
    """tgt[t] = Σ_{k=0..h−1} arr[t+k]（任一 NaN ⇒ NaN）。"""
    out = np.full(len(arr), np.nan)
    for t in range(len(arr) - h + 1):
        w = arr[t:t + h]
        if not np.isnan(w).any():
            out[t] = w.sum()
    return out


def ha_h(y_in, n_train, h, i_max):
    """HA 基准：ha[i] = mean(y_in[s], s ≤ i−h，跳过 NaN），i ∈ [n_train, i_max]。"""
    out = np.empty(i_max - n_train + 1)
    for j, i in enumerate(range(n_train, i_max + 1)):
        w = y_in[:i - h + 1]
        out[j] = np.nanmean(w)
    return out


# ---------------- 4. 锚点 gate（h=1、完整 OOS 行、对 preds_{start}{PTAG}.csv，<6e-7） ----------------
print(f'\n===== C. 锚点 gate（h=1 收益腿 ≡ 主线，{n - N_TRAIN} 行） =====')
preds_stored = pd.read_csv(RES / f'preds_{START}{PTAG}.csv')
for tag, M in RACE_VARIANTS:
    pr0 = subset_ensemble_t(M, y, N_TRAIN, 1, n - 1)[0]
    err = np.abs(pr0 - preds_stored[tag].values).max()
    print(f'  {tag}: 对 preds_{START}{PTAG}.csv max|err| = {err:.2e}')
    assert err < 6e-7, f'{tag} 锚点失败: {err}'
print('  锚点全过 ✓')

# ---------------- 5. 赛马 ----------------
TARGETS = {
    'ret': y,
    'cf': SER['cf'], 'dr': SER['dr'],
    'cf_z': SER['z_cf'], 'dr_z': SER['z_dr'],
    'cf_exp': SER['exp_cf'], 'dr_exp': SER['exp_dr'],
    'cf_rhom': SER['rm_cf'], 'dr_rhom': SER['rm_dr'],
}
def i_max_for(tgt, h):
    valid = np.where(~np.isnan(tgt))[0]
    return min(n - h, valid[-1] - h + 1) if h > 1 else min(n - 1, valid[-1])


# 共同末行（严格配对）：全部目标对齐到 CF 基线的有效范围（h=1→490 / h=3→488 / h=12→479）
I_MAX = {h: i_max_for(SER['cf'], h) for h in HORIZONS}
OOS_DATES = {h: dates[N_TRAIN:I_MAX[h] + 1].reset_index(drop=True) for h in HORIZONS}
SEGMENTS = {'full': lambda d: d == d, 'pre2020': lambda d: d < pd.Timestamp('2020-01-01'),
            'post2020': lambda d: d >= pd.Timestamp('2020-01-01')}

rows = []
print('\n===== D. 渠道归因赛马（全枚举 R²OOS，%） =====')
for h in HORIZONS:
    cells = ['ret', 'cf', 'dr'] if h > 1 else list(TARGETS)
    i_max = I_MAX[h]
    segs = SEGMENTS if h == 1 else {'full': SEGMENTS['full']}
    for tg in cells:
        tgt = fwd_sum(TARGETS[tg], h) if h > 1 else TARGETS[tg]
        y_true = tgt[N_TRAIN:i_max + 1]
        base = ha_h(tgt, N_TRAIN, h, i_max)
        ok0 = ~np.isnan(y_true) & ~np.isnan(base)
        # 1991 起点：expanding-VAR 目标（*_exp）1994-01 才起评（前 37 行 NaN；2001 起点下天然在窗外）
        assert ok0.any(), f'{tg} h={h}：OOS 窗内无有效行'
        if not ok0.all():
            print(f'    [注] {tg} h={h}: 剔除 {int((~ok0).sum())} 个 NaN 行（expanding-VAR 1994 起评）')
        for vtag, M in RACE_VARIANTS:
            prs = subset_ensemble_t(M, tgt, N_TRAIN, h, i_max)
            ok1 = ok0 & ~np.isnan(prs[0])  # 剔除退化训练月的 NaN 预测（枚举/幂集路径三元素相同）
            for sgt, sfn in segs.items():
                sm = sfn(OOS_DATES[h]).values & ok1
                r2s = [R2OOS(y_true[sm], pr[sm], base[sm]) * 100 for pr in prs]
                row = {'target': tg, 'h': h, 'segment': sgt, 'variant': vtag,
                       'r2_s0': r2s[0], 'r2_mean3': np.mean(r2s), 'r2_min3': np.min(r2s),
                       'n_oos': int(sm.sum())}
                if ev.PANEL9:
                    # Clark–West 单边 p（对段内 HA 基准、s0 序列、段 mask 子序列；2026-08-27 补）。
                    # h=1 用默认自动 NW lag；h>1 为前向求和重叠目标，沿用项目重叠口径 NW lag=2h
                    # （run_frequency M3 / run_bgls_bridge / run_liv_frequency 同约定）。
                    row['cw_p'] = ev.clark_west_test(y_true[sm], prs[0][sm], base[sm],
                                                     lag=None if h == 1 else 2 * h)[1]
                rows.append(row)
race = pd.DataFrame(rows)
race.round(4).to_csv(RES / f'cfdr_race{STAG}{PTAG}.csv', index=False)


def piv(h, cells, segment='full'):
    sub = race[(race['h'] == h) & (race['segment'] == segment)].set_index(['target', 'variant'])['r2_mean3'].unstack()
    sub = sub.loc[[c for c in cells if c in sub.index]]
    sub['Δ(d̃−raw)'] = sub['d~_lv'] - sub['raw']
    return sub


main = piv(1, ['ret', 'cf', 'dr'])
print(f'\n--- 主表（h=1，共同有效行 {N_TRAIN}..{I_MAX[1]}，{I_MAX[1] - N_TRAIN + 1} 点；R²OOS 全枚举，%） ---')
print(main.round(3).to_string())
print('\n--- pre/post-2020 分段（h=1；raw 优势仅存在于 pre-2020 收益腿，§36 E2） ---')
for sgt in ('pre2020', 'post2020'):
    print(f'  [{sgt}]')
    print(piv(1, ['ret', 'cf', 'dr'], sgt).round(3).to_string())
print('\n--- horizon 稳健：Δ = d̃−raw（全枚举，%） ---')
for h in (3, 12):
    d = piv(h, ['ret', 'cf', 'dr'])['Δ(d̃−raw)']
    n_ = race[(race['h'] == h) & (race['target'] == 'ret')]['n_oos'].iloc[0]
    print(f'  h={h:2d}（{n_} 点）: ret {d["ret"]:+.3f} | CF {d["cf"]:+.3f} | DR {d["dr"]:+.3f}')
print('\n--- 敏感性（h=1）：z 控制 / expanding-VAR / ρ 月度一致口径 目标 ---')
for cells, tag in ((['cf_z', 'dr_z'], 'z=[tbl,DFY,TMS]'), (['cf_exp', 'dr_exp'], 'expanding-VAR'),
                   (['cf_rhom', 'dr_rhom'], f'ρ_m={RHO_M:.4f}')):
    s = piv(1, cells)
    print(f'  [{tag}]')
    print(s.round(3).to_string())

print('\n判读提示：交叉形态 = Δ(CF) > 0 且 Δ(DR) < 0（d̃ 优势在 CF、raw 优势在 DR）。')
print(f'完成。落盘 results/cfdr_news_series{PTAG}.csv、results/cfdr_race{STAG}{PTAG}.csv')
