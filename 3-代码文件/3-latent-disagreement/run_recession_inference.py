# -*- coding: utf-8 -*-
"""
衰退期推断稳健性：极端段剔除敏感性 + circular block bootstrap + 逐月贡献
============================================================================
口径：--panel9（9 面板 + 幂集 CSR）+ --start 2001（2001 主线，n_train=204，OOS n=288）。
对比对固定为 d~_pair vs raw（主线主变体），损失差 d_t = (y−raw)^2 − (y−d~_pair)^2
（正值 = d~_pair 更好），DM 构造与 run_direct_compare.py 完全相同
（mean(d) 的 Newey-West t 检验，full lag=6 / REC、EXP 子样本 lag=3，双侧正态 p）。

动机（审稿人质疑）：
1. REC 单元格 n=28，DM=3.12 p=.0018 依赖渐近近似；
2. 衰退增益（R² raw 9.46 → d~_pair 12.23）可能由 2008–09（GFC）与 2020（COVID）
   少数极端月驱动。

三部分：
1. 极端段剔除敏感性：REC 单元格在四种口径下重算——(a) 全 28 衰退月（基线锚点）；
   (b) 剔除 COVID 月（2020-02~2020-04，NBER 口径）；(c) 剔除 GFC 月（2007-12~2009-06）；
   (d) 同时剔除两者。mask 子序列上的 DM 为 off-label 用法（子样本不连续、n 最小至 8），
   输出中标注 off_label=True。
2. circular block bootstrap（零假设下重抽：先将损失差分序列按 regime 去均值——REC 月减
   REC 均值、EXP 月减 EXP 均值——再做环形块自助，块长 6 与 12 月，B=5000，种子固定）：
   (i) pooled 规格：对全 OOS 中心化序列重抽，full/REC/EXP 按 usrec 掩码取样本，
   重算 DM 类统计量 T=mean/HAC_se，报告双侧经验 p、mean_d 的 95% basic bootstrap
   区间、T 零分布的 95% 分位区间；(ii) within 规格：仅在 28 个衰退月内向量重抽
   （块跨衰退段边界，off-label），作 REC 小样本对照。掩码子样本 HAC 均为 off-label 近似。
3. 逐月贡献：28 个衰退月的逐月 d_t 列表，标注所属衰退段（2001/GFC/COVID），
   给出各段对 REC 总增益的贡献份额。

基线锚点断言（对照 results/direct_compare_2001_p9.csv，d~_pair vs raw）：
   full DM=2.040 p=.0413；REC DM=3.115 p=.0018；EXP DM=0.916 p=.3597；
   REC R² raw=9.46 / d~_pair=12.23。

输出：results/recession_inference_p9.csv（section 列区分 sensitivity/bootstrap/monthly）
     + 控制台全量打印 + 结论摘要。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # noqa: E402

assert ev.PANEL9 is True, '本脚本仅支持 --panel9（9 面板 + 幂集 CSR）口径，请加 --panel9'

_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=2001, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
RES = HERE / 'results'

B_BOOT = 5000
BLOCKS = (6, 12)
SEED = 20260930

y = ev.y
dates = ev.dates
ha = ev.ha
n_oos = len(y) - N_TRAIN
y_oos = y[N_TRAIN:]
ha_oos = ha[N_TRAIN:]
dates_oos = dates[N_TRAIN:]

preds = pd.read_csv(RES / f'preds_{START}{ev.PTAG}.csv', parse_dates=['date'])
assert (preds['date'].values == dates_oos.values).all()
p_raw = preds['raw'].values.astype(float)
p_dtp = preds['d~_pair'].values.astype(float)

usrec = pd.read_csv(ev.SCNN / 'USREC.csv').iloc[-n_oos:, 0].values
rec = usrec == 1
exp = ~rec
print(f'OOS {n_oos} 月（{dates_oos.iloc[0]:%Y-%m}~{dates_oos.iloc[-1]:%Y-%m}）: '
      f'REC {int(rec.sum())} 月, EXP {int(exp.sum())} 月')

# 损失差（正值 = d~_pair 更好）
d_all = (y_oos - p_raw) ** 2 - (y_oos - p_dtp) ** 2
e2_raw = (y_oos - p_raw) ** 2
e2_dtp = (y_oos - p_dtp) ** 2


def dm_test(d, lag):
    """与 run_direct_compare.dm_test 完全相同的构造：mean(d) 的 NW t 检验，双侧正态 p。"""
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    stat = res.params[0] / res.bse[0]
    p = 2 * (1 - norm.cdf(abs(stat)))
    return d.mean(), stat, p


def r2(m, pred):
    return ev.R2OOS(y_oos[m], pred[m], ha_oos[m]) * 100


def cell(mask, lag):
    """子样本单元格：n / R² raw / R² d~_pair / ΔR²(pt) / mean_d / DM / p。"""
    mean_d, stat, p = dm_test(d_all[mask], lag)
    return {'n': int(mask.sum()), 'R2_raw': r2(mask, p_raw), 'R2_d~_pair': r2(mask, p_dtp),
            'delta_R2_pt': r2(mask, p_dtp) - r2(mask, p_raw),
            'mean_d': mean_d, 'DM_stat': stat, 'p_asym': p}


# ---------------- 基线锚点断言（对照 direct_compare_2001_p9.csv） ----------------
dc = pd.read_csv(RES / f'direct_compare_{START}{ev.PTAG}.csv')
dc = dc[(dc['pair'] == 'd~_pair vs raw') & (dc['series'] == 's0')].set_index('sample')
base_full = cell(np.ones(n_oos, bool), 6)
base_rec = cell(rec, 3)
base_exp = cell(exp, 3)
for nm, got in (('full', base_full), ('REC', base_rec), ('EXP', base_exp)):
    a = dc.loc[nm]
    assert got['n'] == int(a['n']) and abs(got['DM_stat'] - a['DM_stat']) < 1e-5 \
        and abs(got['p_asym'] - a['p_value']) < 1e-5, f'{nm} 锚点与 direct_compare 不一致'
assert abs(base_rec['R2_raw'] - 9.46) < 0.01 and abs(base_rec['R2_d~_pair'] - 12.23) < 0.01
print('锚点断言通过：full DM=2.040 p=.0413 / REC DM=3.115 p=.0018 / EXP DM=0.916 p=.3597；'
      f"REC R² raw={base_rec['R2_raw']:.2f} → d~_pair={base_rec['R2_d~_pair']:.2f}")

# ---------------- 1. 极端段剔除敏感性 ----------------
# NBER 日历口径：GFC 2007-12~2009-06，COVID 2020-02~2020-04。
# 注：管线 usrec 标记相对 NBER 官方平移约 1 个月（GFC 标记 2008-01~2009-06，
# COVID 标记 2020-03~2020-04），剔除按 NBER 日历月执行（未标记月本就不在 REC mask 内）。
ym = dates_oos.dt.to_period('M').values  # numpy Period 数组（位置索引）
_P = pd.Period
gfc = np.array([_P('2007-12') <= p <= _P('2009-06') for p in ym])
covid = np.array([_P('2020-02') <= p <= _P('2020-04') for p in ym])

sens_specs = [('REC_all(n=28基线)', rec),
              ('REC_excl_COVID', rec & ~covid),
              ('REC_excl_GFC', rec & ~gfc),
              ('REC_excl_GFC+COVID', rec & ~gfc & ~covid)]
sens_rows = []
print('\n===== 1. 极端段剔除敏感性（REC 单元格，DM 为 mask 子序列 off-label 用法）=====')
for nm, m in sens_specs:
    c = cell(m, 3)
    sens_rows.append({'section': 'sensitivity', 'spec': nm,
                      'off_label': nm != 'REC_all(n=28基线)',  # 剔除口径 = mask 子序列 off-label
                      **c})
    print(f'[{nm:22s}] n={c["n"]:3d}  R² raw={c["R2_raw"]:6.2f} → d~_pair={c["R2_d~_pair"]:6.2f} '
          f'(Δ {c["delta_R2_pt"]:+.2f}pt)  mean_d={c["mean_d"]:+.6f}  '
          f'DM={c["DM_stat"]:+.3f}  p={c["p_asym"]:.4f}')

# ---------------- 2. circular block bootstrap ----------------
# 零假设下重抽：先将 d 按 regime 去均值（REC 月减 REC 均值、EXP 月减 EXP 均值），
# 使重抽世界满足 H0（各 regime 均值差 = 0），再做环形块自助；T* = mean*/HAC_se*
# 不再减 T_obs（数据已中心化），双侧经验 p = P(|T*| ≥ |T_obs|)。
# mean_d 的 95% 区间 = basic bootstrap CI（[mean_obs−q97.5(mean*), mean_obs−q2.5(mean*)]）。
# 两规格：(i) pooled：对全 OOS 中心化序列重抽，REC/EXP 按 usrec 掩码取条件子样本
# （off-label：掩码子序列不连续，HAC 在其上为近似）；(ii) REC_within：仅在 28 个
# 衰退月（去均值后）内向量重抽——块跨过衰退段边界，同为 off-label，作小样本对照。
rng = np.random.default_rng(SEED)
samples = [('full', np.ones(n_oos, bool), 6), ('REC', rec, 3), ('EXP', exp, 3)]
obs = {nm: dm_test(d_all[m], lag) for nm, m, lag in samples}
d_c = d_all.copy()
d_c[rec] -= d_all[rec].mean()
d_c[exp] -= d_all[exp].mean()


def cbb_indices(n, L, rng):
    """环形块自助索引：起点均匀于 [0,n)，块沿序列环回，截断至 n。"""
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=k)
    idx = (starts[:, None] + np.arange(L)[None, :]) % n
    return idx.ravel()[:n]


def boot_report(nm, m, lag, T_star, M_star, scheme):
    p_boot = float(np.mean(np.abs(T_star) >= abs(obs[nm][1])))
    q = np.percentile(M_star, [2.5, 97.5])
    tq = np.percentile(T_star, [2.5, 97.5])
    ci = (obs[nm][0] - q[1], obs[nm][0] - q[0])  # basic bootstrap CI
    row = {'section': 'bootstrap', 'spec': f'{nm}_{scheme}', 'sample': nm, 'scheme': scheme,
           'block_len': L, 'B': B_BOOT, 'n': int(m.sum()), 'NW_lag': lag,
           'mean_d': obs[nm][0], 'DM_stat': obs[nm][1], 'p_asym': obs[nm][2],
           'p_boot': p_boot, 'mean_d_ci_lo': ci[0], 'mean_d_ci_hi': ci[1],
           'T_null_q025': tq[0], 'T_null_q975': tq[1], 'off_label': nm != 'full'}
    print(f'[{scheme:10s} L={L:2d} {nm:4s}] n={int(m.sum()):3d}  mean_d={obs[nm][0]:+.6f} '
          f'CI95 [{ci[0]:+.6f}, {ci[1]:+.6f}]  DM={obs[nm][1]:+.3f}  '
          f'p_asym={obs[nm][2]:.4f}  p_boot={p_boot:.4f}  '
          f'T* 零分布95%分位 [{tq[0]:+.2f}, {tq[1]:+.2f}]')
    return row


boot_rows = []
print(f'\n===== 2. circular block bootstrap（B={B_BOOT}，块长 {BLOCKS}，种子 {SEED}）=====')
print('（零假设下重抽：d 按 regime 去均值后环形块自助；CI = basic bootstrap；'
      'REC/EXP 掩码子样本与 REC_within 均为 off-label）')
d_rec_c = d_all[rec] - d_all[rec].mean()
n_rec = int(rec.sum())
for L in BLOCKS:
    T_s = {nm: np.empty(B_BOOT) for nm, _, _ in samples}
    M_s = {nm: np.empty(B_BOOT) for nm, _, _ in samples}
    T_w = np.empty(B_BOOT)
    M_w = np.empty(B_BOOT)
    for b in range(B_BOOT):
        d_b = d_c[cbb_indices(n_oos, L, rng)]
        for nm, m, lag in samples:
            mean_b, stat_b, _ = dm_test(d_b[m], lag)
            T_s[nm][b] = stat_b
            M_s[nm][b] = mean_b
        d_w = d_rec_c[cbb_indices(n_rec, L, rng)]
        mean_w, stat_w, _ = dm_test(d_w, 3)
        T_w[b] = stat_w
        M_w[b] = mean_w
    for nm, m, lag in samples:
        boot_rows.append(boot_report(nm, m, lag, T_s[nm], M_s[nm], 'pooled'))
    boot_rows.append(boot_report('REC', rec, 3, T_w, M_w, 'within'))

# ---------------- 3. 逐月贡献 ----------------
def episode(p):
    if p >= pd.Period('2020-02') and p <= pd.Period('2020-04'):
        return 'COVID'
    if p >= pd.Period('2007-12') and p <= pd.Period('2009-06'):
        return 'GFC'
    return '2001'


rec_idx = np.where(rec)[0]
tot = d_all[rec].sum()
mon_rows = []
print('\n===== 3. 衰退月逐月损失差分贡献（正值 = d~_pair 更好）=====')
print(f'{"date":8s} {"episode":6s} {"e2_raw":>9s} {"e2_d~_pair":>10s} {"d_t":>10s} {"share%":>7s}')
for i in rec_idx:
    p = ym[i]
    ep = episode(p)
    share = d_all[i] / tot * 100
    mon_rows.append({'section': 'monthly', 'spec': str(p), 'episode': ep,
                     'e2_raw': e2_raw[i], 'e2_d~_pair': e2_dtp[i], 'd_t': d_all[i],
                     'share_of_REC_gain_pct': share})
    print(f'{str(p):8s} {ep:6s} {e2_raw[i]:9.5f} {e2_dtp[i]:10.5f} {d_all[i]:+10.6f} {share:7.2f}')
mon = pd.DataFrame(mon_rows)
ep_share = mon.groupby('episode')['d_t'].sum() / tot * 100
print('\n分衰退段贡献份额（占 REC 总增益 %）：')
for ep, s in ep_share.items():
    n_ep = int((mon['episode'] == ep).sum())
    print(f'  {ep:6s} n={n_ep:2d}  share={s:+7.2f}%')
top = mon.reindex(mon['d_t'].abs().sort_values(ascending=False).index).head(5)
print('贡献最大的 5 个月份：')
for _, r_ in top.iterrows():
    print(f'  {r_["spec"]} ({r_["episode"]})  d_t={r_["d_t"]:+.6f}  share={r_["share_of_REC_gain_pct"]:+.2f}%')

# ---------------- 落盘 + 结论摘要 ----------------
out = pd.concat([pd.DataFrame(sens_rows), pd.DataFrame(boot_rows), mon],
                ignore_index=True, sort=False)
out.round(8).to_csv(RES / f'recession_inference{ev.PTAG}.csv', index=False)
print(f'\n落盘: results/recession_inference{ev.PTAG}.csv')

print('\n===== 结论摘要 =====')
for r_ in sens_rows:
    print(f"  {r_['spec']:22s} ΔR²={r_['delta_R2_pt']:+.2f}pt  DM={r_['DM_stat']:+.3f}  p={r_['p_asym']:.4f}")
b6 = {r_['spec']: r_ for r_ in boot_rows if r_['block_len'] == 6}
b12 = {r_['spec']: r_ for r_ in boot_rows if r_['block_len'] == 12}
for nm in ('full', 'REC', 'EXP'):
    print(f"  {nm:4s} p_asym={obs[nm][2]:.4f} → p_boot(pooled) L6={b6[f'{nm}_pooled']['p_boot']:.4f} / "
          f"L12={b12[f'{nm}_pooled']['p_boot']:.4f}")
print(f"  REC  p_boot(within) L6={b6['REC_within']['p_boot']:.4f} / "
      f"L12={b12['REC_within']['p_boot']:.4f}")
