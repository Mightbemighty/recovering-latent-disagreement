# -*- coding: utf-8 -*-
"""
post-2020 衰减事件对齐拆解（9 面板 + 幂集 CSR，2001 口径）
=================================================================
背景：主线 d̃_pair 幂集 CSR 全样本 R²OOS=3.956 vs raw 2.706（2001-01~2024-12，n_oos=288，
锚点 results/variants_2001_p9.csv），但增益集中 pre-2020、post-2020 段大幅衰减。
候选机制猜想：2021–22 通胀飙升期 CPI 题 level–disagreement 联动骤增，同题投影此时
过度吸收了衰退期增益赖以存在的内容。本脚本做事件对齐拆解检验该猜想，并排查替代解释
（TBL 利率题在 ZLB 与 2022 加息段的行为）。

分析块：
  A1 逐测度分解：9 测度单变量 expanding OLS（d_raw 与 d̃_pair 两版），pre/post-2020 分段 R²。
  A2 滚动联动：每题 corr(d_k, l_k) 36 月滚动序列（全样本 IS 描述），关键时点数值。
  A3 LOO 归属（post-2020）：幂集 CSR 下逐题剔除，post-2020 段 R² 变化。
  A4 混合臂检验：CPI 臂保持 raw、其余 8 臂 d̃_pair 的 hybrid 输入走幂集 CSR（猜想核心检验）；
     TBL 臂 hybrid 与 CPI+TBL 双臂 raw 为对照。
  A5 增益路径：d̃_pair vs raw 累计 SSE 差分月度序列（2020–2024 逐月打印）。

口径：2001 主线（n_train=204，OOS 2001-01~2024-12），幂集 CSR（确定性），自算 HA，
严格递归（全部经由 run_evidence 的 decontaminate / expanding_preds / subset_ensemble）。
分段 R² 定义：1 − Σ_seg(y−ŷ)² / Σ_seg(y−HA)²（HA 为本管线递归基准的同段取值）。
探索性分析：不修改任何既有文件，仅新增落盘 results/post2020_attenuation_p9.csv。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = ['x', '--panel9']  # 9 面板 + 幂集 CSR 口径（run_evidence 模块级预解析）
import run_evidence as ev  # noqa: E402

assert ev.PANEL9 is True, '须以 --panel9 口径运行（9 测度 SPF-only 面板）'

N_TRAIN = 204  # 2001 主线口径
RES = HERE / 'results'

y, dates, ha = ev.y, ev.dates, ev.ha
n = len(y)
n_oos = n - N_TRAIN
y_oos, ha_oos = y[N_TRAIN:], ha[N_TRAIN:]
dates_oos = dates[N_TRAIN:].reset_index(drop=True)
usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)
POST = (dates_oos >= pd.Timestamp('2020-01-01')).values
PRE = ~POST
SEGS = {'full': np.ones(n_oos, bool), 'pre2020': PRE, 'post2020': POST}

names = [d[2:] for _, d in ev.pairs]  # 面板列序（字母序），断言覆盖 9 题
assert sorted(names) == ['CON', 'CPI', 'CPROF', 'GDP', 'HSG', 'INV', 'IP', 'TBL', 'UEP']
n_p = len(names)
D, L = ev.D, ev.L
D_pair = ev.variants['d~_pair']
print(f'面板列序: {names}')


def r2_seg(pred, mask):
    """分段 R²OOS（%，对同段 HA 基准）。"""
    return ev.R2OOS(y_oos[mask], pred[mask], ha_oos[mask]) * 100


rows = []  # 长表落盘：block / measure / segment / metric / date / value


def put(block, measure, segment, metric, value, date=''):
    rows.append({'block': block, 'measure': measure, 'segment': segment,
                 'metric': metric, 'date': date, 'value': value})


# ---------------- 0. 主线锚点复核（raw / d̃_pair 幂集 CSR） ----------------
pr_raw = ev.subset_ensemble(D, N_TRAIN)[0]
pr_pair = ev.subset_ensemble(D_pair, N_TRAIN)[0]
for tag, pr, anch in (('raw', pr_raw, 2.706), ('d~_pair', pr_pair, 3.956)):
    r2 = ev.R2OOS(y_oos, pr, ha_oos) * 100
    assert abs(r2 - anch) < 0.06, (tag, r2)
    print(f'  [check] {tag:8s} R2OOS（幂集 CSR）= {r2:.3f}（锚点 {anch}）✓')
r_rec, r_exp = ev.R2OOS_by_USREC(y_oos, pr_pair, ha_oos, usrec)
print(f'  [check] d̃_pair REC={r_rec * 100:.2f} EXP={r_exp * 100:.2f}（锚点 12.23 / 1.14）')

print('\n===== 分段总览（幂集 CSR 集成）=====')
for tag, pr in (('raw', pr_raw), ('d~_pair', pr_pair)):
    line = f'  {tag:8s}'
    for seg, m in SEGS.items():
        r2 = r2_seg(pr, m)
        line += f'  {seg}: {r2:+6.3f}'
        put('A0_overview', tag, seg, 'R2OOS_pct', r2)
    print(line)

# ---------------- A1 逐测度分解（单变量 expanding OLS，pre/post-2020） ----------------
print('\n===== A1 逐测度单变量 R²OOS（expanding OLS，分段）=====')
print(f'  {"measure":6s} | {"raw_pre":>8s} {"pair_pre":>8s} {"gain_pre":>9s} | '
      f'{"raw_post":>8s} {"pair_post":>9s} {"gain_post":>9s} | {"gain翻转":>4s}')
for k, nm in enumerate(names):
    pr_d = ev.expanding_preds(D[:, k], y, N_TRAIN)
    pr_dt = ev.expanding_preds(D_pair[:, k], y, N_TRAIN)
    r_d_pre, r_t_pre = r2_seg(pr_d, PRE), r2_seg(pr_dt, PRE)
    r_d_post, r_t_post = r2_seg(pr_d, POST), r2_seg(pr_dt, POST)
    g_pre, g_post = r_t_pre - r_d_pre, r_t_post - r_d_post
    flip = '是' if (g_pre > 0) != (g_post > 0) else ''
    print(f'  {nm:6s} | {r_d_pre:+8.3f} {r_t_pre:+8.3f} {g_pre:+9.3f} | '
          f'{r_d_post:+8.3f} {r_t_post:+9.3f} {g_post:+9.3f} | {flip:>4s}')
    for seg, v in (('pre2020', r_d_pre), ('post2020', r_d_post)):
        put('A1_univariate', nm, seg, 'R2_raw', v)
    for seg, v in (('pre2020', r_t_pre), ('post2020', r_t_post)):
        put('A1_univariate', nm, seg, 'R2_d~_pair', v)
    for seg_name, m in (('pre2020', PRE), ('post2020', POST)):
        put('A1_univariate', nm, seg_name, 'CW_p_raw',
            ev.clark_west_test(y_oos[m], pr_d[m], ha_oos[m])[1])
        put('A1_univariate', nm, seg_name, 'CW_p_d~_pair',
            ev.clark_west_test(y_oos[m], pr_dt[m], ha_oos[m])[1])
    put('A1_univariate', nm, 'pre2020', 'gain', g_pre)
    put('A1_univariate', nm, 'post2020', 'gain', g_post)

# ---------------- A2 滚动联动 corr(d_k, l_k)，36 月窗（全样本 IS 描述） ----------------
print('\n===== A2 滚动 36 月 corr(d_k, l_k)（全样本 IS 描述性质）=====')
WIN = 36
KEY_DATES = ['2008-12-01', '2009-12-01', '2012-12-01', '2015-12-01',  # ZLB 段
             '2020-03-01', '2020-12-01', '2021-06-01', '2021-12-01',  # 疫情/通胀启动
             '2022-06-01', '2022-12-01', '2023-12-01', '2024-12-01']  # 加息段
roll = {}
for k, nm in enumerate(names):
    s = pd.Series(D[:, k]).rolling(WIN).corr(pd.Series(L[:, k])).values
    roll[nm] = s
    for dt_, v in zip(dates, s):
        if not np.isnan(v):
            put('A2_rolling_corr36', nm, 'full_sample', 'corr_d_l', v, str(dt_.date()))
hdr = '  date       ' + ''.join(f'{nm:>8s}' for nm in names)
print(hdr)
for kd in KEY_DATES:
    i = int(np.where(dates == pd.Timestamp(kd))[0][0])
    vals = ''.join((f'{roll[nm][i]:+8.3f}' if not np.isnan(roll[nm][i]) else f'{"nan":>8s}')
                   for nm in names)
    print(f'  {kd[:7]}    {vals}')
# CPI/TBL 题 2021–22 峰值定位
for nm in ('CPI', 'TBL'):
    s = roll[nm]
    m21 = (dates >= '2021-01-01') & (dates <= '2022-12-31')
    idx = np.where(m21)[0]
    pk = idx[np.nanargmax(np.abs(s[idx]))]
    print(f'  [{nm}] 2021–22 |corr| 峰值 = {s[pk]:+.3f}（{dates[pk].date()}）；'
          f'2019-12 = {s[int(np.where(dates == pd.Timestamp("2019-12-01"))[0][0])]:+.3f}')

# ---------------- A3 LOO 归属（post-2020，幂集 CSR 逐题剔除 d̃_pair） ----------------
print('\n===== A3 LOO 归属（d̃_pair 幂集 CSR，剔除第 k 题；分段 R²OOS）=====')
r2_full_post, r2_full_pre = r2_seg(pr_pair, POST), r2_seg(pr_pair, PRE)
print(f'  full d̃_pair: pre2020 {r2_full_pre:+.3f}  post2020 {r2_full_post:+.3f}')
print(f'  {"drop":6s} | {"post2020":>9s} {"Δpost":>8s} | {"pre2020":>8s} {"Δpre":>8s}')
for k, nm in enumerate(names):
    M = np.delete(D_pair, k, axis=1)
    pr = ev.subset_ensemble(M, N_TRAIN)[0]
    r_post, r_pre = r2_seg(pr, POST), r2_seg(pr, PRE)
    print(f'  -{nm:5s} | {r_post:+9.3f} {r_post - r2_full_post:+8.3f} | '
          f'{r_pre:+8.3f} {r_pre - r2_full_pre:+8.3f}')
    put('A3_LOO', nm, 'post2020', 'R2OOS_pct', r_post)
    put('A3_LOO', nm, 'post2020', 'delta_vs_full', r_post - r2_full_post)
    put('A3_LOO', nm, 'pre2020', 'R2OOS_pct', r_pre)
    put('A3_LOO', nm, 'pre2020', 'delta_vs_full', r_pre - r2_full_pre)

# ---------------- A4 混合臂检验（CPI 臂保持 raw，其余 d̃_pair；TBL 对照） ----------------
print('\n===== A4 混合臂检验（幂集 CSR；分段 R²OOS + CW p + REC/EXP）=====')
i_cpi, i_tbl = names.index('CPI'), names.index('TBL')
hybrids = {}
for tag, cols in (('hyb_CPI_raw', [i_cpi]), ('hyb_TBL_raw', [i_tbl]),
                  ('hyb_CPI+TBL_raw', [i_cpi, i_tbl])):
    M = D_pair.copy()
    M[:, cols] = D[:, cols]
    hybrids[tag] = M
print(f'  {"variant":16s} | {"full":>7s} {"pre2020":>8s} {"post2020":>9s} | '
      f'{"CW_p":>6s} {"REC":>7s} {"EXP":>6s}')
for tag, pr in (('raw', pr_raw), ('d~_pair', pr_pair)):
    _, cw_p, _ = ev.clark_west_test(y_oos, pr, ha_oos)
    rr, re_ = ev.R2OOS_by_USREC(y_oos, pr, ha_oos, usrec)
    print(f'  {tag:16s} | {r2_seg(pr, SEGS["full"]):+7.3f} {r2_seg(pr, PRE):+8.3f} '
          f'{r2_seg(pr, POST):+9.3f} | {cw_p:6.4f} {rr * 100:+7.2f} {re_ * 100:+6.2f}')
    for seg, m in SEGS.items():
        put('A4_hybrid', tag, seg, 'R2OOS_pct', r2_seg(pr, m))
    put('A4_hybrid', tag, 'full', 'CW_p', cw_p)
    put('A4_hybrid', tag, 'full', 'REC', rr * 100)
    put('A4_hybrid', tag, 'full', 'EXP', re_ * 100)
for tag, M in hybrids.items():
    pr = ev.subset_ensemble(M, N_TRAIN)[0]
    _, cw_p, _ = ev.clark_west_test(y_oos, pr, ha_oos)
    rr, re_ = ev.R2OOS_by_USREC(y_oos, pr, ha_oos, usrec)
    print(f'  {tag:16s} | {r2_seg(pr, SEGS["full"]):+7.3f} {r2_seg(pr, PRE):+8.3f} '
          f'{r2_seg(pr, POST):+9.3f} | {cw_p:6.4f} {rr * 100:+7.2f} {re_ * 100:+6.2f}')
    for seg, m in SEGS.items():
        put('A4_hybrid', tag, seg, 'R2OOS_pct', r2_seg(pr, m))
    put('A4_hybrid', tag, 'full', 'CW_p', cw_p)
    put('A4_hybrid', tag, 'full', 'REC', rr * 100)
    put('A4_hybrid', tag, 'full', 'EXP', re_ * 100)

# ---------------- A5 增益路径：d̃_pair vs raw 累计 SSE 差分 ----------------
print('\n===== A5 累计 SSE 差分（cumsum(e_raw² − e_d~²)，>0 表示 d̃_pair 领先）=====')
e_raw2 = (y_oos - pr_raw) ** 2
e_pair2 = (y_oos - pr_pair) ** 2
d_sse = e_raw2 - e_pair2
cum = np.cumsum(d_sse)
print(f'  全样本终点: {cum[-1]:+.5f}；2019-12 时点: {cum[PRE][-1]:+.5f}；'
      f'post-2020 净变化: {cum[-1] - cum[PRE][-1]:+.5f}')
print('  2020-01~2024-12 逐月（date, 当月差分, 累计差分）:')
for i in np.where(POST)[0]:
    print(f'    {dates_oos[i].date()}  {d_sse[i]:+.5f}  {cum[i]:+.5f}')
    put('A5_cumsse_path', 'd~_pair-raw', 'post2020', 'monthly_sse_diff', d_sse[i],
        str(dates_oos[i].date()))
    put('A5_cumsse_path', 'd~_pair-raw', 'post2020', 'cum_sse_diff', cum[i],
        str(dates_oos[i].date()))
for i in np.where(PRE)[0]:  # pre 段只落盘不打印
    put('A5_cumsse_path', 'd~_pair-raw', 'pre2020', 'monthly_sse_diff', d_sse[i],
        str(dates_oos[i].date()))
    put('A5_cumsse_path', 'd~_pair-raw', 'pre2020', 'cum_sse_diff', cum[i],
        str(dates_oos[i].date()))
# 关键事件窗净贡献
for a, b, lab in (('2020-02-01', '2020-05-01', '2020-02~05 疫情熔断'),
                  ('2020-06-01', '2020-12-01', '2020-06~12 疫后修复'),
                  ('2021-01-01', '2021-12-01', '2021 通胀启动'),
                  ('2022-01-01', '2022-12-01', '2022 加息段'),
                  ('2023-01-01', '2024-12-01', '2023–24')):
    m = (dates_oos >= a) & (dates_oos <= b)
    print(f'  {lab:22s} 净 SSE 差分 = {d_sse[m.values].sum():+.5f}')

# ---------------- 落盘 ----------------
out = pd.DataFrame(rows)
out.to_csv(RES / 'post2020_attenuation_p9.csv', index=False)
print(f'\n落盘 results/post2020_attenuation_p9.csv（{len(out)} 行，长表：block/measure/segment/metric/date/value）')
