# -*- coding: utf-8 -*-
"""
BW6 外部控制赛马的 real-time vintage 版（H2 稳健性：修订错位质疑）
====================================================================
动机：现值臂 d~_BW6 用 final revised 宏观数据构造（run_bw6.py），审稿人质疑这是
对外部控制的"结构性让分"。本脚本用 ALFRED vintage 数据重造 BW6 外部控制：
- INDPRO 替代 PRINTO01USQ657S（前者 ALFRED vintage 自 1927 起，后者仅 2013 起）；
- PCEDG/PCEND/PCES/PAYEMS 用各月末信息集下的 vintage 列算 pct_change
  （增长率对基期不变，INDPRO 基期随 vintage 变化不影响）；
- USREC 沿用现值（NBER 日期基本不修订）；
- vintage 列在 t 月尚无当月观测时取该 vintage 内最新可得值（≤t），
  天然模拟发布滞后（替代原 lag1 臂）。

口径：--start 2001（ev.STARTS[2001]=204，OOS 2001-01~2024-12，n=288）+ --panel9
（9 测度 SPF-only 面板 + 幂集 CSR）。断言 ev.PANEL9 is True。

递归去污染：ev.decontaminate 假设 Z 固定，vintage 版每个 t 的 Z 历史都不同，
故自写循环——对每个 t ≥ burn(=36)，用 t 月末信息集 vintage 重建全部历史控制值，
OLS 回归 d_k(≤t-1) 对 Z(≤t-1)，算 t 期残差。

锚点：(a) 最终 vintage（2026 年最新）重建的 BW6_vintage 各列增长率与现值 BW6
在面板重叠月的相关系数（INDPRO vs OECD 季度序列有口径差，报告即可）；
(b) 现值臂数字直接读自 results/bw6_2001_p9.csv（复用现值管线落盘，不重造），
表中 d~_BW6 行与该文件逐值一致。

输出：results/bw6_vintage_2001_p9.csv（raw / d~_pair / d~_BW6 / d~_BW6_vintage
四臂 full/REC/EXP R²_OOS + CW p）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # 复用数据加载 / subset_ensemble / 指标（--panel9 模块级预解析）

_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=2001, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
assert START == 2001, '本脚本仅绑定 2001 口径'
assert ev.PANEL9 is True, '必须加 --panel9（9 面板 + 幂集 CSR）'
DATA_DIR = HERE.parent.parent / '1-数据文件'
VIN_DIR = DATA_DIR / 'ALFRED_vintage'

y = ev.y
dates = ev.dates
ha = ev.ha
n_oos = len(y) - N_TRAIN
y_true = y[N_TRAIN:]
base = ha[N_TRAIN:]
assert len(dates) == 492 and dates.iloc[0] == pd.Timestamp('1984-01-01') and dates.iloc[-1] == pd.Timestamp('2024-12-01')
panel_periods = pd.PeriodIndex(dates, freq='M')
print(f'面板: {len(dates)} 月（{panel_periods[0]} ~ {panel_periods[-1]}），OOS n={n_oos}（幂集 CSR, 9 面板）')

# ---------------- 1. 现值 BW6（复刻 run_bw6 构造，仅供锚点 a 对照，不进赛马管线） ----------------
IP = pd.read_csv(DATA_DIR / 'PRINTO01USQ657S.csv')
IP.rename(columns={'observation_date': 'date', 'PRINTO01USQ657S': 'IP_growth'}, inplace=True)
IP['date'] = pd.to_datetime(IP['date'])
IP.set_index('date', inplace=True)
IP = IP.resample('M').ffill()
IP.index = IP.index - pd.offsets.MonthBegin(1)


def _growth(fname):
    df = pd.read_csv(DATA_DIR / f'{fname}.csv')
    df.rename(columns={'observation_date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df[f'{fname}_growth'] = df[fname].pct_change()
    return df


PCEDG, PCEND, PCES, PAYEMS = (_growth(f) for f in ('PCEDG', 'PCEND', 'PCES', 'PAYEMS'))
USREC = pd.read_csv(DATA_DIR / 'USREC.csv')
USREC.rename(columns={'observation_date': 'date'}, inplace=True)
USREC['date'] = pd.to_datetime(USREC['date'])

macro_df = pd.merge(IP[['IP_growth']], PCEDG[['date', 'PCEDG_growth']], on='date')
macro_df = pd.merge(macro_df, PCEND[['date', 'PCEND_growth']], on='date')
macro_df = pd.merge(macro_df, PCES[['date', 'PCES_growth']], on='date')
macro_df = pd.merge(macro_df, PAYEMS[['date', 'PAYEMS_growth']], on='date')
macro_df = pd.merge(macro_df, USREC[['date', 'USREC']], on='date')
MACRO_COLS = ['IP_growth', 'PCEDG_growth', 'PCEND_growth', 'PCES_growth', 'PAYEMS_growth', 'USREC']
mp = pd.DataFrame({'date': dates}).merge(macro_df, on='date', how='left')
assert len(mp) == len(y) and not mp[MACRO_COLS].isna().any().any(), 'BW6 对面板月份有缺失'
BW6_now = mp[MACRO_COLS].values.astype(float)

# ---------------- 2. vintage 宽表 → 每月末 vintage 的增长率矩阵 ----------------
VIN_SERIES = ['INDPRO', 'PCEDG', 'PCEND', 'PCES', 'PAYEMS']  # INDPRO 替代 PRINTO01USQ657S
G_vin, vdates = {}, {}
for s in VIN_SERIES:
    f = VIN_DIR / f'{s}_vintages.csv'
    assert f.exists(), f'缺 {f}（先跑 download_alfred_vintage.py）'
    w = pd.read_csv(f, index_col=0)
    w.index = pd.PeriodIndex(pd.to_datetime(w.index), freq='M')
    w = w[~w.index.duplicated(keep='last')].sort_index()
    vd = pd.to_datetime([c.split('_')[-1] for c in w.columns], format='%Y%m%d')
    vdates[s] = vd
    # 每 vintage 一列增长率：先在连续月度索引上 ffill（t 月无观测 → 该 vintage 内最新可得，
    # 天然模拟发布滞后，替代原 lag1 臂；增长率对基期不变，INDPRO 无需基期调整），
    # 再取面板月。fill_method=None：禁止 pct_change 隐式填充水平值（残缺列不可凭空造增长率）。
    full_idx = pd.period_range(w.index.min(), panel_periods[-1], freq='M')
    G = w.reindex(full_idx).pct_change(fill_method=None).ffill()
    Gr = G.reindex(panel_periods)
    # 残缺 vintage 列（ALFRED 抽取历史段不完整，实测 PCEDG 1999-11-26 / 2003-12-23 两列
    # 仅覆盖 1997-01 起的零星区间）：剔除，回退到前一完整 vintage
    bad = Gr.columns[Gr.isna().any()].tolist()
    if bad:
        print(f'  {s}: 剔除覆盖不全的 vintage 列 {len(bad)} 个: {[b.split("_")[-1] for b in bad]}')
        Gr = Gr.drop(columns=bad)
        vd = pd.to_datetime([c.split('_')[-1] for c in Gr.columns], format='%Y%m%d')
        vdates[s] = vd
    assert not Gr.isna().any().any(), f'{s}: 面板月增长率有 NaN（检查 vintage 覆盖）'
    G_vin[s] = Gr
    print(f'{s}: {w.shape[1]} 个 vintage 列（{vd[0].date()} ~ {vd[-1].date()}）')

# 每月 t 的 vintage 列索引：vintage 日期 ≤ t 月末的最新者
month_ends = dates + pd.offsets.MonthEnd(0)
vidx = {}
for s in VIN_SERIES:
    vd = vdates[s]
    idx = np.searchsorted(vd.values, month_ends.values, side='right') - 1
    assert (idx >= 0).all(), f'{s}: 1984-01 前无可用 vintage'
    vidx[s] = idx

# USREC 现值列（不修订）
usrec_col = mp['USREC'].values.astype(float)


def Z_vintage(t):
    """t 月末信息集下的 BW6_vintage 控制矩阵（行 0..t，历史同为该 vintage 口径）。"""
    cols = [G_vin[s].iloc[:t + 1, vidx[s][t]].values for s in VIN_SERIES]
    return np.column_stack(cols + [usrec_col[:t + 1]])


print('vintage 增长率矩阵构建完成')

# ---------------- 3. 递归去污染（vintage 版自写循环；现值版复用 ev.decontaminate 仅作对照） ----------------
BURN = ev.BURN
D = ev.D


def decontaminate_vintage(Dmat, burn=BURN):
    n = len(Dmat)
    out = np.empty_like(Dmat)
    for t in range(n):
        if t < burn:
            out[t] = Dmat[t] - Dmat[:max(t, 1)].mean(axis=0)
        else:
            Zt = Z_vintage(t)
            assert not np.isnan(Zt).any(), f't={t} Z 有 NaN'
            Zc = np.column_stack([np.ones(t), Zt[:t]])
            G, *_ = np.linalg.lstsq(Zc, Dmat[:t], rcond=None)
            out[t] = Dmat[t] - np.concatenate([[1.0], Zt[t]]) @ G
    return out


D_vin = decontaminate_vintage(D)
print('vintage 递归去污染完成（每 t 重建 Z 历史，burn=36）')

# ---------------- 4. 锚点 (a)：最终 vintage 增长率 vs 现值 BW6 ----------------
print('\n===== 锚点 (a)：最终 vintage vs 现值 BW6（面板月增长率相关） =====')
fin_pairs = [('IP_growth', 'INDPRO'), ('PCEDG_growth', 'PCEDG'), ('PCEND_growth', 'PCEND'),
             ('PCES_growth', 'PCES'), ('PAYEMS_growth', 'PAYEMS')]
anchor_a = {}
for col_now, s in fin_pairs:
    final_col = G_vin[s].columns[-1]  # 2026 年最新 vintage
    g_fin = G_vin[s][final_col].values
    g_now = mp[col_now].values.astype(float)
    r = np.corrcoef(g_fin, g_now)[0, 1]
    anchor_a[f'{s}({col_now})'] = r
    print(f'  {s:8s} vs {col_now:15s}: corr = {r:.4f}')

# 补充（同口径对照）：PRINTO01USQ657S 是 OECD 季度口径（脚本中为季度→月度 ffill，
# 与月度环比直接相关天然偏低）；对齐到季度频率验证——OECD 序列最接近 INDPRO 的
# 季度环比（QoQ），两者相关系数即口径差的量化
w_ind = pd.read_csv(VIN_DIR / 'INDPRO_vintages.csv', index_col=0)
w_ind.index = pd.PeriodIndex(pd.to_datetime(w_ind.index), freq='M')
w_ind = w_ind[~w_ind.index.duplicated(keep='last')].sort_index()
ind_lvl = w_ind[sorted(w_ind.columns)[-1]].dropna()
oecd = pd.read_csv(DATA_DIR / 'PRINTO01USQ657S.csv', parse_dates=['observation_date'],
                   index_col='observation_date')['PRINTO01USQ657S']
oecd.index = oecd.index.to_period('Q')
ind_q = ind_lvl.resample('Q').last()
cmp_q = pd.DataFrame({'oecd': oecd, 'ind_qoq': ind_q.pct_change() * 100}).dropna()
cmp_q = cmp_q.loc['1960Q1':'2024Q4']
r_qoq = cmp_q['oecd'].corr(cmp_q['ind_qoq'])
print(f'  补充（同口径对照）: INDPRO 季度环比 vs OECD IP_growth（1960Q1–2024Q4, n={len(cmp_q)}）: '
      f'corr = {r_qoq:.4f}（口径差：OECD 自估序列 vs Fed G.17，非数据错误）')

# ---------------- 5. 赛马：raw / d~_pair / d~_BW6 / d~_BW6_vintage ----------------
usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)
rf = ev.mkt['RF_l1'] / 100
ha_full_s = pd.Series(ha)
y_full_s = ev.sp['raw']


def run_ensemble(M):
    """与 run_bw6.py 同款幂集 CSR 集成 + 全套指标（9 面板确定性单序列）。"""
    prs = ev.subset_ensemble(M, N_TRAIN)
    pr0 = prs[0]
    r2s = [ev.R2OOS(y_true, pr, base) * 100 for pr in prs]
    cw_s, cw_p, _ = ev.clark_west_test(y_true, pr0, base)
    r_rec, r_exp = ev.R2OOS_by_USREC(y_true, pr0, base, usrec)
    rec_m = usrec.iloc[:, 0].values > 0.5
    fp = pd.Series(np.full(len(y), np.nan))
    fp.iloc[N_TRAIN:] = pr0
    cer3 = ev.cer(fp, y_full_s, ha_full_s, rf, 3, transaction_cost=0, oos_len=n_oos)
    return {'full_s0': r2s[0], 'full_mean3': np.mean(r2s), 'full_min3': np.min(r2s),
            'CW_p': cw_p, 'REC': r_rec * 100, 'EXP': r_exp * 100,
            'CW_p_REC': ev.clark_west_test(y_true[rec_m], pr0[rec_m], base[rec_m])[1],
            'CW_p_EXP': ev.clark_west_test(y_true[~rec_m], pr0[~rec_m], base[~rec_m])[1],
            'CER3_tc0': cer3}


row_vin = run_ensemble(D_vin)
print(f'[d~_BW6_vintage] R2OOS = {row_vin["full_mean3"]:.3f}  CW p={row_vin["CW_p"]:.4f}  '
      f'REC={row_vin["REC"]:.2f} EXP={row_vin["EXP"]:.2f}  CER3_tc0={row_vin["CER3_tc0"]:.3f}')

# 现值三臂直读 results/bw6_2001_p9.csv（锚点 b：复用现值管线落盘，不重造）
res_bw6 = pd.read_csv(HERE / 'results' / f'bw6_{START}{ev.PTAG}.csv', index_col=0)
tab_cols = ['full_s0', 'full_mean3', 'full_min3', 'CW_p', 'REC', 'EXP', 'CER3_tc0']
table = res_bw6.loc[['raw', 'd~_pair', 'd~_BW6'], tab_cols].copy()
table.loc['d~_BW6_vintage'] = pd.Series(row_vin)[tab_cols]
table = table.loc[['raw', 'd~_BW6', 'd~_BW6_vintage', 'd~_pair']]
table.index.name = 'variant'

# REC/EXP 分段 CW p：raw / d~_pair 引自 pair_orth_u14 落盘（同管线同臂），
# d~_BW6 引自 bw6_state 的 USREC 锚点行，vintage 臂用本脚本 run_ensemble 新算值；
# 三处来源均先核对 R² 逐值一致再取 p 值
res_orth = pd.read_csv(HERE / 'results' / f'pair_orth_u14_{START}{ev.PTAG}.csv', index_col=0)
res_state = pd.read_csv(HERE / 'results' / f'bw6_state_{START}{ev.PTAG}.csv', index_col=0)
for nm in ('raw', 'd~_pair'):
    assert abs(res_orth.loc[nm, 'REC'] - table.loc[nm, 'REC']) < 5e-4, f'{nm} REC 不符'
    assert abs(res_orth.loc[nm, 'EXP'] - table.loc[nm, 'EXP']) < 5e-4, f'{nm} EXP 不符'
    table.loc[nm, 'CW_p_REC'] = res_orth.loc[nm, 'CW_p_REC']
    table.loc[nm, 'CW_p_EXP'] = res_orth.loc[nm, 'CW_p_EXP']
assert abs(res_state.loc['USREC(anchor)', 'd~BW6_high'] - table.loc['d~_BW6', 'REC']) < 5e-4
assert abs(res_state.loc['USREC(anchor)', 'd~BW6_low'] - table.loc['d~_BW6', 'EXP']) < 5e-4
table.loc['d~_BW6', 'CW_p_REC'] = res_state.loc['USREC(anchor)', 'd~BW6_high_p']
table.loc['d~_BW6', 'CW_p_EXP'] = res_state.loc['USREC(anchor)', 'd~BW6_low_p']
table.loc['d~_BW6_vintage', 'CW_p_REC'] = row_vin['CW_p_REC']
table.loc['d~_BW6_vintage', 'CW_p_EXP'] = row_vin['CW_p_EXP']

# 锚点 (b) 自检：表中 d~_BW6 行与 bw6_2001_p9.csv 逐值一致（直读，平凡成立但仍断言）
chk = pd.read_csv(HERE / 'results' / f'bw6_{START}{ev.PTAG}.csv', index_col=0)
assert np.allclose(table.loc['d~_BW6', tab_cols].values.astype(float),
                   chk.loc['d~_BW6', tab_cols].values.astype(float), atol=1e-9), '锚点 (b) 失败'
print('锚点 (b) 通过：d~_BW6 行直读自 bw6_2001_p9.csv，逐值一致')

table.round(4).to_csv(HERE / 'results' / f'bw6_vintage_{START}{ev.PTAG}.csv')
print(f'\n===== 四臂对照表（{START} 起点，9 面板 + 幂集 CSR，real-time vintage 列） =====')
print(table.round(3).to_string())
print(f'\n落盘: results/bw6_vintage_{START}{ev.PTAG}.csv')

# ---------------- 6. 发布滞后诊断（报告中用） ----------------
print('\n===== 发布滞后诊断（t 月控制值所用观测的平均滞后，月） =====')
for s in VIN_SERIES:
    w = pd.read_csv(VIN_DIR / f'{s}_vintages.csv', index_col=0)
    w.index = pd.PeriodIndex(pd.to_datetime(w.index), freq='M')
    w = w[~w.index.duplicated(keep='last')].sort_index()
    g = w.pct_change(fill_method=None)
    lags = []
    for t in range(0, len(dates), 12):  # 每年 1 月抽样，省时
        col = G_vin[s].columns[vidx[s][t]]  # 与主管线同一清洗后列序
        last_valid = g[col].dropna().index.max()
        lags.append((panel_periods[t].year - last_valid.year) * 12
                    + (panel_periods[t].month - last_valid.month))
    lags = np.array(lags, dtype=float)
    print(f'  {s:8s}: mean lag = {lags.mean():.2f} 月（p10={np.percentile(lags, 10):.0f}, '
          f'p90={np.percentile(lags, 90):.0f}）')
