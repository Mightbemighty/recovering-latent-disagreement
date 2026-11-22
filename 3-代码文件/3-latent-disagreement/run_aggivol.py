# -*- coding: utf-8 -*-
"""
AggIVOL 重计算（任务 B，2026-08-04）
====================================
定义：每股票每月，月内日度收益对 FF3（Mkt-RF, SMB, HML，含截距）OLS，取残差 std；
横截面等权平均得月度 AggIVOL。要求月内 ≥15 个有效日观测。
方法：充分统计量法——分块读 CRSP 日度（8.7GB），按 (PERMNO, yyyymm) 只聚合
n 与 14 个一阶/二阶/交叉和，由和直接构造 X'X(4×4) 与 X'y，np.linalg.solve 批量求解；
SSE = Σr² − β'X'y，残差方差 = SSE/(n−4)。全程向量化，无 Python 层逐组循环。
过滤器：SecurityType=EQTY & SecuritySubType=COM & PrimaryExch∈{N,A,Q}（同 AF 测度约定）。
样本：1990-01 ~ 2024-12。
产出：aggivol_monthly.csv（date[月末], AggIVOL, n_stocks）。
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / '1-数据文件'
YM_LO, YM_HI = '1990-01', '2024-12'

# ---------------- FF3 日度（percent → decimal） ----------------
ff = pd.read_csv(DATA / 'F-F_Research_Data_Factors_daily.csv', skiprows=4)
ff.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
ff = ff[ff['Date'].astype(str).str.fullmatch(r'\d{8}')]
ff['date'] = pd.to_datetime(ff['Date'], format='%Y%m%d')
for c in ('Mkt-RF', 'SMB', 'HML'):
    ff[c] = pd.to_numeric(ff[c], errors='coerce') / 100.0
ff = ff.set_index('date')[['Mkt-RF', 'SMB', 'HML']].dropna()
print(f'FF3 daily: {ff.index.min().date()} ~ {ff.index.max().date()}, {len(ff)} rows', flush=True)

# ---------------- 分块读 CRSP 日度，块内预聚合 ----------------
USECOLS = ['PERMNO', 'PrimaryExch', 'SecurityType', 'SecuritySubType', 'DlyCalDt', 'DlyRet']
SUM_COLS = ['r', 'r2', 'm', 's', 'h', 'm2', 's2', 'h2', 'rm', 'rs', 'rh', 'ms', 'mh', 'sh']
CKPT = HERE / 'aggivol_sums.pkl'
t0 = time.time()
if CKPT.exists():
    S = pd.read_pickle(CKPT)
    print(f'从检查点加载聚合和: {len(S)} 组', flush=True)
else:
    parts = []
    n_scan = n_keep = 0
    reader = pd.read_csv(DATA / 'daily_stock_return_25.csv', usecols=USECOLS,
                         dtype={'PrimaryExch': 'str', 'SecurityType': 'str', 'SecuritySubType': 'str'},
                         chunksize=5_000_000)
    for ci, ch in enumerate(reader):
        n_scan += len(ch)
        ch = ch[(ch['SecurityType'] == 'EQTY') & (ch['SecuritySubType'] == 'COM')
                & ch['PrimaryExch'].isin(['N', 'A', 'Q'])]
        ch['ym'] = ch['DlyCalDt'].str[:7]
        ch = ch[(ch['ym'] >= YM_LO) & (ch['ym'] <= YM_HI)]
        ch['r'] = pd.to_numeric(ch['DlyRet'], errors='coerce')
        ch['d'] = pd.to_datetime(ch['DlyCalDt'])
        ch = ch.dropna(subset=['r'])
        ch = ch.join(ff, on='d', how='inner')
        n_keep += len(ch)
        if len(ch) == 0:
            continue
        m, s, h = ch['Mkt-RF'].values, ch['SMB'].values, ch['HML'].values
        r = ch['r'].values
        ch = ch[['PERMNO', 'ym']].copy()
        ch['r'] = r; ch['r2'] = r * r
        ch['m'] = m; ch['s'] = s; ch['h'] = h
        ch['m2'] = m * m; ch['s2'] = s * s; ch['h2'] = h * h
        ch['rm'] = r * m; ch['rs'] = r * s; ch['rh'] = r * h
        ch['ms'] = m * s; ch['mh'] = m * h; ch['sh'] = s * h
        g = ch.groupby(['PERMNO', 'ym'], sort=False)
        agg = g[SUM_COLS].sum()
        agg['n'] = g.size()
        parts.append(agg)
        print(f'chunk {ci}: scanned {n_scan/1e6:.0f}M kept {n_keep/1e6:.2f}M, {time.time()-t0:.0f}s', flush=True)

    S = pd.concat(parts).groupby(level=['PERMNO', 'ym']).sum()
    del parts
    S.to_pickle(CKPT)
    print(f'聚合完成: {len(S)} (stock,month) 组, {time.time()-t0:.0f}s（已存检查点）', flush=True)

# ---------------- 充分统计量 → 批量 4×4 OLS ----------------
v = S[SUM_COLS].values.astype(np.float64)
n = S['n'].values.astype(np.float64)
idx = {c: i for i, c in enumerate(SUM_COLS)}

def build_XtX(v, n):
    G = len(v)
    A = np.empty((G, 4, 4))
    A[:, 0, 0] = n
    A[:, 0, 1] = A[:, 1, 0] = v[:, idx['m']]
    A[:, 0, 2] = A[:, 2, 0] = v[:, idx['s']]
    A[:, 0, 3] = A[:, 3, 0] = v[:, idx['h']]
    A[:, 1, 1] = v[:, idx['m2']]; A[:, 1, 2] = A[:, 2, 1] = v[:, idx['ms']]
    A[:, 1, 3] = A[:, 3, 1] = v[:, idx['mh']]
    A[:, 2, 2] = v[:, idx['s2']]; A[:, 2, 3] = A[:, 3, 2] = v[:, idx['sh']]
    A[:, 3, 3] = v[:, idx['h2']]
    return A

A = build_XtX(v, n)
bvec = np.column_stack([v[:, idx['r']], v[:, idx['rm']], v[:, idx['rs']], v[:, idx['rh']]])
# pinv 批量求解：对 n<15 或因子零方差导致的奇异 X'X 稳健（良态矩阵与 solve 等价）
beta = np.einsum('gij,gj->gi', np.linalg.pinv(A), bvec)        # (G,4)
sse = v[:, idx['r2']] - np.einsum('ij,ij->i', beta, bvec)
bad = sse < 0
if bad.any():
    print(f'警告: SSE<0 的组数 = {int(bad.sum())}（置 NaN）', flush=True)
    sse[bad] = np.nan
resid_std = np.sqrt(sse / (n - 4))
resid_std[n < 15] = np.nan
S['ivol'] = resid_std

# ---------------- 月度横截面等权平均 ----------------
mon = S['ivol'].dropna().groupby(level='ym')
aggivol = mon.mean().rename('AggIVOL').to_frame()
aggivol['n_stocks'] = mon.size()
aggivol.index = pd.PeriodIndex(aggivol.index, freq='M').to_timestamp('M')
aggivol.index.name = 'date'
print(f"AggIVOL: {aggivol.index.min().date()} ~ {aggivol.index.max().date()}, {len(aggivol)} 月", flush=True)
print(f"n_stocks: min={int(aggivol['n_stocks'].min())} median={int(aggivol['n_stocks'].median())} "
      f"max={int(aggivol['n_stocks'].max())}", flush=True)
aggivol.round(8).to_csv(HERE / 'aggivol_monthly.csv')
print(f"落盘: {HERE / 'aggivol_monthly.csv'}, 总用时 {time.time()-t0:.0f}s", flush=True)
