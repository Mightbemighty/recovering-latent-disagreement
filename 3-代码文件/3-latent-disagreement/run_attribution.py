# -*- coding: utf-8 -*-
"""
归因检验：聚合方式 × 去污染 × 状态（--start 参数化，默认 1991 主线：OOS 408 月，自算 HA）
====================================================================
回答："子集集成与 lv 去污染各自的收益集中在哪个状态"。

聚合方式：
- A0-EW      ：EW 指数（n_p 列行均值 → 单变量 expanding OLS），确定性
- A0-meanUniv：n_p 个单变量 expanding OLS 预测逐月取均值，确定性
- A1-ens     ：子集集成（21 面板 K=11 全枚举 / --panel9 幂集 CSR，确定性序列；与 preds_{start}.csv 核对）

去污染：raw d vs d~_lv（lv 族投影，burn=36，与主管线一致）。
状态：full / REC / EXP R²OOS（对自算 HA，USREC 分组）。

输出：results/attribution_{start}.csv（单元格 + 集成效应/去污染效应两组差分）。

--panel9（2026-08-19，T6）：去污臂加 d~_pair（9 面板主变体）；指标增 pre/post-2020
分段（T6：pre-2020 Δ 的 A0/A1 聚合层分解，C10 判别）；输出 attribution_{start}_p9.csv。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
RES = HERE / 'results'
PTAG = '_p9' if ev.PANEL9 else ''
NP_ = ev.n_p  # 21（默认）或 9（--panel9）

y = ev.y
ha = ev.ha
n_oos = len(y) - N_TRAIN
y_oos = y[N_TRAIN:]
ha_oos = ha[N_TRAIN:]
dates_oos = ev.dates[N_TRAIN:].reset_index(drop=True)
usrec = pd.read_csv(ev.SCNN / 'USREC.csv').iloc[-n_oos:].reset_index(drop=True)

D_raw = ev.D
D_dlv = ev.variants['d~_lv']
# 去污臂：21 面板 = {raw, d~_lv}（锚点口径）；--panel9 加 d~_pair（9 面板主去污变体）
DECON_ARMS = {'raw': D_raw, 'd~_lv': D_dlv}
if ev.PANEL9:
    DECON_ARMS = {'raw': D_raw, 'd~_pair': ev.variants['d~_pair'], 'd~_lv': D_dlv}

# ---------------- 单元格的预测序列 ----------------
# A0-EW：指数=行均值 → 单变量 expanding OLS
# A0-meanUniv：逐列单变量 expanding OLS 预测逐月取均值
# A1-ens：全枚举/幂集（s0 与落盘核对，容差 6e-7 = 落盘 round(6)）
preds = pd.read_csv(RES / f'preds_{START}{PTAG}.csv', parse_dates=['date'])
p_ew, p_mu, p_ens = {}, {}, {}
for d, M in DECON_ARMS.items():
    p_ew[d] = ev.expanding_preds(M.mean(axis=1), y, N_TRAIN)
    p_mu[d] = np.column_stack([ev.expanding_preds(M[:, k], y, N_TRAIN) for k in range(NP_)]).mean(axis=1)
    prs = ev.subset_ensemble(M, N_TRAIN)
    assert np.abs(prs[0] - preds[d].values).max() < 6e-7, f'{d} 集成与落盘不符'
    p_ens[d] = np.mean(prs, axis=0)

cells = {
    'A0-EW': p_ew,
    'A0-meanUniv': p_mu,
    'A1-ens(mean3)': p_ens,
}

# ---------------- full / REC / EXP R²OOS（--panel9 增 pre/post-2020 分段，T6 口径） ----------------
def r3(p):
    out = {'full': ev.R2OOS(y_oos, p, ha_oos) * 100}
    r_rec, r_exp = ev.R2OOS_by_USREC(y_oos, p, ha_oos, usrec)
    out['REC'], out['EXP'] = r_rec * 100, r_exp * 100
    if ev.PANEL9:
        pre = (dates_oos < pd.Timestamp('2020-01-01')).values
        out['pre2020'] = ev.R2OOS(y_oos[pre], p[pre], ha_oos[pre]) * 100
        out['post2020'] = ev.R2OOS(y_oos[~pre], p[~pre], ha_oos[~pre]) * 100
    return out


# --panel9（2026-08-27）：各设计单元格补 Clark–West 单边 p（对自算 HA 基准；REC/EXP 与
# pre/post-2020 非连续/分段子样本在 mask 子序列上直接调 clark_west_test，先例 =
# MetricsDef.clark_west_test_by_USREC；NW HAC 自动 lag，与主表 variants CW_p 同约定）。
# 派生差值行（ens效应/去污染效应）为星标格间算术，p 列留空（显著性层 = Table 5 DM）。
def cw3(p):
    um = usrec.iloc[:, 0].values
    out = {'full_p': ev.clark_west_test(y_oos, p, ha_oos)[1],
           'REC_p': ev.clark_west_test(y_oos[um == 1], p[um == 1], ha_oos[um == 1])[1],
           'EXP_p': ev.clark_west_test(y_oos[um == 0], p[um == 0], ha_oos[um == 0])[1]}
    pre = (dates_oos < pd.Timestamp('2020-01-01')).values
    out['pre2020_p'] = ev.clark_west_test(y_oos[pre], p[pre], ha_oos[pre])[1]
    out['post2020_p'] = ev.clark_west_test(y_oos[~pre], p[~pre], ha_oos[~pre])[1]
    return out


rows = {}
for a, dd in cells.items():
    for d, p in dd.items():
        row = r3(p)
        if ev.PANEL9:
            row.update(cw3(p))
        rows[f'{a} × {d}'] = row

COLS = ['full', 'REC', 'EXP'] + (['pre2020', 'post2020'] if ev.PANEL9 else [])
PCOLS = [f'{c}_p' for c in COLS] if ev.PANEL9 else []
tab = pd.DataFrame(rows).T[COLS + PCOLS]
tab.index.name = 'cell'
print('===== 单元格 R²OOS(%) =====')
print(tab.round(3).to_string())

# ---------------- 差分 ----------------
eff = {}
# 集成效应 = A1 − A0（同 D 内，两个 A0 基准各算一次）
for a0 in ('A0-EW', 'A0-meanUniv'):
    for d in DECON_ARMS:
        eff[f'ens效应(A1-{a0}) | {d}'] = {
            s: tab.loc[f'A1-ens(mean3) × {d}', s] - tab.loc[f'{a0} × {d}', s] for s in COLS}
# 去污染效应 = 去污臂 − raw（同 A 内）
for a in cells:
    for d in DECON_ARMS:
        if d == 'raw':
            continue
        key = f'去污染效应({d}) | {a}' if ev.PANEL9 else f'去污染效应 | {a}'  # 默认模式行名与锚点文件逐字一致
        eff[key] = {s: tab.loc[f'{a} × {d}', s] - tab.loc[f'{a} × raw', s] for s in COLS}

eff_tab = pd.DataFrame(eff).T[COLS]
eff_tab.index.name = 'effect'
print('\n===== 效应分解（pt）=====')
print(eff_tab.round(3).to_string())

out = pd.concat([tab, eff_tab])
out.round(4).to_csv(RES / f'attribution_{START}{PTAG}.csv')
print(f'\n落盘: results/attribution_{START}{PTAG}.csv')
