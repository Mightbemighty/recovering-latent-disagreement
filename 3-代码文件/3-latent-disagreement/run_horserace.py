# -*- coding: utf-8 -*-
"""
Phase B-1：不可约简赛马（horse race）+ 吸收检验（absorption test）
================================================================
口径：--start 参数化（默认 1991 主线：N_TRAIN=84，OOS=408，1991-01~2024-12；
--start 2001 复现 2001 口径），严格递归——
任何指数构造（PCA 载荷、PLS 权重）与 OLS 系数在预测 t 期时只用 ≤t-1 期数据（expanding）。
benchmark = results/ha_baseline_recomputed.csv 的自算 HA（ha[t]=mean(y[:t])，与 run_evidence 一致）。

三块产出：
1. 竞争聚合器（raw 21 个 d 测度，expanding 严格递归）：EW / PCA(PC1) / PLS(Huang 三步法)，
   各报 R²OOS（对自算 HA）与 CW p。
2. 赛马（encompassing regression，Huang Table 8 式）：OOS 窗口内
   y_t = a + b1*pred_ens_t + b2*pred_comp_t + e，Newey-West(maxlags=6)；
   附 pred_ens 与各竞争器预测的相关系数。pred_ens = preds_{start}.csv 的主变体列。
3. 吸收检验：y_t = a + b1*pred_ens_t + b2*pred_dTBL_t + e（同窗口同 NW）。
   pred_dTBL = d̃_TBL（d_TBL 对全 lv 族 21 列 expanding 投影残差，burn=36）单测度 expanding 一元 OLS。

与 run_svt.py 实现的差异（按任务规格，以本脚本为准）：
- PCA 增加符号定向：每月以训练窗内 PC1 与 y 的相关定号（corr<0 则翻转载荷），
  run_svt.py 直接用 sklearn 的 svd_flip 确定性符号，未对 y 定向。
- PLS 与 run_svt.py 的 pls_huang_expanding_preds 同口径（triPLS：测度对 y 的一元回归斜率作权重，index=X·π/K）。

--panel9（2026-08-20）：9 面板绑定版——竞争聚合器在 9 个 d 测度上构造（D/L 由 run_evidence
模块级预解析切换）；pred_ens 改读 preds_{start}_p9.csv 的 d~_pair 列（9 面板主变体，2026-08-19 起）；
吸收检验 lv 族为 9 列（star 名 d~_TBL(lv9)）；输出 horserace_{start}_p9.csv / absorption_{start}_p9.csv，
不覆写 21 面板锚点文件。

--input {raw,dpair,dlv}（2026-08-21，议程 2 / thoughts/18 T-B 网格补全）：竞争聚合器的输入矩阵
由 raw D 切换为 d̃_pair（逐对 lv 投影残差）或 d̃_lv（全族 lv 投影残差）。非 raw 输入仅限 --panel9
（避免覆写 21 面板锚点 horserace_dlv_2001.csv），且只跑第 1 块（聚合器指标 + 与集成的相关），
encompassing 赛马与吸收检验跳过；输出 horserace_{input}_{start}_p9.csv。默认 --input raw 行为不变。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # 复用数据加载 / decontaminate / expanding_preds / R2OOS / clark_west_test

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_ap0.add_argument('--input', type=str, default='raw', choices=['raw', 'dpair', 'dlv'])
_args0, _ = _ap0.parse_known_args()
START = _args0.start
INPUT = _args0.input  # raw（默认）/ dpair / dlv（议程 2 T-B；限 --panel9）
N_TRAIN = ev.STARTS[START]
NW_LAG = 6
TAG = str(START)
PTAG = ev.PTAG  # ''（默认 21 面板）/ '_p9'（--panel9，由 run_evidence 模块级预解析生效）
ENS_COL = 'd~_pair' if ev.PANEL9 else 'd~_lv'  # 各面板主变体（9 面板 = d~_pair，2026-08-19 起）

y = ev.y
D = ev.D          # raw d 测度（492×n_p；默认 21，--panel9 为 9）
L = ev.L          # lv 族（列数同 n_p）
ha = ev.ha
dates = ev.dates
pairs = ev.pairs

y_oos = y[N_TRAIN:]
ha_oos = ha[N_TRAIN:]
assert not np.isnan(ha_oos).any()

# --input：聚合器输入矩阵（默认 raw = D；议程 2 T-B：dpair = 逐对投影残差 / dlv = 全族投影残差）
if INPUT == 'raw':
    D_in = D
else:
    assert ev.PANEL9, '--input 非 raw 仅限 --panel9（议程 2 T-B 网格；避免覆写 21 面板锚点）'
    if INPUT == 'dpair':
        D_in = np.column_stack([ev.decontaminate(D[:, [k]], L[:, [k]])[:, 0]
                                for k in range(ev.n_p)])
    else:  # dlv
        D_in = ev.decontaminate(D, L)


# ---------------- 竞争聚合器（raw 21 d，expanding 严格递归） ----------------
def ew_expanding_preds(Dmat, y_in, n_train):
    """EW：全部列逐行等权均值 → 单变量 expanding OLS。"""
    return ev.expanding_preds(Dmat.mean(axis=1), y_in, n_train)


def pca_expanding_preds(Dmat, y_in, n_train):
    """PCA：每月用 ≤t-1 数据估计 PC1 载荷（相关阵口径=未再标准化，面板已 Huang 式标准化）；
    符号定向 = 训练窗内 PC1 与 y 相关为负则翻转；再单变量 expanding OLS。"""
    n = len(y_in)
    preds = np.empty(n - n_train)
    for j, i in enumerate(range(n_train - 1, n - 1)):
        Xf = Dmat[:i + 1]
        Xc = Xf - Xf.mean(axis=0)
        # SVD 取第一主成分载荷
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        w = vt[0].copy()
        pc_train = Xc @ w
        if np.corrcoef(pc_train, y_in[:i + 1])[0, 1] < 0:
            w = -w
            pc_train = -pc_train
        beta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(i + 1), pc_train]), y_in[:i + 1], rcond=None)
        pc_next = (Dmat[i + 1] - Xf.mean(axis=0)) @ w
        preds[j] = np.array([1.0, pc_next]) @ beta
    return preds


def pls_huang_expanding_preds(Dmat, y_in, n_train):
    """PLS：Huang 三步法 expanding（triPLS 口径）——每月用 ≤t-1 数据将每个测度对 y 做
    一元时序回归取斜率 π_k，index = X·π/K；再单变量 expanding OLS。"""
    n, K = Dmat.shape
    preds = np.empty(n - n_train)
    for j, i in enumerate(range(n_train - 1, n - 1)):
        Yf = y_in[:i + 1]
        Rc = np.column_stack([np.ones(i + 1), Yf])
        pi = np.linalg.lstsq(Rc, Dmat[:i + 1], rcond=None)[0][1]   # (K,)
        idx = Dmat @ pi / K
        beta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(i + 1), idx[:i + 1]]), Yf, rcond=None)
        preds[j] = np.array([1.0, idx[i + 1]]) @ beta
    return preds


# ---------------- 读集成预测（主变体：21 面板 = d~_lv；--panel9 = d~_pair） ----------------
preds_stored = pd.read_csv(HERE / 'results' / f'preds_{TAG}{PTAG}.csv', parse_dates=['date'])
assert len(preds_stored) == len(y_oos)
assert (preds_stored['date'].values == dates[N_TRAIN:].values).all(), 'OOS 日期不对齐'
pred_ens = preds_stored[ENS_COL].values.astype(float)

# 集成自身指标（对照）
r2_ens = ev.R2OOS(y_oos, pred_ens, ha_oos) * 100
cw_s_ens, cw_p_ens, _ = ev.clark_west_test(y_oos, pred_ens, ha_oos)
print(f'[对照] {ENS_COL} 集成: R2OOS={r2_ens:.3f}, CW p={cw_p_ens:.4f}')


# ---------------- 工具：指标 + encompassing 回归 ----------------
def nw_ols(y_dep, X_cols):
    """OLS + Newey-West(maxlags=NW_LAG)；返回 params 与 tvalues。"""
    X = sm.add_constant(np.column_stack(X_cols))
    res = sm.OLS(y_dep, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAG})
    return res.params, res.tvalues


def metrics(pred):
    r2 = ev.R2OOS(y_oos, pred, ha_oos) * 100
    cw_s, cw_p, cw_sig = ev.clark_west_test(y_oos, pred, ha_oos)
    return r2, cw_s, cw_p, cw_sig


# ---------------- 1+2. 竞争器指标 + 赛马 ----------------
competitors = {
    'EW': ew_expanding_preds(D_in, y, N_TRAIN),
    'PCA': pca_expanding_preds(D_in, y, N_TRAIN),
    'PLS': pls_huang_expanding_preds(D_in, y, N_TRAIN),
}

if INPUT != 'raw':
    # 议程 2 T-B 网格补全：去污染输入上只报聚合器指标（+ 与集成的相关），赛马/吸收跳过
    rows_in = []
    for name, pred in competitors.items():
        r2, cw_s, cw_p, cw_sig = metrics(pred)
        corr = np.corrcoef(pred_ens, pred)[0, 1]
        rows_in.append({'competitor': name, 'R2OOS': r2, 'CW_stat': cw_s, 'CW_p': cw_p,
                        'CW_sig': cw_sig, 'corr_with_ens': corr})
        print(f'[{name:4s}|{INPUT}] R2OOS={r2:7.3f}  CW p={cw_p:.4f}  corr(ens,comp)={corr:+.3f}')
    pd.DataFrame(rows_in).set_index('competitor').round(4).to_csv(
        HERE / 'results' / f'horserace_{INPUT}_{TAG}{PTAG}.csv')
    print(f'\n落盘: results/horserace_{INPUT}_{TAG}{PTAG}.csv（--input {INPUT}：仅聚合器指标）')
    sys.exit(0)

rows = []
for name, pred in competitors.items():
    r2, cw_s, cw_p, cw_sig = metrics(pred)
    corr = np.corrcoef(pred_ens, pred)[0, 1]
    params, tvals = nw_ols(y_oos, [pred_ens, pred])
    rows.append({'competitor': name, 'R2OOS': r2, 'CW_stat': cw_s, 'CW_p': cw_p, 'CW_sig': cw_sig,
                 'corr_with_ens': corr,
                 'b1_ens': params[1], 't1_ens': tvals[1], 'b2_comp': params[2], 't2_comp': tvals[2]})
    print(f'[{name:4s}] R2OOS={r2:7.3f}  CW p={cw_p:.4f}  corr(ens,comp)={corr:+.3f}  '
          f'b1={params[1]:+.3f}(t={tvals[1]:+.2f})  b2={params[2]:+.3f}(t={tvals[2]:+.2f})')

hr = pd.DataFrame(rows).set_index('competitor')
hr.round(4).to_csv(HERE / 'results' / f'horserace_{TAG}{PTAG}.csv')

# ---------------- 3. 吸收检验（d̃_TBL 单测度） ----------------
k_tbl = [k for k, (_, dc) in enumerate(pairs) if dc == 'd_TBL'][0]
stars = {
    # 任务规格主口径：d_TBL 对全 lv 族（21 列；--panel9 为 9 列）expanding 投影残差，burn=36
    f'd~_TBL(lv{ev.n_p})': ev.decontaminate(D[:, [k_tbl]], L)[:, 0],
    # 对照口径：逐对去污染（= run_evidence 的 d~_pair 第 k 列），锚点 R2OOS=4.2223（h1_pairwise_2001.csv；
    # 构造只涉 TBL 自身一对，两面板同值）
    'd~_TBL(pair)': ev.decontaminate(D[:, [k_tbl]], L[:, [k_tbl]])[:, 0],
}
abs_rows = []
for star_name, star_series in stars.items():
    pred_star = ev.expanding_preds(star_series, y, N_TRAIN)
    r2_s, _, cw_p_s, _ = metrics(pred_star)
    corr_s = np.corrcoef(pred_ens, pred_star)[0, 1]
    params, tvals = nw_ols(y_oos, [pred_ens, pred_star])
    abs_rows.append({
        'star': star_name, 'R2OOS_star': r2_s, 'CW_p_star': cw_p_s,
        'R2OOS_ens': r2_ens, 'CW_p_ens': cw_p_ens, 'corr_with_ens': corr_s,
        'b1_ens': params[1], 't1_ens': tvals[1], 'b2_star': params[2], 't2_star': tvals[2],
    })
    print(f'[吸收] {star_name:14s}: R2OOS={r2_s:6.3f} CW p={cw_p_s:.4f} corr={corr_s:+.3f}  '
          f'b1={params[1]:+.3f}(t={tvals[1]:+.2f})  b2={params[2]:+.3f}(t={tvals[2]:+.2f})')
abs_df = pd.DataFrame(abs_rows).set_index('star')
abs_df.round(4).to_csv(HERE / 'results' / f'absorption_{TAG}{PTAG}.csv')

print(f'\n落盘: results/horserace_{TAG}{PTAG}.csv, results/absorption_{TAG}{PTAG}.csv')
