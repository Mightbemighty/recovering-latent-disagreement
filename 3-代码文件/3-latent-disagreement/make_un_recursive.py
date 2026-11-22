# -*- coding: utf-8 -*-
"""
递归版 u14（修复全样本 VAR 前视，results/NOTES.md §28）
========================================================
原构造（3.1_ML_Model v2.ipynb cell 33）：GW14 逐一双变量 VAR（+ mkt_ret，maxlags=6，
BIC 选阶），全样本 1980-08~2024-12 一次拟合 → u_t 含全样本系数信息（前视实锤：
重算 u_DP 与磁盘文件 corr=1.0、max|diff|=1.4e-11）。

本脚本（递归修复版）：
- 系数逐月 expanding 递归，仅用 <t 数据（1980-08 起的历史全部可用）；
- VAR(2) 双变量（econ, mkt_ret）——econ 方程 = OLS: econ_t ~ const + econ_{t-1..2}
  + ret_{t-1..2}，单方程 OLS 与 VAR 逐方程 OLS 数学等价；lag 固定 2（原构造全样本
  BIC 对 14 列的众数选择，对照表随运行输出）；
- 残差只取当期：u_t = log(resid_t^2)；
- INFL 整体 shift(1)（当月 CPI 次月中公布，t 月末仅 t-1 值可得）；EP/BM/DE 沿用
  Goyal 数据集内建的财年滞后，不再额外处理；
- 面板起点 1984-01 行已有 41 个月训练历史（1980-08~1983-12），无需 burn 特例。

输出：un_data_recursive.csv（1984-01~2024-12，492 行，与 all_data_qe 逐月对齐）
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCNN = HERE.parent / '1-sCNNLSTM-and-meta-RSR-notebooks'

econ = pd.read_csv(SCNN / 'econ_data.csv', parse_dates=['date'])
ad = pd.read_csv(SCNN / 'all_data_qe.csv', parse_dates=['date'])

COLS14 = list(econ.columns[1:15])  # DP..INFL
assert COLS14 == ['DP', 'DY', 'EP', 'DE', 'SVAR', 'BM', 'NTIS', 'TBL',
                  'LTY', 'LTR', 'TMS', 'DFY', 'DFR', 'INFL']

econ['INFL'] = econ['INFL'].shift(1)  # 公布滞后
econ = econ.dropna(subset=COLS14 + ['mkt_ret']).reset_index(drop=True)

e14 = econ[COLS14].values.astype(float)
ret = econ['mkt_ret'].values.astype(float)
dates_e = econ['date']

LAG = 2


def recursive_u(col):
    """u_t = log(resid_t^2)，系数用 <t 行 expanding OLS（VAR(2) econ 方程）。"""
    e = e14[:, col]
    n = len(e)
    u = np.full(n, np.nan)
    # 回归矩阵行 t 需要 t-1, t-2 有效 → t 从 LAG 起
    for t in range(LAG, n):
        # 训练行 s ∈ [LAG, t-1]（X 用 s-1/s-2，y = e_s）
        s = np.arange(LAG, t)
        if len(s) < 12:  # 至少需要观测覆盖 5 参数
            continue
        X = np.column_stack([np.ones(len(s)), e[s - 1], e[s - 2], ret[s - 1], ret[s - 2]])
        beta, *_ = np.linalg.lstsq(X, e[s], rcond=None)
        xt = np.array([1.0, e[t - 1], e[t - 2], ret[t - 1], ret[t - 2]])
        r = e[t] - xt @ beta
        u[t] = np.log(r * r + 1e-18)
    return u


U = np.column_stack([recursive_u(c) for c in range(14)])
out = pd.DataFrame(U, columns=[f'u_{c}' for c in COLS14])
out.insert(0, 'date', dates_e)

# 对齐 all_data_qe 的 492 行（1984-01~2024-12）
out = out.merge(ad[['date']], on='date', how='right').sort_values('date').reset_index(drop=True)
assert len(out) == 492 and (out['date'].values == ad['date'].values).all()
assert not out.drop(columns='date').isna().any().any(), '面板内存在 NaN'

out.round(6).to_csv(HERE / 'un_data_recursive.csv', index=False)
print('写出 un_data_recursive.csv:', out.shape)

# 对照：与原全样本版的逐列相关（信息用）
un_old = pd.read_csv(SCNN / 'un_data.csv', parse_dates=['date'])
cors = {c: float(np.corrcoef(out[f'u_{c}'], un_old[f'u_{c}'])[0, 1]) for c in COLS14}
print('与原（前视）版逐列相关：')
for c, r in cors.items():
    print(f'  {c}: {r:.3f}')
print(f'  中位: {np.median(list(cors.values())):.3f}')
