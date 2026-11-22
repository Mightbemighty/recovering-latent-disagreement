# -*- coding: utf-8 -*-
"""
议程 6e：Livingston 利率分歧面板外部验证（2026-08-24；`make_liv_panel.py` 下游）
====================================================================================
目的：普适性检验正面臂——H1/§50/§52 钉死主线信号集中于利率分歧（TBL），ECB SPF 无
利率变量（机制变量缺位的安慰剂臂）；Livingston 为同一受访者类型（专业预测者）、同一
题材（利率）、不同调查 → 若"TBL 承载"是机制而非美国 SPF 数据挖掘产物，Livingston 的
利率分歧应在主线机器下复现（至少方向性）预测力。
面板：`liv4_qe.csv`（TBILL/TBOND × {6M, 12M} 4 测度对，lv=截面均值、dis=截面 IQR
[averaged_inverted_cdf，官方 D1 口径锚定 PASS]，1993-06 起 379 行子网格，rolling-72
标准化）。半年度→月度：6 月调查→7 月起、12 月调查→次年 1 月起 ffill（发布滞后保守）。
机器：主线全机器（expanding 实时去污染 burn=36 + 幂集 CSR 2^4−1=15 子集）；
目标 = 主线 excess_ret_h1（美国市场超额收益，子网格切片对齐）。
口径：n_train=84（OOS 2000-06 起）/ n_train=120（OOS 2003-06 起）。
自检：同机 SPF-9 复现主线锚点 d̃_pair R²OOS = 3.956（2001 口径）方许出数。
输出：results/liv_external_{84,120}.csv。
"""
from pathlib import Path

import numpy as np
import pandas as pd

import run_evidence as ev

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
MS = ['TBILL6M', 'TBILL12M', 'TBOND6M', 'TBOND12M']

y_full, ha_full = ev.y, ev.ha
p9 = pd.read_csv(HERE / 'panel9_qe.csv', parse_dates=['date'])
liv = pd.read_csv(HERE / 'liv4_qe.csv', parse_dates=['date'])
i0 = int(np.where(p9['date'].values == liv['date'].iloc[0].to_datetime64())[0][0])
assert (p9['date'].iloc[i0:].values == liv['date'].values).all()
y, ha = y_full[i0:], ha_full[i0:]
n = len(y)
print(f'Livingston 子网格：{liv["date"].iloc[0]:%Y-%m} 起 {n} 行（主线 iloc {i0}）')

D = liv[[f'd_{m}' for m in MS]].values.astype(float)
L = liv[[f'l_{m}' for m in MS]].values.astype(float)
assert not np.isnan(D).any() and not np.isnan(L).any()
D_pair = np.column_stack([ev.decontaminate(D[:, [k]], L[:, [k]])[:, 0] for k in range(4)])
D_lv = ev.decontaminate(D, L)
variants = {'raw': D, 'd~_pair': D_pair, 'd~_lv': D_lv, 'lv_only': L,
            'd~_lv+l(8)': np.column_stack([D_lv, L])}


def run_start(n_train, tag):
    base, yt = ha[n_train:], y[n_train:]
    rows = []
    for name, M in variants.items():
        pr = ev.powerset_csr_preds(M, y, n_train)
        r2 = ev.R2OOS(yt, pr, base) * 100
        cw_s, cw_p, _ = ev.clark_west_test(yt, pr, base)
        rows.append({'variant': name, 'R2OOS': r2, 'CW_stat': cw_s, 'CW_p': cw_p})
        print(f'  [{tag}] {name:12s} R2OOS={r2:7.3f}  CW p={cw_p:.3f}')
    res = pd.DataFrame(rows).set_index('variant')
    res.round(4).to_csv(OUT / f'liv_external_{tag}.csv')
    return res


def self_check():
    names9 = ['GDP', 'IP', 'CON', 'INV', 'HSG', 'UEP', 'CPI', 'TBL', 'CPROF']
    D9 = p9[[f'd_{m}' for m in names9]].values.astype(float)
    L9 = p9[[f'l_{m}' for m in names9]].values.astype(float)
    D9_pair = np.column_stack([ev.decontaminate(D9[:, [k]], L9[:, [k]])[:, 0] for k in range(9)])
    pr = ev.powerset_csr_preds(D9_pair, y_full, 204)
    r2 = ev.R2OOS(y_full[204:], pr, ha_full[204:]) * 100
    ok = abs(r2 - 3.956) < 0.01
    print(f'[self-check] SPF-9 d~_pair R2OOS = {r2:.4f}（锚点 3.956）→ {"PASS" if ok else "FAIL"}')
    assert ok, '自检失败：机器不等价，禁止出数'


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    self_check()
    for nt, tag in ((84, '84'), (120, '120')):
        print(f'\n===== Livingston 4 测度幂集 CSR（n_train={nt}，OOS {liv["date"].iloc[nt]:%Y-%m} 起）=====')
        run_start(nt, tag)
    print('\n完成（议程 6e）。')
