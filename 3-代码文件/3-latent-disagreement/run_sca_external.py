# -*- coding: utf-8 -*-
"""
议程 6a：SCA-only 面板外部验证（2026-08-21）
============================================
目的：Deng 式外部证据——以 SCA（Michigan Survey of Consumers）6 测度独立面板跑主线全机
器（实时去污染 + 幂集 CSR），检验主线结果的测度来源依赖性。先验：SCA 弱（预测者专业程度
受限 + 直方图类测度 lv–dis 机械共定、A1 独立矩前提不成立）→ 弱结果佐证 SPF-only 选址
（设计文档 §3.4）；如实落盘，无论方向。
面板：SCA = {RPF, EPF, BC, UC, IRC, HOM}（all_data_qe.csv 季末口径，与主线同网格同目标
excess_ret_h1）；幂集 = 2^6−1 = 63 子集等权。变体臂：raw / d~_pair / d~_lv / lv_only /
d~_lv+l(12)，口径 = OOS 2001-01（n_train=204）主线 + 1991（n_train=84）附录。
诊断（机械共定/适定性）：逐测度 corr(l_k,d_k) 与投影 R²（d_k ~ l_k 全样本 IS），SCA vs
SPF-9 对照——直方图类测度的 lv–dis 机械共定使去污染投影不适定（剥除即信号本身）。
自检：同机跑 SPF-9 面板须复现主线锚点 d~_pair R²OOS = 3.956（2001 口径）方许出数。
输出：results/sca_external_{2001,1991}.csv、results/sca_external_diagnostics.csv。
"""
from pathlib import Path

import numpy as np
import pandas as pd

import run_evidence as ev  # 复用数据加载 / decontaminate / powerset_csr_preds / 指标

HERE = Path(__file__).resolve().parent
SCA = ev.SCA  # ['RPF', 'EPF', 'BC', 'UC', 'IRC', 'HOM']
OUT = HERE / 'results'

ad = ev.ad
y = ev.y
dates = ev.dates
D = np.column_stack([ad['d_' + m].values.astype(float) for m in SCA])
L = np.column_stack([ad['l_' + m].values.astype(float) for m in SCA])
n_p = len(SCA)

D_pair = np.column_stack([ev.decontaminate(D[:, [k]], L[:, [k]])[:, 0] for k in range(n_p)])
D_lv = ev.decontaminate(D, L)
variants = {
    'raw': D,
    'd~_pair': D_pair,
    'd~_lv': D_lv,
    'lv_only': L,
    f'd~_lv+l({2 * n_p})': np.column_stack([D_lv, L]),
}
print('SCA 面板变体构建完成（expanding 实时去污染，burn=36；幂集 63 子集）')


def run_start(n_train, tag):
    n_oos = len(y) - n_train
    base = ev.ha[n_train:]
    y_true = y[n_train:]
    usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)
    rf = ev.mkt['RF_l1'] / 100
    ha_full_s = pd.Series(ev.ha)
    y_full_s = ev.sp['raw']
    rows = []
    for name, M in variants.items():
        pr = ev.powerset_csr_preds(M, y, n_train)
        r2 = ev.R2OOS(y_true, pr, base) * 100
        cw_s, cw_p, _ = ev.clark_west_test(y_true, pr, base)
        r_rec, r_exp = ev.R2OOS_by_USREC(y_true, pr, base, usrec)
        fp = pd.Series(np.full(len(y), np.nan))
        fp.iloc[n_train:] = pr
        rows.append({'variant': name, 'R2OOS': r2, 'CW_stat': cw_s, 'CW_p': cw_p,
                     'REC': r_rec * 100, 'EXP': r_exp * 100,
                     'CER3_tc0': ev.cer(fp, y_full_s, ha_full_s, rf, 3, transaction_cost=0.0, oos_len=n_oos)})
        print(f'  [{tag}] {name:14s} R2OOS={r2:7.3f}  CW p={cw_p:.3f}  REC={r_rec*100:7.2f}  EXP={r_exp*100:6.2f}')
    res = pd.DataFrame(rows).set_index('variant')
    res.round(4).to_csv(OUT / f'sca_external_{tag}.csv')
    return res


def diagnostics():
    """逐测度 corr(l_k,d_k) 与投影 R²（全样本 IS）：SCA vs SPF-9 对照。"""
    p9 = pd.read_csv(HERE / 'panel9_qe.csv')
    rows = []
    for src, Dm, Lm, names in (('SCA', D, L, SCA),
                               ('SPF9', p9[[f'd_{m}' for m in ['GDP', 'IP', 'CON', 'INV', 'HSG', 'UEP', 'CPI', 'TBL', 'CPROF']]].values.astype(float),
                                p9[[f'l_{m}' for m in ['GDP', 'IP', 'CON', 'INV', 'HSG', 'UEP', 'CPI', 'TBL', 'CPROF']]].values.astype(float),
                                ['GDP', 'IP', 'CON', 'INV', 'HSG', 'UEP', 'CPI', 'TBL', 'CPROF'])):
        for k, nm in enumerate(names):
            c = np.corrcoef(Lm[:, k], Dm[:, k])[0, 1]
            Xc = np.column_stack([np.ones(len(Dm)), Lm[:, k]])
            b, *_ = np.linalg.lstsq(Xc, Dm[:, k], rcond=None)
            r2_proj = 1 - np.var(Dm[:, k] - Xc @ b) / np.var(Dm[:, k])
            rows.append({'panel': src, 'measure': nm, 'corr_ld': c, 'proj_R2': r2_proj})
    dg = pd.DataFrame(rows).set_index(['panel', 'measure'])
    dg.round(4).to_csv(OUT / 'sca_external_diagnostics.csv')
    print(dg.round(3).to_string())
    return dg


def self_check():
    """同机自检：SPF-9 面板复现主线锚点 d~_pair R²OOS = 3.956（2001 口径，±0.01）。"""
    p9 = pd.read_csv(HERE / 'panel9_qe.csv')
    names9 = ['GDP', 'IP', 'CON', 'INV', 'HSG', 'UEP', 'CPI', 'TBL', 'CPROF']
    D9 = p9[[f'd_{m}' for m in names9]].values.astype(float)
    L9 = p9[[f'l_{m}' for m in names9]].values.astype(float)
    D9_pair = np.column_stack([ev.decontaminate(D9[:, [k]], L9[:, [k]])[:, 0] for k in range(9)])
    pr = ev.powerset_csr_preds(D9_pair, y, 204)
    r2 = ev.R2OOS(y[204:], pr, ev.ha[204:]) * 100
    ok = abs(r2 - 3.956) < 0.01
    print(f'[self-check] SPF-9 d~_pair R2OOS = {r2:.4f}（锚点 3.956）→ {"PASS" if ok else "FAIL"}')
    assert ok, '自检失败：机器不等价，禁止出数'


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    self_check()
    for s, nt in ((2001, 204), (1991, 84)):
        print(f'\n===== SCA-only 幂集 CSR（OOS {s}，n_train={nt}）=====')
        run_start(nt, str(s))
    print('\n===== 诊断：corr(l,d) 与投影 R²（SCA vs SPF-9）=====')
    diagnostics()
    print('\n完成（议程 6a；结果如实落盘，预期弱 = 佐证 SPF-only 选址）。')
