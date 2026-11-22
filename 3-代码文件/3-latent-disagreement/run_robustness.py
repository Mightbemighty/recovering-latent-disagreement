# -*- coding: utf-8 -*-
"""
Phase A 稳健性（--start 参数化，默认 1991 主线 n_train=84；--start 2001 复现 2001 口径 n_train=204）：
burn-in / rolling-γ / ensemble size S。变体锁定 d~_lv（主方案），raw 作参照。
输出 results/robustness_{start}.csv
口径（2026-08-17 起）：burn/rolling-γ/raw 行 = 全枚举 CSR 主线；S 行 = 抽样口径存档（enum=False）。
--panel9（2026-08-20）：9 面板主变体 = d~_pair（同题 lv 投影残差），burn/rolling-γ 网格
全部建在 d~_pair 上，d~_lv 保留一行基准对照；幂集 CSR 为确定性估计量（2^9−1=511 子集
全枚举，无抽样环节），S 敏感性行不适用并注明跳过。输出 results/robustness_{start}_p9.csv。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '1-sCNNLSTM-and-meta-RSR-notebooks'))
from MetricsDef import R2OOS, R2OOS_by_USREC, cer, clark_west_test  # noqa: E402
import run_evidence as ev  # noqa: E402  （模块级已完成数据加载与去污染变体构建）

# ---- 起点开关（2026-08-20：主线 OOS 起点 2001→1991；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
N_OOS = len(ev.y) - N_TRAIN
base = ev.ha[N_TRAIN:]
y_true = ev.y[N_TRAIN:]
usrec = ev.usrec_full.iloc[-N_OOS:].reset_index(drop=True)
rf = ev.mkt['RF_l1'] / 100
ha_full_s = pd.Series(ev.ha)
y_full_s = ev.sp['raw']


def decontaminate_rolling(Dmat, Zmat, burn=36, window=None):
    """window=None → expanding；否则用最近 window 行（不足 burn 前去均值， burn 起用可得行）。"""
    n = len(Dmat)
    out = np.empty_like(Dmat)
    for t in range(n):
        if t < burn:
            out[t] = Dmat[t] - Dmat[:max(t, 1)].mean(axis=0)
        else:
            lo = 0 if window is None else max(0, t - window)
            Zc = np.column_stack([np.ones(t - lo), Zmat[lo:t]])
            G, *_ = np.linalg.lstsq(Zc, Dmat[lo:t], rcond=None)
            out[t] = Dmat[t] - np.concatenate([[1.0], Zmat[t]]) @ G
    return out


def decontaminate_pair(Dmat, Lmat, burn=36, window=None):
    """逐对（同题 lv）去污染：每列 d 仅对其同题 l 投影残差化（--panel9 主变体 d~_pair 的网格版）。"""
    if window is None:
        return np.column_stack([ev.decontaminate(Dmat[:, [k]], Lmat[:, [k]], burn=burn)[:, 0]
                                for k in range(Dmat.shape[1])])
    return np.column_stack([decontaminate_rolling(Dmat[:, [k]], Lmat[:, [k]], burn=burn, window=window)[:, 0]
                            for k in range(Dmat.shape[1])])


def evaluate(M, k=11, b=50, seeds=ev.SEEDS, enum=True):
    """enum=True（默认，2026-08-17 起主线口径）：走 ev.subset_ensemble 全枚举 CSR（确定性）；
    --panel9 下该路径为幂集 CSR（k 被忽略，无 K 概念）。enum=False：S=b×3 种子抽样
    （仅用于 S 敏感性行的抽样口径存档；p9 模式不使用）。
    列名 full_mean3 保留，enum 路径下其值为全枚举口径。"""
    if enum:
        prs = ev.subset_ensemble(M, N_TRAIN, k=k)
        # --panel9 幂集 CSR 为确定性估计量：三种子序列逐值相等（s0≡mean3），
        # CW 在 s0 序列上计算无歧义（与主表 variants CW_p 的 s0 约定一致；2026-08-27 核验断言）
        if ev.PANEL9:
            assert all(np.array_equal(prs[0], pr) for pr in prs[1:]), '幂集 CSR 三种子序列应逐值相等'
        r2s = [R2OOS(y_true, pr, base) * 100 for pr in prs]
        pr0 = prs[0]
    else:
        r2s, pr0 = [], None
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            ss = [sorted(rng.choice(M.shape[1], k, replace=False).tolist()) for _ in range(b)]
            pr = np.column_stack([ev.expanding_preds(M[:, s], ev.y, N_TRAIN) for s in ss]).mean(axis=1)
            r2s.append(R2OOS(y_true, pr, base) * 100)
            if si == 0:
                pr0 = pr
    r_rec, r_exp = R2OOS_by_USREC(y_true, pr0, base, usrec)
    fp = pd.Series(np.full(len(ev.y), np.nan))
    fp.iloc[N_TRAIN:] = pr0
    c3 = cer(fp, y_full_s, ha_full_s, rf, 3, transaction_cost=0.0, oos_len=N_OOS)
    out = {'full_mean3': np.mean(r2s), 'full_min3': np.min(r2s), 'EXP': r_exp * 100, 'CER3_tc0': c3}
    if ev.PANEL9:
        # Clark–West 单边 p（全窗、s0 序列、默认自动 NW lag；2026-08-27 补，供 Table C1 burn 行星标）
        out['CW_p'] = clark_west_test(y_true, pr0, base)[1]
    return out


rows = []
rows.append({'config': 'raw（不去污染）', **evaluate(ev.D)})
if ev.PANEL9:
    # 9 面板（2026-08-20）：主变体 = d~_pair（同题 lv 投影），d~_lv 保留基准对照行。
    for burn in (24, 36, 48):
        M = decontaminate_pair(ev.D, ev.L, burn=burn)
        tag = '（d~_pair，基准）' if burn == 36 else '（d~_pair）'
        rows.append({'config': f'burn={burn}{tag}', **evaluate(M)})
    for W in (60, 120):
        M = decontaminate_pair(ev.D, ev.L, window=W)
        rows.append({'config': f'rolling-γ W={W}（d~_pair）', **evaluate(M)})
    rows.append({'config': 'burn=36（d~_lv 全族对照）', **evaluate(ev.decontaminate(ev.D, ev.L))})
    print('注：--panel9 幂集 CSR 为确定性估计量（511 子集全枚举），无抽样环节，'
          'S 敏感性行不适用（21 面板抽样口径存档行见 robustness_2001.csv）。')
else:
    for burn in (24, 36, 48):
        M = ev.decontaminate(ev.D, ev.L, burn=burn)
        rows.append({'config': f'burn={burn}' + ('（基准）' if burn == 36 else ''), **evaluate(M)})
    for W in (60, 120):
        M = decontaminate_rolling(ev.D, ev.L, window=W)
        rows.append({'config': f'rolling-γ W={W}', **evaluate(M)})
    for S in (25, 100):
        rows.append({'config': f'S={S}（抽样口径存档）', **evaluate(ev.variants['d~_lv'], b=S, enum=False)})

res = pd.DataFrame(rows).set_index('config')
print(res.round(3).to_string())
out_csv = HERE / 'results' / f'robustness_{START}{ev.PTAG}.csv'
res.round(4).to_csv(out_csv)
print(f'\n已存 {out_csv.relative_to(HERE)}')
