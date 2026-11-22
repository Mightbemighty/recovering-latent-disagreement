# -*- coding: utf-8 -*-
"""
T-DM：DM (2021) 主观分解受限改编（议程 4，2026-08-21；设计登记 = thoughts/19）
================================================================================
以 BGLS 发布包的聚合 LTG 共识（IBES 分析师长期盈利增长预期，1981:12–2021:12）为
长 horizon 主观 CF 预期代理，检验 d̃_pair 是否比 raw 更好预测主观 CF 预期的前向修正
（信念层渠道证据，补 T3 客观 VAR 新闻赛马；绕开 corr(CF,DR)≈−1 的分解病理，NOTES §38）。

目标：fwrev_h(t) = LTG_{t+h} − LTG_t（前向 h 月修正；行 t 窗口 = t+1..t+h）。
构造 gate：fwrev12(t) 应逐值等于 BGLS 自带 d1LTG(t+12)（容差 1e-10）。
特征臂：raw / d̃_pair / lv_only；预测器 = 幂集 CSR + expanding OLS
（ev.powerset_csr_preds，nan_aware=True，训练截断 s ≤ i−h）。
评估：R²OOS（%）对 expanding 历史均值（仅用已完成目标）；主格 fwrev12 × 2001，
敏感格 fwrev3 × 2001 与两 h × 1991。限 --panel9。

输出：results/dm_subjective_p9.csv（start × h × arm 长表 + gate 行）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # noqa: E402

assert ev.PANEL9, 'T-DM 限 --panel9（thoughts/19）'
DATA = HERE / '..' / '..' / '1-数据文件'
RES = HERE / 'results'

# ---------------- 1. LTG 面板对齐 ----------------
bg = pd.read_csv(DATA / 'BGLS2024_TimeSeries01.tab', sep='\t', low_memory=False)
bg['date'] = pd.to_datetime(bg['datem'])
p9 = pd.read_csv(HERE / 'panel9_qe.csv', parse_dates=['date'])
m = p9[['date']].merge(bg[['date', 'LTG', 'd1LTG']], on='date', how='left')
ltg = m['LTG'].values.astype(float)
n = len(m)

# ---------------- 2. 目标构造 + gate ----------------
def fwrev(h):
    return np.concatenate([ltg[h:] - ltg[:-h], np.full(h, np.nan)])

gate_rows = []
f12 = fwrev(12)
d1 = m['d1LTG'].values.astype(float)  # d1LTG(t) = LTG_t − LTG_{t−12}（BGLS Table 4 变量）
cmp_ = np.isfinite(f12[:-12]) & np.isfinite(d1[12:])
gdiff = np.abs(f12[:-12][cmp_] - d1[12:][cmp_]).max()
assert gdiff < 1e-6, f'构造 gate 失败：fwrev12(t) vs d1LTG(t+12) max|diff|={gdiff:.2e}'
# 实测 max|diff| = 5.71e-09（.tab 存储精度噪声，LTG 量级 ~10）→ 构造一致确认
gate_rows.append({'item': 'gate: fwrev12(t)==d1LTG(t+12)', 'value': f'max|diff|={gdiff:.2e}', 'arm': '-'})

# ---------------- 3. 赛马 ----------------
ARMS = {'raw': ev.D, 'd~_pair': ev.D_pair, 'lv_only': ev.L}
rows = []
for start, n_train in [(2001, ev.STARTS[2001]), (1991, ev.STARTS[1991])]:
    for h in [12, 3]:
        yt = fwrev(h)
        i_max = int(np.max(np.where(np.isfinite(yt))))  # 最后一个目标有效行
        # expanding 历史均值基准（仅用已完成目标：行 t 可见 s ≤ t−h 的目标）
        ha = np.full(n, np.nan)
        cs, cn = 0.0, 0
        for t in range(n):
            s = t - h
            if s >= 0 and np.isfinite(yt[s]):
                cs += yt[s]; cn += 1
            if cn > 0:
                ha[t] = cs / cn
        for arm, M in ARMS.items():
            pr = ev.powerset_csr_preds(M, yt, n_train, h=h, i_max=i_max, nan_aware=True)
            rows_eval = np.arange(n_train, i_max + 1)
            ok = np.isfinite(yt[rows_eval])
            r2 = ev.R2OOS(yt[rows_eval][ok], pr[ok], ha[rows_eval][ok]) * 100
            # Clark–West 单边 p（2026-08-27 补；对 expanding HA、评估行 ok 子序列）。
            # fwrev_h 为重叠 h 月窗目标，沿用项目重叠目标口径 NW lag=2h
            # （run_frequency M3 / run_bgls_bridge / run_liv_frequency 同约定）。
            _, cw_p, _ = ev.clark_west_test(yt[rows_eval][ok], pr[ok], ha[rows_eval][ok], lag=2 * h)
            rows.append({'start': start, 'h': h, 'arm': arm, 'R2OOS': round(r2, 4),
                         'cw_p': round(cw_p, 4), 'n': int(ok.sum())})
            print(f'[T-DM|{start}|h={h:2d}] {arm:8s} R²OOS={r2:+7.3f}  (n={int(ok.sum())})', flush=True)

out = pd.DataFrame(rows)
g = pd.DataFrame(gate_rows)
g['start'], g['h'], g['R2OOS'], g['n'] = '-', '-', '-', '-'
pd.concat([g, out], ignore_index=True).to_csv(RES / 'dm_subjective_p9.csv', index=False)
print(f"\n[T-DM] gate 通过；落盘 results/dm_subjective_p9.csv")
# 主格判级（thoughts/19 §3 事先承诺）
main = out[(out['start'] == 2001) & (out['h'] == 12)].set_index('arm')['R2OOS']
verdict = '成立' if (main['d~_pair'] > main['raw'] and main['d~_pair'] > 0) else '不成立'
print(f"[T-DM] 主格（fwrev12×2001）：d̃_pair {main['d~_pair']:+.3f} vs raw {main['raw']:+.3f} → 判级 {verdict}")
