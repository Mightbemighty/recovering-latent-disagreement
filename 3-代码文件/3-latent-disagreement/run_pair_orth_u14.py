# -*- coding: utf-8 -*-
"""
d~_pair ⊥ u14 二次投影检验（9 面板，2001 口径）
==============================================
问题：d~_pair（同题 lv 投影残差）的预测力是否其实是残留的感知不确定性
（perceived uncertainty）？做法：对已纯化的 d~_pair 面板再投影一层 u14
（un_data_recursive.csv，14 列递归 JLN 式不确定性），走同款幂集 CSR，
与 raw / d~_pair / d~_u14 三锚点对照（逐值校验，锚点读 results/bw6_2001_p9.csv）。

口径与现成脚本完全一致：sys.argv 锁定 --panel9 --start 2001（n_train=204，
OOS 288 月）；去污染 ev.decontaminate(burn=ev.BURN=36)，集成 ev.subset_ensemble
（9 面板幂集 CSR，确定性三序列同值）。

输出：results/pair_orth_u14_2001_p9.csv（仅新增，不覆写任何现有文件）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = ['run_pair_orth_u14', '--panel9', '--start', '2001']
import run_evidence as ev  # noqa: E402  # 模块级预解析需先锁定 argv

assert ev.PANEL9, '必须 9 面板口径'
N_TRAIN = ev.STARTS[2001]
y, ha, dates = ev.y, ev.ha, ev.dates
n_oos = len(y) - N_TRAIN
y_true, base = y[N_TRAIN:], ha[N_TRAIN:]
assert n_oos == 288
usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)

# ---------------- 1. u14 面板核对（492 行，日期网格与 ev.dates 一致） ----------------
un = pd.read_csv(HERE / 'un_data_recursive.csv', parse_dates=['date'])
assert len(un) == 492 and (un['date'].values == dates.values).all(), 'u14 面板网格不符'
U = un[[c for c in un.columns if c.startswith('u_')]].values.astype(float)
assert U.shape == (492, 14) and not np.isnan(U).any(), 'u14 应有 14 列且无缺失'
# 与 run_evidence 模块级读入的 ev.U 逐值一致（同一文件）
assert np.allclose(U, ev.U)
print(f'u14 面板核对通过：{U.shape[0]} 行 × {U.shape[1]} 列，网格 1984-01~2024-12 与 ev.dates 一致')

# ---------------- 2. 四臂集成（raw / d~_pair / d~_u14 为锚点复核，d~_pair⊥u14 为新臂） ----------------
arms = {
    'raw': ev.D,
    'd~_pair': ev.variants['d~_pair'],
    'd~_u14': ev.decontaminate(ev.D, U, burn=ev.BURN),
    'd~_pair⊥u14': ev.decontaminate(ev.variants['d~_pair'], U, burn=ev.BURN),
}

rows = {}
for name, M in arms.items():
    prs = ev.subset_ensemble(M, N_TRAIN)
    pr0 = prs[0]
    cw_s, cw_p, _ = ev.clark_west_test(y_true, pr0, base)
    r_rec, r_exp = ev.R2OOS_by_USREC(y_true, pr0, base, usrec)
    rec_m = usrec.iloc[:, 0].values > 0.5
    rows[name] = {'full': ev.R2OOS(y_true, pr0, base) * 100,
                  'CW_p': cw_p, 'REC': r_rec * 100, 'EXP': r_exp * 100,
                  'CW_p_REC': ev.clark_west_test(y_true[rec_m], pr0[rec_m], base[rec_m])[1],
                  'CW_p_EXP': ev.clark_west_test(y_true[~rec_m], pr0[~rec_m], base[~rec_m])[1]}
    print(f'[{name:12s}] full={rows[name]["full"]:7.4f}  CW p={cw_p:.4f}  '
          f'REC={rows[name]["REC"]:7.4f}  EXP={rows[name]["EXP"]:7.4f}')

# ---------------- 3. 锚点复核（与 results/bw6_2001_p9.csv 三行逐值一致） ----------------
anchor = pd.read_csv(HERE / 'results' / 'bw6_2001_p9.csv', index_col=0)
for name in ('raw', 'd~_pair', 'd~_u14'):
    for got_k, anc_k in (('full', 'full_s0'), ('CW_p', 'CW_p'), ('REC', 'REC'), ('EXP', 'EXP')):
        got, want = rows[name][got_k], anchor.loc[name, anc_k]
        assert abs(got - want) < 5e-4, f'锚点不符 {name}.{got_k}: {got:.4f} != {want:.4f}'
print('锚点复核通过：raw / d~_pair / d~_u14 与 results/bw6_2001_p9.csv 逐值一致')

# ---------------- 4. 参考量：d~_pair 与 u14 的相关水平（判读用） ----------------
C = np.corrcoef(np.column_stack([ev.variants['d~_pair'], U]).T)[:9, 9:]
print(f'\n参考量 |corr(d~_pair 各列, u14 各列)|：mean={np.abs(C).mean():.3f}  '
      f'median={np.median(np.abs(C)):.3f}  max={np.abs(C).max():.3f}')

# ---------------- 5. 落盘 ----------------
tab = pd.DataFrame(rows).T
tab.index.name = 'variant'
tab.round(4).to_csv(HERE / 'results' / 'pair_orth_u14_2001_p9.csv')
print('\n===== 对照表（OOS 2001-01~2024-12，9 面板幂集 CSR，R2OOS %） =====')
print(tab.round(3).to_string())
print('\n落盘: results/pair_orth_u14_2001_p9.csv')
