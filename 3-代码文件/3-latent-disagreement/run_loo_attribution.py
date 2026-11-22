# -*- coding: utf-8 -*-
"""
LOO 归属分析（leave-one-out attribution）：利率分歧（TBL）支配度量化
====================================================================
动机：审稿人指出主线结果（9 测度幂集 CSR + d̃_pair，2001-01 起 R²OOS 3.956%）高度依赖
利率题（TBL 单变量增益 +1.11pp，其余题增益多为小数点后一位），要求 leave-one-out
归属分析。本脚本在主线口径（--panel9 + --start 2001，n_train=204，n_oos=288）下：

  1. LOO 网格：逐题剔除后剩余 8 题走幂集 CSR（raw 与 d̃_pair 两版输入），
     报 R² full/REC/EXP + Clark–West p，定位剔除伤害最大的题。
  2. 单测度锚点：9 题各自单变量 expanding OLS 的 R²（raw 与 d̃_pair），
     与 results/h1_pairwise_2001_p9.csv（Table 4/B1 源表）逐值核对后引用。
  3. 子集构成：幂集 CSR 的 511 个非空子集，按是否含 TBL 分两组（256 / 255），
     比较两组子集的逐子集 R² 分布（mean/median/分位/正值占比）与组内等权集成的 R²
     ——不含 TBL 组的等权集成按构造恒等于 drop-TBL 面板的幂集 CSR（自检）。
  4. horizon 对照（Q1 季末抽样口径，复用 run_frequency 的季度原生构造）：
     全面板 / drop-TBL / TBL 单测度在季度频率下的 R²，检验 TBL 支配度的频率稳健性。

锚点（NOTES §52 scratch 口径与 variants/h1/frequency 落盘文件）：
  全面板 d̃_pair 3.9561 / raw 2.7063；drop-TBL d̃_pair 0.293；
  h1 单测度 d̃_TBL 4.2223 / raw 3.1085；Q1 全面板 raw 4.9796 / d̃_pair 8.8307、
  d_TBL 单测度 raw 8.3497。

输出：results/loo_attribution_p9.csv + 控制台全量打印。
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # 复用锁定面板 / decontaminate / powerset CSR / 指标（参考 run_bw6.py）

# ---- 起点开关（默认 2001 主线：n_train=204，n_oos=288） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=2001, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]

assert ev.PANEL9 is True, '本脚本限 --panel9（9 测度 SPF-only 幂集 CSR 主线口径）'
assert N_TRAIN == 204, f'LOO 归属分析登记口径为 2001 起点（n_train=204），当前 {N_TRAIN}'

y, dates, ha = ev.y, ev.dates, ev.ha
MEAS = [d[2:] for _, d in ev.pairs]  # 字母序：CON, CPROF, CPI, GDP, HSG, INV, IP, TBL, UEP
K_TBL = MEAS.index('TBL')
n_oos = len(y) - N_TRAIN
y_true, base = y[N_TRAIN:], ha[N_TRAIN:]
usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)

VARIANTS = {'raw': ev.D, 'd~_pair': ev.D_pair}


def metrics(pr):
    """full/REC/EXP R²OOS（%）+ Clark–West p（h=1 默认 lag）。"""
    r2 = ev.R2OOS(y_true, pr, base) * 100
    _, p, _ = ev.clark_west_test(y_true, pr, base)
    r_rec, r_exp = ev.R2OOS_by_USREC(y_true, pr, base, usrec)
    return {'R2': r2, 'REC': r_rec * 100, 'EXP': r_exp * 100, 'CW_p': p}


# ================= 1. LOO 网格 =================
print(f'\n===== 1. LOO 网格（9 面板，{START} 起点，n_train={N_TRAIN}，n_oos={n_oos}，幂集 CSR）=====')
loo_rows, ens_pred = [], {}
for vname, X in VARIANTS.items():
    for drop in [None] + list(range(len(MEAS))):
        if drop is None:
            M, tag = X, 'NONE(全面板)'
        else:
            M = X[:, [j for j in range(X.shape[1]) if j != drop]]
            tag = MEAS[drop]
        pr = ev.subset_ensemble(M, N_TRAIN)[0]  # --panel9 → 幂集 CSR（2^p−1 等权）
        ens_pred[(vname, tag)] = pr
        m = metrics(pr)
        loo_rows.append({'section': 'loo', 'variant': vname, 'drop': tag, **m})
        print(f'  [{vname:8s}] drop={tag:13s} R2={m["R2"]:7.3f}  REC={m["REC"]:7.2f}  '
              f'EXP={m["EXP"]:6.2f}  CW p={m["CW_p"]:.4f}')

loo = pd.DataFrame(loo_rows)
for vname in VARIANTS:
    ref = loo[(loo['variant'] == vname) & (loo['drop'] == 'NONE(全面板)')]['R2'].iloc[0]
    loo.loc[loo['variant'] == vname, 'delta_vs_full'] = loo.loc[loo['variant'] == vname, 'R2'] - ref

# 锚点校验（variants_2001_p9 / NOTES §52 scratch 口径）
a_full_pair = loo[(loo['variant'] == 'd~_pair') & (loo['drop'] == 'NONE(全面板)')]['R2'].iloc[0]
a_full_raw = loo[(loo['variant'] == 'raw') & (loo['drop'] == 'NONE(全面板)')]['R2'].iloc[0]
a_drop_pair = loo[(loo['variant'] == 'd~_pair') & (loo['drop'] == 'TBL')]['R2'].iloc[0]
assert abs(a_full_pair - 3.9561) < 0.01, f'全面板 d~_pair 锚点失配: {a_full_pair}'
assert abs(a_full_raw - 2.7063) < 0.01, f'全面板 raw 锚点失配: {a_full_raw}'
assert abs(a_drop_pair - 0.293) < 0.02, f'drop-TBL d~_pair 锚点失配: {a_drop_pair}'
print('  锚点校验通过：全面板 d~_pair 3.956 / raw 2.706；drop-TBL d~_pair 0.293（NOTES §52）')

# ================= 2. 单测度锚点汇总（核对 h1_pairwise_2001_p9.csv） =================
print('\n===== 2. 单测度单变量 expanding OLS（raw 与 d~_pair，与 h1 落盘表核对）=====')
h1_file = HERE / 'results' / f'h1_pairwise_{START}{ev.PTAG}.csv'
h1 = pd.read_csv(h1_file, index_col=0)
single_rows = []
for k, name in enumerate(MEAS):
    pr_d = ev.expanding_preds(ev.D[:, k], y, N_TRAIN)
    pr_dt = ev.expanding_preds(ev.D_pair[:, k], y, N_TRAIN)
    m_d, m_dt = metrics(pr_d), metrics(pr_dt)
    # 与落盘 h1 表逐值核对（文件 4 位小数舍入）
    assert abs(m_d['R2'] - h1.loc[name, 'R2_d']) < 1e-3, f'{name} raw 单测度与 h1 表失配'
    assert abs(m_dt['R2'] - h1.loc[name, 'R2_d~']) < 1e-3, f'{name} d~_pair 单测度与 h1 表失配'
    single_rows.append({'section': 'single', 'variant': name,
                        'R2': m_dt['R2'], 'R2_raw': m_d['R2'], 'gain': m_dt['R2'] - m_d['R2'],
                        'CW_p': m_dt['CW_p'], 'CWp_raw': m_d['CW_p'],
                        'REC': m_dt['REC'], 'EXP': m_dt['EXP']})
    print(f'  {name:6s} raw R2={m_d["R2"]:7.3f} (CW p={m_d["CW_p"]:.3f})   '
          f'd~_pair R2={m_dt["R2"]:7.3f} (CW p={m_dt["CW_p"]:.3f})   gain={m_dt["R2"] - m_d["R2"]:+.3f}')
single = pd.DataFrame(single_rows)
print(f'  与 {h1_file.name} 逐值核对通过（9/9 测度）')

# ================= 3. 子集构成分析（511 子集 × 是否含 TBL） =================
print('\n===== 3. 子集构成：511 个非空子集按是否含 TBL 分组 =====')
p = len(MEAS)
subsets = [s for r in range(1, p + 1) for s in itertools.combinations(range(p), r)]
assert len(subsets) == 511
subset_rows = []
for vname, X in VARIANTS.items():
    recs = []
    for s in subsets:
        pr = ev.expanding_preds(X[:, list(s)], y, N_TRAIN)
        recs.append({'has_TBL': K_TBL in s, 'k': len(s),
                     'R2': ev.R2OOS(y_true, pr, base) * 100, 'pr': pr})
    df = pd.DataFrame(recs)
    # 自检：511 子集等权平均的 R² 须等于幂集 CSR 主线 R²
    r2_ens = ev.R2OOS(y_true, np.mean(df['pr'].tolist(), axis=0), base) * 100
    r2_ref = a_full_pair if vname == 'd~_pair' else a_full_raw
    assert abs(r2_ens - r2_ref) < 1e-6, f'{vname} 子集等权平均与幂集 CSR 不一致: {r2_ens} vs {r2_ref}'
    for has, grp in ((True, 'with_TBL'), (False, 'without_TBL')):
        sub = df[df['has_TBL'] == has]
        # 组内等权集成（without_TBL 组按构造恒等于 drop-TBL 幂集 CSR）
        pr_grp = np.mean(sub['pr'].tolist(), axis=0)
        m_grp = metrics(pr_grp)
        row = {'section': 'subset', 'variant': vname, 'group': grp,
               'n_subsets': len(sub), 'R2': m_grp['R2'], 'REC': m_grp['REC'],
               'EXP': m_grp['EXP'], 'CW_p': m_grp['CW_p'],
               'subset_R2_mean': sub['R2'].mean(), 'subset_R2_median': sub['R2'].median(),
               'subset_R2_p25': sub['R2'].quantile(0.25), 'subset_R2_p75': sub['R2'].quantile(0.75),
               'subset_R2_min': sub['R2'].min(), 'subset_R2_max': sub['R2'].max(),
               'subset_share_pos': (sub['R2'] > 0).mean() * 100}
        subset_rows.append(row)
        print(f'  [{vname:8s}] {grp:11s} n={len(sub):3d}  逐子集 R2: mean={row["subset_R2_mean"]:6.3f}  '
              f'median={row["subset_R2_median"]:6.3f}  [{row["subset_R2_p25"]:6.3f}, {row["subset_R2_p75"]:6.3f}]  '
              f'min={row["subset_R2_min"]:7.3f}  max={row["subset_R2_max"]:6.3f}  正值占比={row["subset_share_pos"]:.1f}%  '
              f'| 组内等权集成 R2={m_grp["R2"]:7.3f} (CW p={m_grp["CW_p"]:.4f})')
    if vname == 'd~_pair':
        # without_TBL 组内等权集成 ≡ drop-TBL 幂集 CSR（LOO 第 1 节同值）
        assert abs(subset_rows[-1]['R2'] - a_drop_pair) < 1e-6
print('  自检通过：511 子集等权平均 ≡ 幂集 CSR；without_TBL 组内集成 ≡ drop-TBL（LOO）')

# ================= 4. horizon 对照：Q1 季末抽样季度原生口径 =================
print('\n===== 4. Q1 口径（季末抽样，h=1 季度原生，burn=12 季，NW lag=3）=====')
import run_frequency as rf  # 复用 y3 前向 3 月收益目标 / ha_h 同信息集基准

n = len(y)
QE = np.where(dates.dt.month % 3 == 0)[0]
QE = QE[QE <= n - 3]  # 目标需完整
yq = rf.y3[QE]
nq = int(np.sum(QE < N_TRAIN))  # 与主线同一边界（2001 → 68 季训练，95 季 OOS）
Dq, Lq = ev.D[QE], ev.L[QE]
Dpq = np.column_stack([ev.decontaminate(Dq[:, [k]], Lq[:, [k]], burn=12)[:, 0]
                       for k in range(Dq.shape[1])])
mask8 = [j for j in range(p) if j != K_TBL]
yq_true = yq[nq:]
base_q = rf.ha_h(yq, nq, 1)[nq:]
usrec_q = rf.us.iloc[QE[nq:]].reset_index(drop=True)


def metrics_q(pr):
    r2 = ev.R2OOS(yq_true, pr, base_q) * 100
    _, pv, _ = ev.clark_west_test(yq_true, pr, base_q, lag=3)
    r_rec, r_exp = ev.R2OOS_by_USREC(yq_true, pr, base_q, usrec_q)
    return {'R2': r2, 'REC': r_rec * 100, 'EXP': r_exp * 100, 'CW_p': pv}


q1_cells = {
    'Q1 full raw': ev.powerset_csr_preds(Dq, yq, nq, h=1),
    'Q1 full d~_pair': ev.powerset_csr_preds(Dpq, yq, nq, h=1),
    'Q1 drop-TBL raw': ev.powerset_csr_preds(Dq[:, mask8], yq, nq, h=1),
    'Q1 drop-TBL d~_pair': ev.powerset_csr_preds(Dpq[:, mask8], yq, nq, h=1),
    'Q1 TBL single raw': ev.expanding_preds(Dq[:, K_TBL], yq, nq),
    'Q1 TBL single d~_pair': ev.expanding_preds(Dpq[:, K_TBL], yq, nq),
}
q1_rows = []
for name, pr in q1_cells.items():
    m = metrics_q(pr)
    q1_rows.append({'section': 'q1', 'variant': name, **m, 'n_oos': len(yq_true)})
    print(f'  {name:22s} R2={m["R2"]:7.3f}  REC={m["REC"]:7.2f}  EXP={m["EXP"]:6.2f}  CW p={m["CW_p"]:.4f}')
# Q1 锚点（frequency_q1_p9.csv：raw_ens / d~_pair_ens / d_TBL single raw）
assert abs(q1_rows[0]['R2'] - 4.9796) < 0.01, f'Q1 full raw 锚点失配: {q1_rows[0]["R2"]}'
assert abs(q1_rows[1]['R2'] - 8.8307) < 0.01, f'Q1 full d~_pair 锚点失配: {q1_rows[1]["R2"]}'
assert abs(q1_rows[4]['R2'] - 8.3497) < 0.01, f'Q1 TBL single raw 锚点失配: {q1_rows[4]["R2"]}'
print('  Q1 锚点校验通过（frequency_q1_p9.csv：4.980 / 8.831 / 8.350）')

# ================= 汇总落盘 =================
out = pd.concat([loo, single, pd.DataFrame(subset_rows), pd.DataFrame(q1_rows)],
                ignore_index=True)
out_path = HERE / 'results' / f'loo_attribution{ev.PTAG}.csv'
out.round(4).to_csv(out_path, index=False)

print('\n===== LOO 伤害排序（delta vs 全面板，升序 = 剔除伤害最大）=====')
for vname in VARIANTS:
    sub = loo[(loo['variant'] == vname) & (loo['drop'] != 'NONE(全面板)')].sort_values('delta_vs_full')
    print(f'  [{vname}] ' + '  '.join(f'{r["drop"]}:{r["delta_vs_full"]:+.3f}'
                                     for _, r in sub.iterrows()))
print(f'\n落盘: results/loo_attribution{ev.PTAG}.csv')
