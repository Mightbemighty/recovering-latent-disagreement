# -*- coding: utf-8 -*-
"""
实验 5a：shrinkage 聚合器稳健性赛马（9 面板 · 1991 主线，2026-08-20）
==============================================================================
质疑：主结果（d~_pair 去污染增益）是否幂集 CSR 特定？本脚本把三臂测度矩阵
（raw / d~_pair / d~_lv，各 9 列，直读 ev.variants）送入四类 shrinkage 聚合器，
检验去污染边际（d~_pair − raw）的符号是否跨聚合器稳健。

口径：h=1、n_train=84（1991-01–2024-12，408 点 OOS）、expanding 递归（预测行 i+1
只用训练行 s ≤ i）、HA 基准 = ev.ha（ha[t]=mean(y[:t])，信息截至 t−1）。
标准化：expanding 窗内（训练行）均值/样本标准差（ddof=1），同一组 mu/sd 同时施加于
训练行与预测行，无前视。截距经 y 中心化（或 sklearn fit_intercept）处理，不受罚。

四类聚合器（λ 网格 {1, 10, 100, 1000}，施加于标准化后的 X）：
  (i)   panel Ridge      ：9 列联合岭回归，闭式解 beta=(X'X+λI)^{-1}X'y；
  (ii)  panel LASSO      ：sklearn Lasso（坐标下降，目标函数 (1/2n)||r||²+α||β||₁，α=λ）；
  (iii) panel ElasticNet ：sklearn ElasticNet（α=λ，l1_ratio=0.5）；
  (iv)  逐测度单变量 Ridge：每列独立单变量岭回归，9 列预测等权平均。

已知口径冲突（如实记录，不改既定网格）：sklearn 的 LASSO/ElasticNet 目标函数含 1/n
归一化，全零解阈值为 α_max = max|X'y_c|/n；本数据 α_max 量级约 1e-3–1e-2，既定网格
{1,...,1000} 全部落在阈值之上 → 全部 408 窗均为全零解，预测退化为截距项（= 训练窗均值，
与 HA 基准几乎逐点相同，R²OOS ≈ 0）。处理方式：既定网格结果原样报告并逐格记录全零解
发生次数；另加一组**明确标记的补充行**（LASSO_rel / ElasticNet_rel，glmnet 惯例的相对
网格 α = rel × α_max(window)，rel ∈ {0.5, 0.1, 0.05, 0.01}，α_max 逐窗仅用训练行计算，
无前视），使 LASSO/ElasticNet 两臂在非退化区域上仍可回答赛马问题。

对照行：幂集 CSR（直读 results/preds_1991_p9.csv 落盘预测重算 R²OOS，锚点断言
raw 1.79 / d~_pair 2.72 / d~_lv 0.73 ± 0.01）。

输出：results/shrinkage_horserace_1991_p9.csv（行 = 聚合器×λ，列 = 三臂 R²OOS 等）。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import argparse

_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=[2001, 1991])
_args0, _ = _ap0.parse_known_args()
START = _args0.start

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = ['x', '--panel9']
import run_evidence as ev  # noqa: E402

from sklearn.linear_model import ElasticNet, Lasso  # noqa: E402

N_TRAIN = ev.STARTS[START]
y, ha = ev.y, ev.ha
y_true, base = y[N_TRAIN:], ha[N_TRAIN:]
ARMS = {k: ev.variants[k] for k in ('raw', 'd~_pair', 'd~_lv')}
LAMBDAS = [1, 10, 100, 1000]
REL_GRID = [0.5, 0.1, 0.05, 0.01]  # 补充行：glmnet 惯例相对网格（相对逐窗 α_max）


# ---------------- 聚合器引擎（全部 expanding：预测行 i+1，训练行 s ≤ i） ----------------
def _std_params(Xtr):
    """expanding 窗内标准化参数（样本标准差 ddof=1；零方差列回退为 1）。"""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return mu, sd


def panel_ridge_preds(M, lam):
    out = np.empty(len(y) - N_TRAIN)
    for j, i in enumerate(range(N_TRAIN - 1, len(y) - 1)):
        Xtr, ytr = M[:i + 1], y[:i + 1]
        mu, sd = _std_params(Xtr)
        Xs = (Xtr - mu) / sd
        yc = ytr - ytr.mean()
        beta = np.linalg.solve(Xs.T @ Xs + lam * np.eye(M.shape[1]), Xs.T @ yc)
        out[j] = ytr.mean() + ((M[i + 1] - mu) / sd) @ beta
    return out


def panel_sklearn_preds(M, alpha, kind, rel=False):
    """sklearn LASSO/ElasticNet（坐标下降）。rel=True 时 alpha 解释为相对逐窗 α_max 的比例。
    返回 (预测序列, 全零解窗数)。"""
    out = np.empty(len(y) - N_TRAIN)
    zero = 0
    for j, i in enumerate(range(N_TRAIN - 1, len(y) - 1)):
        Xtr, ytr = M[:i + 1], y[:i + 1]
        mu, sd = _std_params(Xtr)
        Xs = (Xtr - mu) / sd
        a = alpha
        if rel:
            a = alpha * np.abs(Xs.T @ (ytr - ytr.mean())).max() / len(ytr)
        mdl = (Lasso(alpha=a, max_iter=20000) if kind == 'lasso'
               else ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000))
        mdl.fit(Xs, ytr)
        if np.all(np.abs(mdl.coef_) < 1e-12):
            zero += 1
        out[j] = mdl.predict(((M[i + 1] - mu) / sd).reshape(1, -1))[0]
    return out, zero


def univar_ridge_preds(M, lam):
    """逐测度单变量岭回归（每列独立标准化+岭），9 列预测等权平均。"""
    out = np.empty(len(y) - N_TRAIN)
    for j, i in enumerate(range(N_TRAIN - 1, len(y) - 1)):
        ytr = y[:i + 1]
        yc = ytr - ytr.mean()
        acc = 0.0
        for c in range(M.shape[1]):
            x = M[:i + 1, c]
            mu, sd = x.mean(), x.std(ddof=1)
            if sd == 0:
                sd = 1.0
            xs = (x - mu) / sd
            b = (xs @ yc) / (xs @ xs + lam)
            acc += ytr.mean() + ((M[i + 1, c] - mu) / sd) * b
        out[j] = acc / M.shape[1]
    return out


# ---------------- 对照行：幂集 CSR（直读落盘预测，锚点断言） ----------------
preds = pd.read_csv(HERE / 'results' / f'preds_{START}_p9.csv')
assert len(preds) == len(y_true)
rows = []
csr_r2 = {}
for arm in ARMS:
    csr_r2[arm] = ev.R2OOS(y_true, preds[arm].values.astype(float), base) * 100
for arm, anchor in (('raw', 1.79 if START == 1991 else 2.71),
                    ('d~_pair', 2.72 if START == 1991 else 3.96),
                    ('d~_lv', 0.73 if START == 1991 else 2.43)):
    assert abs(csr_r2[arm] - anchor) < 0.01, f'CSR 锚点失败: {arm} {csr_r2[arm]:.4f} vs {anchor}'
print(f'对照锚点复核：raw={csr_r2["raw"]:.3f}  d~_pair={csr_r2["d~_pair"]:.3f}  '
      f'd~_lv={csr_r2["d~_lv"]:.3f}（直读 preds_{START}_p9.csv，与 variants_{START}_p9 一致）')
# Clark–West 单边 p（2026-08-27 补，供 Table C2 星标）：ref 行由落盘预测序列（round(6)）现算，
# 并与 variants_{START}_p9.csv 的 CW_p（内存序列现算，round(4) 落盘）对齐核验（容差 1e-3）
_vcw = pd.read_csv(HERE / 'results' / f'variants_{START}_p9.csv', index_col='variant')['CW_p']
csr_p = {}
for arm in ARMS:
    _, _p, _ = ev.clark_west_test(y_true, preds[arm].values.astype(float), base)
    csr_p[arm] = _p
    assert abs(round(_p, 4) - _vcw[arm]) < 1e-3, f'CSR CW_p 对齐失败: {arm} {_p:.4f} vs {_vcw[arm]}'
print(f'对照 CW_p 对齐 variants_{START}_p9：raw={csr_p["raw"]:.4f}  d~_pair={csr_p["d~_pair"]:.4f}  '
      f'd~_lv={csr_p["d~_lv"]:.4f}')
rows.append({'aggregator': 'powerset CSR (ref)', 'lambda': np.nan,
             **csr_r2, **{f'{a}_p': csr_p[a] for a in ARMS},
             'gain_pair-raw': csr_r2['d~_pair'] - csr_r2['raw'], 'zero_sol': np.nan})

# ---------------- 赛马主循环 ----------------
jobs = ([('panel Ridge', lam, ('ridge', lam, False)) for lam in LAMBDAS] +
        [('panel LASSO', lam, ('lasso', lam, False)) for lam in LAMBDAS] +
        [('panel ElasticNet', lam, ('enet', lam, False)) for lam in LAMBDAS] +
        [('panel LASSO_rel', rel, ('lasso', rel, True)) for rel in REL_GRID] +
        [('panel ElasticNet_rel', rel, ('enet', rel, True)) for rel in REL_GRID] +
        [('univar Ridge', lam, ('univ', lam, False)) for lam in LAMBDAS])

for agg_name, lam_tag, (kind, par, rel) in jobs:
    row = {'aggregator': agg_name, 'lambda': lam_tag}
    zero_total = 0
    for arm, M in ARMS.items():
        if kind == 'ridge':
            pr = panel_ridge_preds(M, par)
        elif kind == 'univ':
            pr = univar_ridge_preds(M, par)
        else:
            pr, z = panel_sklearn_preds(M, par, kind, rel=rel)
            zero_total += z
        row[arm] = ev.R2OOS(y_true, pr, base) * 100
        # Clark–West 单边 p（h=1 非重叠，默认自动 NW lag；退化 ZERO-SOL 行同样计算，
        # 其 p 仅为数值填充、渲染层已压表注不展示）
        row[f'{arm}_p'] = ev.clark_west_test(y_true, pr, base)[1]
    row['gain_pair-raw'] = row['d~_pair'] - row['raw']
    row['zero_sol'] = zero_total if kind in ('lasso', 'enet') else np.nan
    rows.append(row)
    zt = f'  全零解窗数(3臂合计/1224)={zero_total}' if kind in ('lasso', 'enet') else ''
    print(f'[{agg_name:20s} λ={lam_tag!s:6}] raw={row["raw"]:7.3f}  d~_pair={row["d~_pair"]:7.3f}  '
          f'd~_lv={row["d~_lv"]:7.3f}  gain(d~_pair−raw)={row["gain_pair-raw"]:+7.3f}{zt}')

tab = pd.DataFrame(rows)
tab_out = tab.round(4)
# CW p 列保留 6 位小数（星标按 clark_west_test 的 ≤ 阈值从 p 派生，4 位舍入在阈值
# 邻域（如实测 1991 口径 LASSO_rel λ=0.1 的 raw_p=0.0100）会模糊 */**/*** 判定）；旧列仍 4 位不变
tab_out[[f'{a}_p' for a in ARMS]] = tab[[f'{a}_p' for a in ARMS]].round(6)
tab_out.to_csv(HERE / 'results' / f'shrinkage_horserace_{START}_p9.csv', index=False)

# ---------------- 判读 ----------------
print('\n===== 判读 =====')
for agg in tab['aggregator'].unique():
    sub = tab[tab['aggregator'] == agg]
    signs = np.sign(sub['gain_pair-raw'].values)
    n_pos = int((sub['gain_pair-raw'] > 0).sum())
    print(f'  {agg:22s}: d~_pair−raw 为正 {n_pos}/{len(sub)} 格'
          f'（范围 {sub["gain_pair-raw"].min():+.3f} – {sub["gain_pair-raw"].max():+.3f}）')
best = tab.loc[tab[ARMS.keys()].max(axis=1).idxmax()]
print(f'\n全表最优格: {best["aggregator"]} λ={best["lambda"]} '
      f'({max(best["raw"], best["d~_pair"], best["d~_lv"]):.3f})；'
      f'CSR 对照 d~_pair={csr_r2["d~_pair"]:.3f} → '
      f'CSR {"仍为全表最优" if csr_r2["d~_pair"] >= tab[ARMS.keys()].max().max() - 1e-9 else "非全表最优"}')
print('落盘 results/shrinkage_horserace_1991_p9.csv')
