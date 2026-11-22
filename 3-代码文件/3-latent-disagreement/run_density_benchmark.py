# -*- coding: utf-8 -*-
"""
密度分歧基准检验（回应审稿人 A3 质疑，2026-09）
================================================
质疑：去污染投影 d_k ⊥ l_k 可能把真分歧信号 D_t 一并剥除（假设 proj(D_t|l_k)≈0 未验证）。
回应：用 SPF 个体密度预测（直方图）构造独立分歧基准 G_t = std(个体密度隐含均值)，
检验 (1) corr(d̃_k, G) 保持高且接近 corr(d_raw, G)；(2) corr(d̃_k, l_k) 远低于 corr(d_raw, l_k)。

数据：1-数据文件/SPF_density_forecasts/Individual_PRGDP.xlsx（实际 GDP 年增长直方图）、
      Individual_PRPGDP.xlsx（GDP 价格指数通胀直方图）。
分箱 era 以 spf-documentation.pdf Tables 6/7 为底，**均经数据非空列模式 + 行和≈100 +
概率质量位置（对照当时实际通胀/增长）三重交叉验证**（详见末尾 era 核对记录）：

PRGDP（年增长率 %；variable number = 列序号）：
  1981Q3–1991Q4：6 箱  [6+, 4~5.9, 2~3.9, 0~1.9, -2~-0.1, <-2]，列 1–6 当年、7–12 次年
  1992Q1–2009Q1：10 箱 [6+, 5~5.9, 4~4.9, 3~3.9, 2~2.9, 1~1.9, 0~0.9, -1~-0.1, -2~-1.1, <-2]，
                 列 1–10 当年、11–20 次年
  2009Q2–2020Q1：11 箱 [6+, 5~5.9, …, -3~-2.1, <-3]，列 1–11/12–22/23–33/34–44 四个 horizon
  2020Q2–2024Q1：11 箱 [16+, 10~15.9, 7~9.9, 4~6.9, 2.5~3.9, 1.5~2.4, 0~1.4, -3~-0.1,
                 -6~-3.1, -12~-6.1, <-12]，同样四组 horizon
  2024Q2–至今：  11 箱 [9+, 7~8.9, 5.5~6.9, 4~5.4, 2.5~3.9, 1.5~2.4, 0~1.4, -1.5~-0.1,
                 -3~-1.6, -5.1~-3.1, <-5.1]

PRPGDP（年通胀率 %；**数据修正文档/直觉两处错位**）：
  1981Q3–1991Q4：6 箱  [10+, 8~9.9, 6~7.9, 4~5.9, 2~3.9, <2]，列 1–6 当年、7–12 次年。
                 （pdftotext 的 Table 6 把两个 15 箱文本块排在 1985Q2–1991Q4 /
                 1992Q1–2013Q4 era 名下，但数据上该两块属 1981Q3 之前的名义 GNP era：
                 1981Q3–1991Q4 每期仅列 1–12 非空且前 6 列行和=100；1992Q1 起才切 20 列。）
  1992Q1–2013Q4：10 箱 [8+, 7~7.9, 6~6.9, 5~5.9, 4~4.9, 3~3.9, 2~2.9, 1~1.9, 0~0.9, <0]，
                 列 1–10 当年、11–20 次年（前 10 列行和=100；质量位置与当年通胀吻合）
  2014Q1–至今：  10 箱 [4+, 3.5~3.9, 3~3.4, 2.5~2.9, 2~2.4, 1.5~1.9, 1~1.4, 0.5~0.9,
                 0~0.4, Will decline(<0)]，列 1–10 当年、11–20 次年

开放端箱中点约定：边界 ± W_typ，W_typ = 该 era 封闭箱宽中位数（1 宽箱 era 即"下界+1.0"，
与任务约定一致；2020Q2 era 16+ → 16+3=19）。概率按行归一化（和不总精确为 100）；
个体行某 horizon 块全非空箱 <2 个或概率和 ≤0 剔除；每期有效受访者 <10 该期记 NaN。
问卷时序存疑期（文档：1985Q1、1986Q1 的年度指代无法确认）：两期全臂置 NaN 并在输出声明。
固定事件问题：当年 horizon 密度分歧年内机械收窄，与面板 d（固定 horizon T..T+4 平均）
口径不同 → 报 quarter-of-year 虚拟变量偏相关，并给当年(A)/次年(B)两臂对照。

映射缝隙披露：面板 GDP 题 = 名义 GDP 水平（Dispersion_1 NGDP sheet），PRGDP = 实际产出
增长；面板 CPI 题 ↔ PRPGDP（GDP 价格指数）是不同价格指数。两臂均为近似对应。

用法：python run_density_benchmark.py --panel9
输出：results/density_benchmark_p9.csv + 控制台完整结果表（不改 NOTES.md）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ap = argparse.ArgumentParser()
_ap.add_argument('--panel9', action='store_true')
ARGS = _ap.parse_args()

import run_evidence as ev  # noqa: E402  模块级预解析 --panel9 后加载 9 测度面板

assert ARGS.panel9 and ev.PANEL9 is True, '本检验仅绑定 9 测度主面板：请加 --panel9'

DATA = HERE.parents[1] / '1-数据文件' / 'SPF_density_forecasts'
MIN_RESP = 10          # 每期最少有效受访者
DROP_PERIODS = {(1985, 1), (1986, 1)}  # 文档：年度指代无法确认，全臂置 NaN
NA = None              # 开放端标记

# ---------------- 分箱 era 定义（数据交叉验证后硬编码） ----------------
# bin = (lo, hi)；NA = 开放端。horizons = 该 era 的年度 horizon 组数（列按 horizon 顺序排布）。
PRGDP_ERAS = [
    ('1981Q3', '1991Q4', [(6, NA), (4, 5.9), (2, 3.9), (0, 1.9), (-2, -0.1), (NA, -2)], 2),
    ('1992Q1', '2009Q1', [(6, NA), (5, 5.9), (4, 4.9), (3, 3.9), (2, 2.9), (1, 1.9),
                          (0, 0.9), (-1, -0.1), (-2, -1.1), (NA, -2)], 2),
    ('2009Q2', '2020Q1', [(6, NA), (5, 5.9), (4, 4.9), (3, 3.9), (2, 2.9), (1, 1.9),
                          (0, 0.9), (-1, -0.1), (-2, -1.1), (-3, -2.1), (NA, -3)], 4),
    ('2020Q2', '2024Q1', [(16, NA), (10, 15.9), (7, 9.9), (4, 6.9), (2.5, 3.9), (1.5, 2.4),
                          (0, 1.4), (-3, -0.1), (-6, -3.1), (-12, -6.1), (NA, -12)], 4),
    ('2024Q2', '2100Q4', [(9, NA), (7, 8.9), (5.5, 6.9), (4, 5.4), (2.5, 3.9), (1.5, 2.4),
                          (0, 1.4), (-1.5, -0.1), (-3, -1.6), (-5.1, -3.1), (NA, -5.1)], 4),
]
PRPGDP_ERAS = [
    ('1981Q3', '1991Q4', [(10, NA), (8, 9.9), (6, 7.9), (4, 5.9), (2, 3.9), (NA, 2)], 2),
    ('1992Q1', '2013Q4', [(8, NA), (7, 7.9), (6, 6.9), (5, 5.9), (4, 4.9), (3, 3.9),
                          (2, 2.9), (1, 1.9), (0, 0.9), (NA, 0)], 2),
    ('2014Q1', '2100Q4', [(4, NA), (3.5, 3.9), (3, 3.4), (2.5, 2.9), (2, 2.4), (1.5, 1.9),
                          (1, 1.4), (0.5, 0.9), (0, 0.4), (NA, 0)], 2),
]


def era_key(y, q):
    return y * 10 + q


def bin_midpoints(bins):
    """封闭箱取 (lo+hi)/2；开放端 = 边界 ± W_typ（W_typ = 封闭箱宽中位数）。"""
    widths = [hi - lo for lo, hi in bins if lo is not NA and hi is not NA]
    w_typ = float(np.median(widths))
    mids = []
    for lo, hi in bins:
        if lo is NA:
            mids.append(hi - w_typ)
        elif hi is NA:
            mids.append(lo + w_typ)
        else:
            mids.append((lo + hi) / 2.0)
    return np.array(mids)


def find_era(eras, y, q):
    k = era_key(y, q)
    for lo_q, hi_q, bins, nh in eras:
        ly, lq = int(lo_q[:4]), int(lo_q[-1])
        hy, hq = int(hi_q[:4]), int(hi_q[-1])
        if era_key(ly, lq) <= k <= era_key(hy, hq):
            return bins, nh
    return None


def block_moments(prob_block, mids):
    """单期单 horizon：个体密度隐含均值/方差 → G = std(μ_i) + LS 分解描述量。
    prob_block: n×nbins（含 NaN 列/行）；返回 dict 或 None（有效受访者不足）。"""
    mus, vs, probs = [], [], []
    for row in prob_block:
        m = ~np.isnan(row)
        if m.sum() < 2 or row[m].sum() <= 0:
            continue
        p = row[m] / row[m].sum()
        mid = mids[m]
        mu = float(p @ mid)
        mus.append(mu)
        vs.append(float(p @ (mid ** 2) - mu ** 2))
        full = np.zeros(len(row))
        full[m] = p
        probs.append(full)
    n = len(mus)
    if n < MIN_RESP:
        return None
    mus, vs = np.array(mus), np.array(vs)
    pooled_p = np.mean(probs, axis=0)
    pooled_mu = float(pooled_p @ mids)
    agg_unc = float(pooled_p @ (mids ** 2) - pooled_mu ** 2)  # var(总直方图)
    dis = float(mus.var(ddof=1))                              # disagreement = var(个体均值)
    return {'n': n, 'G': float(mus.std(ddof=1)), 'dis': dis,
            'mean_ind_var': float(vs.mean()), 'agg_unc': agg_unc,
            'share': dis / agg_unc if agg_unc > 0 else np.nan}


def build_density_g(fname, sheet, prefix, eras):
    """逐期逐 horizon 构造 G 基准；返回 (YEAR, QUARTER) 索引的 DataFrame。"""
    df = pd.read_excel(DATA / fname, sheet_name=sheet)
    pcols = [c for c in df.columns if c.startswith(prefix)]
    rows = {}
    for (y, q), sub in df.groupby(['YEAR', 'QUARTER']):
        spec = find_era(eras, int(y), int(q))
        if spec is None:
            continue
        bins, nh = spec
        nb = len(bins)
        mids = bin_midpoints(bins)
        rec = {}
        for h, tag in enumerate(['A', 'B']):      # A = 当年，B = 次年
            if h >= nh:
                continue
            blk = sub[pcols[h * nb:(h + 1) * nb]].values.astype(float)
            r = block_moments(blk, mids)
            if r is None:
                rec.update({f'n_{tag}': int((~np.isnan(blk).all(axis=1)).sum()),
                            f'G_{tag}': np.nan, f'share_{tag}': np.nan})
            else:
                rec.update({f'n_{tag}': r['n'], f'G_{tag}': r['G'], f'share_{tag}': r['share'],
                            f'dis_{tag}': r['dis'], f'mean_ind_var_{tag}': r['mean_ind_var'],
                            f'agg_unc_{tag}': r['agg_unc']})
        rows[(int(y), int(q))] = rec
    out = pd.DataFrame(rows).T
    out.index.names = ['year', 'quarter']
    for p in DROP_PERIODS:
        if p in out.index:
            gcols = [c for c in out.columns if c.startswith(('G_', 'share_', 'dis_'))]
            out.loc[p, gcols] = np.nan
    return out.sort_index()


# ---------------- 面板对接（复用 run_evidence 数据与去污染函数） ----------------
names = [d[2:] for _, d in ev.pairs]          # 字母序：CON, CPI, CPROF, GDP, HSG, INV, IP, TBL, UEP
IDX = {nm: names.index(nm) for nm in names}
D_PAIR = np.column_stack([ev.decontaminate(ev.D[:, [k]], ev.L[:, [k]])[:, 0]
                          for k in range(ev.n_p)])   # 主变体：同题投影残差
D_LV = ev.decontaminate(ev.D, ev.L)                  # 对照：全族投影残差

dates = ev.dates
qe_mask = dates.dt.month.isin([3, 6, 9, 12])         # 季度末月 ↔ survey 季度（季末对齐口径）
pnl = pd.DataFrame({
    'year': dates[qe_mask].dt.year.values,
    'quarter': dates[qe_mask].dt.quarter.values,
})
for nm in ['GDP', 'CPI']:
    k = IDX[nm]
    pnl[f'l_{nm}'] = ev.L[qe_mask.values, k]
    pnl[f'd_raw_{nm}'] = ev.D[qe_mask.values, k]
    pnl[f'd_pair_{nm}'] = D_PAIR[qe_mask.values, k]
    pnl[f'd_lv_{nm}'] = D_LV[qe_mask.values, k]
pnl = pnl.set_index(['year', 'quarter'])

# ---------------- 构造 G 并合并 ----------------
G_RGDP = build_density_g('Individual_PRGDP.xlsx', 'PRGDP', 'PRGDP', PRGDP_ERAS)
G_PGDP = build_density_g('Individual_PRPGDP.xlsx', 'PRPGDP', 'PRPGDP', PRPGDP_ERAS)
G_RGDP.columns = [f'PRGDP_{c}' for c in G_RGDP.columns]
G_PGDP.columns = [f'PRPGDP_{c}' for c in G_PGDP.columns]

merged = pnl.join(G_RGDP).join(G_PGDP)
merged = merged.loc[(slice(1984, 2024), slice(None)), :]

# ---------------- 检验统计 ----------------
ARMS = [('GDP', 'PRGDP'), ('CPI', 'PRPGDP')]


def corr_report(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())
    r, p = sstats.pearsonr(x[m], y[m])
    return r, p, int(m.sum())


def partial_corr_qoy(x, y, qtr):
    """quarter-of-year 虚拟变量偏相关：两边各自对 QOY dummies 取残差后相关。"""
    m = ~(np.isnan(x) | np.isnan(y))
    xm, ym, qm = x[m], y[m], qtr[m].astype(int)
    Qd = np.zeros((len(qm), 4))
    Qd[np.arange(len(qm)), qm - 1] = 1.0
    rx = xm - Qd @ np.linalg.lstsq(Qd, xm, rcond=None)[0]
    ry = ym - Qd @ np.linalg.lstsq(Qd, ym, rcond=None)[0]
    r, p = sstats.pearsonr(rx, ry)
    return r, p, int(m.sum())


def ols_report(yv, X):
    m = ~np.isnan(yv) & ~np.isnan(X).any(axis=1)
    ym, Xm = yv[m], np.column_stack([np.ones(m.sum()), X[m]])
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    resid = ym - Xm @ beta
    dof = len(ym) - Xm.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
    tss = ((ym - ym.mean()) ** 2).sum()
    return beta, beta / se, 1 - resid @ resid / tss, int(m.sum())


qtr_arr = merged.index.get_level_values('quarter').values
half_id = np.where(merged.index.get_level_values('year') <= 2004, 'H1(≤2004)', 'H2(≥2005)')
out_csv = HERE / 'results' / 'density_benchmark_p9.csv'  # 落盘在 PRCCPI 诊断列并入之后

print('=' * 96)
print('密度分歧基准检验（9 面板，季度 1984Q1–2024Q4；问卷存疑期 1985Q1/1986Q1 已置 NaN）')
print('面板列序:', names)
print('有效受访者数范围: PRGDP n=%d–%d / PRPGDP n=%d–%d（<10 的期已置 NaN）'
      % (np.nanmin(merged['PRGDP_n_A']), np.nanmax(merged['PRGDP_n_A']),
         np.nanmin(merged['PRPGDP_n_A']), np.nanmax(merged['PRPGDP_n_A'])))
print('输出:', out_csv)

for pv, gv in ARMS:
    for arm in ['A', 'B']:
        G = merged[f'{gv}_G_{arm}'].values.astype(float)
        d_raw = merged[f'd_raw_{pv}'].values.astype(float)
        d_pair = merged[f'd_pair_{pv}'].values.astype(float)
        d_lv = merged[f'd_lv_{pv}'].values.astype(float)
        lv = merged[f'l_{pv}'].values.astype(float)
        share = merged[f'{gv}_share_{arm}'].values.astype(float)
        print('\n' + '-' * 96)
        print(f'【{pv} ↔ {gv}，horizon 臂 {arm}（{"当年" if arm == "A" else "次年"}）】  '
              f'LS 分解 disagreement 份额中位数 = {np.nanmedian(share):.3f}')
        rows = []
        for lab, sub in [('全样本', np.ones(len(G), bool)),
                         ('前半', half_id == 'H1(≤2004)'), ('后半', half_id == 'H2(≥2005)')]:
            r1 = corr_report(d_raw[sub], G[sub])
            r2 = corr_report(d_pair[sub], G[sub])
            r3 = corr_report(d_lv[sub], G[sub])
            r4 = corr_report(d_raw[sub], lv[sub])
            r5 = corr_report(d_pair[sub], lv[sub])
            pc1 = partial_corr_qoy(d_raw[sub], G[sub], qtr_arr[sub])
            pc2 = partial_corr_qoy(d_pair[sub], G[sub], qtr_arr[sub])
            rows.append((lab, r1, r2, r3, r4, r5, pc1, pc2))
        hdr = f'{"样本":<10}{"corr(d_raw,G)":>16}{"corr(d~pair,G)":>16}{"corr(d~lv,G)":>15}' \
              f'{"corr(d_raw,l)":>16}{"corr(d~pair,l)":>16}{"pcorr(d_raw,G|QOY)":>20}{"pcorr(d~pair,G|QOY)":>21}'
        print(hdr)
        for lab, r1, r2, r3, r4, r5, pc1, pc2 in rows:
            fmt = lambda t: f'{t[0]:>7.3f}(p={t[1]:.3f},n={t[2]})' if not np.isnan(t[0]) else '     NaN'
            print(f'{lab:<10}{fmt(r1):>16}{fmt(r2):>16}{fmt(r3):>15}{fmt(r4):>16}{fmt(r5):>16}'
                  f'{fmt(pc1):>20}{fmt(pc2):>21}')
        # 回归载荷：G ~ [l, d_raw] vs G ~ [l, d~pair]（全样本）
        for dl, dd in [('d_raw', d_raw), ('d_pair', d_pair), ('d_lv', d_lv)]:
            beta, t, r2_, n_ = ols_report(G, np.column_stack([lv, dd]))
            print(f'  G ~ [l, {dl}]: beta_l={beta[1]:.3f}(t={t[1]:.2f})  '
                  f'beta_d={beta[2]:.3f}(t={t[2]:.2f})  R2={r2_:.3f}  n={n_}')

# ---------------- 诊断：post-1992（剔除 6 箱粗 era）+ PRCCPI 交叉核对 ----------------
# PRCCPI（核心 CPI，Q4/Q4 增长）：2007Q1 起，10 箱同 PRPGDP 2014 era 结构，列 1–10/11–20。
PRCCPI_ERAS = [
    ('2007Q1', '2100Q4', [(4, NA), (3.5, 3.9), (3, 3.4), (2.5, 2.9), (2, 2.4), (1.5, 1.9),
                          (1, 1.4), (0.5, 0.9), (0, 0.4), (NA, 0)], 2),
]
G_CCPI = build_density_g('Individual_PRCCPI.xlsx', 'PRCCPI', 'PRCCPI', PRCCPI_ERAS)
G_CCPI.columns = [f'PRCCPI_{c}' for c in G_CCPI.columns]
merged = merged.join(G_CCPI)
merged.reset_index().to_csv(out_csv, index=False)

print('\n' + '-' * 96)
print('【诊断 1：post-1992 子样本（剔除 1984–1991 的 6 箱粗 era，箱粗会衰减相关）】')
sub92 = merged.index.get_level_values('year') >= 1992
for pv, gv in ARMS:
    for arm in ['A', 'B']:
        r1 = corr_report(merged[f'd_raw_{pv}'].values[sub92], merged[f'{gv}_G_{arm}'].values[sub92])
        r2 = corr_report(merged[f'd_pair_{pv}'].values[sub92], merged[f'{gv}_G_{arm}'].values[sub92])
        print(f'  {pv}↔{gv} 臂{arm}: corr(d_raw,G)={r1[0]:.3f}(p={r1[1]:.4f},n={r1[2]})  '
              f'corr(d~pair,G)={r2[0]:.3f}(p={r2[1]:.4f},n={r2[2]})')

print('\n【诊断 2：CPI 臂的 PRCCPI（核心 CPI）交叉核对，2007Q1 起公共子样本】')
sub07 = merged.index.get_level_values('year') >= 2007
for gv in ['PRPGDP', 'PRCCPI']:
    for arm in ['A', 'B']:
        r1 = corr_report(merged['d_raw_CPI'].values[sub07], merged[f'{gv}_G_{arm}'].values[sub07])
        r2 = corr_report(merged['d_pair_CPI'].values[sub07], merged[f'{gv}_G_{arm}'].values[sub07])
        print(f'  CPI↔{gv} 臂{arm}: corr(d_raw,G)={r1[0]:.3f}(p={r1[1]:.4f},n={r1[2]})  '
              f'corr(d~pair,G)={r2[0]:.3f}(p={r2[1]:.4f},n={r2[2]})')

# era 核对记录（数据验证摘要，备查）
print('\n' + '=' * 96)
print('era 核对依据（数据非空模式）：PRGDP 列数边界 1981Q3→12 列(6×2)、1992Q1→20 列(10×2)、'
      '2009Q2→44 列(11×4)；2020Q2/2024Q1/2024Q2 仅箱定义切换（列数不变，依文档 Table 7）。')
print('PRPGDP 1981Q3–1991Q4 每期仅列 1–12 非空且前 6 列行和=100 → 6 箱×2 horizon'
      '（文档 pdftotext 中两个 15 箱文本块实为 1981Q3 之前 era，数据证伪其 1985Q2–1991Q4 归属）；'
      '1992Q1–2013Q4 前 10 列行和=100 → 10 箱 [8+,…,<0]（概率质量位置与当年通胀吻合）。')
