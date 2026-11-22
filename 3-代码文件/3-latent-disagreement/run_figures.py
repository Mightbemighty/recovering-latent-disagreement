# -*- coding: utf-8 -*-
"""
Phase B-3：论文图表首批产出 + 主线分窗指标（--start 参数化，默认 1991 主线：N_TRAIN=84，OOS=408）
=================================================
口径：自算 HA。所有图英文标注、朴素学术风、savefig.dpi=1000，PDF 使用嵌入式 TrueType 字体，PDF+PNG 双格式，输出到 figures/。

图清单：
1. fig_corr_heatmap   — dis 测度相关矩阵热图（21 面板 SPF13/SCA6/AF2 分组+分隔线，标注中位数；
                        与起点无关，文件名不带年份标签）
2. fig_pca_scree      — PCA 累计方差曲线（dis/lv/all），与起点无关，文件名不带年份标签
3. fig_cumsse         — Goyal-Welch 累计平方误差差曲线（主变体 vs raw，NBER 衰退阴影）；
                        与起点相关，1991 起点文件名加 _1991
4. fig_rec_exp_bars   — REC/EXP 分组 R²OOS 条形图（raw / d~_u14 / d~_lv / d~_pair）；
                        与起点相关，1991 起点文件名加 _1991
表：results/segment_{start}.csv — raw 与主变体（s0 与 mean3）分窗 R²OOS

--panel9（2026-08-20）：读 preds/variants/stylized_corr_matrix 的 _p9 版本，主变体
d~_pair（d~_lv 保留对照，条形图与分窗表均含两者），热图 9 测度全属 SPF 族无分组分隔线；
输出 figures/*_p9.{pdf,png} 与 results/segment_{start}_p9.csv，不覆写 21 面板图。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
STAG = '' if START == 2001 else f'_{START}'  # 与起点相关的图/表：2001 起点沿用旧文件名（回归兼容）
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
RES = HERE / 'results'
P9 = ev.PANEL9
PTAG = ev.PTAG
MAIN = 'd~_pair' if P9 else 'd~_lv'  # 9 面板主变体 = d~_pair（d~_lv 退居对照）

plt.rcParams.update({
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'figure.dpi': 200, 'savefig.dpi': 1000,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'legend.frameon': False, 'font.family': 'serif',
    'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
})

y = ev.y
dates = ev.dates
ha = ev.ha
n_oos = len(y) - N_TRAIN
y_oos = y[N_TRAIN:]
ha_oos = ha[N_TRAIN:]
dates_oos = dates[N_TRAIN:]

preds = pd.read_csv(RES / f'preds_{START}{PTAG}.csv', parse_dates=['date'])
assert (preds['date'].values == dates_oos.values).all()
pr_main_s0 = preds[MAIN].values.astype(float)
pr_raw_s0 = preds['raw'].values.astype(float)

# mean3 序列：重算集成（s0 与落盘值核对；--panel9 幂集口径下 s0=mean3）
prs_main = ev.subset_ensemble(ev.variants[MAIN], N_TRAIN)
prs_raw = ev.subset_ensemble(ev.D, N_TRAIN)
# preds_{start}{PTAG}.csv 落盘时 round(6)，容差取 6e-7
assert np.abs(prs_main[0] - pr_main_s0).max() < 6e-7 and np.abs(prs_raw[0] - pr_raw_s0).max() < 6e-7
pr_main_m3 = np.mean(prs_main, axis=0)
pr_raw_m3 = np.mean(prs_raw, axis=0)
# 9 面板对照序列：d~_lv（分窗表与条形图保留对照）
prs_lv = ev.subset_ensemble(ev.variants['d~_lv'], N_TRAIN) if P9 else prs_main
print(f's0 与 preds_{START}{PTAG}.csv 核对一致')


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved figures/{name}.pdf/.png')


# ---------------- 1. 相关矩阵热图（21 面板：SPF13/SCA6/AF2 分组；9 面板：全 SPF 族无分组） ----------------
C = pd.read_csv(RES / f'stylized_corr_matrix{PTAG}.csv', index_col=0)
order = [d[2:] for _, d in ev.pairs] if P9 else ev.SPF + ev.SCA + ev.AF
assert set(order) == set(C.columns)
Cm = C.loc[order, order].values
n_m = len(order)
iu = np.triu_indices(n_m, 1)
med = np.median(Cm[iu])
fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(Cm, cmap='Greys', vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
ax.set_xticklabels(order, rotation=90, fontsize=7)
ax.set_yticklabels(order, fontsize=7)
if P9:
    # 2026-09 期刊可读性修订：图内标题/说明行删除（由 caption 承载），median 保留为图内标注
    ax.set_ylabel('Disagreement measure', fontsize=8)
    ax.text(0.995, 1.02, f'median = {med:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8)
else:
    for b in (12.5, 18.5):  # SPF|SCA|AF 分隔线
        ax.axhline(b, color='k', lw=1.0); ax.axvline(b, color='k', lw=1.0)
    ax.text(6, -1.6, 'SPF', ha='center', fontsize=8, fontweight='bold')
    ax.text(15.5, -1.6, 'SCA', ha='center', fontsize=8, fontweight='bold')
    ax.text(19.5, -1.6, 'AF', ha='center', fontsize=8, fontweight='bold')
    ax.set_title(f'Pairwise correlations of {n_m} disagreement measures (median = {med:.3f})', fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
save(fig, f'fig_corr_heatmap{PTAG}')

# ---------------- 2. PCA 累计方差曲线 ----------------
n_p = ev.n_p  # 21（默认）或 9（--panel9）
sets = {f'dis{n_p}': ev.D, f'lv{n_p}': ev.L, f'all{2 * n_p}': np.column_stack([ev.D, ev.L])}
# 2026-09 修订：p9 图例改读者可读命名（21 面板沿用旧短标签）
LEG9 = {f'dis{n_p}': f'Disagreement ({n_p})', f'lv{n_p}': f'Level ({n_p})',
        f'all{2 * n_p}': f'Pooled ({2 * n_p})'}
fig, ax = plt.subplots(figsize=(5.6, 3.6))
style_list = [('k', '-', 'o'), ('0.45', '-', 's'), ('0.55', '--', '^')]  # dis / lv / all
for (tag, X), (c, ls, mk) in zip(sets.items(), style_list):
    Cx = np.corrcoef(X.T)
    evv = np.linalg.eigvalsh(Cx)[::-1]
    share = evv / evv.sum()
    cum = np.cumsum(share)
    ax.plot(range(1, len(cum) + 1), cum * 100, color=c, ls=ls, marker=mk, ms=3, lw=1.2,
            label=LEG9[tag] if P9 else tag)
    if tag == f'dis{n_p}':
        ax.annotate(f'PC1 = {share[0]*100:.1f}%', xy=(1, share[0]*100), xytext=(3.0, 40),
                    fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6, color='0.3'))
        n80 = int(np.searchsorted(cum, 0.80) + 1)
        ax.annotate(f'{n80} PCs for 80%', xy=(n80, 80), xytext=(n80 + 1.5, 66),
                    fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6, color='0.3'))
ax.axhline(80, color='0.6', lw=0.7, ls=':')
ax.set_xlabel('Number of principal components')
ax.set_ylabel('Cumulative variance share (%)')
ax.set_xticks(list(range(1, 19, 2)) if P9 else [1, 3, 6, 9, 12, 15, 18, 21])  # p9：均匀刻度，覆盖 all18 曲线全程
ax.legend(loc='lower right', fontsize=8)
save(fig, f'fig_pca_scree{PTAG}')

# ---------------- 3. 累计平方误差差曲线（Goyal-Welch） ----------------
usrec = pd.read_csv(ev.SCNN / 'USREC.csv').iloc[-n_oos:, 0].values
d_main = (y_oos - ha_oos) ** 2 - (y_oos - pr_main_s0) ** 2
d_raw = (y_oos - ha_oos) ** 2 - (y_oos - pr_raw_s0) ** 2
dd = np.asarray(dates_oos)  # pandas 2.x Series 与 matplotlib 3.5 _check_1d 不兼容，绘图一律转 numpy
fig, ax = plt.subplots(figsize=(6.4, 3.4))
# 衰退阴影
in_rec = False
for i, r in enumerate(usrec):
    if r == 1 and not in_rec:
        start = i; in_rec = True
    if in_rec and (r == 0 or i == len(usrec) - 1):
        ax.axvspan(dd[start], dd[i], color='0.85', zorder=0)
        in_rec = False
main_label = r'$\tilde{d}_{\mathrm{pair}}$ ensemble' if P9 else r'$\tilde{d}_{lv}$ ensemble'
ax.plot(dd, np.cumsum(d_main), 'k-', lw=1.3, label=main_label)
ax.plot(dd, np.cumsum(d_raw), color='0.45', ls='--', lw=1.1, label='Raw ensemble' if P9 else 'raw ensemble')
ax.axhline(0, color='0.6', lw=0.7)
if P9:
    # 2026-09 修订：HA 展开为 historical average；图内标题删除（由 caption 承载）
    ax.set_ylabel(r'Cumulative $\Delta$SSE vs historical average')
else:
    ax.set_ylabel(r'Cumulative $\Delta$SSE vs HA')
    ax.set_title(f'Cumulative squared-error difference, OOS {START}-01 to 2024-12 (shaded: NBER recessions)',
                 fontsize=9)
ax.legend(loc='upper left', fontsize=8)
save(fig, f'fig_cumsse{STAG}{PTAG}')

# ---------------- 4. REC/EXP 分组条形图 ----------------
res = pd.read_csv(RES / f'variants_{START}{PTAG}.csv', index_col=0)
groups = ['raw', 'd~_u14', 'd~_lv', 'd~_pair'] if P9 else ['raw', 'd~_u14', 'd~_lv']
labels = {'raw': 'Raw', 'd~_u14': r'$\tilde{d}_{\mathrm{u14}}$', 'd~_lv': r'$\tilde{d}_{\mathrm{lv}}$',
          'd~_pair': r'$\tilde{d}_{\mathrm{pair}}$'}
rec = [res.loc[g, 'REC'] for g in groups]
expv = [res.loc[g, 'EXP'] for g in groups]
x = np.arange(len(groups)); w = 0.34
fig, ax = plt.subplots(figsize=(5.4 if P9 else 4.6, 3.2))
b1 = ax.bar(x - w / 2, rec, w, color='0.25', label='Recession')
b2 = ax.bar(x + w / 2, expv, w, color='0.75', label='Expansion')
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + (0.35 if v >= 0 else -0.75),
                f'{v:+.2f}', ha='center', fontsize=7.5)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([labels[g] for g in groups])
ax.set_ylabel(r'$R^2_{OOS}$ (%)')
ax.set_ylim(min(expv) - 2, max(rec) + 4.5)
ax.legend(fontsize=8, loc='upper center', ncol=2)
save(fig, f'fig_rec_exp_bars{STAG}{PTAG}')

# ---------------- 5. 分窗指标 ----------------
dt = dates_oos
mask = {
    'full': np.ones(n_oos, bool),
    'ex-2009': dt.dt.year != 2009,
    'ex-2008-09': ~dt.dt.year.isin([2008, 2009]),
    f'{START}-2018': dt.dt.year <= 2018,
    '2019-2024': dt.dt.year >= 2019,
}
series = {'raw_s0': pr_raw_s0, 'raw_mean3': pr_raw_m3,
          f'{MAIN}_s0': pr_main_s0, f'{MAIN}_mean3': pr_main_m3}
if P9:  # 9 面板：d~_lv 对照行（幂集口径确定性，s0=mean3）
    series['d~_lv_s0'] = prs_lv[0]
    series['d~_lv_mean3'] = np.mean(prs_lv, axis=0)
rows = []
for sname, pr in series.items():
    row = {'series': sname}
    for wname, m in mask.items():
        row[wname] = ev.R2OOS(y_oos[m], pr[m], ha_oos[m]) * 100
        row[f'n_{wname}'] = int(m.sum())
        if P9:
            # Clark–West 单边 p（窗 mask/切片子序列；h=1 非重叠，默认自动 NW lag；2026-08-27 补，
            # 供 Table A2 segment 块星标；ex-2009/ex-2008-09 为非连续 mask，先例 =
            # MetricsDef.clark_west_test_by_USREC）
            row[f'{wname}_p'] = ev.clark_west_test(y_oos[m], pr[m], ha_oos[m])[1]
    rows.append(row)
seg = pd.DataFrame(rows).set_index('series')
seg.round(4).to_csv(RES / f'segment_{START}{PTAG}.csv')
print(f'\n===== segment_{START}{PTAG}.csv =====')
print(seg.round(3).to_string())

# cumSSE 衰退/扩张段斜率（NOTES 判读用）
for nm, d in ((MAIN, d_main), ('raw', d_raw)):
    r = d[usrec == 1].sum() / (usrec == 1).sum() * 1e4
    e = d[usrec == 0].sum() / (usrec == 0).sum() * 1e4
    print(f'cumSSE 月均增量({nm}): REC {r:+.2f}e-4 / EXP {e:+.2f}e-4')
print(f'\n完成。figures/ 4 张图（PDF+PNG，后缀 "{STAG}{PTAG}"）+ results/segment_{START}{PTAG}.csv')
