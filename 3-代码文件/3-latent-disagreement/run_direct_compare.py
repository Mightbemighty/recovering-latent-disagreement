# -*- coding: utf-8 -*-
"""
d~ 集成 vs raw 集成的直接比较（--start 参数化，默认 1991 主线：OOS 408 月，自算 HA）
==================================================================
回答："去污染相对 raw 的增益是否显著、何时显著"。

1. Diebold-Mariano：损失差 d_t = (y-pred_raw)^2 - (y-pred_d~)^2（正值 = d~ 更好），
   mean(d) 的 Newey-West(lag=6) t 检验，双侧 p。s0 与 mean3 各一次（全枚举口径下二者同值）。
2. 状态分组 DM：USREC=1 / =0 子样本分别做（小样本 NW lag=3）。
3. figures/fig_cumsse_relative：cumsum(d_t)（全枚举序列）+ NBER 衰退阴影 + 零线。
4. figures/fig_rolling_r2oos：60 个月滚动窗 R²OOS（对同期 HA），raw vs d~（全枚举序列）。

--panel9（2026-08-19 面板重构，thoughts/17 执行清单第 6 步）：输入改 preds_{start}_p9.csv
（幂集 CSR 口径），对比对扩展为主变体 d~_pair 与 raw / d~_lv / lv_only 的全两两 DM
（检验 H2 翻转与 EXP 修复签名迁移）；图与摘要的主对比 = d~_pair vs raw（21 面板 d~_lv
vs raw 的对应位）。输出一律带 _p9 后缀，不覆写 21 面板文件。
起点参数化（2026-08-20）：--start ∈ {2001, 1991}（默认 1991 主线）；输出 csv 带
_{start} 标签；图文件 2001 起点保持旧名、1991 起点加 _1991。

输出：results/direct_compare_{start}{_p9}.csv + 两张图。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
STAG = '' if START == 2001 else f'_{START}'  # 图文件：2001 起点沿用旧名（回归兼容）
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
RES = HERE / 'results'

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

preds = pd.read_csv(RES / f'preds_{START}{ev.PTAG}.csv', parse_dates=['date'])
assert (preds['date'].values == dates_oos.values).all()

# 对比对（d_t = e_a² − e_b²，正 mean_d = b 更好）。21 面板：主线问题 = 去污染 d~_lv vs raw；
# --panel9：主变体 d~_pair 与 raw/d~_lv/lv_only 全两两（H2 翻转 + EXP 签名迁移检验）。
if ev.PANEL9:
    _pool = ['raw', 'd~_pair', 'd~_lv', 'lv_only']
    PAIRS = [(a, b) for i, a in enumerate(_pool) for b in _pool[i + 1:]]
    MAIN_PAIR = ('raw', 'd~_pair')   # 图与摘要的主对比（21 面板 d~_lv vs raw 的对应位）
else:
    PAIRS = [('raw', 'd~_lv')]
    MAIN_PAIR = ('raw', 'd~_lv')
TEX = {'raw': 'raw', 'd~_lv': r'$\tilde{d}_{\mathrm{lv}}$', 'd~_pair': r'$\tilde{d}_{\mathrm{pair}}$',
       'lv_only': 'lv only'}
VARS = sorted({nm for pair in PAIRS for nm in pair})

pr = {f'{nm}_s0': preds[nm].values.astype(float) for nm in VARS}
# mean3 序列（与 run_figures 同法；s0 与落盘核对，容差 6e-7 = 落盘 round(6)；全枚举/幂集口径下 s0=mean3）
for nm in VARS:
    prs_nm = ev.subset_ensemble(ev.variants[nm], N_TRAIN)
    assert np.abs(prs_nm[0] - pr[f'{nm}_s0']).max() < 6e-7, f'{nm} 重算与落盘列不一致'
    pr[f'{nm}_mean3'] = np.mean(prs_nm, axis=0)

usrec = pd.read_csv(ev.SCNN / 'USREC.csv').iloc[-n_oos:, 0].values
print(f'OOS {n_oos} 月: REC {int(usrec.sum())} 月, EXP {int((usrec == 0).sum())} 月')


def dm_test(d, lag):
    """mean(d) 的 NW t 检验；返回 mean, stat, 双侧 p。"""
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    from scipy.stats import norm
    stat = res.params[0] / res.bse[0]
    p = 2 * (1 - norm.cdf(abs(stat)))
    return d.mean(), stat, p


# ---------------- 1+2. DM 检验（全体 / REC / EXP × s0 / mean3 × 对比对） ----------------
rows = []
for a, b in PAIRS:
    for tag in ('s0', 'mean3'):
        d_all = (y_oos - pr[f'{a}_{tag}']) ** 2 - (y_oos - pr[f'{b}_{tag}']) ** 2
        for sub, m, lag in (('full', np.ones(n_oos, bool), 6),
                            ('REC', usrec == 1, 3),
                            ('EXP', usrec == 0, 3)):
            mean_d, stat, p = dm_test(d_all[m], lag)
            mse_ha = np.mean((y_oos[m] - ha_oos[m]) ** 2)
            dr2 = mean_d / mse_ha * 100   # 该子样本内的 ΔR²OOS（pt）
            row = {'series': tag, 'sample': sub, 'n': int(m.sum()), 'NW_lag': lag,
                   'mean_d': mean_d, 'delta_R2OOS_pt': dr2, 'DM_stat': stat, 'p_value': p}
            if ev.PANEL9:
                row = {'pair': f'{b} vs {a}', **row}
            rows.append(row)
            ptag = f'{b} vs {a}  ' if ev.PANEL9 else ''
            print(f'[{ptag}{tag:5s} {sub:4s}] n={int(m.sum()):3d}  mean(d)={mean_d:+.6f} '
                  f'(ΔR²OOS {dr2:+.2f}pt)  DM={stat:+.3f}  p={p:.4f}')
dc = pd.DataFrame(rows)
dc.round(6).to_csv(RES / f'direct_compare_{START}{ev.PTAG}.csv', index=False)

# ---------------- 3. 相对 cumSSE 图（mean3） ----------------
a_main, b_main = MAIN_PAIR
d_m3 = (y_oos - pr[f'{a_main}_mean3']) ** 2 - (y_oos - pr[f'{b_main}_mean3']) ** 2


def shade_rec(ax):
    in_rec = False
    for i, r in enumerate(usrec):
        if r == 1 and not in_rec:
            start = i; in_rec = True
        if in_rec and (r == 0 or i == len(usrec) - 1):
            ax.axvspan(dates_oos.iloc[start], dates_oos.iloc[i], color='0.85', zorder=0)
            in_rec = False


fig, ax = plt.subplots(figsize=(6.4, 3.4))
shade_rec(ax)
ax.plot(np.asarray(dates_oos), np.cumsum(d_m3), 'k-', lw=1.3)  # np.asarray：本机 pandas 拒绝 Series[:, None]（matplotlib 3.5 内部调用），转换后语义不变
ax.axhline(0, color='0.6', lw=0.7)
# 2026-09 修订：图内标题删除（由 caption 承载）；raw 标注为 raw ensemble
ax.set_ylabel('Cumulative SSE (raw minus decontaminated)')
fig.savefig(FIG / f'fig_cumsse_relative{STAG}{ev.PTAG}.pdf', bbox_inches='tight')
fig.savefig(FIG / f'fig_cumsse_relative{STAG}{ev.PTAG}.png', bbox_inches='tight')
plt.close(fig)
print(f'  saved figures/fig_cumsse_relative{STAG}{ev.PTAG}.pdf/.png')

# ---------------- 4. 60 月滚动 R²OOS ----------------
W = 60
roll = {a_main: [], b_main: []}
roll_dates = []
for end in range(W, n_oos + 1):
    sl = slice(end - W, end)
    roll[a_main].append(ev.R2OOS(y_oos[sl], pr[f'{a_main}_mean3'][sl], ha_oos[sl]) * 100)
    roll[b_main].append(ev.R2OOS(y_oos[sl], pr[f'{b_main}_mean3'][sl], ha_oos[sl]) * 100)
    roll_dates.append(dates_oos.iloc[end - 1])
roll_dates = pd.DatetimeIndex(roll_dates)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
in_rec = False
for i, r in enumerate(usrec):
    if r == 1 and not in_rec:
        start = i; in_rec = True
    if in_rec and (r == 0 or i == len(usrec) - 1):
        lo = max(start, W - 1)
        if lo <= i:
            ax.axvspan(dates_oos.iloc[lo], dates_oos.iloc[i], color='0.85', zorder=0)
        in_rec = False
ax.plot(np.asarray(roll_dates), roll[b_main], 'k-', lw=1.3, label=f'{TEX[b_main]} ensemble')  # np.asarray 同上
ax.plot(np.asarray(roll_dates), roll[a_main], color='0.45', ls='--', lw=1.1,
        label='Raw ensemble' if a_main == 'raw' else f'{TEX[a_main]} ensemble')
ax.axhline(0, color='0.6', lw=0.7)
ax.set_ylabel(r'Rolling 60-month $R^2_{OOS}$ (%)')
ax.legend(loc='upper left', fontsize=8)
# 2026-09 修订：图内标题删除（口径信息由 caption 承载）
fig.savefig(FIG / f'fig_rolling_r2oos{STAG}{ev.PTAG}.pdf', bbox_inches='tight')
fig.savefig(FIG / f'fig_rolling_r2oos{STAG}{ev.PTAG}.png', bbox_inches='tight')
plt.close(fig)
print(f'  saved figures/fig_rolling_r2oos{STAG}{ev.PTAG}.pdf/.png')

# ---------------- 曲线形态摘要（NOTES 用） ----------------
cs = np.cumsum(d_m3)
# 2026-09 修订：relative cumSSE 为正月份占比落盘（独立文件，不动 direct_compare csv 的 schema）
share_pos = float((cs > 0).mean())
pd.DataFrame({'pair': [f'{b_main} minus {a_main}'], 'metric': ['cumsse_relative_positive_share'],
              'value': [round(share_pos, 6)], 'n_months': [n_oos]}).to_csv(
    RES / f'cumsse_relative_share_{START}{ev.PTAG}.csv', index=False)
print(f'relative cumSSE > 0 月份占比: {share_pos * 100:.1f}%（n={n_oos}）'
      f' → results/cumsse_relative_share_{START}{ev.PTAG}.csv')
yr = dates_oos.dt.year.values
if ev.PANEL9:
    print(f'主对比对：{b_main} vs {a_main}（正值 = {b_main} 更好）')
for a, b in ((START, 2007), (2008, 2009), (2010, 2019), (2020, 2024)):
    m = (yr >= a) & (yr <= b)
    print(f'cumsum(d) {a}-{b}: {cs[m][-1] - (cs[m][0] - d_m3[m][0]):+.5f}')
print(f'\n完成。results/direct_compare_{START}{ev.PTAG}.csv + 2 张图')
