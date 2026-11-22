# -*- coding: utf-8 -*-
"""
9 测度 SPF-only 面板构建（设计文档 v4.1 §3.4 面板选址，2026-08-19）
==================================================================
定稿面板（R1–R5，事前登记 2026-08-19）：{GDP, IP, CON, INV, HSG, UEP, CPI, TBL}
（level 形式）+ {CPROF}（growth 形式，用户裁定：level 形式题内 corr +0.79 系名义规模
机械缩放，growth +0.43 与 UEP 先例相当）。

变换链逐项复刻 notebook 管线（季末口径），以便 CPROF 与存量测度同炉出：
  1.1（lv）：meanLevel_26.xlsx 按列名取 VAR2..VAR6、meanGrowth_26.xlsx 取
      dVAR2..dVAR6（skipna 均值）；
  1.2（dis）：Dispersion_{1,2,3}_26.xlsx，usecols = VAR_D1/D2(T)..(T+4)，skipna 均值；
  对齐：季度 → 季度末月（Huang et al. 2021 保守假设，不做 -1 月偏移）→ resample M ffill；
  nb2：截断 [1983-01, 2024-12]（现行面板因 AF 起点形成的实际截断）→ rolling-72
      （min_periods=12）逐列标准化 → 去前 12 行 → 1984-01 起 492 行网格。

验证（锚点纪律）：同链重建全部 13 个存量 SPF 测度对（含 5 个 growth 形式），
逐列对照 all_data_qe.csv，max|diff| < 1e-8 方许落盘 CPROF。
输出：panel9_qe.csv（date, excess_ret_h1, l_/d_ × 9，492 行）+ 控制台验证表。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / '1-数据文件'
SCNN = HERE.parent / '1-sCNNLSTM-and-meta-RSR-notebooks'
HORIZONS = ['T', 'T+1', 'T+2', 'T+3', 'T+4']
TRUNC_START, TRUNC_END = '1983-01-01', '2024-12-31'   # nb2 实际截断（AF 起点绑定）；去 12 行后 = 1984-01

# 1.2 的 dis 来源映射：短名 -> (文件, sheet, 列前缀, D1/D2)
DIS_LEVEL_SRC = {  # level 形式（D1 列）：GDP/HSG/UEP/CPI/TBL ← D1 文件；IP/CON/INV ← D3 文件
    'GDP': ('Dispersion_1_26.xlsx', 'NGDP', 'NGDP'),
    'IP': ('Dispersion_3_26.xlsx', 'INDPROD', 'INDPROD'),
    'CON': ('Dispersion_3_26.xlsx', 'RCONSUM', 'RCONSUM'),
    'INV': ('Dispersion_3_26.xlsx', 'RNRESIN', 'RNRESIN'),
    'HSG': ('Dispersion_1_26.xlsx', 'HOUSING', 'HOUSING'),
    'UEP': ('Dispersion_1_26.xlsx', 'UNEMP', 'UNEMP'),
    'CPI': ('Dispersion_1_26.xlsx', 'CPI', 'CPI'),
    'TBL': ('Dispersion_1_26.xlsx', 'TBILL', 'TBILL'),
}
DIS_GROWTH_SRC = {  # growth 形式（D2 列，均 D2 文件）
    'GDPg': ('NGDP', 'NGDP'), 'IPg': ('INDPROD', 'INDPROD'), 'CONg': ('RCONSUM', 'RCONSUM'),
    'INVg': ('RNRESIN', 'RNRESIN'), 'HSGg': ('HOUSING', 'HOUSING'), 'CPROF': ('CPROF', 'CPROF'),
}
LV_LEVEL_SHEETS = {'GDP': 'NGDP', 'IP': 'INDPROD', 'CON': 'RCONSUM', 'INV': 'RNRESIN',
                   'HSG': 'HOUSING', 'UEP': 'UNEMP', 'CPI': 'CPI', 'TBL': 'TBILL'}
LV_GROWTH_SHEETS = {'GDPg': 'NGDP', 'IPg': 'INDPROD', 'CONg': 'RCONSUM', 'INVg': 'RNRESIN',
                    'HSGg': 'HOUSING', 'CPROF': 'CPROF'}


def qe_monthly(dates_q, values):
    """季度序列 → 季度末月对齐 → 月度 ffill（1.1/1.2 逐字口径）。"""
    measure = pd.DataFrame({'date': pd.PeriodIndex(dates_q, freq='Q'), 'v': values})
    measure['date'] = measure['date'].dt.to_timestamp().dt.to_period('Q').dt.end_time.dt.to_period('M')
    assert not measure['date'].duplicated().any()
    return measure.set_index('date')['v'].resample('M').ffill()


def read_lv(fname, sheet, kind):
    """1.1：level 取 {SHEET}2..{SHEET}6；growth 取 d{sheet}2..d{sheet}6（skipna 均值）。"""
    if kind == 'level':
        cols = [f'{sheet}{i}' for i in range(2, 7)]
    else:
        cols = [f'd{sheet.lower()}{i}' for i in range(2, 7)]
    df = pd.read_excel(DATA / fname, sheet_name=sheet, usecols=['YEAR', 'QUARTER'] + cols)
    q = df['YEAR'].astype(str) + 'Q' + df['QUARTER'].astype(str)
    return qe_monthly(q, df[cols].mean(axis=1, skipna=True))


def read_dis(fname, sheet, prefix, kind):
    """1.2：usecols = prefix_D{kind}(T..T+4)，skiprows=9，skipna 均值。"""
    cols = ['Survey_Date(T)'] + [f'{prefix}_D{kind}({h})' for h in HORIZONS]
    df = pd.read_excel(DATA / fname, sheet_name=sheet, skiprows=9, usecols=cols,
                       dtype={c: float for c in cols[1:]})
    return qe_monthly(df['Survey_Date(T)'], df.iloc[:, 1:6].mean(axis=1, skipna=True))


def data_std(df):
    """nb2 data_std 逐字：rolling(72, min_periods=12) 逐列 z-score。"""
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)
        rm = df[col].rolling(window=72, min_periods=12).mean()
        rs = df[col].rolling(window=72, min_periods=12).std()
        df[col] = (df[col] - rm) / rs
    return df


# ---------------- 1. 重建全部 13 存量 SPF 测度对 + CPROF ----------------
lv_series, dis_series = {}, {}
for short, sheet in LV_LEVEL_SHEETS.items():
    lv_series[short] = read_lv('meanLevel_26.xlsx', sheet, 'level')
for short, sheet in LV_GROWTH_SHEETS.items():
    lv_series[short] = read_lv('meanGrowth_26.xlsx', sheet, 'growth')
for short, (f, sheet, prefix) in DIS_LEVEL_SRC.items():
    dis_series[short] = read_dis(f, sheet, prefix, 1)
for short, (sheet, prefix) in DIS_GROWTH_SRC.items():
    dis_series[short] = read_dis('Dispersion_2_26.xlsx', sheet, prefix, 2)

names = list(lv_series)
panel = pd.DataFrame({'date': lv_series['GDP'].index.to_timestamp()})
for nm in names:
    panel['l_' + nm] = lv_series[nm].values
    panel['d_' + nm] = dis_series[nm].reindex(lv_series['GDP'].index).values

# ---------------- 2. nb2 口径：截断 → rolling-72 标准化 → 去前 12 行 ----------------
panel = panel[(panel['date'] >= TRUNC_START) & (panel['date'] <= TRUNC_END)].reset_index(drop=True)
panel_raw = panel.copy()   # 标准化前快照（Table 1 原始口径描述统计用，2026-08-26）
panel = data_std(panel).iloc[12:].reset_index(drop=True)
ad = pd.read_csv(SCNN / 'all_data_qe.csv', parse_dates=['date'])
assert len(panel) == len(ad) == 492, f'网格错位: {len(panel)} vs {len(ad)}'
assert (panel['date'].values == ad['date'].values).all(), '日期网格与 all_data_qe 不一致'
assert not panel[[c for c in panel.columns if c != 'date']].isna().any().any(), '测度列存在 NaN'

# ---------------- 3. 锚点验证：13 存量测度对 vs all_data_qe ----------------
print('===== 验证：同链重建 vs all_data_qe.csv（max|diff|） =====')
worst = 0.0
for nm in ['GDP', 'GDPg', 'IP', 'IPg', 'CON', 'CONg', 'INV', 'INVg', 'HSG', 'HSGg',
           'UEP', 'CPI', 'TBL']:
    for pre in ('l_', 'd_'):
        d = np.abs(panel[pre + nm].values - ad[pre + nm].values).max()
        worst = max(worst, d)
        tag = '✓' if d < 1e-8 else '✗'
        print(f'  {pre}{nm:6s} max|diff| = {d:.2e} {tag}')
assert worst < 1e-8, f'复刻失败: worst = {worst}'
print(f'全部 26 列 max|diff| = {worst:.2e} < 1e-8 ✓ —— 变换链复刻成立，CPROF 同链可信')

# ---------------- 4. 落盘 9 测度面板 ----------------
PANEL9 = ['GDP', 'IP', 'CON', 'INV', 'HSG', 'UEP', 'CPI', 'TBL', 'CPROF']
out = pd.DataFrame({'date': panel['date'], 'excess_ret_h1': ad['excess_ret_h1']})
for nm in PANEL9:
    out['l_' + nm] = panel['l_' + nm]
    out['d_' + nm] = panel['d_' + nm]
out.round(10).to_csv(HERE / 'panel9_qe.csv', index=False)
print(f'\n落盘 panel9_qe.csv：{out.shape[0]} 行 × {out.shape[1]} 列（9 测度对 + date + excess_ret_h1）')

# ---------------- 4b. 落盘 9 测度原始面板（标准化前，与 panel9_qe 同 492 行网格） ----------------
src = panel_raw[panel_raw['date'].isin(out['date'])].reset_index(drop=True)
raw9 = src[['date'] + [f'{p}{nm}' for nm in PANEL9 for p in ('l_', 'd_')]].copy()
assert (raw9['date'].values == out['date'].values).all(), 'raw 面板日期网格与 panel9_qe 不一致'
assert not raw9[[c for c in raw9.columns if c != 'date']].isna().any().any(), 'raw 面板存在 NaN'
raw9.round(10).to_csv(HERE / 'panel9_qe_raw.csv', index=False)
print(f'落盘 panel9_qe_raw.csv：{raw9.shape[0]} 行 × {raw9.shape[1]} 列（标准化前，供 Table 1 原始口径描述统计）')

# ---------------- 5. 描述性核查：题内 lv–dis 相关 + 两两相关中位 ----------------
sub = out[(out['date'] >= '1984-01-01') & (out['date'] <= '2022-12-31')]
print('\n===== 9 测度题内 lv–dis 相关（1984–2022，标准化后） =====')
for nm in PANEL9:
    print(f'  {nm:6s} {sub["l_" + nm].corr(sub["d_" + nm]):+.3f}')
Dmat = out[[f'd_{nm}' for nm in PANEL9]].values
C = np.corrcoef(Dmat.T)
pw = C[np.triu_indices(9, 1)]
print(f'\n9 测度两两相关中位（全样本）= {np.median(pw):.3f}'
      f'（预登记期望：低于 SPF13 基线 0.443；8 测度参照 0.327）')
