# -*- coding: utf-8 -*-
"""
议程 6e 面板构建：Livingston 利率分歧面板（2026-08-24）
==========================================================
目的：普适性检验的正面臂——主线信号集中于利率分歧（TBL，H1 单测度 d̃ +4.22；
NOTES §50/§52 变量构成归因），ECB SPF 无利率变量故未检验到机制；Livingston Survey
（费城联储，半年度 6/12 月）自 1992-06 起有 TBILL（3 月期国库券）与 TBOND（10 年期
国债）利率水平的个体预测——同一受访者总体类型（专业预测者）、同一题材（利率）、
不同调查 → 检验"TBL 承载"机制的跨调查普适性。
设计：变量 = TBILL/TBOND × horizon {6M, 12M}（近/中两个 horizon，利率水平预测，
与主线 d_TBL level 形式兼容），4 测度对；lv = 截面均值、dis = 截面 IQR（75th−25th，
对齐主线 D1 口径与官方 Dispersion1 定义）。半年度→月度对齐：6 月调查 → 7 月起、
12 月调查 → 次年 1 月起 ffill 6 个月（发布滞后保守让 1 个月，无前视）。
变换链：rolling-72（min_periods=12）逐列 z-score（复刻 make_panel9）；面板起点
1993-06（1992-06 起 12 行烧入）。锚点核对：个体算 lv/dis vs 官方 means / D1。
输出：`liv4_qe.csv`（月度，1993-06–2024-12，列 l_/d_ × 4 测度）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / '1-数据文件'
MEASURES = [('TBILL', '6M'), ('TBILL', '12M'), ('TBOND', '6M'), ('TBOND', '12M')]

# ---------------- 1. 个体预测 → 每期截面 lv/dis ----------------
xl = pd.ExcelFile(DATA / 'livingston_individualdata.xlsx')
frames = {}
for var, hz in MEASURES:
    df = xl.parse(var)
    df.columns = [str(c).strip() for c in df.columns]
    col = f'{var}_{hz}'
    sub = df[['Date', col]].dropna()
    sub['Date'] = pd.to_datetime(sub['Date'])
    g = sub.groupby('Date')[col]
    lv = g.mean().rename(f'l_{var}{hz}')
    # 官方 D1 口径核验：75th−25th 用 averaged_inverted_cdf 插值（max|Δ|=0.01、corr=1.0000）
    dis = g.apply(lambda s: np.percentile(s, 75, method='averaged_inverted_cdf')
                  - np.percentile(s, 25, method='averaged_inverted_cdf')).rename(f'd_{var}{hz}')
    n = g.size().rename(f'n_{var}{hz}')
    frames[f'{var}{hz}'] = pd.concat([lv, dis, n], axis=1)

panel_h = pd.concat(frames.values(), axis=1)  # 半年度网格（6/12 月）
print(f'半年度网格：{panel_h.index[0]:%Y-%m} .. {panel_h.index[-1]:%Y-%m}，{len(panel_h)} 期')
for m in ['TBILL6M', 'TBILL12M', 'TBOND6M', 'TBOND12M']:
    nn = panel_h[f'n_{m}']
    print(f'  {m}: 首期 {panel_h[f"l_{m}"].first_valid_index():%Y-%m}，每期人数中位 {nn.median():.0f}（min {nn.min():.0f}）')

# ---------------- 2. 锚点核对：个体算 vs 官方 mean / D1 ----------------
means = pd.read_excel(DATA / 'livingston_means.xlsx', sheet_name='TBILL', header=0)
means.columns = [str(c).strip() for c in means.columns]
means['Date'] = pd.to_datetime(means['Date'])
means = means.set_index('Date')
d1 = pd.read_excel(DATA / 'livingston_Dispersion1.xlsx', sheet_name='TBILL', header=5)
d1.columns = [str(c).strip() for c in d1.columns]
d1['Date'] = pd.to_datetime(d1['Date'])
d1 = d1.set_index('Date')
joined = panel_h.join(means[['TBILL_6M']].rename(columns={'TBILL_6M': 'lv_official'}), how='inner') \
                  .join(d1[['TBILL_6M_D1']].rename(columns={'TBILL_6M_D1': 'd_official'}), how='inner')
diff_lv = (joined['l_TBILL6M'] - joined['lv_official']).abs().max()
diff_d = (joined['d_TBILL6M'] - joined['d_official']).abs().max()
ok = max(diff_lv, diff_d) < 0.06
print(f'锚点核对（TBILL 6M，{len(joined)} 期）：max|Δlv| = {diff_lv:.4f}，max|Δdis| = {diff_d:.4f} '
      f'→ {"PASS" if ok else "FAIL"}')
assert ok, '锚点核对失败'

# ---------------- 3. 半年 → 月度（发布滞后保守：6 月→7 月、12 月→次年 1 月）----------------
rows = {}
for dt, row in panel_h.iterrows():
    avail = pd.Timestamp(dt.year, dt.month + 1, 1) if dt.month == 6 else pd.Timestamp(dt.year + 1, 1, 1)
    rows[avail] = row[[c for c in panel_h.columns if not c.startswith('n_')]]
monthly = pd.DataFrame(rows).T.sort_index()
full_idx = pd.date_range('1992-07-01', '2024-12-01', freq='MS')
monthly = monthly.reindex(full_idx).ffill()
assert not monthly.isna().any().any(), 'ffill 后仍有缺失'

# ---------------- 4. rolling-72 标准化 + 网格对齐 ----------------
def data_std(df):
    out = df.copy()
    for col in df.columns:
        rm = df[col].rolling(window=72, min_periods=12).mean()
        rs = df[col].rolling(window=72, min_periods=12).std()
        out[col] = (df[col] - rm) / rs
    return out

std = data_std(monthly).dropna()
print(f'标准化后面板：{std.index[0]:%Y-%m} .. {std.index[-1]:%Y-%m}，{len(std)} 行')
p9 = pd.read_csv(HERE / 'panel9_qe.csv', parse_dates=['date'])
grid = p9['date']
assert std.index.isin(grid).all(), '面板日期不在主线网格内'
i0 = int(np.where(grid.values == std.index[0].to_datetime64())[0][0])
print(f'主线网格切片起点 iloc = {i0}（{grid.iloc[i0]:%Y-%m}）')
std.round(6).to_csv(HERE / 'liv4_qe.csv', index_label='date')

# ---------------- 5. sanity：与主线 SPF 同题材测度的相关 ----------------
for c_new, c_ref in [('l_TBILL6M', 'l_TBL'), ('d_TBILL6M', 'd_TBL')]:
    a = std[c_new].values
    b = p9[c_ref].iloc[i0:].values
    print(f'corr({c_new}, {c_ref}) = {np.corrcoef(a, b)[0, 1]:+.3f}')
for v in ['TBILL6M', 'TBILL12M', 'TBOND6M', 'TBOND12M']:
    r = std[f'l_{v}'].corr(std[f'd_{v}'])
    print(f'{v}: corr(l,d) = {r:+.3f}')
print('已写 liv4_qe.csv')
