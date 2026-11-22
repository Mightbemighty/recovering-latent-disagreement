# -*- coding: utf-8 -*-
"""
SPF 信息可得性时间表核查（run_bw6_vintage.py 配套）
====================================================
来源：费城联储 "Dates of Previous Surveys"（spf-release-dates.txt，
本地缓存 1-数据文件/SPF_timing/spf-release-dates.txt；原始 URL：
https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/
survey-of-professional-forecasters/spf-release-dates.txt）。

链条：survey 季度 YYYY Qk → 发布日（News Release Date）
     → 季末月（Q1→3月、Q2→6月、Q3→9月、Q4→12月，面板季末映射口径）
     → 预测月（季末月+1）。
验证：发布日 < 季末月末 < 预测月（保守性）；lag = 发布日 → 预测月首日的天数。

注意：1990Q2 之前的真实截止/发布日不可考（文件首行声明），1983Q4–1990Q1 行留空并注明。
输出：results/spf_timing_check.csv（季度、截止日、发布日、映射月、预测月、lag 天数）。
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / '1-数据文件' / 'SPF_timing' / 'spf-release-dates.txt'
OUT = HERE / 'results' / 'spf_timing_check.csv'

# ---------------- 1. 解析官方 txt ----------------
text = RAW.read_text(encoding='utf-8')
pat = re.compile(r'^(?:(\d{4}))?\s+(Q[1-4])\s+(\d{1,2}/\d{1,2}/\d{2})\**\s+(\d{1,2}/\d{1,2}/\d{2})\**\s*$')
records = []
year = None
for line in text.splitlines():
    m = pat.match(line.rstrip())
    if not m:
        continue
    if m.group(1):
        year = int(m.group(1))
    q = int(m.group(2)[1])
    dl = pd.to_datetime(m.group(3), format='%m/%d/%y')
    rl = pd.to_datetime(m.group(4), format='%m/%d/%y')
    records.append({'year': year, 'q': q, 'deadline': dl, 'release': rl})
dates_df = pd.DataFrame(records).set_index(['year', 'q'])
print(f'官方发布日解析：{len(dates_df)} 个季度（{dates_df.index[0]} ~ {dates_df.index[-1]}）')

# ---------------- 2. 1983Q4–2024Q4 全季度链条 ----------------
rows = []
for y in range(1983, 2025):
    for q in (([4] if y == 1983 else [1, 2, 3, 4])):
        qe_month = q * 3                      # 季末月：Q1→3 ... Q4→12
        qe_end = pd.Timestamp(y, qe_month, 1) + pd.offsets.MonthEnd(0)
        pred_month = (pd.Timestamp(y, qe_month, 1) + pd.offsets.MonthBegin(1))
        row = {'survey_quarter': f'{y}Q{q}', 'deadline': None, 'release': None,
               'qe_month': f'{y}-{qe_month:02d}', 'pred_month': pred_month.strftime('%Y-%m'),
               'lag_days': None, 'note': ''}
        if (y, q) in dates_df.index:
            r = dates_df.loc[(y, q)]
            row['deadline'] = r['deadline'].strftime('%Y-%m-%d')
            row['release'] = r['release'].strftime('%Y-%m-%d')
            row['lag_days'] = (pred_month - r['release']).days
            if not (r['release'] < qe_end):
                row['note'] = '发布日 >= 季末月末（破坏保守性！）'
        else:
            row['note'] = 'pre-1990Q2：真实日期不可考（官方文件声明）'
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

# ---------------- 3. 报告 ----------------
known = out.dropna(subset=['lag_days'])
oos = known[known['survey_quarter'].between('2000Q4', '2024Q4')]  # OOS 2001-01 起预测所用调查
print(f'\n全段（1990Q2–2024Q4，n={len(known)}）lag 天数: min={known["lag_days"].min()}  '
      f'median={known["lag_days"].median():.0f}  max={known["lag_days"].max()}')
print(f'OOS 相关段（2000Q4–2024Q4，n={len(oos)}）lag 天数: min={oos["lag_days"].min()}  '
      f'median={oos["lag_days"].median():.0f}  max={oos["lag_days"].max()}')
viol = out[out['note'].str.contains('破坏', na=False)]
print(f'保守性违例数: {len(viol)}')
if len(viol):
    print(viol.to_string())
print(f'\n最早 5 个可得季度:')
print(known.head(5).to_string(index=False))
print(f'\n落盘: {OUT}')
