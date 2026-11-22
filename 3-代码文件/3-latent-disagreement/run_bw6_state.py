# -*- coding: utf-8 -*-
"""
d~_BW6 臂的状态电池补算（9 面板，2001 口径）
============================================
在 run_state_battery.py 的状态切分框架下，为 d~_BW6 设定补算 11 个状态变量
（+USREC 锚点行）高/低态下的 OOS R2OOS(%)，并与 results/state_battery_2001_p9.csv
中 raw / d~_pair / d~_lv 三臂同行数值合并输出对照表。

口径（全部继承现成锚点脚本）：
- BW6 矩阵构造：照抄 run_bw6.py 第 56–98 行（PRINTO01USQ657S 季度 ffill +
  PCEDG/PCEND/PCES/PAYEMS 月度 pct_change + USREC，contemporaneous 主口径）；
- 去污染/集成：ev.decontaminate(ev.D, BW6, burn=ev.BURN) + ev.subset_ensemble(M, 204)；
- 锚点校验：full/REC/EXP 必须与 results/bw6_2001_p9.csv 的 d~_BW6 行逐值一致；
- 状态切分：照抄 run_state_battery.py（OOS 窗口内当月中位数分组、月末→Period 对齐）。

运行前须将 sys.argv 锁定为 --panel9 --start 2001（n_train=204，OOS 288 月）。
输出：results/bw6_state_2001_p9.csv（仅新增，不覆写任何现有文件）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / '1-数据文件'
sys.path.insert(0, str(HERE))
sys.argv = ['run_bw6_state', '--panel9', '--start', '2001']
import run_evidence as ev  # noqa: E402  # 模块级预解析需先锁定 argv

assert ev.PANEL9, '必须 9 面板口径'
N_TRAIN = ev.STARTS[2001]
y = ev.y
dates = ev.dates
ha = ev.ha
n_oos = len(y) - N_TRAIN
y_oos = y[N_TRAIN:]
ha_oos = ha[N_TRAIN:]
assert n_oos == 288

# ---------------- 1. 重建 BW6（照抄 run_bw6.py 主口径构造） ----------------
IP = pd.read_csv(DATA_DIR / 'PRINTO01USQ657S.csv')
IP.rename(columns={'observation_date': 'date', 'PRINTO01USQ657S': 'IP_growth'}, inplace=True)
IP['date'] = pd.to_datetime(IP['date'])
IP.set_index('date', inplace=True)
IP = IP.resample('M').ffill()
IP.index = IP.index - pd.offsets.MonthBegin(1)


def _growth(fname):
    df = pd.read_csv(DATA_DIR / f'{fname}.csv')
    df.rename(columns={'observation_date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df[f'{fname}_growth'] = df[fname].pct_change()
    return df


PCEDG, PCEND, PCES, PAYEMS = (_growth(f) for f in ('PCEDG', 'PCEND', 'PCES', 'PAYEMS'))
USREC = pd.read_csv(DATA_DIR / 'USREC.csv')
USREC.rename(columns={'observation_date': 'date'}, inplace=True)
USREC['date'] = pd.to_datetime(USREC['date'])

macro_df = pd.merge(IP[['IP_growth']], PCEDG[['date', 'PCEDG_growth']], on='date')
macro_df = pd.merge(macro_df, PCEND[['date', 'PCEND_growth']], on='date')
macro_df = pd.merge(macro_df, PCES[['date', 'PCES_growth']], on='date')
macro_df = pd.merge(macro_df, PAYEMS[['date', 'PAYEMS_growth']], on='date')
macro_df = pd.merge(macro_df, USREC[['date', 'USREC']], on='date')
MACRO_COLS = ['IP_growth', 'PCEDG_growth', 'PCEND_growth', 'PCES_growth', 'PAYEMS_growth', 'USREC']

mp = pd.DataFrame({'date': dates}).merge(macro_df, on='date', how='left')
assert len(mp) == len(y) and not mp[MACRO_COLS].isna().any().any(), 'BW6 对面板月份有缺失'
BW6 = mp[MACRO_COLS].values.astype(float)

# ---------------- 2. d~_BW6 去污染 + 幂集 CSR 集成 + 锚点校验 ----------------
M_bw6 = ev.decontaminate(ev.D, BW6, burn=ev.BURN)
prs = ev.subset_ensemble(M_bw6, N_TRAIN)
pr0 = prs[0]
p_mean = np.mean(prs, axis=0)  # 与 run_state_battery.py 的 P[nm] 口径一致

full = ev.R2OOS(y_oos, pr0, ha_oos) * 100
usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)
rec, exp = ev.R2OOS_by_USREC(y_oos, pr0, ha_oos, usrec)
anchor = pd.read_csv(HERE / 'results' / 'bw6_2001_p9.csv', index_col=0).loc['d~_BW6']
for name, got, want in (('full', full, anchor['full_s0']),
                        ('REC', rec * 100, anchor['REC']),
                        ('EXP', exp * 100, anchor['EXP'])):
    assert abs(got - want) < 5e-4, f'锚点不符 {name}: {got:.4f} != {want:.4f}'
print(f'锚点校验通过：full={full:.4f}  REC={rec * 100:.4f}  EXP={exp * 100:.4f}')

# ---------------- 3. 状态切分（照抄 run_state_battery.py 任务 C 的对齐与分组） ----------------
panel = pd.read_csv(HERE / 'state_battery_data.csv', parse_dates=['date'])
sp = panel.set_index(panel['date'].dt.to_period('M'))
per_oos = dates[N_TRAIN:].dt.to_period('M')
usrec_bat = pd.read_csv(ev.SCNN / 'USREC.csv').iloc[-n_oos:, 0].values.astype(float)

bat = pd.read_csv(HERE / 'results' / 'state_battery_2001_p9.csv', index_col=0)
rows = []
for v in [c for c in panel.columns if c != 'date']:
    s = sp[v].reindex(per_oos).values.astype(float)
    valid = ~np.isnan(s)
    med = np.nanmedian(s)
    high = valid & (s > med)
    low = valid & (s <= med)
    rows.append({'state': v,
                 'bw6_high': ev.R2OOS(y_oos[high], p_mean[high], ha_oos[high]) * 100,
                 'bw6_low': ev.R2OOS(y_oos[low], p_mean[low], ha_oos[low]) * 100,
                 'bw6_high_p': ev.clark_west_test(y_oos[high], p_mean[high], ha_oos[high])[1],
                 'bw6_low_p': ev.clark_west_test(y_oos[low], p_mean[low], ha_oos[low])[1]})
# USREC 锚点行（>0.5 = 衰退）
high = usrec_bat > 0.5
rows.append({'state': 'USREC(anchor)',
             'bw6_high': ev.R2OOS(y_oos[high], p_mean[high], ha_oos[high]) * 100,
             'bw6_low': ev.R2OOS(y_oos[~high], p_mean[~high], ha_oos[~high]) * 100,
             'bw6_high_p': ev.clark_west_test(y_oos[high], p_mean[high], ha_oos[high])[1],
             'bw6_low_p': ev.clark_west_test(y_oos[~high], p_mean[~high], ha_oos[~high])[1]})

bw6 = pd.DataFrame(rows).set_index('state')
# USREC 锚点行的 high/low 应与锚点校验的 REC/EXP 一致
assert abs(bw6.loc['USREC(anchor)', 'bw6_high'] - rec * 100) < 1e-9
assert abs(bw6.loc['USREC(anchor)', 'bw6_low'] - exp * 100) < 1e-9

# ---------------- 4. 合并对照表（state × {raw, d~pair, d~lv, d~BW6} × {high, low}） ----------------
tab = pd.DataFrame(index=bw6.index)
for nm, cols in (('raw', ('raw_high', 'raw_low')),
                 ('d~pair', ('d~pair_high', 'd~pair_low')),
                 ('d~lv', ('d~lv_high', 'd~lv_low'))):
    tab[f'{nm}_high'] = bat[cols[0]]
    tab[f'{nm}_low'] = bat[cols[1]]
tab['d~BW6_high'] = bw6['bw6_high']
tab['d~BW6_low'] = bw6['bw6_low']
tab['d~BW6_high_p'] = bw6['bw6_high_p']
tab['d~BW6_low_p'] = bw6['bw6_low_p']
# 高/低态差距（high − low），供形态判读
for nm in ('raw', 'd~pair', 'd~lv', 'd~BW6'):
    tab[f'gapHL_{nm}'] = tab[f'{nm}_high'] - tab[f'{nm}_low']
tab.round(4).to_csv(HERE / 'results' / 'bw6_state_2001_p9.csv')
print('\n===== d~_BW6 状态电池对照表（OOS 2001-01~2024-12，9 面板幂集 CSR，R2OOS %） =====')
print(tab.round(2).to_string())
print('\n落盘: results/bw6_state_2001_p9.csv')
