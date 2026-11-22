# -*- coding: utf-8 -*-
"""
状态电池（Phase B 收官，2026-08-04）
====================================
任务 A：状态面板 state_battery_data.csv（月末时间戳，连续月频 1960-07~2025-12）
  VIX_avg（月均文件直读，与日度自算核对）/ VIX_eom（日度月末值）/
  VoV（月内日度 Δlog(VIX) 的 std，≥10 个有效差分）/ CATFIN / PLS_SENT / PLS_SENT_orth /
  JLN_macro_h1 / JLN_fin_h1 / JLN_real_h1 / AggIVOL（若 aggivol_monthly.csv 已存在）。
任务 C：状态电池表 results/state_battery_{start}.csv
  raw / d~ 集成（21 面板 K=11 全枚举 / --panel9 幂集 CSR），OOS 由 --start 给出（默认 1991），
  自算 HA；每个状态变量按当月中位数分高/低组，
  报两组 R²OOS 与 gap(d~−raw)；附各变量对 USREC 的 OOS 相关与变量间最大两两相关。
日期约定：状态面板用月末 Timestamp（yyyymm→月末）；与 y/preds（all_data_qe 月初约定）
  的合并按 year-month Period 对齐。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / '1-数据文件'
sys.path.insert(0, str(HERE))
import run_evidence as ev

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
RES = HERE / 'results'

# ================= 任务 A：状态面板 =================
idx = pd.date_range('1960-07-31', '2025-12-31', freq='M')
panel = pd.DataFrame({'date': idx})
coverage = []


def add_var(series_me, name):
    """series_me: month-end Timestamp index 的 Series。"""
    global panel
    s = series_me[~series_me.index.duplicated()]
    panel[name] = panel['date'].map(s)
    win = panel[(panel['date'] >= '1984-01-31') & (panel['date'] <= '2024-12-31')]
    v = win[name]
    coverage.append({'var': name, 'start': str(s.first_valid_index().date()),
                     'end': str(s.last_valid_index().date()),
                     'n_valid_84_24': int(v.notna().sum()), 'n_missing_84_24': int(v.isna().sum())})


# --- VIX ---
vix_d = pd.read_csv(DATA / 'VIXCLS_daily.csv', parse_dates=['observation_date'])
vix_d = vix_d.set_index('observation_date')['VIXCLS'].dropna()
vix_avg_file = pd.read_csv(DATA / 'VIXCLS_mthly_avg.csv', parse_dates=['observation_date'])
vix_avg_file = vix_avg_file.set_index('observation_date')['VIXCLS']
vix_avg_file.index = vix_avg_file.index.to_period('M').to_timestamp('M')
add_var(vix_avg_file, 'VIX_avg')
# 核对：日度自算月均 vs 月均文件
vix_avg_calc = vix_d.groupby(vix_d.index.to_period('M')).mean()
vix_avg_calc.index = vix_avg_calc.index.to_timestamp('M')
chk = pd.concat([vix_avg_file, vix_avg_calc], axis=1, keys=['file', 'calc']).dropna()
assert np.abs(chk['file'] - chk['calc']).max() < 0.02, 'VIX 月均文件与日度自算不符'
print(f"VIX_avg 核对通过（max|file-calc|={np.abs(chk['file'] - chk['calc']).max():.4f}）")
vix_eom = vix_d.groupby(vix_d.index.to_period('M')).last()
vix_eom.index = vix_eom.index.to_timestamp('M')
add_var(vix_eom, 'VIX_eom')
# VoV：月内 Δlog(VIX) 的 std，≥10 个有效差分
dlv = np.log(vix_d).diff().dropna()
vov = dlv.groupby(dlv.index.to_period('M')).agg(['std', 'count'])
vov.loc[vov['count'] < 10, 'std'] = np.nan
vov_s = vov['std']
vov_s.index = vov_s.index.to_timestamp('M')
add_var(vov_s, 'VoV')

# --- CATFIN（'Jan-73' → 月末；'33%' → 0.33） ---
cat = pd.read_csv(DATA / 'CATFIN_2025.csv')
cat['date'] = pd.to_datetime(cat['yyyymm'], format='%b-%y').dt.to_period('M').dt.to_timestamp('M')
cat['CATFIN'] = pd.to_numeric(cat['CATFIN'].astype(str).str.rstrip('%'), errors='coerce') / 100.0
add_var(cat.set_index('date')['CATFIN'], 'CATFIN')

# --- PLS sentiment（截止 2023-12） ---
pls = pd.read_csv(DATA / 'PLS_Sentiment_indices(PLS).csv')
pls.columns = [c.strip() for c in pls.columns]
pls['date'] = pd.to_datetime(pls['yyyymm'].astype(str), format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
pls['PLS_SENT'] = pd.to_numeric(pls['PLS_SENT'], errors='coerce')
pls['PLS_SENT_orth'] = pd.to_numeric(pls['PLS_SENT othogonalized (to macro econ vars)'], errors='coerce')
add_var(pls.set_index('date')['PLS_SENT'], 'PLS_SENT')
add_var(pls.set_index('date')['PLS_SENT_orth'], 'PLS_SENT_orth')

# --- JLN 不确定性 ×3（只用 h=1） ---
for f, nm in (('Macro', 'JLN_macro_h1'), ('Financial', 'JLN_fin_h1'), ('Real', 'JLN_real_h1')):
    df = pd.read_excel(DATA / f'{f}UncertaintyToCirculate.xlsx', sheet_name=f'{f} Uncertainty')
    df['date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp('M')
    add_var(df.set_index('date')['h=1'].astype(float), nm)

# --- AggIVOL（若任务 B 已完成） ---
f_iv = HERE / 'aggivol_monthly.csv'
if f_iv.exists():
    iv = pd.read_csv(f_iv, parse_dates=['date']).set_index('date')['AggIVOL']
    add_var(iv, 'AggIVOL')
    print('AggIVOL 已并入状态面板')
else:
    print('aggivol_monthly.csv 不存在，本轮状态面板不含 AggIVOL')

panel.round(6).to_csv(HERE / 'state_battery_data.csv', index=False)
cov = pd.DataFrame(coverage).set_index('var')
print('\n===== 变量覆盖率（1984-01~2024-12 = 492 月窗口） =====')
print(cov.to_string())
print(f'\n落盘: state_battery_data.csv, {len(panel)} 行 × {panel.shape[1]} 列')

# ================= 任务 C：状态电池表 =================
y = ev.y
ha = ev.ha
n_oos = len(y) - N_TRAIN
y_oos = y[N_TRAIN:]
ha_oos = ha[N_TRAIN:]
per_oos = ev.dates[N_TRAIN:].dt.to_period('M')

# 变体口径：默认 21 面板报 raw / d~_lv（历史锚点）；--panel9 报 raw / d~_pair（主变体）
# / d~_lv（对照），列名相应扩展，输出带 _p9 后缀不覆写 21 面板锚点文件。
if ev.PANEL9:
    var_mats = [('raw', ev.D, 'raw'), ('d~pair', ev.variants['d~_pair'], 'd~_pair'),
                ('d~lv', ev.variants['d~_lv'], 'd~_lv')]
else:
    var_mats = [('raw', ev.D, 'raw'), ('d~', ev.variants['d~_lv'], 'd~_lv')]
preds = pd.read_csv(RES / f'preds_{START}{ev.PTAG}.csv', parse_dates=['date'])
P = {}
for nm, M, col in var_mats:
    prs = ev.subset_ensemble(M, N_TRAIN)
    assert np.abs(prs[0] - preds[col].values).max() < 6e-7
    P[nm] = np.mean(prs, axis=0)

sp = panel.set_index(panel['date'].dt.to_period('M'))
usrec = pd.read_csv(ev.SCNN / 'USREC.csv').iloc[-n_oos:, 0].values.astype(float)

STATE_VARS = [c for c in panel.columns if c != 'date']
rows = []
state_oos = {}
for v in STATE_VARS:
    s = sp[v].reindex(per_oos).values.astype(float)
    state_oos[v] = s
    valid = ~np.isnan(s)
    med = np.nanmedian(s)
    high = valid & (s > med)
    low = valid & (s <= med)
    cov_note = f'{int(valid.sum())}/{n_oos}'
    row = {'state': v, 'coverage': cov_note, 'median': med,
           'n_high': int(high.sum()), 'n_low': int(low.sum()),
           'corr_USREC': np.corrcoef(s[valid], usrec[valid])[0, 1]}
    for gname, g in (('high', high), ('low', low)):
        for nm in P:
            row[f'{nm}_{gname}'] = ev.R2OOS(y_oos[g], P[nm][g], ha_oos[g]) * 100
            if ev.PANEL9:
                # Clark–West 单边 p：状态高/低 mask 选出的（非连续）子序列上直接检验
                # （先例 = MetricsDef.clark_west_test_by_USREC；NW HAC 自动 lag）
                row[f'{nm}_{gname}_p'] = ev.clark_west_test(y_oos[g], P[nm][g], ha_oos[g])[1]
    for nm in P:
        if nm == 'raw':
            continue
        sfx = '' if nm == 'd~' else f'_{nm}'  # 默认口径保持 gap_high/gap_low 列名不变
        row[f'gap{sfx}_high'] = row[f'{nm}_high'] - row['raw_high']
        row[f'gap{sfx}_low'] = row[f'{nm}_low'] - row['raw_low']
    rows.append(row)
# USREC 锚点行（>0.5 = 衰退）
high = usrec > 0.5
row = {'state': 'USREC(anchor)', 'coverage': f'{n_oos}/{n_oos}', 'median': 0.0,
       'n_high': int(high.sum()), 'n_low': int((~high).sum()), 'corr_USREC': 1.0}
for gname, g in (('high', high), ('low', ~high)):
    for nm in P:
        row[f'{nm}_{gname}'] = ev.R2OOS(y_oos[g], P[nm][g], ha_oos[g]) * 100
        if ev.PANEL9:
            row[f'{nm}_{gname}_p'] = ev.clark_west_test(y_oos[g], P[nm][g], ha_oos[g])[1]
for nm in P:
    if nm == 'raw':
        continue
    sfx = '' if nm == 'd~' else f'_{nm}'
    row[f'gap{sfx}_high'] = row[f'{nm}_high'] - row['raw_high']
    row[f'gap{sfx}_low'] = row[f'{nm}_low'] - row['raw_low']
rows.append(row)

bat = pd.DataFrame(rows).set_index('state')
bat_out = bat.round(4)
if ev.PANEL9:
    # CW p 列保留 6 位小数（支撑 make_table_drafts 的 1e-6 独立复核锚点）；旧列仍 4 位不变
    _pcols = [c for c in bat.columns if c.endswith('_p')]
    bat_out[_pcols] = bat[_pcols].round(6)
bat_out.to_csv(RES / f'state_battery_{START}{ev.PTAG}.csv')
print(f'\n===== 状态电池表（OOS {START}-01~2024-12，{"幂集 CSR" if ev.PANEL9 else "全枚举"}，自算 HA{", 9 面板" if ev.PANEL9 else ""}） =====')
print(bat.round(3).to_string())

# 变量间两两相关上限（OOS，不含 USREC）
M = np.column_stack([state_oos[v] for v in STATE_VARS])
Cm = pd.DataFrame(M, columns=STATE_VARS).corr()
np.fill_diagonal(Cm.values, np.nan)
mx = Cm.abs().max()
print('\n变量间最大 |两两相关|（每变量）:')
print(mx.round(3).sort_values(ascending=False).to_string())
print(f'\n落盘: results/state_battery_{START}{ev.PTAG}.csv')
