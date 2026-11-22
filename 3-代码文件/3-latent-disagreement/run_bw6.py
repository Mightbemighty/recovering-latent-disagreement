# -*- coding: utf-8 -*-
"""
Phase B-2：Huang/Baker-Wurgler 式宏观去污染（BW6 正交化）vs lv 匹配控制的对照
==============================================================================
口径：--start 参数化（默认 1991 主线：N_TRAIN=84，OOS=408；--start 2001 复现 2001 口径），
严格递归（expanding，burn=36），自算 HA。
BW6 = Baker & Wurgler (2006) 六宏观变量：IP growth（FRED PRINTO01USQ657S，季度→月度 ffill）、
PCEDG/PCEND/PCES/PAYEMS growth（月度 pct_change）、USREC。构造复刻 nb2
`2_level_dis_data_prep v2.ipynb` 的 `#fundamental information` 节（cell 13），
并与该 notebook 保存的 cell 输出抽样核对（1959-02、2024-12 两行）。

两个口径：
- 主口径 d~_BW6      ：contemporaneous（与 nb2 一致；宏观发布滞后未处理，见 NOTES 声明）
- 稳健列 d~_BW6(lag1)：BW6 整体滞后 1 个月（模拟发布滞后）

可选参考列 nb2_purged(ref)：直接读 nb2 落盘的 `dispersion_data_test.csv`
（= nb2 递归 purge 变体）走同款集成。注意该文件三点口径差异：
(i) midquarter 对齐（已验证与 dispersion_data_test_midquarter_backup.csv 逐字节相同，
    信息比季末面板早 1 个月）；(ii) 递归 purge 的 t 期残差用 ≤t（含当期）OLS 估计，
    是 in-sample 残差且早期无 burn（观测<参数时残差≈0）；(iii) purge 残差未再标准化。

输出：results/bw6_{start}.csv + NOTES.md 第 7 节（手工追加）。

--panel9（2026-08-20）：9 面板绑定版——D/L 自动切 9 测度、集成自动走幂集 CSR
（均由 run_evidence 模块级预解析分派）；nb2_purged(ref) 参考列跳过（nb2 落盘为 21 测度
midquarter 口径且缺 d_CPROF，不作 9 面板参照）；对照表参照行改读 variants_{start}_p9.csv
并增列 d~_pair（9 面板主变体）；输出 bw6_{start}_p9.csv，不覆写 21 面板锚点文件。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # 复用数据加载 / decontaminate / expanding_preds / subset_ensemble / 指标

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
DATA_DIR = HERE.parent.parent / '1-数据文件'
PTAG = ev.PTAG  # ''（默认 21 面板）/ '_p9'（--panel9，由 run_evidence 模块级预解析生效）

y = ev.y
dates = ev.dates
ha = ev.ha
n_oos = len(y) - N_TRAIN
y_true = y[N_TRAIN:]
base = ha[N_TRAIN:]

# ---------------- 1. 重建 macro_df（复刻 nb2 cell 13） ----------------
IP = pd.read_csv(DATA_DIR / 'PRINTO01USQ657S.csv')
IP.rename(columns={'observation_date': 'date', 'PRINTO01USQ657S': 'IP_growth'}, inplace=True)
IP['date'] = pd.to_datetime(IP['date'])
IP.set_index('date', inplace=True)
IP = IP.resample('M').ffill()
IP.index = IP.index - pd.offsets.MonthBegin(1)

def _growth(fname, col):
    df = pd.read_csv(DATA_DIR / f'{fname}.csv')
    df.rename(columns={'observation_date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df[f'{fname}_growth'] = df[fname].pct_change()
    return df

PCEDG, PCEND, PCES, PAYEMS = (_growth(f, f) for f in ('PCEDG', 'PCEND', 'PCES', 'PAYEMS'))
USREC = pd.read_csv(DATA_DIR / 'USREC.csv')
USREC.rename(columns={'observation_date': 'date'}, inplace=True)
USREC['date'] = pd.to_datetime(USREC['date'])

macro_df = pd.merge(IP[['IP_growth']], PCEDG[['date', 'PCEDG_growth']], on='date')
macro_df = pd.merge(macro_df, PCEND[['date', 'PCEND_growth']], on='date')
macro_df = pd.merge(macro_df, PCES[['date', 'PCES_growth']], on='date')
macro_df = pd.merge(macro_df, PAYEMS[['date', 'PAYEMS_growth']], on='date')
macro_df = pd.merge(macro_df, USREC[['date', 'USREC']], on='date')
MACRO_COLS = ['IP_growth', 'PCEDG_growth', 'PCEND_growth', 'PCES_growth', 'PAYEMS_growth', 'USREC']

# 与 nb2 cell 13 保存的输出抽样核对
def _chk(dt, col, expect):
    v = macro_df.loc[macro_df['date'] == dt, col].iloc[0]
    assert abs(v - expect) < 5e-5, f'{dt} {col}: {v} != {expect}'

_chk('1959-02-01', 'IP_growth', 4.3933);  _chk('1959-02-01', 'PCEDG_growth', 0.0449)
_chk('1959-02-01', 'PAYEMS_growth', 0.0040)
_chk('2024-12-01', 'IP_growth', -0.3750); _chk('2024-12-01', 'PCEDG_growth', 0.0134)
_chk('2024-12-01', 'PCES_growth', 0.0066); _chk('2024-12-01', 'USREC', 0.0)
print('macro_df 重建与 nb2 cell 13 输出抽样核对通过（1959-02、2024-12）')

# 对齐到面板月份（1984-01~2024-12，月初）
mp = pd.DataFrame({'date': dates}).merge(macro_df, on='date', how='left')
assert len(mp) == len(y) and not mp[MACRO_COLS].isna().any().any(), 'BW6 对面板月份有缺失'
BW6 = mp[MACRO_COLS].values.astype(float)
BW6_lag1 = pd.DataFrame(BW6, columns=MACRO_COLS).shift(1).bfill().values  # 首行回填，影响可忽略

# ---------------- 2. 严格递归 BW6 正交化 + 3. 子集集成 ----------------
usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)
rf = ev.mkt['RF_l1'] / 100
ha_full_s = pd.Series(ha)
y_full_s = ev.sp['raw']


def run_ensemble(M):
    """主管线同款集成（默认 21 面板 = K=11 全枚举 CSR；--panel9 = 幂集 CSR，确定性三序列同值）
    + 全套指标（s0 序列口径，与 variants_2001 一致）。"""
    prs = ev.subset_ensemble(M, N_TRAIN)
    pr0 = prs[0]
    r2s = [ev.R2OOS(y_true, pr, base) * 100 for pr in prs]
    cw_s, cw_p, _ = ev.clark_west_test(y_true, pr0, base)
    r_rec, r_exp = ev.R2OOS_by_USREC(y_true, pr0, base, usrec)
    rec_m = usrec.iloc[:, 0].values > 0.5
    fp = pd.Series(np.full(len(y), np.nan))
    fp.iloc[N_TRAIN:] = pr0
    cer3 = ev.cer(fp, y_full_s, ha_full_s, rf, 3, transaction_cost=0, oos_len=n_oos)
    return {'full_s0': r2s[0], 'full_mean3': np.mean(r2s), 'full_min3': np.min(r2s),
            'CW_p': cw_p, 'REC': r_rec * 100, 'EXP': r_exp * 100,
            'CW_p_REC': ev.clark_west_test(y_true[rec_m], pr0[rec_m], base[rec_m])[1],
            'CW_p_EXP': ev.clark_west_test(y_true[~rec_m], pr0[~rec_m], base[~rec_m])[1],
            'CER3_tc0': cer3}


variants_bw6 = {
    'd~_BW6': ev.decontaminate(ev.D, BW6, burn=ev.BURN),
    'd~_BW6(lag1)': ev.decontaminate(ev.D, BW6_lag1, burn=ev.BURN),
}
rows = {}
ens_tag = '幂集 CSR' if ev.PANEL9 else '全枚举'
for name, M in variants_bw6.items():
    rows[name] = run_ensemble(M)
    r = rows[name]
    print(f'[{name:12s}] R2OOS = {r["full_mean3"]:.3f}（{ens_tag}，s0/min3 同值）  '
          f'CW p={r["CW_p"]:.4f}  REC={r["REC"]:.2f} EXP={r["EXP"]:.2f}  CER3_tc0={r["CER3_tc0"]:.3f}')

# ---------------- 5. 可选参考列：nb2 落盘的递归 purged 分歧（midquarter；仅 21 面板口径） ----------------
SCNN = ev.SCNN
f_purged = SCNN / 'dispersion_data_test.csv'
if ev.PANEL9:
    print('nb2_purged(ref) 跳过：nb2 落盘为 21 测度 midquarter 口径且缺 d_CPROF，不作 9 面板参照')
elif f_purged.exists():
    dt = pd.read_csv(f_purged, parse_dates=['date'])
    assert len(dt) == len(y) and (dt['date'].values == dates.values).all()
    d_cols = [d for _, d in ev.pairs]
    D_nb2 = dt[d_cols].values.astype(float)   # 已是 purge 残差，直接进集成
    rows['nb2_purged(ref)'] = run_ensemble(D_nb2)
    r = rows['nb2_purged(ref)']
    print(f'[nb2_purged  ] R2OOS = {r["full_mean3"]:.3f}（{ens_tag}，s0/min3 同值）  '
          f'CW p={r["CW_p"]:.4f}  REC={r["REC"]:.2f} EXP={r["EXP"]:.2f}  CER3_tc0={r["CER3_tc0"]:.3f}')
else:
    print('nb2 purged 文件不存在，跳过参考列')

# ---------------- 4. 对照表（raw / d~_u14 / d~_lv 等从 variants_{start}{PTAG}.csv 直读） ----------------
res = pd.read_csv(HERE / 'results' / f'variants_{START}{PTAG}.csv', index_col=0)
tab_cols = ['full_s0', 'full_mean3', 'full_min3', 'CW_p', 'REC', 'EXP', 'CER3_tc0']
table = pd.DataFrame(rows).T[tab_cols]
ref_names = ('raw', 'd~_pair', 'd~_u14', 'd~_lv') if ev.PANEL9 else ('raw', 'd~_u14', 'd~_lv')
for name in ref_names:
    table.loc[name] = res.loc[name, tab_cols]
row_order = (['raw', 'd~_BW6', 'd~_BW6(lag1)', 'd~_pair', 'd~_u14', 'd~_lv'] if ev.PANEL9
             else ['raw', 'd~_BW6', 'd~_BW6(lag1)', 'nb2_purged(ref)', 'd~_u14', 'd~_lv'])
table = table.loc[row_order]
table.index.name = 'variant'
table.round(4).to_csv(HERE / 'results' / f'bw6_{START}{PTAG}.csv')
print(f'\n===== 对照表（{START} 起点{"，9 面板 + 幂集 CSR" if ev.PANEL9 else ""}）=====')
print(table.round(3).to_string())
print(f'\n落盘: results/bw6_{START}{PTAG}.csv')
