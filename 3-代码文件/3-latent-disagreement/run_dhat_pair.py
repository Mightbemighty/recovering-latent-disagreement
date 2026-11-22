# -*- coding: utf-8 -*-
"""
实验 d̂_pair 镜像：pair 剥离侧为空的直接证据（9 面板，2026-08-20）
==============================================================================
目的：为"pair 剥离侧为空"提供直接证据，对照 T1（run_t1_lv_channel.py）中 d̂_lv 的镜像。
构造：d̂_pair 面板 = D − d~_pair 面板，即逐列同题 lv 投影的**拟合值**（ev.variants
['d~_pair'] 是残差；decontaminate 口径下 burn=36 前段拟合值 = 历史均值——t 行残差 =
D[t] − mean(D[:max(t,1)])，故拟合值即该历史均值；burn 后 = expanding 投影拟合）。
自洽 gate（与 T1 同款）：(c1) 逐列 decontaminate_fit(D[:,k], L[:,k]) ≡ D − d~_pair
（atol 1e-10）；(c2) fit + resid ≡ D（由构造恒成立，仍断言）。

检验：
  (i) 1991 主线口径：d̂_pair 幂集 CSR（2^9−1 子集等权，n_train=84，expanding）对
      h=1 超额收益的 R²OOS（预期 ≈0 或负）与 Clark–West p；
 (ii) T1 口径复刻：N_TRAIN=204、评估窗口/共同末行（h=1→490 / h=12→479）、HA 基准、
      分段（full / pre2020 / post2020）、nan_aware 训练全部与 run_t1_lv_channel.py
      逐字一致（fwd_sum / ha_h / i_max_for 为其逐字副本；目标序列读既有落盘
      results/cfdr_news_series.csv），跑 h=1 ret/cf/dr 与 h=12 dr。
      与 T1 落盘 results/t1_lv_channel_p9.csv 的 d̂_lv 行并排打印作口径对照。

输出：results/dhat_pair_1991_p9.csv（长表：predictor/n_train/target/h/segment/r2/cw_p/n_oos）。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = ['x', '--panel9']
import run_evidence as ev  # noqa: E402

R2OOS = ev.R2OOS
RES = HERE / 'results'
y, dates, D, L = ev.y, ev.dates, ev.D, ev.L
n = len(y)


# ---------------- 0. d̂_pair 构造 + 自洽 gate ----------------
def decontaminate_fit(Dmat, Zmat, burn=ev.BURN):
    """decontaminate 的镜像：expanding 投影的**拟合值**（t 行用 <t 行估计；burn 前=历史均值）。
    逐字复制 run_t1_lv_channel.py 的同名函数。"""
    n_ = len(Dmat)
    out = np.empty_like(Dmat)
    for t in range(n_):
        if t < burn:
            out[t] = Dmat[:max(t, 1)].mean(axis=0)
        else:
            Zc = np.column_stack([np.ones(t), Zmat[:t]])
            G, *_ = np.linalg.lstsq(Zc, Dmat[:t], rcond=None)
            out[t] = np.concatenate([[1.0], Zmat[t]]) @ G
    return out


D_hat_pair = D - ev.variants['d~_pair']
# gate(c1)：与逐列同题 lv 投影拟合值逐点一致
D_hat_pair_chk = np.column_stack([decontaminate_fit(D[:, [k]], L[:, [k]])[:, 0]
                                  for k in range(D.shape[1])])
assert np.allclose(D_hat_pair, D_hat_pair_chk, atol=1e-10), 'D − d~_pair != 逐列 decontaminate_fit'
# gate(c2)：fit + resid ≡ D（由构造恒成立）
assert np.allclose(D_hat_pair + ev.variants['d~_pair'], D, atol=1e-10)
print('gates：d̂_pair ≡ D − d~_pair ≡ 逐列同题 lv 投影拟合值 ✓；fit + resid ≡ D ✓')

rows = []

# ---------------- (i) 1991 主线口径：h=1 ret ----------------
N_TRAIN_91 = ev.STARTS[1991]  # 84
pr91 = ev.powerset_csr_preds(D_hat_pair, y, N_TRAIN_91)
r2_91 = R2OOS(y[N_TRAIN_91:], pr91, ev.ha[N_TRAIN_91:]) * 100
cw_s, cw_p, _ = ev.clark_west_test(y[N_TRAIN_91:], pr91, ev.ha[N_TRAIN_91:])
rows.append({'predictor': 'dhat_pair', 'n_train': N_TRAIN_91, 'target': 'ret', 'h': 1,
             'segment': 'full', 'r2': r2_91, 'cw_p': cw_p, 'n_oos': len(pr91)})
print(f'\n(i) 1991 主线（n_train=84，408 点）：d̂_pair × h=1 ret  R²OOS = {r2_91:.4f}  '
      f'CW p = {cw_p:.4f}（预期 ≈0 或负）')

# ---------------- (ii) T1 口径复刻：h=1 ret/cf/dr + h=12 dr ----------------
N_TRAIN_T1 = 204

# 渠道新闻目标（读既有落盘，与 cfdr_race / T1 同网格）
news = pd.read_csv(RES / 'cfdr_news_series.csv', parse_dates=['date'])
assert len(news) == n and (news['date'].values == dates.values).all(), '新闻序列网格错位'
SER = {c: news[c].values.astype(float) for c in news.columns if c != 'date'}
TARGETS = {'ret': y, 'cf': SER['cf_news'], 'dr': SER['dr_news']}


def fwd_sum(arr, h):
    """tgt[t] = Σ_{k=0..h−1} arr[t+k]（任一 NaN ⇒ NaN）。逐字复制 run_t1_lv_channel.py。"""
    out = np.full(len(arr), np.nan)
    for t in range(len(arr) - h + 1):
        w = arr[t:t + h]
        if not np.isnan(w).any():
            out[t] = w.sum()
    return out


def ha_h(y_in, n_train, h, i_max):
    """HA 基准：ha[i] = mean(y_in[s], s ≤ i−h，跳过 NaN），i ∈ [n_train, i_max]。
    逐字复制 run_t1_lv_channel.py。"""
    out = np.empty(i_max - n_train + 1)
    for j, i in enumerate(range(n_train, i_max + 1)):
        out[j] = np.nanmean(y_in[:i - h + 1])
    return out


def i_max_for(tgt, h):
    """逐字复制 run_t1_lv_channel.py。"""
    valid = np.where(~np.isnan(tgt))[0]
    return min(n - h, valid[-1] - h + 1) if h > 1 else min(n - 1, valid[-1])


I_MAX = {h: i_max_for(TARGETS['cf'], h) for h in (1, 12)}
SEGMENTS = {'full': lambda d: d == d, 'pre2020': lambda d: d < pd.Timestamp('2020-01-01'),
            'post2020': lambda d: d >= pd.Timestamp('2020-01-01')}

print('\n(ii) T1 口径（n_train=204，幂集 CSR，与 t1_lv_channel_p9.csv 同网格）：')
for h, cells in ((1, ['ret', 'cf', 'dr']), (12, ['dr'])):
    i_max = I_MAX[h]
    oos_dates = dates[N_TRAIN_T1:i_max + 1].reset_index(drop=True)
    for tg in cells:
        tgt_full = fwd_sum(TARGETS[tg], h) if h > 1 else TARGETS[tg]
        y_true = tgt_full[N_TRAIN_T1:i_max + 1]
        base = ha_h(tgt_full, N_TRAIN_T1, h, i_max)
        assert not np.isnan(y_true).any() and not np.isnan(base).any()
        pr = ev.powerset_csr_preds(D_hat_pair, tgt_full, N_TRAIN_T1, h=h, i_max=i_max,
                                   nan_aware=True)
        segs = SEGMENTS if h == 1 else {'full': SEGMENTS['full']}
        for sgt, sfn in segs.items():
            sm = sfn(oos_dates).values
            r2 = R2OOS(y_true[sm], pr[sm], base[sm]) * 100
            cw = np.nan
            if tg == 'ret':
                _, cw, _ = ev.clark_west_test(y_true[sm], pr[sm], base[sm])
            rows.append({'predictor': 'dhat_pair', 'n_train': N_TRAIN_T1, 'target': tg,
                         'h': h, 'segment': sgt, 'r2': r2, 'cw_p': cw, 'n_oos': int(sm.sum())})
        print(f'  h={h} target={tg} 完成（i_max={i_max}，{i_max - N_TRAIN_T1 + 1} 点）')

tab = pd.DataFrame(rows)
tab.round(4).to_csv(RES / 'dhat_pair_1991_p9.csv', index=False)

# ---------------- 与 T1 d̂_lv 行并排对照 ----------------
t1 = pd.read_csv(RES / 't1_lv_channel_p9.csv')
t1 = t1[t1['predictor'] == 'dhat_lv'].set_index(['target', 'h', 'segment'])
mine = tab[tab['n_train'] == N_TRAIN_T1].set_index(['target', 'h', 'segment'])
cmp = mine[['r2', 'n_oos']].rename(columns={'r2': 'dhat_pair_r2'})
cmp['dhat_lv_r2(T1)'] = t1['r2']
print('\n===== 与 T1 d̂_lv 行对照（n_train=204，同网格 R²OOS，%） =====')
print(cmp.round(3).to_string())
print('\n判读提示：d̂_pair 各格 R²OOS ≈0 或负（且与 d̂_lv 镜像同为非正）→ "pair 剥离侧为空"'
      '获直接证据；若 h=12 dr 显著为正则提示剥离侧含长 horizon 折现率内容，需上报。')
print('落盘 results/dhat_pair_1991_p9.csv')
