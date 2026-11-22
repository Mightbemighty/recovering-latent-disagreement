# -*- coding: utf-8 -*-
"""
HLW (Huang, Li & Wang 2021, JFE) 同数据同窗口对齐赛马（9 面板主线口径；审稿人同场比较要求）
================================================================================
动机：稿件原有比较为拼接口径——引 Huang 原文月度 1991–2018 R²OS = 1.56%（其 Table 3，
BW6-purged 24 测度 PLS，OOS 1991:02–2018:12，收益月标签）对照我方 1991 起点窗 2.72%。
本脚本在**同一数据、同一窗口、同一基准**下做三场赛马：

  窗口：月度 h=1，OOS 预测行 1991-01 ~ 2018-12（dates 口径，n_oos=336，
        与 results/segment_1991_p9.csv 的 '1991-2018' 段逐行一致），
        n_train=84（我方 1991 口径），HA 基准自算（ha[t]=mean(y[:t])，同 run_evidence）。
  臂 1a：HLW 式 PLS 分歧指数，clean triPLS 口径（π_k 与预测回归均只用 s ≤ i−h 的
         已完成样本对，逐字口径复制 run_huang_bridge.pls_h），raw 24 测度面板
         （21 + DSUV/DIVOL/DOID，复用 results/huang_market_measures.csv 复刻产物）。
  臂 1b：同 1a，raw 9 测度面板（panel9_qe.csv，SPF-only）。
  臂 2 ：我方 raw 幂集 CSR（9 面板，2^9−1=511 子集等权，ev.powerset_csr_preds）。
  臂 3 ：我方 d̃_pair 幂集 CSR（9 面板主线变体；同题 lv 投影残差，ev.decontaminate 逐列）。

自洽 gate：
  (g1) 本脚本重算的 raw / d~_pair 全 OOS 预测序列与 results/preds_1991_p9.csv
       落盘值逐点一致（容差 6e-7，落盘 round(6)）；
  (g2) 锚点复现确认——本脚本 d̃_pair 1991–2018 段 R²OS/CW p 与既有锚点文件
       results/segment_1991_p9.csv（d~_pair_mean3 行 = 3.437 / p=.0089）一致；
       raw 行（2.5952 / p=.0222）同步核对。
归因材料：读既有 results/huang_bridge.csv 与 results/huang_pls_lookahead.csv，
  打印 Huang 原窗（n=335，收益月 1991:02–2018:12）下 clean/bw6/leak 各口径 PLS 值，
  供"原文 1.56% 与我方数据下表现之差异"的归因（测度集/窗口/口径三维度）。

输出：results/hlw_aligned_p9.csv + 控制台赛马表与锚点核对。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # 默认载入 21 面板（臂 1a 需要 ev.D）；9 面板单独读 panel9_qe.csv

RES = HERE / 'results'
y, dates, ha = ev.y, ev.dates, ev.ha
n = len(y)
pm = pd.PeriodIndex(pd.to_datetime(pd.Series(dates)), freq='M')

# ---------------- 9 测度面板（panel9_qe.csv，SPF-only） ----------------
p9 = pd.read_csv(HERE / 'panel9_qe.csv', parse_dates=['date'])
assert len(p9) == n and (p9['date'].values == dates.values).all(), 'panel9 日期网格与主线不一致'
assert np.allclose(p9['excess_ret_h1'].values.astype(float), y, atol=1e-8), 'panel9 目标序列不一致'
l9 = sorted(c for c in p9.columns if c.startswith('l_'))
d9 = ['d_' + c[2:] for c in l9]
D9 = p9[d9].values.astype(float)
L9 = p9[l9].values.astype(float)
assert not np.isnan(D9).any() and not np.isnan(L9).any()
# d̃_pair：逐列同题 lv 投影残差（expanding 实时去污染，burn=36，同 ev.variants['d~_pair'] 构造）
D_pair9 = np.column_stack([ev.decontaminate(D9[:, [k]], L9[:, [k]])[:, 0] for k in range(D9.shape[1])])
print(f'9 面板: {[c[2:] for c in l9]}', flush=True)

# ---------------- 24 测度面板（复制 run_huang_bridge：21 + DSUV/DIVOL/DOID，rolling-72 标准化） ----------------
mm = pd.read_csv(RES / 'huang_market_measures.csv')
mm['pm'] = pd.PeriodIndex(mm['date'].astype(str).str[:7], freq='M')
mm = mm.set_index('pm').drop(columns=['date'])
mm = mm.reindex(pd.period_range('1968-01', '2024-12', freq='M'))
for c in mm.columns:  # Huang 式 rolling 6 年标准化（min 12），与桥接复刻同口径
    mm[c] = (mm[c] - mm[c].rolling(72, min_periods=12).mean()) / \
            mm[c].rolling(72, min_periods=12).std()
mm = mm.reindex(pm)
M24 = np.column_stack([ev.D, mm[['DSUV', 'DIVOL', 'DOID']].values.astype(float)])
print(f'24 面板: {n} 行 × {M24.shape[1]} 测度（21 + DSUV/DIVOL/DOID）', flush=True)

# ---------------- 窗口 ----------------
N_TRAIN = ev.STARTS[1991]                       # 84
END = pm.get_loc(pd.Period('2018-12', 'M'))     # 419；OOS 预测行 84..419
y_true = y[N_TRAIN:END + 1]
base = ha[N_TRAIN:END + 1]
assert not np.isnan(base).any() and len(y_true) == 336
print(f'窗口: 预测行 {pm[N_TRAIN]} ~ {pm[END]}，n_train={N_TRAIN}，n_oos={len(y_true)}', flush=True)


# ---------------- HLW 式 PLS（clean triPLS；口径逐字复制 run_huang_bridge.pls_h，加 i_max） ----------------
def pls_h(M, y_in, n_train, h=1, i_max=None):
    """Huang 三步法（triPLS 口径）缺失感知版：π_k 逐列用可用样本对（s ≤ i−h）；
    指数 = 当月可用测度的 π 加权均值；再 expanding OLS（≤ i−h）。"""
    n_, K = M.shape
    if i_max is None:
        i_max = n_ - h
    preds = np.empty(i_max - n_train + 1)
    for j, i in enumerate(range(n_train, i_max + 1)):
        tr = np.arange(0, i - h + 1)
        ytr = y_in[tr]
        pi = np.zeros(K)
        for k in range(K):
            ok = ~np.isnan(M[tr, k])
            if ok.sum() < 24:      # 样本太少则不估（π_k=0，该测度不贡献）
                continue
            Rc = np.column_stack([np.ones(ok.sum()), ytr[ok]])
            pi[k] = np.linalg.lstsq(Rc, M[tr, k][ok], rcond=None)[0][1]
        avail_tr = ~np.isnan(M[tr])
        avail_i = ~np.isnan(M[i])
        denom_tr = avail_tr.sum(axis=1).clip(min=1)
        idx_tr = np.nansum(M[tr] * pi, axis=1) / denom_tr
        idx_i = np.nansum(M[i] * pi) / max(avail_i.sum(), 1)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ytr)), idx_tr]),
                                   ytr, rcond=None)
        preds[j] = np.array([1.0, idx_i]) @ beta
    return preds


# ---------------- 评估（h=1 非重叠，CW 用默认自动 NW lag，与 segment_1991_p9.csv 口径一致） ----------------
def evaluate(pr_window, tag):
    r2 = ev.R2OOS(y_true, pr_window, base) * 100
    cw_s, cw_p, cw_sig = ev.clark_west_test(y_true, pr_window, base)
    print(f'  {tag:34s} R2OS={r2:7.3f}  CW stat={cw_s:6.3f}  p={cw_p:.4f} {cw_sig}', flush=True)
    return {'R2OS': r2, 'CW_stat': cw_s, 'CW_p': cw_p, 'CW_sig': cw_sig}


rows = []
print('\n===== 赛马（同数据/同窗口/同 HA 基准；h=1 月度，clean 无前视口径） =====', flush=True)

# 臂 1a/1b：HLW PLS clean，raw
for tag, M, mset in (('HLW-PLS (clean triPLS)', M24, '24 (21+DSUV/DIVOL/DOID)'),
                     ('HLW-PLS (clean triPLS)', D9, '9 (SPF-only)')):
    pr = pls_h(M, y, N_TRAIN, h=1, i_max=END)
    met = evaluate(pr, f'{tag} [{mset}]')
    rows.append({'arm': 'HLW-PLS', 'measure_set': mset, 'predictor': 'raw',
                 'aggregator': 'PLS single index (expanding, clean pi)',
                 'n_train': N_TRAIN, 'window': '1991-01~2018-12', 'n_oos': len(y_true),
                 'source': 'this script', **met})

# 臂 2/3：幂集 CSR（9 面板）；全 OOS 序列重算以做 gate (g1)，再切片入窗
pr_csr = {}
for tag, M in (('raw', D9), ('d~_pair', D_pair9)):
    pr_full = ev.powerset_csr_preds(M, y, N_TRAIN)   # 行 84..491（408 点）
    stored = pd.read_csv(RES / 'preds_1991_p9.csv', parse_dates=['date'])
    assert (stored['date'].values == dates[N_TRAIN:].values).all()
    diff = np.abs(stored[tag].values.astype(float) - pr_full).max()
    assert diff < 6e-7, f'gate(g1) 失败：{tag} 与 preds_1991_p9.csv 最大差 {diff:.2e}'
    print(f'gate(g1) {tag}: 与 preds_1991_p9.csv 逐点一致（max|diff|={diff:.2e}）', flush=True)
    pr_csr[tag] = pr_full[:END - N_TRAIN + 1]
    met = evaluate(pr_csr[tag], f'Powerset CSR (9, 511 subsets) [{tag}]')
    rows.append({'arm': f'CSR-{tag}', 'measure_set': '9 (SPF-only)', 'predictor': tag,
                 'aggregator': 'powerset CSR (2^9-1 subsets, equal weight)',
                 'n_train': N_TRAIN, 'window': '1991-01~2018-12', 'n_oos': len(y_true),
                 'source': 'this script', **met})

# ---------------- gate (g2)：锚点复现确认（segment_1991_p9.csv，Table A2 来源） ----------------
print('\n===== 锚点核对（既有 results/segment_1991_p9.csv，1991-2018 段） =====', flush=True)
seg = pd.read_csv(RES / 'segment_1991_p9.csv').set_index('series')
ANCHOR = {'raw': ('raw_mean3', 3.44), 'd~_pair': ('d~_pair_mean3', 3.44)}
anchor_ok = True
for tag, (srow, _ref) in ANCHOR.items():
    a_r2, a_p, a_n = seg.loc[srow, ['1991-2018', '1991-2018_p', 'n_1991-2018']]
    r2_new = ev.R2OOS(y_true, pr_csr[tag], base) * 100
    _, p_new, _ = ev.clark_west_test(y_true, pr_csr[tag], base)
    ok = abs(r2_new - a_r2) < 5e-4 and abs(p_new - a_p) < 5e-4 and int(a_n) == 336
    anchor_ok &= ok
    print(f'  {tag:8s} 锚点 R2OS={a_r2:.4f} (p={a_p:.4f}, n={int(a_n)})  |  '
          f'本脚本 R2OS={r2_new:.4f} (p={p_new:.4f})  ->  {"✓ 一致" if ok else "✗ 不一致"}',
          flush=True)
    rows.append({'arm': f'CSR-{tag}', 'measure_set': '9 (SPF-only)', 'predictor': tag,
                 'aggregator': 'powerset CSR', 'n_train': N_TRAIN,
                 'window': '1991-2018', 'n_oos': int(a_n),
                 'source': 'segment_1991_p9.csv (anchor)', 'R2OS': a_r2,
                 'CW_stat': np.nan, 'CW_p': a_p, 'CW_sig': ''})
assert anchor_ok, 'gate(g2) 失败：锚点未复现'
print(f'  d̃_pair 1991–2018 既有锚点值 3.44% 复现确认 ✓（精确值 {seg.loc["d~_pair_mean3", "1991-2018"]:.4f}）',
      flush=True)

tab = pd.DataFrame(rows)
tab.round(4).to_csv(RES / 'hlw_aligned_p9.csv', index=False)

# ---------------- 归因材料：Huang 原窗（n=335）下既有桥接/前视检验结果 ----------------
print('\n===== 归因材料（既有落盘；Huang 原窗 OOS 收益月 1991:02–2018:12，n_oos=335） =====', flush=True)
hb = pd.read_csv(RES / 'huang_bridge.csv')
sel = hb[(hb['align'] == 'qe') & (hb['index'] == 'PLS') & (hb['h'] == 1)]
for _, r in sel.iterrows():
    print(f'  huang_bridge.csv  qe/{r["purge"]:3s}/PLS/h=1: R2OS={r["R2OS"]:7.3f}  '
          f'CW p={r["CW_p"]:.3f}  (Huang 原文 {r["Huang_R2OS"]:.2f})', flush=True)
hl = pd.read_csv(RES / 'huang_pls_lookahead.csv')
sel2 = hl[hl['h'] == 1]
for _, r in sel2.iterrows():
    print(f'  huang_pls_lookahead.csv  {r["purge"]:3s}/h=1/pi={r["pi_mode"]:9s}: '
          f'R2OS={r["R2OS"]:7.3f}  CW p={r["CW_p"]:.4f}', flush=True)
print('  注：huang_bridge/lookahead 的 CW 用 NW lag=2h=2；本脚本赛马表用默认自动 lag（锚点口径）。',
      flush=True)

print('\n===== results/hlw_aligned_p9.csv =====', flush=True)
print(tab.round(4).to_string(index=False), flush=True)
print(f'\n已写 results/hlw_aligned_p9.csv（{len(tab)} 行）', flush=True)
