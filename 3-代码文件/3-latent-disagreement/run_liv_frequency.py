# -*- coding: utf-8 -*-
"""
议程 6e 追加：Livingston 利率分歧面板的 horizon 检验（2026-08-25；`run_liv_external.py` 下游）
======================================================================================
动机：§53 的 Livingston 臂在主线月度 h=1 下 raw 双口径显著而 d̃_pair 跨口径不稳
（n_train=84 翻负）；Livingston 为半年度调查（ffill 6 个月稀释月度网格时间分辨率），
且其 TBILL/TBOND 预测本身为 6M/12M horizon——horizon 匹配原理提示 h=3/h=6/h=12 口径
可能更适合该面板。本脚本把同一面板与机器（expanding 实时去污染 burn=36 + 幂集 CSR
15 子集）搬到 h=3/h=6/h=12 重叠目标上检验，口径全部复刻 run_frequency.py 的 M3 约定：
- 目标：yh[t] = Σ_{j=0..h−1} y[t+j]（y = excess_ret_h1，log 口径求和精确），主网格
  全长构造后切片子网格（1993-06 起，与 run_liv_external 同一对齐断言）。
- 无前视：预测行 i 训练只用行 s ≤ i−h（目标已完成）；HA 基准同信息集
  ha_h[i] = mean(yh[s], s ≤ i−h)（主网格全长累计后切片，基准保留 1984 起历史）。
- 推断：重叠目标的 CW 检验用 Newey-West lag = 2h（run_frequency M3 约定：h=3→6）。
  注意：h=1 格的 CW p 因此与 §53（自动 lag）略有出入（如 raw nt=84：0.058 vs 0.037），
  R²OOS 逐值一致。
- 单测度臂：4 测度各自 d̃_pair/raw expanding OLS，检验"单测度弱、集成强"结构的 horizon 依赖。
- 训练窗扫描（2026-08-25，回应 n_train=84/120 双口径的任意性质疑）：n_train 自 84
  （7 年，与主线 1991 口径最小训练窗同长）至 180（15 年）步长 12，h ∈ {1,3,6}，
  raw / d̃_pair 两臂——h=6 结论须不依赖任何单一 cutoff 方成立。
- ridge 网格（2026-08-25，检验 CSR 保守性假说）：panel Ridge λ ∈ {1,10,100,1000}
  （expanding 窗内标准化、截距不受罚，引擎复刻 run_shrinkage_horserace.py 加 horizon
  截断），h ∈ {1,3,6,12} × n_train ∈ {84,120}，raw / d̃_pair——去污染增益随收缩强度
  的形态（主线 §6.3.3：增益随收缩放松单调上升）在本面板是否同构。
自检：同机 SPF-9 复现主线锚点 d̃_pair R²OOS = 3.956（2001 口径 h=1）方许出数。
输出：results/liv_freq_h{3,6,12}_{84,120}.csv（五变体）、liv_freq_single_h{3,6,12}_{84,120}.csv
      （单测度臂）、liv_freq_sweep.csv（训练窗扫描）、liv_freq_ridge.csv（ridge 网格）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

import run_evidence as ev
from run_frequency import ha_h, expanding_preds_h  # 复用频率附录口径（horizon 截断 + HA 同信息集）

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
MS = ['TBILL6M', 'TBILL12M', 'TBOND6M', 'TBOND12M']

y_full, ha_full = ev.y, ev.ha
n_full = len(y_full)


# ---- 主网格 h 期累积目标（log 收益求和，精确）：yh[t] = Σ_{j<h} y[t+j]，t ≤ n−h ----
def build_yh(h):
    yh = np.full(n_full, np.nan)
    acc = np.zeros(n_full)
    for j in range(h):
        acc += np.roll(y_full, -j)
    yh[:n_full - h + 1] = acc[:n_full - h + 1]
    return yh


p9 = pd.read_csv(HERE / 'panel9_qe.csv', parse_dates=['date'])
liv = pd.read_csv(HERE / 'liv4_qe.csv', parse_dates=['date'])
i0 = int(np.where(p9['date'].values == liv['date'].iloc[0].to_datetime64())[0][0])
assert (p9['date'].iloc[i0:].values == liv['date'].values).all()
n = len(liv)
print(f'Livingston 子网格：{liv["date"].iloc[0]:%Y-%m} 起 {n} 行（主线 iloc {i0}）')

D = liv[[f'd_{m}' for m in MS]].values.astype(float)
L = liv[[f'l_{m}' for m in MS]].values.astype(float)
assert not np.isnan(D).any() and not np.isnan(L).any()
D_pair = np.column_stack([ev.decontaminate(D[:, [k]], L[:, [k]])[:, 0] for k in range(4)])
D_lv = ev.decontaminate(D, L)
variants = {'raw': D, 'd~_pair': D_pair, 'd~_lv': D_lv, 'lv_only': L,
            'd~_lv+l(8)': np.column_stack([D_lv, L])}


def panel_ridge_h(M, yh, n_train, h, lam):
    """panel Ridge + horizon 截断：预测行 i 训练只用行 s ≤ i−h（目标已完成）；
    expanding 窗内标准化（ddof=1），截距经 y 中心化不受罚。复刻 run_shrinkage_horserace.py。"""
    last = len(yh) - h
    out = np.empty(last - n_train + 1)
    for j, i in enumerate(range(n_train, last + 1)):
        Xtr, ytr = M[:i - h + 1], yh[:i - h + 1]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (Xtr - mu) / sd
        yc = ytr - ytr.mean()
        beta = np.linalg.solve(Xs.T @ Xs + lam * np.eye(M.shape[1]), Xs.T @ yc)
        out[j] = ytr.mean() + ((M[i] - mu) / sd) @ beta
    return out


def _eval_grid(h):
    """返回该 horizon 的子网格目标、基准与末行。"""
    yh_full = build_yh(h)
    base_full = ha_h(yh_full, 0, h)[i0:]  # 全长 HA（n_train 参数不参与 ha_h 计算）
    return yh_full[i0:], base_full, n - h


def run_horizon(h, n_train, tag):
    yh, base_full, last = _eval_grid(h)
    y_true = yh[n_train:last + 1]
    base = base_full[n_train:last + 1]
    assert not np.isnan(y_true).any() and not np.isnan(base).any()
    rows = []
    for name, M in variants.items():
        pr = ev.powerset_csr_preds(M, yh, n_train, h=h)
        assert len(pr) == len(y_true)
        r2 = ev.R2OOS(y_true, pr, base) * 100
        cw_s, cw_p, _ = ev.clark_west_test(y_true, pr, base, lag=2 * h)
        rows.append({'variant': name, 'R2OOS': r2, 'CW_stat': cw_s, 'CW_p': cw_p})
        print(f'  [{tag}] {name:12s} R2OOS={r2:7.3f}  CW p={cw_p:.3f}')
    res = pd.DataFrame(rows).set_index('variant')
    res.round(4).to_csv(OUT / f'liv_freq_{tag}.csv')

    # 单测度臂（d̃_pair 与 raw，expanding OLS）
    srows = []
    for k, m in enumerate(MS):
        for vn, M1 in ((f'raw_{m}', D[:, [k]]), (f'd~_pair_{m}', D_pair[:, [k]])):
            pr = np.asarray(expanding_preds_h(M1, yh, n_train, h))
            r2 = ev.R2OOS(y_true, pr, base) * 100
            cw_s, cw_p, _ = ev.clark_west_test(y_true, pr, base, lag=2 * h)
            srows.append({'arm': vn, 'R2OOS': r2, 'CW_stat': cw_s, 'CW_p': cw_p})
            print(f'  [{tag}] single {vn:16s} R2OOS={r2:7.3f}  CW p={cw_p:.3f}')
    sres = pd.DataFrame(srows).set_index('arm')
    sres.round(4).to_csv(OUT / f'liv_freq_single_{tag}.csv')
    return res, sres


def run_sweep():
    """训练窗扫描：n_train ∈ {84,96,...,180} × h ∈ {1,3,6}，raw / d̃_pair。"""
    rows = []
    for h in (1, 3, 6):
        yh, base_full, last = _eval_grid(h)
        for nt in range(84, 181, 12):
            yt, bs = yh[nt:last + 1], base_full[nt:last + 1]
            row = {'h': h, 'n_train': nt, 'n_oos': len(yt)}
            for name in ('raw', 'd~_pair'):
                pr = ev.powerset_csr_preds(variants[name], yh, nt, h=h)
                row[f'R2_{name}'] = ev.R2OOS(yt, pr, bs) * 100
                _, p, _ = ev.clark_west_test(yt, pr, bs, lag=2 * h)
                row[f'CWp_{name}'] = p
            row['gain_pair-raw'] = row['R2_d~_pair'] - row['R2_raw']
            rows.append(row)
            print(f'  [sweep] h={h} nt={nt:3d}  raw={row["R2_raw"]:7.2f} (p={row["CWp_raw"]:.3f})  '
                  f'pair={row["R2_d~_pair"]:7.2f} (p={row["CWp_d~_pair"]:.3f})  Δ={row["gain_pair-raw"]:+6.2f}')
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / 'liv_freq_sweep.csv', index=False)
    return res


def run_ridge_grid():
    """ridge 网格：h ∈ {1,3,6,12} × n_train ∈ {84,120} × λ ∈ {1,10,100,1000}，raw / d̃_pair。"""
    rows = []
    for h in (1, 3, 6, 12):
        yh, base_full, last = _eval_grid(h)
        for nt in (84, 120):
            yt, bs = yh[nt:last + 1], base_full[nt:last + 1]
            for lam in (1, 10, 100, 1000):
                row = {'h': h, 'n_train': nt, 'lambda': lam}
                for name in ('raw', 'd~_pair'):
                    pr = panel_ridge_h(variants[name], yh, nt, h, lam)
                    row[f'R2_{name}'] = ev.R2OOS(yt, pr, bs) * 100
                row['gain_pair-raw'] = row['R2_d~_pair'] - row['R2_raw']
                rows.append(row)
            print(f'  [ridge] h={h:2d} nt={nt:3d}  Δ(pair−raw): ' +
                  '  '.join(f'λ={r["lambda"]:4.0f}:{r["gain_pair-raw"]:+6.2f}'
                            for r in rows[-4:]))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / 'liv_freq_ridge.csv', index=False)
    return res


def self_check():
    names9 = ['GDP', 'IP', 'CON', 'INV', 'HSG', 'UEP', 'CPI', 'TBL', 'CPROF']
    D9 = p9[[f'd_{m}' for m in names9]].values.astype(float)
    L9 = p9[[f'l_{m}' for m in names9]].values.astype(float)
    D9_pair = np.column_stack([ev.decontaminate(D9[:, [k]], L9[:, [k]])[:, 0] for k in range(9)])
    pr = ev.powerset_csr_preds(D9_pair, y_full, 204)
    r2 = ev.R2OOS(y_full[204:], pr, ha_full[204:]) * 100
    ok = abs(r2 - 3.956) < 0.01
    print(f'[self-check] SPF-9 d~_pair R2OOS = {r2:.4f}（锚点 3.956）→ {"PASS" if ok else "FAIL"}')
    assert ok, '自检失败：机器不等价，禁止出数'


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    self_check()
    for h in (3, 6, 12):
        for nt in (84, 120):
            tag = f'h{h}_{nt}'
            print(f'\n===== Livingston 4 测度幂集 CSR（h={h}，n_train={nt}，'
                  f'OOS {liv["date"].iloc[nt]:%Y-%m} 起，n_oos={n - h - nt + 1}，NW lag={2 * h}）=====')
            run_horizon(h, nt, tag)
    print('\n===== 训练窗扫描（n_train 84–180 步长 12，h ∈ {1,3,6}）=====')
    run_sweep()
    print('\n===== ridge 网格（h ∈ {1,3,6,12}，n_train ∈ {84,120}，λ ∈ {1,10,100,1000}）=====')
    run_ridge_grid()
    print('\n完成（议程 6e 追加：horizon 检验 + 训练窗扫描 + ridge 网格）。')
