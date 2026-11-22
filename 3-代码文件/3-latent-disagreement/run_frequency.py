# -*- coding: utf-8 -*-
"""
频率附录管线（Paper 2 附录：与 Deng et al. 2024 季度证据的桥接）
================================================================
动机：H4（lv 无预测力）是月度 h=1 的操作事实；Deng et al. (2024, JFQA) 发现 SPF 共识
水平的 PLS 指数在季度频率预测市场收益。本附录把"频率"分解为两个机制：
  (i) horizon 效应：M1(主线) → M3，抽样不变、horizon 1月→3月（重叠）；
  (ii) 新鲜度效应：M3 → Q1，horizon 相同（都是未来 3 个月收益）、月度抽样→季末抽样
       （SPF 在月度面板上 2/3 观测为 ffill 陈旧信息，季度抽样后每个观测都是当季新调查；
       SCA/AF 天然月度新鲜 → M3↔Q1 差应≈0，构成内建安慰剂）。
年度频率不做（24 个非重叠点功效不足），留待新工作。

设计
----
- M3：月度抽样，目标 = 未来 3 个月 log 超额收益之和（excess_ret_h1 为 log 口径，求和精确）。
  预测第 t 行时训练只用目标已完成的样本对（s ≤ t−3；h=1 时退化为 s ≤ t−1 = 主线口径），
  PLS 载荷 / 去污染 γ / OLS / HA 全部同规则——严格无前视（重叠只影响推断：CW 用 NW lag=6）。
- Q1：季末行抽样（月份 3/6/9/12），目标 = 同一行起的 3 个月收益之和（= 下一季度，非重叠）。
  全部构造季度原生化：去污染 burn=12 季（≡ 主线 36 月）、expanding OLS、50×11 子集集成、
  triPLS 均在季度子面板上运行。CW 用 NW lag=3。
- 样本边界与主线一致（--start 参数化，2026-08-20 起默认 1991 主线）：月度 n_train=ev.STARTS[start]
  （1991：OOS 行 84..489，406 点；2001：行 204..489，286 点）；
  季度 n_train=QE 中小于月度 n_train 的季数（1991：28 季，OOS 135 季，预测时点 1991-03..2024-09；
  2001：68 季，OOS 95 季，预测时点 2001-03..2024-09）。
- HA 自算、仅用已完成目标（同信息集）；REC/EXP 按预测时点 USREC（季度衰退点 ~10 个，仅描述性）。
- 自检（--selfcheck）：h=1 配置须复现主线锚点（仅 2001 起点 21 面板启用，variants_2001.csv
  全枚举口径：lv_only ≈ −0.908 / raw ≈ 2.748 / d̃_lv ≈ 4.004），误差 >0.02 即报错；
  其余起点/面板组合跳过自检（h=1 自洽性由 cfdr_race / t1 的落盘 gate 覆盖）。

--panel9（2026-08-19 面板重构，thoughts/17 执行清单第 6 步）：
- 面板切换由 run_evidence 模块级预解析完成（9 测度 SPF-only，panel9_qe.csv）；
  集成规则随 ev.subset_ensemble 同口径改幂集 CSR（2^p−1 非空子集等权，带 h 截断）。
- lv 侧子池格（SPF13/SCA6/AF2）不适用：9 面板全部测度属 SPF，无 SCA/AF
  月度新鲜安慰剂；lv 侧只保留全池 lv_ens_all9 / lv_PLS_all9 两格。
- 自检锚点为 21 面板口径，--panel9 下跳过（9 面板 h=1 主线锚点见 variants_2001_p9.csv）。
- 输出带 ev.PTAG（_p9）后缀，不覆写 21 面板文件。

起点参数化（2026-08-20）：--start ∈ {2001, 1991}，默认 1991 主线；产出文件名在
2001 起点保持旧名（回归兼容），1991 起点加 _1991（frequency_{m3,q1}[_1991]{_p9}.csv）。

输出：results/frequency_{m3,q1}[_1991]{_p9}.csv + results/frequency_preds_{m3,q1}[_1991]{_p9}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # 复用锁定面板加载 / decontaminate / MetricsDef / 种子约定

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
STAG = '' if START == 2001 else f'_{START}'  # 产出文件：2001 起点沿用旧名（回归兼容）

y, D, L, U = ev.y, ev.D, ev.L, ev.U
dates, pairs = ev.dates, ev.pairs
n = len(y)

MEAS = [l[2:] for l, _ in pairs]  # 列序 = 字母序（run_evidence sorted(l_cols)）
IDX = {'all21': list(range(21)),
       'SPF13': [k for k, m in enumerate(MEAS) if m in ev.SPF],
       'SCA6': [k for k, m in enumerate(MEAS) if m in ev.SCA],
       'AF2': [k for k, m in enumerate(MEAS) if m in ev.AF]}
K_TBL = MEAS.index('TBL')

# 前向 3 个月目标（log 收益求和，精确）：y3[t] = y[t]+y[t+1]+y[t+2]，t ≤ n−3
y3 = np.full(n, np.nan)
y3[:n - 2] = y[:n - 2] + y[1:n - 1] + y[2:]

# USREC 与面板逐行对齐（USREC.csv 可能更长，取末 492 行）
us = ev.usrec_full.iloc[-n:].reset_index(drop=True)
assert len(us) == n


# ---------------- 带 horizon 截断的核心函数（h=1 时与主线逐点一致） ----------------
def expanding_preds_h(X_in, y_in, n_train, h):
    """预测行 i ∈ [n_train, len−h]，训练只用行 s ≤ i−h（目标已完成）。"""
    X_in = np.atleast_2d(X_in).T if X_in.ndim == 1 else X_in
    Xc = np.column_stack([np.ones(len(y_in)), X_in])
    last = len(y_in) - h
    preds = np.empty(last - n_train + 1)
    for j, i in enumerate(range(n_train, last + 1)):
        beta, *_ = np.linalg.lstsq(Xc[:i - h + 1], y_in[:i - h + 1], rcond=None)
        preds[j] = Xc[i] @ beta
    return preds


def pls_huang_h(Xmat, y_in, n_train, h):
    """triPLS（Huang 三步法口径）+ horizon 截断：预测行 i 时 π 与 OLS 只用行 s ≤ i−h。"""
    n_, K = Xmat.shape
    last = n_ - h
    preds = np.empty(last - n_train + 1)
    for j, i in enumerate(range(n_train, last + 1)):
        tr = slice(0, i - h + 1)
        Yf = y_in[tr]
        Rc = np.column_stack([np.ones(len(Yf)), Yf])
        pi = np.linalg.lstsq(Rc, Xmat[tr], rcond=None)[0][1]
        idx_tr = Xmat[tr] @ pi / K
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Yf)), idx_tr]), Yf, rcond=None)
        preds[j] = np.array([1.0, Xmat[i] @ pi / K]) @ beta
    return preds


def subset_ensemble_h(M, y_in, n_train, h, k=ev.K, seeds=ev.SEEDS, b=ev.B):
    """主线口径（2026-08-17 起）：C(p,k) ≤ ev.ENUM_CAP 时全枚举 CSR（确定性）；
    超上限保留 S=b×3 抽样。--panel9 口径（2026-08-19 起）：幂集 CSR
    （2^p−1 非空子集等权，与 ev.subset_ensemble 同规则，支持 h 截断）。"""
    if ev.PANEL9:
        pr = ev.powerset_csr_preds(M, y_in, n_train, h=h)
        return [pr, pr, pr]
    kk = min(k, M.shape[1])
    if ev.comb(M.shape[1], kk) <= ev.ENUM_CAP:
        pr = ev.csr_enum_preds(M, y_in, n_train, kk, h=h)
        return [pr, pr, pr]
    prs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ss = [sorted(rng.choice(M.shape[1], kk, replace=False).tolist()) for _ in range(b)]
        prs.append(np.column_stack([expanding_preds_h(M[:, s], y_in, n_train, h) for s in ss]).mean(axis=1))
    return prs


def ha_h(y_in, n_train, h):
    """HA 基准：ha[i] = mean(y_in[s], s ≤ i−h)（仅用已完成目标，与模型同信息集）。"""
    cs = np.concatenate([[0.0], np.nancumsum(y_in)])
    idx = np.arange(len(y_in)) - h + 1          # i − h + 1 个已完成观测
    out = np.full(len(y_in), np.nan)
    valid = idx >= 1
    out[valid] = cs[idx[valid]] / idx[valid]
    return out


# ---------------- 单元格评估 ----------------
def eval_cells(cells, y_in, n_train, h, cw_lag, usrec_oos, oos_dates, tag):
    """cells: dict name -> (matrix, kind)；kind ∈ {'ens','pls','single'}。
    返回 (汇总 DataFrame, seed0 预测 DataFrame)。"""
    y_true = y_in[n_train:len(y_in) - h + 1]
    base = ha_h(y_in, n_train, h)[n_train:len(y_in) - h + 1]
    assert not np.isnan(y_true).any() and not np.isnan(base).any()
    rows, preds_store = [], {}
    for name, (M, kind) in cells.items():
        if kind == 'ens':
            prs = subset_ensemble_h(M, y_in, n_train, h)
        elif kind == 'pls':
            prs = [pls_huang_h(M, y_in, n_train, h)]
        else:
            prs = [expanding_preds_h(M, y_in, n_train, h)]
        r2s = [ev.R2OOS(y_true, pr, base) * 100 for pr in prs]
        pr0 = prs[0]
        cw_s, cw_p, cw_sig = ev.clark_west_test(y_true, pr0, base, lag=cw_lag)
        r_rec, r_exp = ev.R2OOS_by_USREC(y_true, pr0, base, usrec_oos)
        rows.append({'cell': name, 'kind': kind, 'n_oos': len(y_true),
                     'R2_s0': r2s[0], 'R2_meanS': np.mean(r2s), 'R2_minS': np.min(r2s),
                     'CW_stat': cw_s, 'CW_p': cw_p, 'CW_sig': cw_sig,
                     'REC': r_rec * 100, 'EXP': r_exp * 100})
        preds_store[name] = pr0
        print(f'  [{tag}] {name:16s} R2(meanS)={np.mean(r2s):7.3f}  CW p={cw_p:.3f}  '
              f'REC={r_rec * 100:7.2f}  EXP={r_exp * 100:6.2f}')
    res = pd.DataFrame(rows).set_index('cell')
    pr = pd.DataFrame(preds_store, index=oos_dates)
    return res, pr


# ---------------- 变体集合 ----------------
def build_cells(Dm, Lm, Um, burn):
    d_lv = ev.decontaminate(Dm, Lm, burn=burn)
    d_pair = np.column_stack([ev.decontaminate(Dm[:, [k]], Lm[:, [k]], burn=burn)[:, 0]
                              for k in range(Dm.shape[1])])
    d_u14 = ev.decontaminate(Dm, Um, burn=burn)
    d_tbl = ev.decontaminate(Dm[:, [K_TBL]], Lm, burn=burn)
    cells = {
        # dis 侧
        'raw_ens': (Dm, 'ens'),
        'd~_lv_ens': (d_lv, 'ens'),
        'd~_pair_ens': (d_pair, 'ens'),
        'd~_u14_ens': (d_u14, 'ens'),
        'd_PLS_raw': (Dm, 'pls'),
        'd_TBL': (Dm[:, [K_TBL]], 'single'),
        'd~_TBL': (d_tbl, 'single'),
        # lv 侧
        f'lv_ens_all{Dm.shape[1]}': (Lm, 'ens'),
    }
    if not ev.PANEL9:
        # lv 侧子池（21 面板独有：SCA/AF 月度新鲜安慰剂；9 面板全属 SPF，无此分解）
        cells['lv_ens_SPF13'] = (Lm[:, IDX['SPF13']], 'ens')
        cells['lv_ens_SCA6'] = (Lm[:, IDX['SCA6']], 'ens')
        cells['lv_ens_AF2'] = (Lm[:, IDX['AF2']], 'ens')
    cells[f'lv_PLS_all{Dm.shape[1]}'] = (Lm, 'pls')
    if not ev.PANEL9:
        cells['lv_PLS_SPF13'] = (Lm[:, IDX['SPF13']], 'pls')
        cells['lv_PLS_SCA6'] = (Lm[:, IDX['SCA6']], 'pls')
        cells['lv_PLS_AF2'] = (Lm[:, IDX['AF2']], 'pls')
    return cells


# ---------------- M3：月度抽样，h=3 重叠 ----------------
def run_m3(outdir):
    n_train, h = N_TRAIN, 3
    cells = build_cells(ev.D, ev.L, U, burn=ev.BURN)  # 预测器构造与主线完全一致（月度原生）
    usrec_oos = us.iloc[n_train:n - h + 1].reset_index(drop=True)
    oos_dates = dates[n_train:n - h + 1]
    res, pr = eval_cells(cells, y3, n_train, h, cw_lag=6, usrec_oos=usrec_oos,
                         oos_dates=oos_dates, tag='M3')
    res.round(4).to_csv(outdir / f'frequency_m3{STAG}{ev.PTAG}.csv')
    pr.round(6).to_csv(outdir / f'frequency_preds_m3{STAG}{ev.PTAG}.csv')
    return res


# ---------------- Q1：季末抽样，季度原生构造 ----------------
def run_q1(outdir):
    QE = np.where(dates.dt.month % 3 == 0)[0]
    QE = QE[QE <= n - 3]                       # 目标需完整：行 i ≤ n−3
    Dq, Lq, Uq, yq = D[QE], L[QE], U[QE], y3[QE]
    n_train = int(np.sum(QE < N_TRAIN))        # 与主线同一边界（行 N_TRAIN = start-01；1991 → 28 季）
    cells = build_cells(Dq, Lq, Uq, burn=12)   # 季度原生：burn=12 季 ≡ 主线 36 月
    oos_rows = QE[n_train:]
    usrec_oos = us.iloc[oos_rows].reset_index(drop=True)
    oos_dates = dates[oos_rows]
    res, pr = eval_cells(cells, yq, n_train, 1, cw_lag=3, usrec_oos=usrec_oos,
                         oos_dates=oos_dates, tag='Q1')
    res.round(4).to_csv(outdir / f'frequency_q1{STAG}{ev.PTAG}.csv')
    pr.round(6).to_csv(outdir / f'frequency_preds_q1{STAG}{ev.PTAG}.csv')
    print(f'  [Q1] 季度样本 {len(QE)} 季，n_train={n_train}，OOS={len(yq) - n_train} 季 '
          f'({oos_dates.iloc[0].date()}..{oos_dates.iloc[-1].date()}，预测时点)')
    return res


# ---------------- 自检：h=1 须复现主线锚点（仅 2001 起点 21 面板启用） ----------------
def selfcheck():
    if ev.PANEL9 or START != 2001:
        # 锚点为 2001 起点 21 面板全枚举口径；其余组合跳过（h=1 自洽由 cfdr_race/t1 gate 覆盖；
        # 9 面板 h=1 主线锚点见 variants_{start}_p9.csv）
        print(f'  [selfcheck] --panel9 或 --start {START}：2001/21面板锚点不适用，跳过')
        return
    anchors = {'lv_only': (ev.variants['lv_only'], 11, -0.908),
               'raw': (ev.D, ev.K, 2.748),
               'd~_lv': (ev.variants['d~_lv'], ev.K, 4.004)}  # 2026-08-17 全枚举主线口径（旧 mean3：-0.697/2.695/3.796）
    ok = True
    for name, (M, k, ref) in anchors.items():
        prs = subset_ensemble_h(M, y, N_TRAIN, 1, k=k)
        y_true = y[N_TRAIN:]
        base = ev.ha[N_TRAIN:]
        r2m = np.mean([ev.R2OOS(y_true, pr, base) * 100 for pr in prs])
        stat = 'OK' if abs(r2m - ref) < 0.02 else 'MISMATCH'
        ok &= stat == 'OK'
        print(f'  [selfcheck h=1] {name:8s} mean3 R2OOS = {r2m:8.4f}  (锚点 {ref})  {stat}')
    if not ok:
        raise SystemExit('自检失败：h=1 未复现主线锚点，检查截断/对齐逻辑')
    print('自检通过（h=1 与主线逐点一致）')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--start', type=int, default=START, choices=sorted(ev.STARTS))  # 生效值在模块级预解析；此处为通过参数校验
    ap.add_argument('--panel9', action='store_true')  # 切换由 run_evidence 模块级预解析读取；此处仅为通过参数校验
    args = ap.parse_args()
    outdir = HERE / 'results'
    outdir.mkdir(exist_ok=True)
    if args.selfcheck:
        selfcheck()
    print('\n===== M3（月度抽样，h=3 重叠，NW lag=6）=====')
    run_m3(outdir)
    print('\n===== Q1（季末抽样，季度原生，NW lag=3）=====')
    run_q1(outdir)
    print(f'\n落盘: results/frequency_{{m3,q1}}{STAG}{ev.PTAG}.csv, results/frequency_preds_{{m3,q1}}{STAG}{ev.PTAG}.csv')
