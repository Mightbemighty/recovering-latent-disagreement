# -*- coding: utf-8 -*-
"""
K 扫描 + 固定 K 全枚举 CSR（complete subset regression，EGT 精确实现）
=====================================================================
动机（2026-08-14 导师反馈 Q4/Q5）：随机 50×11 被质疑 arbitrary——
为什么是 11、为什么是 50、为什么不取全部组合。本脚本把子集集成替换为
**固定 K 的全枚举 CSR**：C(21,K) 个子集逐月 expanding OLS、等权平均，
无种子、无 S、结果确定；并对 K ∈ {1,5,8,11,15,18,21} 扫描展示敏感性高原。

口径与 run_evidence.py 逐点一致：季末面板、OOS 起点由 --start 给出
（默认 1991，n_train=84；--start 2001 = 旧主线 n_train=204）、
expanding 递归（i+1 期预测用 ≤i 数据）、自算 HA 基准（ha[t]=mean(y[:t])）。
实现：按月聚合充分统计量 G=X'X / c=X'y，子集正规方程分块批量求解
（np.linalg.solve；奇异块回退 lstsq），与 lstsq 语义差异以 k=21 全样本
对照 expanding_preds 校验。

验证锚点：
- K=1 全枚举 ≡ A0-meanUniv（attribution_2001.csv：raw +0.21 / d~_lv +0.59）
- K=11 全枚举 ≈ S=50 抽样 mean3（raw 2.695 / d~_lv 3.796，MC 误差内）
- K=21 全枚举 ≡ expanding_preds(全 21 列)（kitchen-sink OLS）

输出：results/kscan_{start}.csv（指标表）+ results/preds_kscan_{start}.csv（预测序列）

--panel9（2026-08-19）：9 面板 K 层剖面（K=1..9 全层），校验改 K=9 对
expanding_preds(全 9 列)；输出带 _p9 后缀。供幂集 CSR 的层权披露与
§21 结论 4（大 K 层 REC 集中）的 9 面板对应检查。
"""
import argparse
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # noqa: E402  （模块级完成数据加载与变体构建；csr_enum_preds 2026-08-17 移入其中）

# ---- 起点开关（2026-08-20：默认 1991 主线 n_train=84；--start 2001 复现 2001 口径） ----
# 模块级预解析：run_kscan_burn / run_kscan_burn_grid 以 import 方式复用 N_TRAIN/metric_row
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--start', type=int, default=1991, choices=sorted(ev.STARTS))
_args0, _ = _ap0.parse_known_args()
START = _args0.start
N_TRAIN = ev.STARTS[START]
P = ev.D.shape[1]  # 21（默认）或 9（--panel9，经 run_evidence 模块级预解析生效）
KS = [1, 5, 8, 11, 15, 18, 21] if P == 21 else list(range(1, P + 1))
PTAG = '_p9' if ev.PANEL9 else ''
VARIANTS = {'raw': ev.variants['raw'], 'd~_lv': ev.variants['d~_lv']}
if ev.PANEL9:
    VARIANTS['d~_pair'] = ev.variants['d~_pair']  # 9 面板主变体（2026-08-19 起），层剖面披露必备
RES = HERE / 'results'


def metric_row(vtag, k, n_sub, pr, y_oos, ha_oos, usrec, fp_template):
    r_full = ev.R2OOS(y_oos, pr, ha_oos) * 100
    cw_s, cw_p, _ = ev.clark_west_test(y_oos, pr, ha_oos)
    r_rec, r_exp = ev.R2OOS_by_USREC(y_oos, pr, ha_oos, usrec)
    fp = fp_template.copy()
    fp.iloc[N_TRAIN:] = pr
    row = {'variant': vtag, 'K': k, 'n_subsets': n_sub,
           'full': r_full, 'CW_stat': cw_s, 'CW_p': cw_p,
           'REC': r_rec * 100, 'EXP': r_exp * 100, 'gap': (r_rec - r_exp) * 100}
    n_oos = len(y_oos)
    for g in (3, 5):
        row[f'CER{g}_tc0'] = ev.cer(fp, ev.sp['raw'], pd.Series(ev.ha), ev.mkt['RF_l1'] / 100,
                                    g, transaction_cost=0.0, oos_len=n_oos)
    row['CER3_tc50'] = ev.cer(fp, ev.sp['raw'], pd.Series(ev.ha), ev.mkt['RF_l1'] / 100,
                              3, transaction_cost=0.005, oos_len=n_oos)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', nargs='+', type=int, default=KS)
    ap.add_argument('--start', type=int, default=START, choices=sorted(ev.STARTS))  # 生效值在模块级预解析；此处为通过参数校验
    ap.add_argument('--panel9', action='store_true')  # 切换由 run_evidence 模块级预解析读取；此处仅为通过参数校验
    args = ap.parse_args()

    y = ev.y
    n_oos = len(y) - N_TRAIN
    y_oos, ha_oos = y[N_TRAIN:], ev.ha[N_TRAIN:]
    usrec = ev.usrec_full.iloc[-n_oos:].reset_index(drop=True)
    fp_template = pd.Series(np.full(len(y), np.nan))

    rows, preds_store = [], {}
    for vtag, M in VARIANTS.items():
        for k in args.ks:
            t0 = time.time()
            pr = ev.csr_enum_preds(M, y, N_TRAIN, k)
            dt = time.time() - t0
            n_sub = comb(M.shape[1], k)
            row = metric_row(vtag, k, n_sub, pr, y_oos, ha_oos, usrec, fp_template)
            rows.append(row)
            preds_store[f'{vtag}_k{k}'] = pr
            print(f'  {vtag:6s} K={k:2d} (C={n_sub:>7,})  full={row["full"]:7.3f}  '
                  f'CW p={row["CW_p"]:.3f}  REC={row["REC"]:7.2f}  EXP={row["EXP"]:6.2f}  '
                  f'CER3_tc0={row["CER3_tc0"]:6.2f}   [{dt:.0f}s]', flush=True)

            # ---- 校验 ----
            if k == P:  # 全池 k=p：与 lstsq 全样本实现逐点对照
                ref = ev.expanding_preds(M, y, N_TRAIN)
                d = np.abs(pr - ref).max()
                assert d < 1e-6, f'{vtag} K={P} 与 lstsq 偏差 {d:.2e}'
                print(f'    [check] K={P} vs expanding_preds max|Δ|={d:.2e} ✓')
            if k == 1:  # 与 meanUniv（逐列单变量 OLS 取均值）逐点对照
                ref = np.column_stack([ev.expanding_preds(M[:, c], y, N_TRAIN)
                                       for c in range(M.shape[1])]).mean(axis=1)
                d = np.abs(pr - ref).max()
                assert d < 1e-6, f'{vtag} K=1 与 meanUniv 偏差 {d:.2e}'
                print(f'    [check] K=1 vs meanUniv max|Δ|={d:.2e} ✓')

    res = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    res.round(4).to_csv(RES / f'kscan_{START}{PTAG}.csv', index=False)
    pd.DataFrame(preds_store, index=ev.dates[N_TRAIN:]).round(6).to_csv(RES / f'preds_kscan_{START}{PTAG}.csv')
    print(f'\n落盘: results/kscan_{START}{PTAG}.csv, results/preds_kscan_{START}{PTAG}.csv')

    # 对照锚点提示
    if ev.PANEL9:
        print(f'\n锚点提示：幂集 CSR 主线值见 variants_{START}_p9.csv（powerset_csr_preds，'
              '应落于 K 层包络之内）；K 层剖面图 run_kscan_fig.py --panel9。')
    else:
        print('\n锚点对照：raw S=50 mean3 = 2.695；d~_lv S=50 mean3 = 3.796；'
              'meanUniv raw = +0.21 / d~_lv = +0.59（attribution_2001.csv）')


if __name__ == '__main__':
    main()
