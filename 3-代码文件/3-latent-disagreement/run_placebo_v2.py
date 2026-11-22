# -*- coding: utf-8 -*-
"""
Placebo v2（9 面板，2001 主线口径）：结构保留型安慰剂面板
================================================================================
动机：替代旧版 T-D 安慰剂（run_measure_superiority.py，B=200、9 列独立 AR(1)），
回应审稿人三点批评：(i) 未保留实际面板约 0.32 的横截面相关结构；(ii) n=200 的
分辨率下限为 p=0.005；(iii) 未交代 level 列如何模拟。本脚本 B=1000（p 分辨率
0.001），level 列与 disagreement 列**联合**处理（18 列真实面板整体重抽样/模拟）。

口径：限 --panel9（由 run_evidence 模块级预解析分派，断言 ev.PANEL9）；
OOS 2001-01（n_train=204，n_oos=288），HA 基准与主线一致（ha[t]=mean(y[:t])）。
实际锚点（本脚本启动时自算并断言，非硬编码）：raw 2.706 / d̃_pair 3.956 /
增益 +1.25（pp）。

两种安慰剂生成方式（各 B 次，均对全部 18 列 [D | L] 联合操作）：
1. cbb12 / cbb24（主）：环形块自助（circular block bootstrap, CBB）。选择说明：
   采用**简单环形块自助**（固定块长、起点均匀、取模回绕）而非 Politis–Romano
   平稳 bootstrap（几何随机块长）——前者实现透明、块长即审稿语境下的显式规格
   （12 / 24 月两个规格），后者只是块长随机化的混合，结论层面无实质差异。
   块内时序、全部横截面相关与 level–disagreement 联动结构完整保留，
   仅破坏面板与真实收益 y 的时间对齐。
2. ar1corr（辅）：相关结构 AR(1) 模拟。逐列经验 AR(1)（x_t = c + rho x_{t-1} + e，
   全样本校准——placebo 生成非实时对象，沿用 T-D 先例并声明），创新向量按
   18×18 残差协方差的 Cholesky 因子联合生成（同时保留创新方差与横截面相关），
   burn-in 100 步。

每次复制：模拟面板 → raw 臂（D_sim 直接过幂集 CSR）与 d̃_pair 臂（同题 lv 投影
去污染 burn=36 后过幂集 CSR），对**真实** y 算 OOS R²；经验
p = (1 + #{placebo ≥ 实际值}) / (B+1)。

性能：单次复制（decontaminate + 双臂幂集 CSR）实测约 0.5s，远低于 15s 阈值，
故不做充分统计量层优化；B=1000×3 规格用 multiprocessing（Windows spawn，
默认 workers=6）。随机种子 base 12345，每 (规格, rep) 由 SeedSequence 派生。

输出：results/placebo_v2_p9.csv（每 rep 一行：method / block_len / rep / seed /
R2_raw / R2_dpair / gain）+ 控制台分布汇总与经验 p。主程序会覆盖输出目录中的同名文件，必须在隔离副本运行。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # noqa: E402  数据加载 / decontaminate / powerset_csr_preds / R2OOS

_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--panel9', action='store_true')
_ap0.add_argument('--B', type=int, default=1000)
_ap0.add_argument('--workers', type=int, default=6)
_ap0.add_argument('--base-seed', type=int, default=12345)
_args0, _ = _ap0.parse_known_args()
assert ev.PANEL9, 'placebo v2 限 --panel9（9 面板幂集 CSR 口径）'

N_TRAIN = ev.STARTS[2001]  # 2001 主线口径（n_train=204，n_oos=288）
y, ha, D, L = ev.y, ev.ha, ev.D, ev.L
n_p = ev.n_p
P = np.column_stack([D, L])  # 18 列联合面板（d 前 9，l 后 9）
N = len(y)
RES = HERE / 'results'

# 安慰剂规格：(方法标签, 块长/参数)。cbb = 环形块自助；ar1corr = 相关结构 AR(1)。
SPECS = [('cbb12', 12), ('cbb24', 24), ('ar1corr', None)]


# ---------------- 安慰剂面板生成 ----------------
def sim_cbb(rng, block_len):
    """简单环形块自助：起点 iid Uniform{0..N-1}，固定块长，取模回绕，截断至 N 行。"""
    m = int(np.ceil(N / block_len))
    starts = rng.integers(0, N, m)
    idx = (starts[:, None] + np.arange(block_len)).ravel()[:N] % N
    return P[idx]


def _ar1_fit_joint(M):
    """逐列 AR(1)（截距 + 斜率），返回 (c, rho, 残差协方差的 Cholesky)。"""
    x0, x1 = M[:-1], M[1:]
    coef = np.empty((2, M.shape[1]))
    E = np.empty_like(x1)
    for j in range(M.shape[1]):
        A = np.column_stack([np.ones(len(x0)), x0[:, j]])
        coef[:, j], *_ = np.linalg.lstsq(A, x1[:, j], rcond=None)
        E[:, j] = x1[:, j] - A @ coef[:, j]
    Sig = np.cov(E.T, ddof=2)
    return coef[0], coef[1], np.linalg.cholesky(Sig)


def sim_ar1corr(rng, _unused, burn_in=100):
    """相关结构 AR(1)：创新 = Cholesky(残差协方差) @ z，保留横截面相关。"""
    c, rho, Cf = _AR1
    p = P.shape[1]
    T = N + burn_in
    z = rng.standard_normal((T, p))
    X = np.empty((T, p))
    X[0] = c / (1 - rho) + Cf @ z[0]
    for t in range(1, T):
        X[t] = c + rho * X[t - 1] + Cf @ z[t]
    return X[burn_in:]


_AR1 = None  # 模块级缓存（worker 初始化时设定）


def _init_worker():
    """spawn 子进程：重建 AR(1) 校准缓存（D/L/P 等经 fork/spawn 全局继承）。"""
    global _AR1
    if _AR1 is None:
        _AR1 = _ar1_fit_joint(P)


# ---------------- 单次复制：双臂管线 ----------------
def run_pair(Msim):
    """模拟面板 → raw / d̃_pair 两臂幂集 CSR，对真实 y 的 OOS R²（%）。"""
    Ds, Ls = Msim[:, :n_p], Msim[:, n_p:]
    y_oos, ha_oos = y[N_TRAIN:], ha[N_TRAIN:]
    r2_raw = ev.R2OOS(y_oos, ev.powerset_csr_preds(Ds, y, N_TRAIN), ha_oos) * 100
    Dp = np.column_stack([ev.decontaminate(Ds[:, [k]], Ls[:, [k]])[:, 0]
                          for k in range(n_p)])
    r2_d = ev.R2OOS(y_oos, ev.powerset_csr_preds(Dp, y, N_TRAIN), ha_oos) * 100
    return r2_raw, r2_d


def _do_rep(task):
    method, param, rep, seed = task
    _init_worker()
    rng = np.random.default_rng(seed)
    Msim = sim_cbb(rng, param) if method.startswith('cbb') else sim_ar1corr(rng, param)
    t0 = time.perf_counter()
    r2_raw, r2_d = run_pair(Msim)
    dt = time.perf_counter() - t0
    return {'method': method, 'block_len': param if param is not None else '',
            'rep': rep, 'seed': seed, 'R2_raw': r2_raw, 'R2_dpair': r2_d,
            'gain': r2_d - r2_raw, 'sec': dt}


def main():
    B = _args0.B
    # 实际锚点自算并断言（对照任务书声明值，非硬编码进 p 值计算）
    t0 = time.perf_counter()
    act_raw, act_d = run_pair(P)
    print(f'实际值自算: raw={act_raw:.4f}（锚 2.706）  d̃_pair={act_d:.4f}（锚 3.956）  '
          f'gain={act_d - act_raw:.4f}（锚 +1.25）  [{time.perf_counter() - t0:.2f}s]')
    assert abs(act_raw - 2.706) < 0.01 and abs(act_d - 3.956) < 0.01, '实际值与主线锚点不符'

    tasks = []
    for si, (method, param) in enumerate(SPECS):
        for rep in range(B):
            seed = int(np.random.SeedSequence(
                [_args0.base_seed, si, rep]).generate_state(1)[0])
            tasks.append((method, param, rep, seed))

    rows, t_start = [], time.perf_counter()
    if _args0.workers > 1:
        import multiprocessing as mp
        with mp.Pool(_args0.workers, initializer=_init_worker) as pool:
            for j, r in enumerate(pool.imap_unordered(_do_rep, tasks, chunksize=8)):
                rows.append(r)
                if (j + 1) % 100 == 0:
                    el = time.perf_counter() - t_start
                    print(f'  {j + 1}/{len(tasks)} 完成，{el:.0f}s'
                          f'（均 {(el / (j + 1)) * _args0.workers:.2f}s/复制/核）', flush=True)
    else:
        for j, t in enumerate(tasks):
            rows.append(_do_rep(t))
            if (j + 1) % 20 == 0:
                print(f'  {j + 1}/{len(tasks)} 完成，{time.perf_counter() - t_start:.0f}s',
                      flush=True)
    wall = time.perf_counter() - t_start
    df = pd.DataFrame(rows).sort_values(['method', 'rep']).reset_index(drop=True)

    fp = RES / 'placebo_v2_p9.csv'
    df.drop(columns=['sec']).round(4).to_csv(fp, index=False)
    print(f'\n总耗时 {wall:.0f}s（workers={_args0.workers}，B={B}×{len(SPECS)} 规格）；'
          f'单次复制均值 {df["sec"].mean():.2f}s')
    print(f'落盘: results/placebo_v2_p9.csv')

    actuals = {'R2_raw': act_raw, 'R2_dpair': act_d, 'gain': act_d - act_raw}
    print('\n===== 安慰剂分布汇总与经验 p（p = (1 + #{placebo ≥ 实际值})/(B+1)）=====')
    for method, _ in SPECS:
        sub = df[df['method'] == method]
        print(f'\n--- {method}（n={len(sub)}）---')
        for stat in ['R2_raw', 'R2_dpair', 'gain']:
            v = sub[stat].values
            emp_p = (1 + (v >= actuals[stat]).sum()) / (len(v) + 1)
            print(f'  {stat:8s} 均值={v.mean():+7.3f} sd={v.std():6.3f} '
                  f'q50={np.median(v):+7.3f} q90={np.percentile(v, 90):+7.3f} '
                  f'q95={np.percentile(v, 95):+7.3f} q99={np.percentile(v, 99):+7.3f} '
                  f'max={v.max():+7.3f} | 实际={actuals[stat]:+7.3f} 经验p={emp_p:.4f}')


if __name__ == '__main__':
    main()
