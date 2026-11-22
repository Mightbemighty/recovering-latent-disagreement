# -*- coding: utf-8 -*-
"""
测度优越性论证电池 T-A / T-A' / T-C / T-D（议程 2，2026-08-21；设计登记 = thoughts/18）
================================================================================
主线口径：9 面板幂集 CSR + OOS 2001-01（n_train=204）；T-A 附 21 面板功效补充与
1991 口径对照（T-A 内部双起点各写一文件）；T-A'/T-C/T-D 固定 2001 口径（设计登记 §5/§7）。

- T-A 剂量-反应（HK19 借鉴 §3）：dose_k = 全样本 IS 投影 R²（pair 口径主 /
  族口径敏感性）；gain_k 读 results/h1_pairwise_{start}{ptag}.csv（R2_d~ − R2_d，
  单测度一元 expanding OLS）；9 / 21 测度截面 Spearman + Pearson。
- T-A' 子集层面剂量-反应（thoughts/18 §7，T-A 阴性的裁决检验）：511 个非空子集
  逐子集 R²OOS（raw / d̃_pair 双臂），dose(S) = 成员 pair 投影 R² 均值；overall 与
  分 k 层 Spearman / Pearson。自检：k=1 层复现 h1_pairwise 锚点；层均值预测按
  C(9,k)/511 二项权重合成复现幂集 CSR 锚点 3.9561。限 --panel9。
- T-C 分域检验（DJM08 Eq.5 / Schwert 式）：域 = lv CSR 集成实时预测 ŷ_lv 的
  expanding 中位数二分（低域 = lv 模型看不到正收益信号的月份；ŷ_lv[t] 与其历史
  均为预测时点已知信息，对 y[t] 无前视）。域内 R²OOS（HA 基准不变）与 CW
  （NW lag=2，子序列非连续月为近似，注明）。前提检验 = 同域 lv_only 应无力。
  敏感性：lv_ens 水平值 expanding 三分位（中域 = lv 平静域）。限 --panel9。
- T-D placebo 面板（HK19 借鉴 §4）：B=200 个 placebo 面板（9 列独立 AR(1)，逐列
  全样本校准真实 d 列 AR(1) 系数与残差 std——placebo 生成非实时对象，全样本校准
  可接受并声明），L 保留真实面板，跑完整管线（pair 投影 → 幂集 CSR → R²OOS）；
  臂 = {raw, d̃_pair}；敏感性 (b) = L 同换噪声（B=100，仅 d̃_pair 臂）。
  经验 p = (1 + #{placebo ≥ 真实值}) / (B+1)。限 --panel9。

输出（不覆写既有锚点；{ptag} = '_p9' / ''）：
  results/measure_superiority_dose_{2001,1991}{ptag}.csv   （T-A）
  results/measure_superiority_dose_subset_2001{ptag}.csv   （T-A'）
  results/measure_superiority_domain_2001{ptag}.csv        （T-C）
  results/measure_superiority_placebo_2001{ptag}.csv       （T-D）
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_evidence as ev  # noqa: E402  数据加载 / decontaminate / powerset_csr_preds / R2OOS / clark_west_test

_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--test', type=str, default='all', choices=['A', 'Ap', 'C', 'D', 'all'])
_args0, _ = _ap0.parse_known_args()
TEST = _args0.test

RES = HERE / 'results'
PTAG = ev.PTAG            # '_p9'（--panel9）/ ''（默认 21 面板）
N_TRAIN = ev.STARTS[2001]  # T-C/T-D 固定 2001 主线口径（204）
y, ha, D, L = ev.y, ev.ha, ev.D, ev.L
n_p = ev.n_p
pairs = ev.pairs


# ---------------- T-A 剂量-反应 ----------------
def test_a():
    # 剂量：全样本 IS 投影 R²（与起点无关，两个 --start 文件共用同一剂量列）
    dose_pair, dose_fam = [], []
    for k in range(n_p):
        dose_pair.append(np.corrcoef(D[:, k], L[:, k])[0, 1] ** 2)
        X = np.column_stack([np.ones(len(L)), L])
        b, *_ = np.linalg.lstsq(X, D[:, k], rcond=None)
        r = D[:, k] - X @ b
        dose_fam.append(1 - r.var() / D[:, k].var())
    for start in [2001, 1991]:
        pw = pd.read_csv(RES / f'h1_pairwise_{start}{PTAG}.csv')
        assert list(pw['measure']) == [d[2:] for _, d in pairs], '测度顺序与面板不对齐'
        out = pw.copy()
        out['dose_pair'] = np.round(dose_pair, 4)
        out['dose_fam'] = np.round(dose_fam, 4)
        assert np.allclose(out['gain'], out['R2_d~'] - out['R2_d'], atol=1e-3)
        for dcol in ['dose_pair', 'dose_fam']:
            rho, rho_p = spearmanr(out[dcol], out['gain'])
            r, r_p = pearsonr(out[dcol], out['gain'])
            print(f'[T-A|{start}{PTAG}] {dcol}: Spearman rho={rho:+.3f} (p={rho_p:.3f}), '
                  f'Pearson r={r:+.3f} (p={r_p:.3f}), n={len(out)}')
        out.round(4).to_csv(RES / f'measure_superiority_dose_{start}{PTAG}.csv', index=False)
    print(f'[T-A] 落盘 results/measure_superiority_dose_{{2001,1991}}{PTAG}.csv', flush=True)


# ---------------- T-A' 子集层面剂量-反应 ----------------
def _subset_preds_matrix(M, y_in, n_train, k):
    """固定 k 的逐子集 expanding OLS 预测矩阵（n_oos × C(p,k)），语义与 ev.csr_enum_preds
    逐点一致（预测行 i 训练用行 s < i），仅不做跨子集平均；与 expanding_preds 同源。
    返回 (子集成员索引数组, 预测矩阵)。"""
    p = M.shape[1]
    subs = np.array(list(itertools.combinations(range(p), k)), dtype=np.int64)
    nS = len(subs)
    idx = np.concatenate([np.zeros((nS, 1), dtype=np.int64), subs + 1], axis=1)  # 0 = 截距
    Xc = np.column_stack([np.ones(len(y_in)), M])
    out = np.empty((len(y_in) - n_train, nS))
    for j, i in enumerate(range(n_train, len(y_in))):
        A, yt = Xc[:i], y_in[:i]
        G = A.T @ A
        c = A.T @ yt
        xn = Xc[i]
        Gs = G[idx[:, :, None], idx[:, None, :]]
        cs = c[idx]
        try:
            beta = np.linalg.solve(Gs, cs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = (np.linalg.pinv(Gs) @ cs[..., None])[..., 0]
        out[j] = (xn[idx] * beta).sum(axis=1)
    return subs, out


def test_a_prime():
    """T-A'（thoughts/18 §7）：剂量-反应在 CSR 实际使用层（子集）是否成立——裁决 T-A 阴性的
    竞争解释（单测度基预测器太弱淹没信号 vs 增益弥散）。限 --panel9、2001 主线口径。
    511 子集 × {raw, d̃_pair} 逐子集 R²OOS；dose(S) = 成员测度 pair 投影 R² 均值（与 T-A 同源）。
    自检：(i) k=1 层 raw R² 逐点复现 h1_pairwise 锚点（atol 2e-3）；
         (ii) 各 k 层 d̃_pair 均值预测按 C(9,k)/511 二项权重合成 = 幂集 CSR 预测 → R²OOS≈3.9561。"""
    assert ev.PANEL9, "T-A' 限 --panel9（thoughts/18 §7）"
    D_pair = ev.D_pair
    dose = np.array([np.corrcoef(D[:, k], L[:, k])[0, 1] ** 2 for k in range(n_p)])
    mnames = [d[2:] for _, d in pairs]
    y_oos, ha_oos = y[N_TRAIN:], ha[N_TRAIN:]
    r2 = lambda pr: ev.R2OOS(y_oos, pr, ha_oos) * 100
    rows, csr_pred = [], np.zeros(len(y_oos))
    n_sub_total = 2 ** n_p - 1
    for k in range(1, n_p + 1):
        subs_r, Pr = _subset_preds_matrix(D, y, N_TRAIN, k)
        subs_d, Pd = _subset_preds_matrix(D_pair, y, N_TRAIN, k)
        assert (subs_r == subs_d).all()
        nS = len(subs_r)
        csr_pred += Pd.mean(axis=1) * (nS / n_sub_total)
        Rr = np.array([r2(Pr[:, s]) for s in range(nS)])
        Rd = np.array([r2(Pd[:, s]) for s in range(nS)])
        if k == 1:  # 自检 (i)
            h1 = pd.read_csv(RES / f'h1_pairwise_2001{PTAG}.csv')
            for s in range(n_p):
                assert abs(Rr[s] - float(h1.loc[h1['measure'] == mnames[s], 'R2_d'].iloc[0])) < 2e-3, \
                    f'k=1 raw 自检失败: {mnames[s]}'
        for s, mem in enumerate(subs_r):
            rows.append({'k': k, 'members': '+'.join(mnames[int(m)] for m in mem),
                         'dose': float(dose[mem].mean()), 'R2_raw': Rr[s], 'R2_dpair': Rd[s],
                         'gain': Rd[s] - Rr[s]})
    df = pd.DataFrame(rows)
    fp = RES / f'measure_superiority_dose_subset_2001{PTAG}.csv'
    df.round(4).to_csv(fp, index=False)
    csr_syn = r2(csr_pred)  # 自检 (ii)：二项加权合成 = 幂集 CSR
    assert abs(csr_syn - 3.9561) < 0.01, f'CSR 合成自检失败: {csr_syn:.4f}'
    rho, rho_p = spearmanr(df['dose'], df['gain'])
    r, r_p = pearsonr(df['dose'], df['gain'])
    print(f"[T-A'|2001{PTAG}] overall: Spearman rho={rho:+.3f} (p={rho_p:.3f}), "
          f'Pearson r={r:+.3f} (p={r_p:.3f}), n={len(df)}')
    print(f"  自检通过：k=1 复现 h1_pairwise 锚点；CSR 合成 R²OOS={csr_syn:.4f}≈3.9561")
    for k in range(1, n_p + 1):
        sub = df[df['k'] == k]
        msg = f'  k={k}: n={len(sub)}, mean gain={sub["gain"].mean():+.2f}, gain>0 占比 {100 * (sub["gain"] > 0).mean():.0f}%'
        if len(sub) > 2:
            rs, ps_ = spearmanr(sub['dose'], sub['gain'])
            msg += f', rho={rs:+.3f} (p={ps_:.3f})'
        print(msg)
    print(f"[T-A'] 落盘 {fp.name}", flush=True)


# ---------------- T-C 分域检验 ----------------
def _domain_row(domain_def, dom_name, dom, series, pr, y_oos, ha_oos):
    act, pm, hm = y_oos[dom], pr[dom], ha_oos[dom]
    r2 = ev.R2OOS(act, pm, hm) * 100
    cw_s, cw_p, _ = ev.clark_west_test(act, pm, hm, lag=2)  # 子序列非连续月，NW 为近似
    return {'domain_def': domain_def, 'domain': dom_name, 'series': series,
            'R2OOS': r2, 'CW_stat': cw_s, 'CW_p': cw_p, 'n': int(dom.sum())}


def test_c():
    assert ev.PANEL9, 'T-C 限 --panel9（thoughts/18 §3）'
    preds = pd.read_csv(RES / f'preds_2001{PTAG}.csv', parse_dates=['date'])
    y_oos, ha_oos = y[N_TRAIN:], ha[N_TRAIN:]
    SERIES = {'d~_pair': preds['d~_pair'].values.astype(float),
              'raw': preds['raw'].values.astype(float),
              'lv_only': preds['lv_only'].values.astype(float)}
    pr_lv = SERIES['lv_only']
    rows = []
    # 主口径：ŷ_lv expanding 中位数二分（含 t 本身——预测时点已知信息）
    med = np.array([np.median(pr_lv[:j + 1]) for j in range(len(pr_lv))])
    domains = {'low_yhat_lv': pr_lv < med, 'high_yhat_lv': pr_lv >= med}
    # 敏感性：lv_ens 水平值 expanding 三分位
    lv_ens = L.mean(axis=1)[N_TRAIN:]
    q33 = np.array([np.quantile(lv_ens[:j + 1], 1 / 3) for j in range(len(lv_ens))])
    q67 = np.array([np.quantile(lv_ens[:j + 1], 2 / 3) for j in range(len(lv_ens))])
    domains_s = {'lv_low': lv_ens < q33, 'lv_mid': (lv_ens >= q33) & (lv_ens < q67),
                 'lv_high': lv_ens >= q67}
    for def_name, doms in [('yhat_median', domains), ('lv_tercile', domains_s)]:
        for dom_name, dom in doms.items():
            for sname, pr in SERIES.items():
                rows.append(_domain_row(def_name, dom_name, dom, sname, pr, y_oos, ha_oos))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(RES / f'measure_superiority_domain_2001{PTAG}.csv', index=False)
    print(out.to_string(index=False))
    print(f'[T-C] 落盘 results/measure_superiority_domain_2001{PTAG}.csv', flush=True)


# ---------------- T-D placebo 面板 ----------------
def _ar1_fit(M):
    """逐列 AR(1)：x_t = c + rho x_{t-1} + e；返回 (c, rho, sigma_e) 列表。"""
    out = []
    for k in range(M.shape[1]):
        x0, x1 = M[:-1, k], M[1:, k]
        A = np.column_stack([np.ones(len(x0)), x0])
        b, *_ = np.linalg.lstsq(A, x1, rcond=None)
        e = x1 - A @ b
        out.append((b[0], b[1], e.std(ddof=2)))
    return out


def _ar1_sim(params, n, rng, burn_in=100):
    """逐列模拟 AR(1)（lfilter：x_t = rho x_{t-1} + u_t，u_t = c + sigma z）。"""
    M = np.empty((n, len(params)))
    for k, (c, rho, sig) in enumerate(params):
        u = c + sig * rng.standard_normal(n + burn_in)
        x = lfilter([1.0], [1.0, -rho], u)
        M[:, k] = x[burn_in:]
    return M


def _pipeline_r2(Dm, Lm, y_oos, ha_oos):
    """完整管线：pair 投影 → 幂集 CSR → R²OOS（pt）。"""
    Dp = np.column_stack([ev.decontaminate(Dm[:, [k]], Lm[:, [k]])[:, 0]
                          for k in range(Dm.shape[1])])
    pr = ev.powerset_csr_preds(Dp, y, N_TRAIN)
    return ev.R2OOS(y_oos, pr, ha_oos) * 100


def test_d(B=200, B_b=100, seed=20260821):
    assert ev.PANEL9, 'T-D 限 --panel9（thoughts/18 §4）'
    y_oos, ha_oos = y[N_TRAIN:], ha[N_TRAIN:]
    preds = pd.read_csv(RES / f'preds_2001{PTAG}.csv')
    real = {'raw': ev.R2OOS(y_oos, preds['raw'].values.astype(float), ha_oos) * 100,
            'd~_pair': ev.R2OOS(y_oos, preds['d~_pair'].values.astype(float), ha_oos) * 100}
    print(f"[T-D] 真实值: raw={real['raw']:.4f}, d~_pair={real['d~_pair']:.4f}", flush=True)
    rng = np.random.default_rng(seed)
    par_d, par_l = _ar1_fit(D), _ar1_fit(L)
    n = len(y)
    rows = [{'id': 'REAL', 'arm': a, 'L_noise': False, 'R2OOS': v} for a, v in real.items()]
    for b in range(B):  # 主口径 (a)：D 噪声 + L 真实
        Dpl = _ar1_sim(par_d, n, rng)
        rows.append({'id': f'pl{b:03d}', 'arm': 'raw', 'L_noise': False,
                     'R2OOS': ev.R2OOS(y_oos, ev.powerset_csr_preds(Dpl, y, N_TRAIN),
                                       ha_oos) * 100})
        rows.append({'id': f'pl{b:03d}', 'arm': 'd~_pair', 'L_noise': False,
                     'R2OOS': _pipeline_r2(Dpl, L, y_oos, ha_oos)})
        if (b + 1) % 20 == 0:
            print(f'[T-D] placebo (a) {b + 1}/{B}', flush=True)
    for b in range(B_b):  # 敏感性 (b)：D/L 同换噪声，仅 d~_pair 臂
        Dpl = _ar1_sim(par_d, n, rng)
        Lpl = _ar1_sim(par_l, n, rng)
        rows.append({'id': f'plb{b:03d}', 'arm': 'd~_pair', 'L_noise': True,
                     'R2OOS': _pipeline_r2(Dpl, Lpl, y_oos, ha_oos)})
        if (b + 1) % 20 == 0:
            print(f'[T-D] placebo (b) {b + 1}/{B_b}', flush=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(RES / f'measure_superiority_placebo_2001{PTAG}.csv', index=False)
    for lnoise, tag in [(False, '(a) L真实'), (True, '(b) L噪声')]:
        for arm in ['raw', 'd~_pair']:
            pl = out[(out['L_noise'] == lnoise) & (out['arm'] == arm)
                     & (out['id'] != 'REAL')]['R2OOS'].values
            if len(pl) == 0:
                continue
            rv = real[arm]
            emp_p = (1 + (pl >= rv).sum()) / (len(pl) + 1)
            print(f'[T-D|{tag}|{arm}] placebo n={len(pl)} 中位={np.median(pl):+.3f} '
                  f'均值={pl.mean():+.3f} 真实={rv:+.3f} 经验p={emp_p:.4f}', flush=True)
    print(f'[T-D] 落盘 results/measure_superiority_placebo_2001{PTAG}.csv', flush=True)


if __name__ == '__main__':
    if TEST in ['A', 'all']:
        test_a()
    if not ev.PANEL9 and TEST in ['Ap', 'C', 'D', 'all']:
        print("[跳过] T-A'/T-C/T-D 限 --panel9；默认（21 面板）模式仅跑 T-A", flush=True)
    else:
        if TEST in ['Ap', 'all']:
            test_a_prime()
        if TEST in ['C', 'all']:
            test_c()
        if TEST in ['D', 'all']:
            test_d()
