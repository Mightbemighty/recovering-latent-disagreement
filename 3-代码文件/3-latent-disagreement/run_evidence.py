# -*- coding: utf-8 -*-
"""
Paper 2 证据管线（3-latent-disagreement）
==========================================
纪律（README 第 1 条）：IS 全样本报告；OOS 估计严格递归——去污染 γ 与 OLS 系数只用 ≤t 数据。
当前 JOF 主线为 --starts 2001（n_train=204）；--starts 1991 为附录扩展口径。运行时必须显式传参。
HA 基准：本管线从锁定面板重算，ha[t] = mean(y[:t])（与模型同信息集，数据截至 t-1）；
         存档 baseline_prediction.csv 仅用于锚点对照（约定 = mean(y[:t-1]) + 早期 vintage 噪声）。
锚点（2001-start，2026-08-17 全枚举主线）：d̃_lv ≈ 4.00 / CER3_tc0 ≈ 7.99 / EXP ≈ +1.18；
         raw ≈ 2.75（K=11 全枚举 CSR，C(21,11)=352,716 子集等权平均，确定性无种子）。
         历史：mean3 抽样口径（S=50×3 种子，2026-08-17 退役为稳健性，备份
         results/backup_mean3_2026-08-17/）d̃_lv 3.80 / CER3 8.11；d̃_u14/d̃_lv+u14 于
         2026-08-15 切换递归面板 un_data_recursive.csv（旧 un_data.csv 前视废弃，NOTES §28/§29）。
         全枚举理论依据：与 mean3 为同一估计量（S→∞ 极限），差异仅 MC 误差（§6.16 同价性复核）。
面板开关（2026-08-19，设计文档 v4.1 §3.4）：--panel9 切换 9 测度 SPF-only 面板
         （panel9_qe.csv），集成规则改幂集 CSR（2^p−1 非空子集等权）；输出带 _p9 后缀。
输出：results/*.csv + figures/*.png
"""
import argparse
import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SCNN = HERE.parent / '1-sCNNLSTM-and-meta-RSR-notebooks'
sys.path.insert(0, str(SCNN))
from MetricsDef import R2OOS, R2OOS_by_USREC, cer, sharpe_ratio, clark_west_test  # noqa: E402

B, K = 50, 11
SEEDS = [20260730, 20260731, 1]
BURN = 36
SPF = ['GDP', 'GDPg', 'IPg', 'IP', 'CONg', 'CON', 'INVg', 'INV', 'HSG', 'HSGg', 'UEP', 'CPI', 'TBL']
SCA = ['RPF', 'EPF', 'BC', 'UC', 'IRC', 'HOM']
AF = ['Yu', 'HS']
STARTS = {2001: 204, 1991: 84}

# ---------------- 面板开关（2026-08-19，设计文档 v4.1 §3.4 面板选址） ----------------
# --panel9：切换 9 测度 SPF-only 面板（panel9_qe.csv，make_panel9.py 构建，26 列
# 对 all_data_qe 锚点验证通过）；集成规则改**幂集 CSR**（全部 2^p−1 非空子集等权，
# 二项层权使质量集中 K/2，p=9 时 k=1 与 k=9 两层合计权重仅 2.0%）。默认 21 面板
# + 固定 K=11 全枚举口径不变。9 面板输出文件一律带 _p9 后缀，不覆写既有锚点文件。
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument('--panel9', action='store_true')
_args0, _ = _ap0.parse_known_args()
PANEL9 = _args0.panel9
PTAG = '_p9' if PANEL9 else ''

# ---------------- 数据 ----------------
if PANEL9:
    ad = pd.read_csv(HERE / 'panel9_qe.csv')
else:
    ad = pd.read_csv(SCNN / 'all_data_qe.csv')
un = pd.read_csv(HERE / 'un_data_recursive.csv', parse_dates=['date'])  # 2026-08-15 递归版（NOTES §28）
assert len(un) == len(ad) == 492 and (un['date'].values == pd.to_datetime(ad['date']).values).all()
sp = pd.read_csv(SCNN / 'SP500_h1.csv', parse_dates=['date'])
mkt = pd.read_csv(SCNN / 'mkt_RF_truncated.csv')
usrec_full = pd.read_csv(SCNN / 'USREC.csv')
base_stored = pd.read_csv(SCNN / 'baseline_prediction.csv')['h1'].values.astype(float)

y = ad['excess_ret_h1'].values.astype(float)
assert np.allclose(y, sp['raw'].values.astype(float), atol=1e-8)
dates = pd.to_datetime(ad['date'])

l_cols = sorted([c for c in ad.columns if c.startswith('l_')])
pairs = [(l, 'd_' + l[2:]) for l in l_cols]
n_p = len(pairs)  # 21（默认面板）或 9（--panel9）
u_cols = [c for c in un.columns if c.startswith('u_')]
D = np.column_stack([ad[d].values.astype(float) for _, d in pairs])
L = np.column_stack([ad[l].values.astype(float) for l, _ in pairs])
U = un[u_cols].values.astype(float)
assert not np.isnan(U).any() and not np.isnan(D).any() and not np.isnan(L).any()

# HA 基准（本管线约定：ha[t] = mean(y[:t])，信息截至 t-1；t=0 无定义）
cs = np.concatenate([[0.0], np.cumsum(y)])
ha = np.full(len(y), np.nan)
ha[1:] = cs[1:-1] / np.arange(1, len(y))


# ---------------- 核心函数（与 E2 语义一致） ----------------
def expanding_preds(X_in, y_in, n_train):
    X_in = np.atleast_2d(X_in).T if X_in.ndim == 1 else X_in
    Xc = np.column_stack([np.ones(len(y_in)), X_in])
    preds = np.empty(len(y_in) - n_train)
    for j, i in enumerate(range(n_train - 1, len(y_in) - 1)):
        beta, *_ = np.linalg.lstsq(Xc[:i + 1], y_in[:i + 1], rcond=None)
        preds[j] = Xc[i + 1] @ beta
    return preds


def decontaminate(Dmat, Zmat, burn=BURN):
    """逐列实时 expanding 去污染：t 行残差用 <t 行估计的投影系数；burn 前仅去均值。"""
    n = len(Dmat)
    out = np.empty_like(Dmat)
    for t in range(n):
        if t < burn:
            out[t] = Dmat[t] - Dmat[:max(t, 1)].mean(axis=0)
        else:
            Zc = np.column_stack([np.ones(t), Zmat[:t]])
            G, *_ = np.linalg.lstsq(Zc, Dmat[:t], rcond=None)
            out[t] = Dmat[t] - np.concatenate([[1.0], Zmat[t]]) @ G
    return out


def csr_enum_preds(M, y_in, n_train, k, h=1, i_max=None, nan_aware=False, block=20000):
    """固定 k 的全枚举 CSR 预测序列（等权平均全部 C(p,k) 个子集的 expanding OLS）。
    预测行 i ∈ [n_train, i_max]（默认 len(y_in)−h），训练只用行 s ≤ i−h（目标已完成）；
    nan_aware=True 时训练行额外剔除 y_in 为 NaN 的行（CF/DR 新闻序列用）；
    有效训练行 < k+2 的月预测为 NaN（退化防护，下游评估掩码剔除）。
    h=1 且无 NaN 时与主线 expanding_preds 逐点一致。
    实现：按月聚合充分统计量 G=X'X / c=X'y，子集正规方程分块批量求解（np.linalg.solve；
    奇异块回退 lstsq）。与 expanding_preds 的语义一致性由 run_kscan.py 的 K=1/K=21 校验保证。"""
    p = M.shape[1]
    subs = np.array(list(itertools.combinations(range(p), k)), dtype=np.int64)
    nS = len(subs)
    idx = np.concatenate([np.zeros((nS, 1), dtype=np.int64), subs + 1], axis=1)  # 0 = 截距
    Xc = np.column_stack([np.ones(len(y_in)), M])
    if i_max is None:
        i_max = len(y_in) - h
    out = np.empty(i_max - n_train + 1)
    for j, i in enumerate(range(n_train, i_max + 1)):
        end = i - h + 1  # 训练行 s ≤ i−h（s < end）
        if nan_aware:
            ok = ~np.isnan(y_in[:end])
            if ok.sum() < k + 2:
                # 有效训练行 ≤ 参数数（k+1）时子集回归退化：预测置 NaN 并由下游评估掩码剔除。
                # （1991 起点下 expanding-VAR 新闻目标 1994-01 才起评，早期月触发；
                # 2001 起点各链有效行 ≥84 从不触发，行为不变。）
                out[j] = np.nan
                continue
            A, yt = Xc[:end][ok], y_in[:end][ok]
        else:
            A, yt = Xc[:end], y_in[:end]
        G = A.T @ A
        c = A.T @ yt
        xn = Xc[i]
        acc = 0.0
        for b0 in range(0, nS, block):
            ii = idx[b0:b0 + block]
            Gs = G[ii[:, :, None], ii[:, None, :]]
            cs = c[ii]
            try:
                beta = np.linalg.solve(Gs, cs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                # 奇异 Gram（如 1991 起点下 expanding-VAR 目标早期月有效训练行为 0/极少）：
                # pinv 支持批处理并给出最小范数最小二乘解（原 lstsq 回退对 3 维 Gs 必报错，
                # 属从未触发过的潜在 bug；2001 口径各链从未进入此分支，修复不改变既有结果）。
                beta = (np.linalg.pinv(Gs) @ cs[..., None])[..., 0]
            acc += (xn[ii] * beta).sum(axis=1).sum()
        out[j] = acc / nS
    return out


ENUM_CAP = 1_000_000  # C(p,k) 可枚举上限：21 列池全部 k 可枚举（K=11 = 352,716）；42 列池保留抽样


def powerset_csr_preds(M, y_in, n_train, h=1, i_max=None, nan_aware=False):
    """幂集 CSR（--panel9 口径）：全部 2^p−1 个非空子集等权平均，
    = 各 k 层（固定 k 全枚举层均值）按 C(p,k) 二项系数加权，k=1..p 全枚举。
    层权集中 K/2 附近（p=9：k=4/5 两层合计 49.3%，k=1 与 k=9 合计仅 2.0%），
    即 EGT complete subset regression 的原始构造。
    h / i_max / nan_aware 与 csr_enum_preds 同语义（CF/DR 渠道检验用）。"""
    p = M.shape[1]
    tot = (1 << p) - 1  # 2^p − 1 = Σ_k C(p,k)
    acc = 0.0
    for k in range(1, p + 1):
        acc = acc + (comb(p, k) / tot) * csr_enum_preds(M, y_in, n_train, k, h=h,
                                                        i_max=i_max, nan_aware=nan_aware)
    return acc


def subset_ensemble(M, n_train, k=K, b=B, seeds=SEEDS):
    """子集集成预测。主线口径（2026-08-17 起）：C(p,k) ≤ ENUM_CAP 时走**全枚举 CSR**
    （等权平均全部子集，无种子、无 S、确定性）；超出上限的池（42 列 k=22，C≈5.4e11）
    保留 S=50×3 种子抽样。--panel9 口径（2026-08-19 起）：幂集 CSR（powerset_csr_preds）。
    返回 3 元素 list（全枚举/幂集路径下三元素为同一序列），
    下游 full_s0 / full_mean3 列语义不变。"""
    if PANEL9:
        return [powerset_csr_preds(M, y, n_train)] * 3
    kk = min(k, M.shape[1])
    if comb(M.shape[1], kk) <= ENUM_CAP:
        pr = csr_enum_preds(M, y, n_train, kk)
        return [pr, pr, pr]
    prs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ss = [sorted(rng.choice(M.shape[1], kk, replace=False).tolist()) for _ in range(b)]
        pr = np.column_stack([expanding_preds(M[:, s], y, n_train) for s in ss]).mean(axis=1)
        prs.append(pr)
    return prs


# ---------------- 变体构建（与 start 无关，全样本实时） ----------------
D_pair = np.column_stack([decontaminate(D[:, [k]], L[:, [k]])[:, 0] for k in range(n_p)])
variants = {
    'raw': D,
    'd~_pair': D_pair,
    'd~_lv': decontaminate(D, L),
    'd~_u14': decontaminate(D, U),
    'd~_lv+u14': decontaminate(D, np.column_stack([L, U])),
    'lv_only': L,
    f'd~_lv+l({2 * n_p})': np.column_stack([decontaminate(D, L), L]),
}
POOL_K = {'lv_only': 11, f'd~_lv+l({2 * n_p})': 22}
print('变体构建完成（expanding 实时去污染，burn=36）')


# ---------------- 单 start 运行 ----------------
def run_start(n_train, tag, outdir):
    n_oos = len(y) - n_train
    base = ha[n_train:]
    assert not np.isnan(base).any()
    y_true = y[n_train:]
    usrec = usrec_full.iloc[-n_oos:].reset_index(drop=True)
    rf = mkt['RF_l1'] / 100
    ha_full_s = pd.Series(ha)
    y_full_s = sp['raw']

    rows, preds_store = [], {}
    for name, M in variants.items():
        kk = POOL_K.get(name, K)
        prs = subset_ensemble(M, n_train, k=kk)
        pr0 = prs[0]
        r2s = [R2OOS(y_true, pr, base) * 100 for pr in prs]
        cw_s, cw_p, cw_sig = clark_west_test(y_true, pr0, base)
        r_rec, r_exp = R2OOS_by_USREC(y_true, pr0, base, usrec)
        fp = pd.Series(np.full(len(y), np.nan))
        fp.iloc[n_train:] = pr0
        row = {'variant': name, 'full_s0': r2s[0], 'full_mean3': np.mean(r2s), 'full_min3': np.min(r2s),
               'CW_stat': cw_s, 'CW_p': cw_p, 'CW_sig': cw_sig,
               'REC': r_rec * 100, 'EXP': r_exp * 100, 'gap': (r_rec - r_exp) * 100}
        for g in (3, 5):
            for tc, tc_tag in ((0.0, 'tc0'), (0.005, 'tc50')):
                row[f'CER{g}_{tc_tag}'] = cer(fp, y_full_s, ha_full_s, rf, g, transaction_cost=tc, oos_len=n_oos)
                row[f'dSR{g}_{tc_tag}'] = sharpe_ratio(fp, y_full_s, ha_full_s, rf, g, transaction_cost=tc, oos_len=n_oos)
        rows.append(row)
        preds_store[name] = pr0
        print(f'  [{tag}] {name:14s} R2OOS(mean3)={np.mean(r2s):6.3f}  CW p={cw_p:.3f}')
    res = pd.DataFrame(rows).set_index('variant')

    # H1：逐对单变量（raw d vs 逐对去污染 d~_pair）
    h1_rows = []
    for k, (lc, dc) in enumerate(pairs):
        pr_d = expanding_preds(D[:, k], y, n_train)
        pr_dt = expanding_preds(D_pair[:, k], y, n_train)
        r2_d = R2OOS(y_true, pr_d, base) * 100
        r2_dt = R2OOS(y_true, pr_dt, base) * 100
        _, p_d, _ = clark_west_test(y_true, pr_d, base)
        _, p_dt, _ = clark_west_test(y_true, pr_dt, base)
        h1_rows.append({'measure': dc[2:], 'R2_d': r2_d, 'R2_d~': r2_dt, 'gain': r2_dt - r2_d,
                        'CWp_d': p_d, 'CWp_d~': p_dt,
                        'abs_corr_ld': abs(np.corrcoef(L[:, k], D[:, k])[0, 1])})
    h1 = pd.DataFrame(h1_rows).set_index('measure')
    wins = (h1['gain'] > 0).sum()
    cg = np.corrcoef(h1['gain'], h1['abs_corr_ld'])[0, 1]
    print(f'  [{tag}] H1: d~ 胜 d = {wins}/{n_p}, corr(gain,|corr|)={cg:.3f}')

    if PANEL9 and n_train == STARTS[2001]:
        raw9 = pd.read_csv(HERE / 'panel9_qe_raw.csv')
        assert len(raw9) == len(ad) and (pd.to_datetime(raw9['date']).values == dates.values).all()
        raw_abs_corr = np.array([
            abs(np.corrcoef(raw9[l].values.astype(float), raw9[d].values.astype(float))[0, 1])
            for l, d in pairs
        ])
        rho, p_value = spearmanr(raw_abs_corr, h1['gain'].to_numpy())
        h1[['gain']].assign(abs_corr_ld_raw_units=raw_abs_corr).round(6).to_csv(
            outdir / 'h1_spearman_inputs_2001_p9.csv')
        pd.DataFrame([{'n_questions': n_p, 'spearman_rho': rho, 'spearman_p': p_value}]).to_csv(
            outdir / 'h1_spearman_2001_p9.csv', index=False)
        print(f'  [{tag}] H1 raw-unit Spearman: rho={rho:.6f}, p={p_value:.6f}')

    # 落盘（--panel9 一律带 _p9 后缀，不覆写 21 面板锚点文件）
    outdir.mkdir(exist_ok=True)
    res.round(4).to_csv(outdir / f'variants_{tag}{PTAG}.csv')
    h1.round(4).to_csv(outdir / f'h1_pairwise_{tag}{PTAG}.csv')
    pd.DataFrame(preds_store, index=dates[n_train:]).round(6).to_csv(outdir / f'preds_{tag}{PTAG}.csv')
    pd.Series(ha, index=dates).to_csv(outdir / 'ha_baseline_recomputed.csv')
    return res, h1


# ---------------- stylized facts（全样本描述统计，与 start 无关） ----------------
def stylized_facts(outdir):
    names = [d[2:] for _, d in pairs]
    # 2026-08-19 修复分组错位：pairs 为字母序（BC,CON,...,Yu），旧版位置假设 ['SPF']*13+...
    # 与实际列序不符，组内/组间均值全错（如"SPF 0.231"实为 BC..INVg 混杂集；正确按名分组
    # SPF mean 0.422 / SCA 0.063 / AF 0.761，与 thoughts/11 §1 一致）。按名映射分组：
    if PANEL9:
        grp = ['SPF'] * n_p  # 9 面板全部测度均属 SPF 族，无跨源分组
    else:
        _g = {m: g for g, ms in (('SPF', SPF), ('SCA', SCA), ('AF', AF)) for m in ms}
        grp = [_g[nm] for nm in names]
    C = np.corrcoef(D.T)
    iu = np.triu_indices(n_p, 1)
    pw = C[iu]
    ptag = '，9 面板（panel9_qe）' if PANEL9 else ''
    lines = [f'=== stylized facts（全样本 492 月，1984-01~2024-12，IS 描述性质{ptag}）===',
             f'dis 两两相关: median={np.median(pw):.3f} p10={np.percentile(pw, 10):.3f}']
    present = [g for g in ('SPF', 'SCA', 'AF') if g in grp]
    for g in present:
        idx = [i for i, gg in enumerate(grp) if gg == g]
        sub = C[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        lines.append(f'  组内 {g}: mean={sub.mean():.3f}')
    for a, b in (('SPF', 'SCA'), ('SPF', 'AF'), ('SCA', 'AF')):
        if a in present and b in present:
            ia = [i for i, gg in enumerate(grp) if gg == a]
            ib = [i for i, gg in enumerate(grp) if gg == b]
            lines.append(f'  组间 {a}-{b}: mean={C[np.ix_(ia, ib)].mean():.3f}')
    half = len(y) // 2
    for lo, hi, tg in ((0, half, '前半'), (half, len(y), '后半')):
        Ch = np.corrcoef(D[lo:hi].T)[iu]
        lines.append(f'  split-half {tg}: median={np.median(Ch):.3f}')
    # PCA 谱（dis / lv / all）
    pca_rows = []
    for tagX, X in ((f'dis{n_p}', D), (f'lv{n_p}', L), (f'all{2 * n_p}', np.column_stack([D, L]))):
        Cx = np.corrcoef(X.T)
        ev = np.linalg.eigvalsh(Cx)[::-1]
        share = ev / ev.sum()
        n80 = int(np.searchsorted(np.cumsum(share), 0.80) + 1)
        pca_rows.append({'set': tagX, 'PC1_share': share[0], 'n_to_80pct': n80})
        lines.append(f'  PCA {tagX}: PC1={share[0] * 100:.1f}%, 80% 需 {n80} 成分')
    pd.DataFrame(pca_rows).round(4).to_csv(outdir / f'stylized_pca{PTAG}.csv', index=False)
    pd.DataFrame(C, index=names, columns=names).round(4).to_csv(outdir / f'stylized_corr_matrix{PTAG}.csv')
    txt = '\n'.join(lines)
    (outdir / f'stylized_facts{PTAG}.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--starts', nargs='+', type=int, default=[1991], choices=list(STARTS))  # 2026-08-20：主线默认仅 1991
    ap.add_argument('--skip-facts', action='store_true')
    ap.add_argument('--panel9', action='store_true')  # 实际切换在模块级预解析；此处仅为通过参数校验
    args = ap.parse_args()
    outdir = HERE / 'results'
    for s in args.starts:
        print(f'\n===== OOS start {s}（n_train={STARTS[s]}）=====')
        run_start(STARTS[s], str(s), outdir)
    if not args.skip_facts:
        stylized_facts(outdir)
    if PANEL9:
        print('\n完成（9 面板 + 幂集 CSR 口径，输出带 _p9 后缀）。')
    else:
        print('\n完成。对照锚点（2001, 旧基准 E2 记录）：d~_lv mean3≈3.83, CER3_tc0≈8.17, EXP≈+1.23')
