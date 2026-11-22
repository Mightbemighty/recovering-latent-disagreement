# -*- coding: utf-8 -*-
"""Finite utility repair from stored Paper 2 forecasts, without fitting models.

The date column is the forecast origin t; excess_ret_h1 is the realized
log excess return at t+1. Input predictions and the historical-average
benchmark remain in their original log-excess units. The utility experiment
uses the explicitly fixed plug-in conversion, not a Jensen correction.

Only this script's output directory is written. No project pipeline or
MetricsDef module is imported or executed. Run with the existing Anaconda
Python; on Windows its Library/bin may need to precede PATH for matplotlib.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
N_TRAIN, N_OOS, WINDOW = 204, 288, 120
GAMMAS = (3, 5)
TCS = ((0.0, "tc0"), (0.005, "tc50"))
STRATEGIES = ("raw", "d~_pair", "HA")
UTILITY_COLS = [f"{m}{g}_{tc}" for g in GAMMAS for _, tc in TCS
                for m in ("CER", "dSR")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def monthly_cer(returns, gamma):
    return np.mean(returns) - gamma * np.var(returns, ddof=1) / 2


def annual_sr(excess):
    return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(12)


def summary(returns, rf_target, gamma):
    return {"cer_ann_pp": float(1200 * monthly_cer(returns, gamma)),
            "sr_ann": float(annual_sr(returns - rf_target)),
            "mean_ann_pct": float(1200 * np.mean(returns)),
            "vol_ann_pct": float(100 * np.sqrt(12) * np.std(returns, ddof=1))}


def portfolio_path(prediction, rf_origin, rf_target, rm_target, variance, gamma, tc):
    simple_excess_prediction = (1 + rf_origin) * np.expm1(prediction)
    weight = np.clip(simple_excess_prediction / (gamma * variance), 0, 1.5)
    previous_weight = np.r_[0.0, weight[:-1]]
    turnover = np.abs(weight - previous_weight)
    cost = tc * turnover
    gross = rf_target + weight * (rm_target - rf_target)
    returns = gross - cost
    return {"pred_simple_excess_plugin": simple_excess_prediction,
            "weight": weight, "previous_weight": previous_weight,
            "turnover": turnover, "transaction_cost": cost,
            "gross_return": gross, "portfolio_return": returns,
            "portfolio_excess_return": returns - rf_target}


def scalar_path(prediction, origins, targets, market, rf, gamma, tc):
    """Independent scalar, calendar-selected path with Python sample moments."""
    out, weights, turns = [], [], []
    previous = 0.0
    for pred, origin, target in zip(prediction, origins, targets):
        first = origin - pd.offsets.MonthBegin(WINDOW - 1)
        history = [float(market.loc[d]) for d in pd.date_range(first, origin, freq="MS")]
        assert len(history) == WINDOW
        avg = math.fsum(history) / len(history)
        var = math.fsum((x - avg) ** 2 for x in history) / (len(history) - 1)
        pred_excess = (1 + float(rf.loc[origin])) * math.expm1(float(pred))
        weight = max(0.0, min(1.5, pred_excess / (gamma * var)))
        turn = abs(weight - previous)
        realized = float(rf.loc[target]) + weight * (float(market.loc[target])
                                                    - float(rf.loc[target])) - tc * turn
        out.append(realized)
        weights.append(weight)
        turns.append(turn)
        previous = weight
    return np.array(out), np.array(weights), np.array(turns)


def scalar_cer(returns, gamma):
    values = [float(x) for x in returns]
    avg = math.fsum(values) / len(values)
    variance = math.fsum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return avg - gamma * variance / 2


def reconstruct_legacy(pred, y, ha, legacy_rf):
    """Literal old arithmetic, using stored predictions rather than refitting."""
    pseudo_market = np.expm1(y)
    variance = np.array([np.var(pseudo_market[i-WINDOW:i], ddof=1)
                         for i in range(N_TRAIN, len(y))])
    realized, rf = pseudo_market[N_TRAIN:], legacy_rf[N_TRAIN:]
    paths, rows = {}, []
    for strategy, prediction in {**pred, "HA": ha[N_TRAIN:]}.items():
        for gamma in GAMMAS:
            weight = np.clip(np.expm1(prediction) / (gamma * variance), 0, 1.5)
            turn = np.abs(weight - np.r_[0.0, weight[:-1]])
            for tc, tag in TCS:
                paths[(strategy, gamma, tag)] = rf + weight * (realized - rf) - tc * turn
    for strategy in pred:
        row = {"portfolio": strategy + "+CSR"}
        for gamma in GAMMAS:
            for _, tag in TCS:
                r = paths[(strategy, gamma, tag)]
                b = paths[("HA", gamma, tag)]
                row[f"CER{gamma}_{tag}"] = 1200 * (monthly_cer(r, gamma) - monthly_cer(b, gamma))
                row[f"dSR{gamma}_{tag}"] = annual_sr(r-rf) - annual_sr(b-rf)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_pair(port, args):
    """Paired circular blocks; estimated weights and monthly costs stay fixed.

    The centered null draws are delta_star - delta_hat. The two-sided p-value
    uses their absolute magnitude and a finite-replication +1 correction.
    Primary intervals are the equal-tail basic intervals obtained by inverting
    these centered errors. Percentile intervals are supplementary.
    """
    rng = np.random.default_rng(args.seed)
    starts = rng.integers(0, N_OOS,
                          size=(args.bootstrap_reps, math.ceil(N_OOS/args.block_length)))
    idx = ((starts[:, :, None] + np.arange(args.block_length)) % N_OOS).reshape(
        args.bootstrap_reps, -1)[:, :N_OOS]
    rows, draws = [], []
    for gamma in GAMMAS:
        for tc, tag in TCS:
            raw = port[("raw", gamma, tag)]["portfolio_return"]
            pair = port[("d~_pair", gamma, tag)]["portfolio_return"]
            observed = 1200 * (monthly_cer(pair, gamma) - monthly_cer(raw, gamma))
            rb, pb = raw[idx], pair[idx]
            boot = 1200 * (pb.mean(axis=1) - rb.mean(axis=1)
                           - gamma/2 * (pb.var(axis=1, ddof=1) - rb.var(axis=1, ddof=1)))
            errors = boot - observed
            err_lo, err_hi = np.quantile(errors, [0.025, 0.975])
            pct_lo, pct_hi = np.quantile(boot, [0.025, 0.975])
            p = (1 + np.count_nonzero(np.abs(errors) >= abs(observed))) / (args.bootstrap_reps + 1)
            rows.append({"gamma": gamma, "tc": tc, "tc_tag": tag,
                         "difference_pair_minus_raw_ann_pp": float(observed),
                         "ci95_basic_lo_ann_pp": float(observed-err_hi),
                         "ci95_basic_hi_ann_pp": float(observed-err_lo),
                         "p_two_sided_centered": float(p),
                         "ci95_percentile_lo_ann_pp": float(pct_lo),
                         "ci95_percentile_hi_ann_pp": float(pct_hi),
                         "bootstrap_mean_ann_pp": float(boot.mean()),
                         "bootstrap_sd_ann_pp": float(boot.std(ddof=1)),
                         "seed": args.seed, "block_length": args.block_length,
                         "replications": args.bootstrap_reps,
                         "conditional_on_estimated_weights": True})
            draws.append(pd.DataFrame({"gamma": gamma, "tc": tc, "replication":
                                       np.arange(1, args.bootstrap_reps+1),
                                       "difference_star_ann_pp": boot,
                                       "centered_null_draw_ann_pp": errors}))
    return pd.DataFrame(rows), pd.concat(draws, ignore_index=True), idx


def create_figures(recursive, rolling, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.dpi": 320, "legend.frameon": False})
    labels = {"raw": "Return forecasts using raw disagreement",
              "d~_pair": "Return forecasts using decontaminated disagreement"}
    colors = {"raw": "#777777", "d~_pair": "#155A7A"}
    definitions = []
    specs = [
        ("cer_path", "Recursive certainty-equivalent gains over historical-average forecasts",
         "Annualized certainty-equivalent gain\n(percentage points)", "dcer_vs_ha_ann_pp"),
        ("sr_path", "Recursive Sharpe-ratio gains over historical-average forecasts",
         "Annualized Sharpe-ratio gain", "dsr_vs_ha_ann"),
        ("rolling60_cer_difference", "Rolling 60-month certainty-equivalent difference",
         "Decontaminated minus raw portfolio\ncertainty-equivalent return\n(percentage points)", None)]
    for name, title, ylabel, value in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.28, top=0.79)
        fig.suptitle(title, fontsize=10.2, y=0.97)
        if value:
            for strategy in ("raw", "d~_pair"):
                data = recursive[(recursive.gamma == 3) & (recursive.tc == 0)
                                 & (recursive.strategy == strategy)]
                ax.plot(pd.to_datetime(data.forecast_origin).to_numpy(), data[value].to_numpy(),
                        color=colors[strategy], ls="--" if strategy == "raw" else "-",
                        linewidth=1.3, label=labels[strategy])
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
                       bbox_to_anchor=(0.55, 0.91), fontsize=8.3, ncol=1)
        else:
            data = rolling[(rolling.gamma == 3) & (rolling.tc == 0)]
            ax.plot(pd.to_datetime(data.forecast_origin).to_numpy(), data.dcer_pair_minus_raw_ann_pp.to_numpy(),
                    color=colors["d~_pair"], linewidth=1.3)
        ax.axhline(0, color="0.55", lw=0.65)
        ax.axvline(pd.Timestamp("2020-01-01"), color="0.55", lw=0.7, ls=":")
        ax.set_xlabel("Forecast origin month")
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=8)
        ax.margins(x=0.015)
        notes = ("Notes: Monthly forecast origins: January 2001 to December 2024; realized returns occur one month later.\n"
                 "Forecasts average all 511 nonempty subsets of nine raw or own-question decontaminated measures.\n"
                 "Portfolio returns combine the market price return and the risk-free return, with equity weights in [0, 1.5].\n"
                 + ("Risk aversion = 3; no transaction costs.\n" if value else
                    "Risk aversion = 3; no transaction costs. Certainty-equivalent differences are annualized.\n")
                 + ("Recursive estimates use the first 13 months onward. " if value else
                    "Each estimate uses 60 realized monthly returns. ")
                 + "The vertical line marks January 2020.")
        fig.text(0.06, 0.04, notes, ha="left", va="bottom", fontsize=7.0, linespacing=1.6)
        fig.canvas.draw()
        visible = [{"role": "title", "text": title},
                   {"role": "axis_x", "text": ax.get_xlabel()},
                   {"role": "axis_y", "text": ax.get_ylabel()},
                   {"role": "note", "text": notes}]
        visible.extend({"role": "legend", "text": label} for label in
                       (list(labels.values()) if value else []))
        visible.extend({"role": "tick", "text": tick.get_text()} for tick in
                       list(ax.get_xticklabels()) + list(ax.get_yticklabels()) if tick.get_text())
        for extension in ("pdf", "png"):
            fig.savefig(outdir / f"{name}.{extension}", facecolor="white")
        definitions.append({"artifact_id": name,
                            "artifact_state": "candidate", "manuscript_state": "not-integrated",
                            "png": str(outdir / f"{name}.png"),
                            "png_sha256": sha256(outdir / f"{name}.png"),
                            "pdf": str(outdir / f"{name}.pdf"),
                            "pdf_sha256": sha256(outdir / f"{name}.pdf"),
                            "numeric_payload": str(outdir / ("recursive_paths.csv" if value else "rolling60_differences.csv")),
                            "gamma": 3, "tc": 0,
                            "visible_labels": visible,
                            "csr_definition": "Equal-weighted forecasts from all 511 nonempty subsets of nine disagreement measures.",
                            "decontaminated_definition": "Each disagreement measure is recursively projected on the consensus forecast of the same survey question.",
                            "cer_definition": "1200 times the difference in monthly mean-variance certainty equivalents; sample variance uses ddof=1.",
                            "sharpe_definition": "Mean simple portfolio excess return divided by its sample standard deviation, multiplied by sqrt(12)."})
        plt.close(fig)
    return definitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path,
                        default=HERE / "results" / "utility_repair_20261030")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--block-length", type=int, default=12)
    args = parser.parse_args()
    allowed = (HERE / "results" / "utility_repair_20261030").resolve()
    args.outdir = args.outdir.resolve()
    assert args.outdir == allowed or allowed in args.outdir.parents, "Output must remain in the authorized result directory."
    assert args.bootstrap_reps > 0 and 0 < args.block_length <= N_OOS
    args.outdir.mkdir(parents=True, exist_ok=True)

    sources = {"panel": HERE / "panel9_qe.csv",
               "predictions": HERE / "results" / "preds_2001_p9.csv",
               "market": ROOT / "1-数据文件" / "SP500_monthly_1970m2_2025m12.csv",
               "risk_free": ROOT / "1-数据文件" / "F-F_Research_Data_Factors_26.csv",
               "legacy_rf": HERE.parent / "1-sCNNLSTM-and-meta-RSR-notebooks" / "mkt_RF_truncated.csv",
               "legacy_y": HERE.parent / "1-sCNNLSTM-and-meta-RSR-notebooks" / "SP500_h1.csv",
               "legacy_grid": HERE / "results" / "utility_grid_2001_p9.csv",
               "legacy_variants": HERE / "results" / "variants_2001_p9.csv",
               "legacy_metrics_source": HERE.parent / "1-sCNNLSTM-and-meta-RSR-notebooks" / "MetricsDef.py"}
    before = {key: sha256(path) for key, path in sources.items()}
    panel = pd.read_csv(sources["panel"], usecols=["date", "excess_ret_h1"], parse_dates=["date"])
    disk = pd.read_csv(sources["predictions"], usecols=["date", "raw", "d~_pair"], parse_dates=["date"])
    y = panel.excess_ret_h1.to_numpy()
    origins = pd.DatetimeIndex(panel.date)
    targets = origins + pd.offsets.MonthBegin(1)
    assert len(panel) == 492 and len(disk) == N_OOS
    assert origins.equals(pd.date_range("1984-01-01", "2024-12-01", freq="MS"))
    assert pd.DatetimeIndex(disk.date).equals(origins[N_TRAIN:])
    assert origins[N_TRAIN] == pd.Timestamp("2001-01-01")
    assert targets[N_TRAIN] == pd.Timestamp("2001-02-01")
    assert targets[-1] == pd.Timestamp("2025-01-01")
    assert ((targets.to_period("M").asi8 - origins.to_period("M").asi8) == 1).all()
    ha = np.r_[np.nan, np.cumsum(y)[:-1]/np.arange(1, len(y))]
    assert np.max(np.abs(ha[N_TRAIN:] - np.array([np.mean(y[:i]) for i in range(N_TRAIN, len(y))]))) < 1e-15
    pred = {s: disk[s].to_numpy() for s in ("raw", "d~_pair")}
    predictions = {**pred, "HA": ha[N_TRAIN:]}
    assert all(np.isfinite(p).all() for p in predictions.values())

    ff = pd.read_csv(sources["risk_free"], skiprows=4, usecols=["date", "RF"], dtype={"date": str})
    assert ff.date.str.fullmatch(r"\d{6}").all()
    ff["date"] = pd.to_datetime(ff.date, format="%Y%m")
    rf = ff.set_index("date")["RF"].sort_index()/100
    raw_market = pd.read_csv(sources["market"], usecols=["Date", "Rate"], parse_dates=["Date"])
    market = raw_market.set_index("Date")["Rate"].sort_index()/100
    assert rf.index.is_unique and market.index.is_unique
    rf_origin_full = rf.reindex(origins).to_numpy()
    rf_target_full = rf.reindex(targets).to_numpy()
    rm_target_full = market.reindex(targets).to_numpy()
    assert np.isfinite(np.r_[rf_origin_full, rf_target_full, rm_target_full]).all()
    reconstructed = (1 + rf_target_full)*np.exp(y)-1
    identity_error = float(np.max(np.abs(reconstructed-rm_target_full)))
    assert identity_error < 1e-9
    log_identity_error = float(np.max(np.abs(y-(np.log1p(rm_target_full)-np.log1p(rf_target_full)))))
    assert log_identity_error < 1e-9
    # The information end date for every variance estimate is exactly origin t.
    # The corresponding y window contains completed target returns through t.
    variance, history_start, history_end = [], [], []
    for i in range(N_TRAIN, len(panel)):
        dates = pd.date_range(origins[i]-pd.offsets.MonthBegin(WINDOW-1), origins[i], freq="MS")
        history = market.reindex(dates)
        assert len(history) == WINDOW and history.notna().all()
        assert dates[-1] == origins[i] < targets[i]
        assert dates[0] == targets[i-WINDOW]
        variance.append(history.var(ddof=1))
        history_start.append(dates[0])
        history_end.append(dates[-1])
    variance = np.array(variance)
    assert (variance > 0).all()
    oos_origins, oos_targets = origins[N_TRAIN:], targets[N_TRAIN:]
    rf_origin, rf_target, rm_target = rf_origin_full[N_TRAIN:], rf_target_full[N_TRAIN:], rm_target_full[N_TRAIN:]
    calendar = pd.DataFrame({"forecast_origin": oos_origins, "realized_target": oos_targets,
                             "y_log_excess": y[N_TRAIN:], "ha_log_excess": ha[N_TRAIN:],
                             "raw_log_excess_forecast": pred["raw"],
                             "pair_log_excess_forecast": pred["d~_pair"],
                             "rf_origin": rf_origin, "rf_target": rf_target,
                             "rm_target": rm_target, "rm_target_reconstructed": reconstructed[N_TRAIN:],
                             "variance_120m": variance, "variance_history_start": history_start,
                             "variance_history_end": history_end, "variance_history_count": WINDOW,
                             "ha_last_realized_target": oos_origins})
    calendar.to_csv(args.outdir / "calendar_inputs.csv", index=False, float_format="%.15g")

    legacy_rf = pd.read_csv(sources["legacy_rf"], usecols=["RF_l1"])["RF_l1"].to_numpy()/100
    legacy_y = pd.read_csv(sources["legacy_y"], usecols=["raw"])["raw"].to_numpy()
    assert len(legacy_rf) == len(panel) and np.max(np.abs(y-legacy_y)) < 1e-9
    legacy = reconstruct_legacy(pred, y, ha, legacy_rf)
    old = pd.read_csv(sources["legacy_grid"]).set_index("portfolio")
    old_v = pd.read_csv(sources["legacy_variants"]).set_index("variant")
    legacy_checks = []
    for _, row in legacy.iterrows():
        for column in UTILITY_COLS:
            got, reference = float(row[column]), float(old.loc[row.portfolio, column])
            reference_v = float(old_v.loc[row.portfolio.removesuffix("+CSR"), column])
            assert abs(got-reference) < 0.002 and abs(got-reference_v) < 0.002, (row.portfolio, column, got, reference)
            legacy_checks.append({"portfolio": row.portfolio, "metric": column,
                                  "reconstructed_legacy": got, "stored_legacy": reference,
                                  "stored_variant": reference_v, "absolute_error": abs(got-reference)})
    legacy.to_csv(args.outdir / "legacy_reconstructed_grid.csv", index=False, float_format="%.12g")
    pd.DataFrame(legacy_checks).to_csv(args.outdir / "legacy_cell_checks.csv", index=False, float_format="%.12g")

    port, monthly, stats_rows = {}, [], []
    scalar_errors = []
    for strategy, prediction in predictions.items():
        for gamma in GAMMAS:
            for tc, tag in TCS:
                path = portfolio_path(prediction, rf_origin, rf_target, rm_target, variance, gamma, tc)
                port[(strategy, gamma, tag)] = path
                assert np.all((path["weight"] >= 0) & (path["weight"] <= 1.5))
                assert abs(path["transaction_cost"][0]-tc*path["weight"][0]) < 1e-15
                assert np.max(np.abs(path["gross_return"]-path["portfolio_return"]-path["transaction_cost"])) < 1e-15
                # Independent scalar calendar-selection and arithmetic cover all 12 paths.
                scalar, scalar_w, scalar_t = scalar_path(prediction, oos_origins, oos_targets, market, rf, gamma, tc)
                maxerr = max(float(np.max(np.abs(scalar-path["portfolio_return"]))),
                             float(np.max(np.abs(scalar_w-path["weight"]))),
                             float(np.max(np.abs(scalar_t-path["turnover"]))))
                assert maxerr < 1e-12
                scalar_cer_error = abs(scalar_cer(scalar, gamma)-monthly_cer(path["portfolio_return"], gamma))
                assert scalar_cer_error < 1e-14
                scalar_errors.append({"strategy": strategy, "gamma": gamma, "tc": tc,
                                      "max_path_error": maxerr, "monthly_cer_error": scalar_cer_error})
                frame = calendar.copy()
                frame.insert(2, "strategy", strategy)
                frame.insert(3, "gamma", gamma)
                frame.insert(4, "tc", tc)
                frame["log_excess_forecast"] = prediction
                for key, values in path.items():
                    frame[key] = values
                monthly.append(frame)
                statistics = summary(path["portfolio_return"], rf_target, gamma)
                stats_rows.append({"strategy": strategy, "gamma": gamma, "tc": tc, **statistics,
                                   "mean_weight": float(path["weight"].mean()),
                                   "mean_turnover": float(path["turnover"].mean()),
                                   "floor_share": float((path["weight"] == 0).mean()),
                                   "cap_share": float((path["weight"] == 1.5).mean())})
            no_cost = port[(strategy, gamma, "tc0")]["portfolio_return"]
            with_cost = port[(strategy, gamma, "tc50")]["portfolio_return"]
            # Cost must reduce each strategy's absolute CER, not necessarily its CER gain over HA.
            assert monthly_cer(with_cost, gamma) < monthly_cer(no_cost, gamma)
    pd.concat(monthly, ignore_index=True).to_csv(args.outdir / "monthly_portfolios.csv", index=False, float_format="%.15g")
    pd.DataFrame(stats_rows).to_csv(args.outdir / "portfolio_statistics.csv", index=False, float_format="%.12g")
    pd.DataFrame(scalar_errors).to_csv(args.outdir / "scalar_cross_checks.csv", index=False, float_format="%.12g")

    grid_rows = []
    for strategy in ("raw", "d~_pair"):
        row = {"portfolio": strategy+"+CSR"}
        for gamma in GAMMAS:
            for _, tag in TCS:
                p = port[(strategy, gamma, tag)]["portfolio_return"]
                b = port[("HA", gamma, tag)]["portfolio_return"]
                row[f"CER{gamma}_{tag}"] = 1200*(monthly_cer(p, gamma)-monthly_cer(b, gamma))
                row[f"dSR{gamma}_{tag}"] = annual_sr(p-rf_target)-annual_sr(b-rf_target)
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(args.outdir / "utility_grid.csv", index=False, float_format="%.12g")
    write_json(args.outdir / "utility_grid.json", {"cer_unit": "annualized percentage points",
               "sharpe_unit": "annualized dimensionless difference", "baseline": "HA at matching gamma and tc",
               "columns": UTILITY_COLS, "rows": grid.to_dict("records")})

    boot, draws, boot_idx = bootstrap_pair(port, args)
    boot.to_csv(args.outdir / "bootstrap_pair_minus_raw.csv", index=False, float_format="%.12g")
    draws.to_csv(args.outdir / "bootstrap_draws.csv", index=False, float_format="%.12g")
    write_json(args.outdir / "bootstrap_pair_minus_raw.json", boot.to_dict("records"))
    np.save(args.outdir / "bootstrap_indices.npy", boot_idx)

    recursive, rolling, segments = [], [], []
    for gamma in GAMMAS:
        for tc, tag in TCS:
            raw = port[("raw", gamma, tag)]["portfolio_return"]
            pair = port[("d~_pair", gamma, tag)]["portfolio_return"]
            benchmark = port[("HA", gamma, tag)]["portfolio_return"]
            for strategy in ("raw", "d~_pair"):
                returns = port[(strategy, gamma, tag)]["portfolio_return"]
                for end in range(13, N_OOS+1):
                    recursive.append({"strategy": strategy, "gamma": gamma, "tc": tc,
                                      "forecast_origin": oos_origins[end-1], "realized_target": oos_targets[end-1],
                                      "n_months": end,
                                      "dcer_vs_ha_ann_pp": 1200*(monthly_cer(returns[:end], gamma)-monthly_cer(benchmark[:end], gamma)),
                                      "dsr_vs_ha_ann": annual_sr(returns[:end]-rf_target[:end])-annual_sr(benchmark[:end]-rf_target[:end])})
            for end in range(60, N_OOS+1):
                sl = slice(end-60, end)
                rolling.append({"gamma": gamma, "tc": tc, "forecast_origin": oos_origins[end-1],
                                "realized_target": oos_targets[end-1], "origin_window_start": oos_origins[end-60],
                                "target_window_start": oos_targets[end-60], "n_months": 60,
                                "dcer_pair_minus_raw_ann_pp": 1200*(monthly_cer(pair[sl], gamma)-monthly_cer(raw[sl], gamma)),
                                "dsr_pair_minus_raw_ann": annual_sr(pair[sl]-rf_target[sl])-annual_sr(raw[sl]-rf_target[sl])})
            for name, mask in (("full", np.ones(N_OOS, dtype=bool)),
                               ("pre_2020_origin", oos_origins < pd.Timestamp("2020-01-01")),
                               ("post_2020_origin", oos_origins >= pd.Timestamp("2020-01-01"))):
                for strategy in STRATEGIES:
                    returns = port[(strategy, gamma, tag)]["portfolio_return"]
                    stat = summary(returns[mask], rf_target[mask], gamma)
                    base = summary(benchmark[mask], rf_target[mask], gamma)
                    segments.append({"segment": name, "strategy": strategy, "gamma": gamma, "tc": tc,
                                     "origin_start": oos_origins[mask][0], "origin_end": oos_origins[mask][-1],
                                     "target_start": oos_targets[mask][0], "target_end": oos_targets[mask][-1],
                                     "n_months": int(mask.sum()), **stat,
                                     "dcer_vs_ha_ann_pp": stat["cer_ann_pp"]-base["cer_ann_pp"],
                                     "dsr_vs_ha_ann": stat["sr_ann"]-base["sr_ann"]})
    recursive, rolling, segments = pd.DataFrame(recursive), pd.DataFrame(rolling), pd.DataFrame(segments)
    assert len(recursive) == 2*2*2*(N_OOS-12) and len(rolling) == 2*2*(N_OOS-59)
    assert set(segments[segments.segment == "pre_2020_origin"].n_months) == {228}
    assert set(segments[segments.segment == "post_2020_origin"].n_months) == {60}
    recursive.to_csv(args.outdir / "recursive_paths.csv", index=False, float_format="%.12g")
    rolling.to_csv(args.outdir / "rolling60_differences.csv", index=False, float_format="%.12g")
    segments.to_csv(args.outdir / "segments.csv", index=False, float_format="%.12g")
    figures = create_figures(recursive, rolling, args.outdir)
    write_json(args.outdir / "figure_source_labels.json", figures)

    denominator = np.sum((y[N_TRAIN:]-ha[N_TRAIN:])**2)
    r2 = {s: float(100*(1-np.sum((y[N_TRAIN:]-p)**2)/denominator)) for s, p in pred.items()}
    assert abs(r2["raw"]-2.706) < 0.002 and abs(r2["d~_pair"]-3.956) < 0.002
    after = {key: sha256(path) for key, path in sources.items()}
    assert before == after
    check_report = {"status": "PASS", "market_reconstruction_max_abs_error": identity_error,
                    "log_excess_identity_max_abs_error": log_identity_error,
                    "r2oos_percent": r2, "legacy_max_cell_error": max(x["absolute_error"] for x in legacy_checks),
                    "legacy_tolerance": 0.002,
                    "scalar_max_path_error": max(x["max_path_error"] for x in scalar_errors),
                    "input_hashes_unchanged": before == after,
                    "assertions": ["492 full sample / 204 training / 288 OOS",
                                   "Forecast origins 2001-01--2024-12; realized targets 2001-02--2025-01",
                                   "HA only uses completed y targets through origin",
                                   "120 actual monthly market price returns ending at origin; no target-month variance input",
                                   "Origin RF only in plug-in forecast; target RF only in realized returns",
                                   "All 16 legacy CSR utility metrics recovered from stored forecasts",
                                   "All 12 new paths match independent scalar calendar arithmetic",
                                   "First cost equals tc * abs(initial weight - 0)",
                                   "Positive trading costs lower every strategy's absolute CER",
                                   "R2 anchors unchanged; original CW/DM files untouched",
                                   "No existing project module imported; no prediction generated"]}
    write_json(args.outdir / "checks.json", check_report)

    lines = ["# Paper 2 有限效用修复：复核记录", "",
             "本次仅使用已存预测重算组合效用；未估计模型、未生成预测、未调用 LLM，未改写原始数据、既有预测、共享 MetricsDef 或论文。产物状态为 candidate / not-integrated。", "",
             "## 定义与时间轴", "",
             "- 全样本 492 月；初始训练 204 月；样本外预测起点为 2001-01–2024-12，共 288 月。date 是预测起点 t，y 是 t+1 的对数超额收益；实际回报月份为 2001-02–2025-01。",
             "- RF 实际来源为 1-数据文件/F-F_Research_Data_Factors_26.csv，RF/100。市场简单价格收益来自 SP500_monthly_1970m2_2025m12.csv 的 Rate/100，不含股息。这里沿用原目标对应的 SP500 价格序列。",
             "- 原 y = log(1+Rm_target)−log(1+RF_target)。原预测及 HA 不变；HA[i]=mean(y[:i])，最后已实现目标月份为起点 t。",
             "- 简单超额预测 = (1+RF_origin)×expm1(log-excess prediction)，HA 同法。这是固定的 plug-in 转换，并非 Jensen 调整，也未估计条件期望的重变换修正。",
             "- 方差取截至起点 t 已实现最近 120 月真实简单市场价格收益的样本方差（ddof=1）；w=clip(pred_simple_excess/(γ×variance),0,1.5)，γ∈{3,5}。",
             "- 初始权重为 0；转手=|w_t−w_(t−1)|；成本=tc×转手，tc∈{0,0.005}。Rp=RF_target+w×(Rm_target−RF_target)−成本。",
             "- 月度 CER=mean(Rp)−γ×var(Rp)/2；ΔCER=1200×(CER_strategy−CER_HA)，单位为年化百分点（旧脚本误标为基点，旧数值的乘数同为 1200）。年化 SR 使用简单超额收益 Rp−RF_target，乘 sqrt(12)。",
             "- pre/post 按预测起点 2020-01 划分，分别 228/60 月；分段与 rolling 均切取连续实际组合收益，不在分段处把权重重新设为 0。", "",
             "## 修复后的 CSR 主表", "",
             "数值顺序：γ=3/tc0，γ=3/tc50，γ=5/tc0，γ=5/tc50；每格为 ΔCER 年化百分点 / ΔSR。", ""]
    for row in grid_rows:
        vals = [f"{row[f'CER{g}_{tag}']:.6f} / {row[f'dSR{g}_{tag}']:.6f}" for g in GAMMAS for _, tag in TCS]
        lines.append(f"- {row['portfolio']}：" + "；".join(vals) + "。")
    lines += ["", "## 配对区块 bootstrap", "",
              f"固定种子 {args.seed}、环形块长 {args.block_length} 月、{args.bootstrap_reps} 次，联合重采样 raw/pair 两组合已经实现的月收益；所有设定共用同一组抽样索引。推断条件于已估计权重与相应成本，未包含预测模型或权重估计的不确定性。",
              "差值=pair−raw。令 d 为原样本差值、d* 为区块重采样差值；中心化零假设抽样为 d*−d。双侧 p=[1+Σ1(|d*−d|≥|d|)]/(B+1)。主 95% 区间为 basic 区间 [d−q.975(d*−d), d−q.025(d*−d)]，另存百分位区间供核对。", ""]
    for row in boot.to_dict("records"):
        lines.append(f"- γ={row['gamma']}，{row['tc_tag']}：差值 {row['difference_pair_minus_raw_ann_pp']:.6f}，95% basic CI [{row['ci95_basic_lo_ann_pp']:.6f}, {row['ci95_basic_hi_ann_pp']:.6f}]，双侧 p={row['p_two_sided_centered']:.4f}。")
    lines += ["", "## 复核结果", "",
              f"- 市场收益恒等重构最大误差 {identity_error:.3e}；对数超额收益恒等式最大误差 {log_identity_error:.3e}。",
              f"- 用现存预测复算旧 2×8 指标最大误差 {check_report['legacy_max_cell_error']:.6g}，均低于 0.002 容差；未执行旧预测管线。",
              f"- 所有 12 条新组合路径与独立逐月标量计算一致，最大误差 {check_report['scalar_max_path_error']:.3e}；CER 再用 Python 标量均值与样本方差核对。",
              "- 端点、逐月连续、t→t+1、120 月窗口、无前视、首月成本及正成本降低各组合绝对 CER 的断言全部通过。相对 HA 的 CER 差不必随成本单调，未把该性质误作必要条件。",
              f"- R²OOS 锚点：raw={r2['raw']:.9f}%，pair={r2['d~_pair']:.9f}%。R²、CW、DM 仍针对原对数超额收益预测；本次未更换原目标或模型比较样本。",
              "- 9 个输入与旧实现文件的 SHA-256 在运行前后相同，见 manifest.json。",
              "- 递归 CER/SR 从第 13 个月起；rolling 为 60 个月，共 229 个窗口/设定。3 张图展示 γ=3、tc=0，完整 4 设定数值均另存。英文图例使用 raw/decontaminated disagreement，CSR 在定义文件中展开为所有非空子集的等权平均。",
              "- 未找到可以直接复用的已保存 EW 月度预测，因此本次新表仅列 CSR raw/pair；原 EW 效用结果留作历史对照，未重建 EW 预测。",
              "- 图像及标签基础信息已生成；正式读者标签审阅、最终渲染确认及论文整合由主代理另行完成。", ""]
    (args.outdir / "verification.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "artifact_state": "candidate", "manuscript_state": "not-integrated",
                "protocol": {"train": N_TRAIN, "oos": N_OOS, "variance_window": WINDOW,
                             "variance_ddof": 1, "gammas": list(GAMMAS), "transaction_costs": [x[0] for x in TCS],
                             "weight_bounds": [0, 1.5], "initial_weight": 0,
                             "seed": args.seed, "bootstrap_reps": args.bootstrap_reps,
                             "bootstrap_block": args.block_length,
                             "ci_method": "centered basic; supplementary percentile",
                             "prediction_conversion": "(1+RF_origin)*expm1(log_excess_forecast); plug-in, no Jensen adjustment"},
                "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
                "inputs": [{"key": key, "path": str(path.resolve()), "sha256_before": before[key],
                            "sha256_after": after[key], "unchanged": before[key] == after[key]}
                           for key, path in sources.items()],
                "outputs": [{"path": str(path.resolve()), "sha256": sha256(path)}
                            for path in sorted(args.outdir.iterdir()) if path.is_file() and path.name != "manifest.json"]}
    write_json(args.outdir / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "outdir": str(args.outdir), "r2": r2,
                      "grid": grid_rows, "bootstrap": boot.to_dict("records"),
                      "checks": check_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
