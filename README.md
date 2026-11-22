# Recovering Latent Disagreement: research source code

Research source code for **“Recovering Latent Disagreement: Decontaminated Survey Disagreement and Stock Market Return Predictability.”** The repository preserves the current working-tree source files corresponding to the Journal of Forecasting main document and Supporting Information.

## Status and scope

- Thirty-five Python source files cover the main 12 tables and 9 figures, Supporting Information 13 tables and 1 figure, and related in-text analyses through static source mapping.
- This is **research source code, not a validated replication package**. No clean-environment run, full estimation, placebo/bootstrap batch, or numerical table/figure comparison was performed for this release.
- Raw data, derived panels, forecasts, saved predictions, caches, checkpoints, logs, generated figures, `results/`, internal reviews, and post-publication exploratory projects are excluded.
- CRSP, I/B/E/S, OptionMetrics, and mixed panels containing restricted or unverified-redistribution content are not distributed. Public data must be obtained from original providers as described in `DATA.md`.

## Directory convention

The repository mirrors the relative source layout because the scripts derive their input and output roots from `__file__`:

```text
3-代码文件/1-sCNNLSTM-and-meta-RSR-notebooks/MetricsDef.py
3-代码文件/3-latent-disagreement/*.py
3-代码文件/3-latent-disagreement/results/     # empty output placeholder
3-代码文件/3-latent-disagreement/figures/     # empty output placeholder
1-数据文件/                                  # empty source-data placeholder
```

Do not flatten these directories without updating the scripts' path logic. Place user-supplied source and intermediate files according to `DATA.md`.

## Important execution warning

Many scripts have no main guard and read data or write outputs on import. Do **not** import all modules as a test. Run them only in an isolated copy with user-supplied inputs and a separate output location. Several scripts overwrite fixed filenames. `download_alfred_vintage.py` performs network requests only when explicitly run. See `RUNBOOK.md`.

## Current main specification

- Nine SPF-only level/disagreement pairs.
- Quarter-end monthly panel, 1984-01–2024-12.
- Main out-of-sample forecast origins: 2001-01–2024-12; initial training length 204 and 288 out-of-sample origins.
- Expanding recursive estimation; same-question decontamination; powerset complete subset regression over all 511 nonempty predictor subsets.
- The 1991-start analysis is an appendix specification. Always pass `--panel9` and explicit start arguments; do not rely on historical defaults or docstrings.

## Setup

Python 3.9 or later is required by `run_utility_repair_p9.py`. The release does not provide a unified locked environment. Install package names from `requirements.txt` in an isolated environment and record actual versions before numerical comparisons.

```bash
python -m pip install -r requirements.txt
```

See `RUNBOOK.md` for ordered commands and `TABLE_FIGURE_MAP.md` for static paper-to-source mappings.

## Known source-only limitations

The 35 files are the minimal direct evidence source set, not a clean runnable closure:

1. `make_panel9.py` reads the user-supplied mixed `all_data_qe.csv` for dates, returns, and anchor checks. SCA external validation also uses that mixed panel. Those inputs are not included.
2. Current utility repair reads historical `utility_grid_2001_p9.csv` and variants for legacy comparison. Those retired utility artifacts are excluded; current Table 11 must not be sourced from the older `MetricsDef.cer` implementation.
3. `run_dhat_pair.py` additionally reads `cfdr_news_series.csv` and `t1_lv_channel_p9.csv` for secondary T1/channel comparisons.
4. `run_hlw_aligned.py` contains both the required SPF9 arm and historical 24-measure/bridge comparisons; the latter require excluded CRSP/OptionMetrics-related and exploratory artifacts.
5. AggIVOL requires licensed CRSP daily stock returns. Without `aggivol_monthly.csv`, state-battery code can produce an incomplete state panel; always verify the expected 11-state coverage rather than treating execution as full Table 9/C3 reproduction.
6. Table 1 descriptive-statistic export, some Table 8A/D1 summaries, and final figure renaming/installing still include console or manual steps.
7. Full 1,000-replication placebo, 5,000-replication bootstrap, and all numerical table/figure outputs remain unrun in this release.

These limitations do not prevent source review, but they prevent an unqualified reproducibility claim. No numerical output is marked reproduced here.

## Data and code availability

Follow `DATA.md` for source descriptions and access conditions, required local files, and restricted-data handling. Publicly accessible does not automatically mean redistributable. Users must record provider, version/vintage, access date, license/terms, and local hashes before running comparisons.

## License and updates

Code is released under the MIT License; see `LICENSE`. Third-party data and dependencies retain their own terms.

Preserve a tagged review version and checksums. Future corrections or clean reproduction work should create a new tagged version with updated `PROVENANCE.md`, `MANIFEST.json`, and `SHA256SUMS.txt`, rather than silently replacing the reviewed source.
