# Data sources and restricted-data handling

No data files are included. Paths below follow the repository's mirrored relative layout. Users must supply inputs in an isolated local copy and must not commit them.

## Public or published sources

| Provider/source | Local examples | Access notes |
|---|---|---|
| Survey of Professional Forecasters, Federal Reserve Bank of Philadelphia | `1-数据文件/meanLevel_26.xlsx`, `meanGrowth_26.xlsx`, `Dispersion_{1,2,3}_26.xlsx` | Obtain official consensus/dispersion workbooks; record versions and access dates. |
| Philadelphia Fed real-time data and publication calendars | Real-time macro series and SPF timing inputs | Preserve realtime vintages and release-date conventions. |
| SPF density forecasts | `1-数据文件/SPF_density_forecasts/Individual_PRGDP.xlsx`, `Individual_PRPGDP.xlsx`, `Individual_PRCCPI.xlsx` | Individual histogram records are not redistributed here. |
| Livingston Survey, Federal Reserve Bank of Philadelphia | `livingston_individualdata.xlsx`, `livingston_means.xlsx`, `livingston_Dispersion1.xlsx` | Obtain official individual and summary workbooks. |
| Michigan Survey of Consumers | Six categorical-response measures used through user-supplied intermediate panels | Follow the provider's access and redistribution terms. |
| FRED/ALFRED and NBER | FRED controls, ALFRED vintages, recession dates | `download_alfred_vintage.py` makes explicit network requests and needs no API key; preserve vintages. |
| Welch–Goyal predictor data | `1-数据文件/PredictorData2024.xlsx` and macro predictors | Obtain from the published source and cite the associated work. |
| Bordalo et al. replication package | `1-数据文件/BGLS2024_TimeSeries01.tab` | Public archive DOI `10.7910/DVN/WN1FE1`; check archive license/terms before redistribution. |
| Kenneth French Data Library | FF3 daily/monthly factors and RF | Obtain original files and record vintages. |
| CATFIN, PLS sentiment, JLN uncertainty, VIX | State-battery source files | Use original published sources and their access terms. |

## Licensed or restricted sources

- **CRSP**: `1-数据文件/daily_stock_return_25.csv` and other CRSP records require an authorized subscription. Raw records, stock-month checkpoints, and redistributable extracts must not be uploaded. AggIVOL cannot be fully regenerated without authorized inputs.
- **I/B/E/S and OptionMetrics**: not required as raw inputs for every current main estimate, but historical mixed panels and excluded bridge projects may contain their derived content. Do not upload mixed panels without a field-level license review.
- **Mixed derived panels**: `all_data_qe.csv`, `panel9_qe.csv`, `panel9_qe_raw.csv`, `un_data_recursive.csv`, return/RF files, and prediction files are excluded even when some columns originate in public sources, because their combined provenance and redistribution rights are not uniformly cleared.

## Generated local intermediates

The following are produced or consumed locally and are not source files in this repository:

- `3-代码文件/1-sCNNLSTM-and-meta-RSR-notebooks/all_data_qe.csv`, `SP500_h1.csv`, `mkt_RF_truncated.csv`, `USREC.csv`, `baseline_prediction.csv`, `un_data.csv`.
- `3-代码文件/3-latent-disagreement/panel9_qe.csv`, `panel9_qe_raw.csv`, `un_data_recursive.csv`, `liv4_qe.csv`, `aggivol_monthly.csv`.
- Historical comparison inputs such as `utility_grid_2001_p9.csv`, `cfdr_news_series.csv`, `t1_lv_channel_p9.csv`, `huang_market_measures.csv`, `preds_1991_p9.csv`, and `segment_1991_p9.csv`.

Users must establish the provenance of each input independently. Successful execution with substitute or updated files is not exact reproduction of the manuscript sample.

## Before running

1. Record provider URL, access date, version/vintage, license or terms, and local SHA-256 for every supplied input.
2. Keep restricted files under local access controls and outside public upload directories.
3. Check row counts, date grids, missingness, target timing, and realized-return month alignment.
4. Verify that state coverage includes all intended variables, including AggIVOL, before interpreting state tables.
5. Do not publish generated logs, forecasts, or aggregate derivatives without reviewing whether they reveal restricted records.
