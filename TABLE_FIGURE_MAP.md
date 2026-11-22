# Static paper-to-source map

This is a static correspondence map, not a reproduction report. All paths below are under `3-代码文件/3-latent-disagreement/` except `MetricsDef.py`, which is under `3-代码文件/1-sCNNLSTM-and-meta-RSR-notebooks/`.

| Paper object | Current source entry points | Output/verification boundary |
|---|---|---|
| Table 1 | `make_panel9.py` | Produces raw-unit panel, not the final descriptive-statistic table export. |
| Table 2 | `run_evidence.py` | Stylized correlations/PCA outputs; not rerun. |
| Tables 3 and A1 | `run_evidence.py`, `run_bw6.py` | Main and 1991 variants/BW6; not rerun. |
| Table 4 | `run_evidence.py` | Pairwise comparisons; not rerun. |
| Table 5 | `run_horserace.py`, `run_evidence.py` | EW/PCA/PLS and CSR comparison; not rerun. |
| Tables 6 and A2 | `run_direct_compare.py`, `run_figures.py` | DM/segment results; not rerun. |
| Table 7 | `run_evidence.py`, `run_dhat_pair.py` | Level-only and fitted-side analyses; secondary T1 dependencies excluded. |
| Table 8A | `run_measure_superiority.py --test Ap` | Dose-subset results; some summary steps remain console/manual. |
| Table 8B | `run_placebo_v2.py` | Corrected per-column AR(1) source retained; 1,000 reps not rerun. |
| Tables 9 and C3 | `run_state_battery.py`, `run_aggivol.py` | AggIVOL needs licensed CRSP; incomplete-state execution is not full reproduction. |
| Table 10 | `run_attribution.py` | State attribution; not rerun. |
| Table 11 | `run_utility_repair_p9.py` | Sole current utility source; legacy comparison inputs excluded. |
| Table 12A | `run_cfdr_race.py` | News and CF/DR forecast comparisons; not rerun. |
| Table 12B | `run_dm_subjective.py` | BGLS published-package input; not rerun. |
| Table B1 | `run_frequency.py` | Frequency appendix; not rerun. |
| Table C1 | `run_kscan.py`, `run_robustness.py` | k-layer and robustness results; legacy utility columns are not Table 11 sources. |
| Table C2 | `run_shrinkage_horserace.py` | Shrinkage comparison; not the complete CSR–ridge correlation claim generator. |
| Table C4 | `run_pair_orth_u14.py` | Sequential projection; not rerun. |
| Table C5 | `run_loo_attribution.py` | Leave-one-out analysis; not rerun. |
| Table C6 | `run_post2020_attenuation.py` | Post-2020 analysis; not rerun. |
| Table C7 | `download_alfred_vintage.py`, `run_bw6_vintage.py` | Explicit network acquisition and mixed-vintage analysis; not run. |
| Table D1 | `run_density_benchmark.py` | Density benchmark; some correlation summaries remain console/manual. |
| Table E1 | `run_sca_external.py` | Requires user-supplied mixed panel; not rerun. |
| Table E2 | `make_liv_panel.py`, `run_liv_external.py`, `run_liv_frequency.py` | Livingston panel and horizons; not rerun. |
| Main Figures 1, 2, 3, 6 and SI Figure 1 | `run_figures.py` | Current root script; not repair-work copies. |
| Main Figures 4 and 5 | `run_direct_compare.py` | Current root script; not rerun. |
| Main Figures 7, 8, 9 | `run_utility_repair_p9.py` | Figure rename/install step remains external. |
| Recession robustness prose | `run_recession_inference.py` | Exclusion/bootstrap source; 5,000 reps not rerun. |
| SPF timing exception prose | `run_spf_timing_check.py` | Publication-date checks; not rerun. |
| Same-window PLS prose | `run_hlw_aligned.py` | Required SPF9 arm plus excluded historical bridge dependencies; not rerun. |
