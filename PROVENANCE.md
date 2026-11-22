# Provenance and verification boundary

## Source identity

The 35 source files were copied from the current local working tree, not from Git HEAD. This preserves the untracked current `run_utility_repair_p9.py` and modified current placebo/figure scripts. No `results/`, figure-repair work copies, internal audits, archived 21-panel projects, or later contribution-followup code are included.

`MANIFEST.json` records source and release hashes. `SHA256SUMS.txt` covers release files.

## Release transformations

- Preserve relative source directories required by `__file__` path logic.
- Localize the restricted-data root in `run_aggivol.py` and remove aggregate-panel row previews.
- Align `run_evidence.py` documentation with the current 2001 main specification; runtime arguments remain explicit.
- Correct the placebo driver's output-overwrite documentation.
- Add source/data/run documentation, an unpinned dependency list, output exclusions, and MIT licensing.

No model, estimator, sample, forecast, inference, or utility formula was otherwise intentionally changed for this source release.

## Checks performed

- 35 Python files copied and syntax-compiled.
- No personal absolute path or common secret pattern detected in the copied source.
- No data, forecast, cache, log, generated figure, or internal review file included.
- Static paper-to-source mapping recorded in `TABLE_FIGURE_MAP.md`.

## Checks not performed

No module-import sweep, real-data run, estimation, network download, CRSP processing, full placebo/bootstrap batch, or numerical table/figure comparison was performed. No result is marked reproduced.

The source set retains disclosed dependencies on excluded mixed panels and historical comparison artifacts. Table 1 and some summaries/figure installation steps remain manual or console-based. The package must therefore be described as research source code, not a validated replication package.

## Future updates

Preserve this version. Any decoupling, clean-environment execution, input-version registration, or numerical comparison should be recorded in a new tagged version with updated hashes and an explicit list of reproduced and unreproduced objects.
