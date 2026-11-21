# Dataset creation and preprocessing

Source selected; final foreground and benchmark values traced.

## Selected components

- `histokit`
- `patcher`
- `benchmark`
- `reporting`

## Original execution/source locations

- `hypatia:/home/dm236/repos/histokit`
- `hypatia:/home/dm236/repos/patcher`
- `hypatia:/home/dm236/repos/icaird-cervical-benchmark`
- `local:scripts`

## Evidence

- All 5 foreground method rows match the Hypatia methods_performance_summary.csv.
- All 20 embedding/image benchmark rows, including confidence intervals, match the Hypatia bootstrap export.
- MIL table matches the saved 10,000-resample MIL bootstrap table.
- Local and Hypatia histokit Python/shell/configuration files are identical in the preservation snapshots.

## Boundaries and findings

- Patch/image benchmark bootstrap export records 1000 resamples, while the thesis states 10000; MIL records 10000.
- Private annotation/patch inputs are required to retrain preprocessing and benchmark models.

Source hashes and original copy locations are recorded in `../SOURCE_MANIFEST.json`. The full tracing report is `../PROVENANCE.md`.
