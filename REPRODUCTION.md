# Verified report reproduction

Use each project's own lockfile and installation instructions for training. The reporting checks used the environment recorded in `REPORTING_ENVIRONMENT.json`. Private images, embeddings, labels and model checkpoints are not part of this source tree.

## Chapter 3

The selected benchmark generates model predictions with the configs under `chapter03/benchmark/configs`. The traced prediction runs are `icaird120k_annotation_labels_seed0_20260615`, `icaird120k_annotation_label_images_seed0_20260615`, and `icaird_mil_uni2h_bags_gpu_20260616_1335` on Hypatia. Original prediction paths are recorded in `evidence/chapter3-benchmark-checks.json`.

For the patch/image bootstrap export, the traced parameter is `--resamples 1000 --seed 20260702`; do not silently substitute the thesis prose's 10000 if reproducing its reported intervals. The separate MIL table uses 10000. The generators are `chapter03/reporting/make_icaird_thesis_results_table.py` and `make_mil_bootstrap_table.py`.

Foreground evaluation runs through `chapter03/histokit/scripts/summarize_fgbg_method_results.py`, then `summarize_background_segmentation_methods.py`; pass `--output-dir` to a fresh directory. See the copied script defaults for the exact private input paths.

## Chapter 4

`chapter04/reporting/render_full_quality_tables.py` reads `chapter04/final-results/quality-results-with-nufd.csv` and regenerates the quality tables. Plot scripts similarly read the preserved final-result and per-seed utility exports. These inputs are the thesis reporting version.

Regenerating normalization from embeddings requires the source manifests/caches recorded in the original metadata. Re-exporting the current live per-run JSON tree fails because records are missing and some surviving metric values have changed; this is documented, not treated as a successful check.

`generation` and `evaluation` are separate synpath package environments. Do not install both into one environment and assume import resolution identifies the intended version. Use the saved Docker/Slurm launch scripts with explicitly selected source mounts and configs.

## Chapter 5

Set RUN to the archived directory `hproj-2026-05-25_16-41-27-bb2a5203-icaird100k`, not an arbitrary latest run. Write outputs to a fresh directory.

```bash
python chapter05/reporting/make_threshold_tables.py   --thresholds "$RUN/curve/thresholds_results.csv"   --intrinsic-dims "$RUN/intrinsic_dims.csv"   --curve-dir "$RUN/curve" --output-dir "$OUT/thresholds"

python chapter05/reporting/summarise_generalisation_for_thesis.py   --run-root "$RUN" --output-dir "$OUT/generalisation"

python chapter05/reporting/make_generalisation_summary_package.py   --run-root "$RUN" --output-dir "$OUT/summary"
```

The final summary command reads the saved `RUN/tables/generalisation_thesis_summary/generalisation_threshold_validation_full_appendix.csv`. It was checked against the published numbers; it does not rerun model training. Preserve that archived input alongside any newly generated summaries.

## Chapter 6 and Appendix B

Copy the complete archived `clustering_stability.csv` to a fresh report directory. It contains all four clustering methods; the local `clustering_results.csv` does not.

```bash
PYTHONPATH=chapter06/clustering/src python   chapter06/clustering/plots/plot_rq1_dimension_metric_families.py   --run-dir "$OUT/clustering"
```

This reproduces the numeric plotted-values CSV exactly. The appendix generator is `cluster_exp.rq1_appendix.generate_rq1_appendix_dimension_tables(run_dir=..., stability_csv=...)`; its two LaTeX tables reproduced byte-for-byte.

`chapter06/reporting/plot_class_global_vs_low_d_global_grid.py` uses the copied aggregate inputs to reproduce the low-dimensional conditioning deltas and figures. The saved per-seed CSV replaces dependence on an expired temporary export. The configs in `synthesis-configs` specify the 42/50 PCA dimensions and k=24. Actual synthesis uses the Chapter 4 generator/evaluator stages.

## Historical appendices

Appendix C uses the explicitly named paper notebooks, pinned to the recovered pathgen commit. Their saved metrics were checked before outputs were removed from the publication copies. Use the private original clone to inspect those saved outputs.

Appendix E uses the pinned wsipipe and wsipipe-isyntax projects; follow their own package/dependency documentation.
