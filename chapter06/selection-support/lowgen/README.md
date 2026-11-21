# Lowgen

Fast low-dimensional synthetic-data experiments for the histopathology embedding
work.

## Fast Cluster-Conditioned Audit

Use this when you need a quick answer to:

> Do clustering validation metrics predict the quality of feature-space samples
> generated from those clusters?

The audit reuses three expensive artifacts instead of recomputing them:

- dimensionality targets from the DGX/Hypatia follow-up run;
- cached lowgen representations under `paths.output_root / paths.run_name`;
- `clustering_experiment` outputs, especially `clustering_results.csv` and the
  referenced `labels_*.npz` files.

Minimal flow:

```bash
lowgen dgx import-dimensions --config lowgen/config/dgx_v2.yml
lowgen dgx build-manifest --config lowgen/config/dgx_v2.yml
lowgen dgx cluster-conditioning-audit \
  --config lowgen/config/dgx_v2.yml \
  --clustering-results /path/to/clustering_results.csv \
  --dataset icaird100k \
  --encoder uni2 \
  --projector umap \
  --target-name intrinsic_mle \
  --seed 0 \
  --max-cluster-configs 12 \
  --synthetic-total 5000 \
  --generator-kind cluster_guided_full_diag_gaussian_probe
```

Use `roc_auc_99` when that target exists in both the lowgen manifest and the
clustering results; otherwise use the matching `intrinsic_mle` fallback.

Outputs are written to
`<run_dir>/cluster_conditioning_audit/`:

- `selected_cluster_configs.csv`: the selected metric-spread clustering rows;
- `cluster_guided_full_embedding_results.csv`: one row per clustering
  configuration, with validation metrics, coverage, full-space fidelity, and
  diversity columns for a cluster-guided full-embedding diagonal Gaussian probe;
- `cluster_metric_quality_correlations.csv`: Pearson and Spearman correlations
  between clustering metrics and synthetic quality metrics, grouped by
  generator;
- `skipped_rows.csv`: cluster rows that could not be aligned or evaluated;
- `metadata.json`: provenance for the audit.

The default cluster-conditioning generator is
`cluster_guided_full_diag_gaussian_probe`.
It is local to this audit rather than part of the main lowgen generator family:
for each reduced-space cluster, it samples full-dimensional embeddings from a
diagonal Gaussian fitted to the full embeddings assigned to that cluster. For a
fast exemplar run, cap `--max-cluster-configs` and keep
`include_noise_cluster: false`.

## DGX Container + Dask

For larger cluster-conditioned probes, build and run inside a RAPIDS container
on the DGX rather than the shared venv:

```bash
cd /raid/experiments/lowgen_v2/lowgen
scripts/build_dgx_container.sh lowgen-dgx:latest
scripts/run_dgx_container.sh dgx cluster-conditioning-audit \
  --config config/dgx_v2.yml \
  --clustering-results /raid/experiments/lowgen_v2/output/lowgen-v2-trainsized-bic-k/cluster_guided_full_embedding_inputs/umap/clustering_results_selected.csv \
  --output-dir /raid/experiments/lowgen_v2/output/lowgen-v2-trainsized-bic-k/cluster_guided_full_embedding_exemplar_umap_dask \
  --dataset icaird100k \
  --encoder uni2 \
  --projector umap \
  --target-name intrinsic_mle \
  --seed 0 \
  --max-cluster-configs 12 \
  --synthetic-total 5000 \
  --generator-kind cluster_guided_full_diag_gaussian_probe \
  --n-workers 8 \
  --dask-chunk-size 2 \
  --force
```

`--n-workers` enables row-chunk Dask execution for the cluster audit. With
`runtime.require_cuml: true`, the runner starts a `LocalCUDACluster`, so the
container should include `cupy`, `cuml`, `dask-cuda`, and `dask-cudf`.
