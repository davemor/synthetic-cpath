# Clustering low-dimensional representations

Clustering and conditioning stages selected; final outputs reproduced.

## Selected components

- `clustering`
- `selection-support`
- `conditioning`
- `synthesis-configs`
- `reporting`

## Original execution/source locations

- `hypatia:/home/dm236/repos/histology-clustering-experiment`
- `wopr:/home/dm236/repos/histology-clustering-experiment`
- `local:clustering_experiment`
- `dgx:/raid/dm236/synpath_balanced_global_clusters`
- `dgx:/raid/experiments/lowgen_v2`

## Evidence

- Hypatia ran projection/clustering tasks; the WOPR script explicitly rsyncs the spherical tasks and runs the combined 32-worker summary.
- 51 WOPR summary/trend/table output files match the local Chapter6 results by SHA-256.
- The local clustering computational Python modules match the WOPR source; its extra CLI/appendix module provides the final appendix tables.
- Both Appendix B full dimension tables regenerate byte-for-byte.
- Combined PCA/UMAP plotted-values CSV regenerates byte-for-byte. PNG bytes differ in the current rendering environment.
- The low-dimensional conditioning comparison recomputes all 24 quality-delta rows and 72 utility-delta rows numerically.
- Saved synthesis configs specify UNI2/PCA/KMeans k=24, dimension 42 for iCAIRD and 50 for pathmnist224/Kather.

## Boundaries and findings

- The local res/clustering_results.csv is only a 4704-row kmeans/hdbscan subset. Do not use it as the complete clustering dataset; clustering_stability.csv contains all four methods and reproduces the thesis outputs.
- The low-dimensional generator stages reuse Chapter 4 generation/evaluation code with the Chapter 6 saved configs.
- Selection support is preserved, but the thesis description of evenly weighted six metrics needs reconciliation with the saved balanced_compromise/Pareto selection metadata.

Source hashes and original copy locations are recorded in `../SOURCE_MANIFEST.json`. The full tracing report is `../PROVENANCE.md`.
