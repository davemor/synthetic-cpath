# Histology Clustering Experiment

Standalone clustering experiment for iCAIRD 100K, Kather 100K, and Spider
foundation-model embeddings.

The project consumes hproj-style embedding caches and dimensionality-reduction
outputs, but it does not import or extend hproj internals.

## Main Commands

```bash
cluster-exp validate-runtime
cluster-exp make-projection-manifest --config configs/hypatia.yml --output /path/to/projection_tasks.jsonl
cluster-exp run-projection-task --config configs/hypatia.yml --manifest /path/to/projection_tasks.jsonl --task-index 0
cluster-exp make-manifest --config configs/hypatia.yml --output /path/to/tasks.jsonl
cluster-exp run-task --config configs/hypatia.yml --manifest /path/to/tasks.jsonl --task-index 0
cluster-exp summarise --run-dir /path/to/run
```

On Hypatia, use the Slurm scripts in `slurm/`. They target `gpu.A100` and
validate RAPIDS/cuML runtime support before running GPU work.

Typical Hypatia flow:

```bash
sbatch slurm/prepare_projections_a100.sbatch
sbatch --dependency=afterok:<projection-manifest-job> slurm/run_projection_array_a100.sbatch
sbatch slurm/make_manifest_a100.sbatch
wc -l /home/dm236/scratch/histology_clustering/manifests/clustering_tasks.jsonl
sbatch slurm/run_clustering_array_a100.sbatch
sbatch slurm/summarise_a100.sbatch
```

Projection cache artifacts are written once per dataset, encoder, projector,
dimension, seed, and source fingerprint. Clustering tasks read that cache and
never refit PCA/UMAP. Each clustering task evaluates the reduced grid for one
representation and writes one labels artifact per clustering configuration.

Final efficient clustering grid:

- K-Means: `k=[4,8,16,24,32,48,64,96,128,192,256]`, `k-means++`, `n_init=10`, `max_iter=300`
- HDBSCAN: `min_cluster_size=[50,100,200]`, `min_samples=25`, `cluster_selection_method=eom`
- Fit subsample: deterministic random `10%` of each representation, with `min=10000` and `max=50000`
