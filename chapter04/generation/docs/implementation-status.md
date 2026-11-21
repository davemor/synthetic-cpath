# Implementation status

This document records what exists in the current project skeleton and what is
still deliberately pending.

## Implemented

- `uv` Python package metadata and `src/` package layout.
- Typer CLI entry points for all planned stages.
- Config loading with simple `${oc.env:NAME,default}` interpolation.
- Pipeline runner that expands experiment configs into concrete jobs.
- Stage classes with explicit output artifacts and resume checks.
- Dataset, condition, and synthetic manifest schemas.
- PathMNIST+ manifest preparation through the official `medmnist` API.
- SPIDER manifest preparation through Hugging Face snapshot download and
  extracted metadata/image trees.
- Condition manifest generation for:
  - `true_label`
  - `unconditioned`
  - `pseudo_label`, when cluster cache CSV exists
  - `label_global_cluster`, when global cluster cache CSV exists
  - `label_local_cluster`, when within-class cluster cache CSV exists
- UNI embedding extraction via `timm` / Hugging Face.
- Global KMeans pseudo-label selection.
- Within-class KMeans local-cluster labels.
- StyleGAN2-ADA training and sampling wrappers.
- Stable Diffusion 1.5 LoRA training and sampling wrappers.
- UNI-FID/KID, UNI precision/recall, cluster coverage, privacy nearest-neighbor
  metrics, and optional clean-fid Inception-FID.
- ResNet-18 utility training and evaluation.
- Report aggregation into CSV files and `report.md`.
- Report stage that creates initial `results.csv` and `report.md`.
- Setup scripts for PathMNIST+ and SPIDER downloads.
- Hypatia Slurm templates.

## Pending

- Full visual figure generation in the report.
- Empirical validation of SPIDER metadata parsing against the downloaded files
  on Hypatia.
- Empirical validation of the external StyleGAN2-ADA command line on Hypatia.
- Empirical validation of the external Diffusers LoRA command line on Hypatia.
- Runtime tuning of generator budgets for the available GPU queue.

The project is now ready for the next phase: run the smoke pipeline on Hypatia,
fix any dataset-specific parsing issues exposed by the real downloaded files,
then start full dataset/model runs.
