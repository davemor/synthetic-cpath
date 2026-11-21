# synpath

Synthetic histopathology generation experiments with true labels,
unconditioned generation, and UNI-derived pseudo-labels.

The project is designed as a config-driven command line benchmark. It prepares
patch datasets, caches pathology foundation-model embeddings, selects KMeans
pseudo-labels, trains external generative models, evaluates generated patches,
and aggregates results into CSV files, figures, and a short Markdown report.

## Study

The main study evaluates whether pseudo-label conditioning can recover much of
the fidelity and diversity benefit of true-label conditioning when annotations
are unavailable.

Datasets are treated independently:

- PathMNIST+ 224 / Kather100K
- SPIDER skin
- SPIDER colorectal
- SPIDER thorax
- SPIDER breast
- iCAIRD 100K, later via manifest CSV and ImageFolder-compatible patches

Generation arms are run per dataset:

- StyleGAN2-ADA, true-label conditional
- StyleGAN2-ADA, unconditioned
- StyleGAN2-ADA, UNI KMeans pseudo-label conditional
- Stable Diffusion 1.5 LoRA, true-label prompted
- Stable Diffusion 1.5 LoRA, unconditioned prompted
- Stable Diffusion 1.5 LoRA, UNI KMeans pseudo-label prompted

Utility classifiers are only run for labelled data:

- real train only
- true-label synthetic only
- real plus true-label synthetic

Pseudo-label and unconditioned generated samples are not used for supervised
pathology classification.

## Install

This repository expects to be run on an Ubuntu NVIDIA DGX or similar CUDA
system. Development on macOS is supported for configuration, reports, and tests,
but model training is expected to run on Linux with GPUs.

```bash
uv sync
```

External model tooling is installed separately:

```bash
bash scripts/setup/install_stylegan2ada.sh
bash scripts/setup/install_sd_lora_env.sh
```

The StyleGAN setup creates a dedicated environment at
`${SYNPATH_EXTERNAL_ROOT:-external}/stylegan2ada-env`. Set
`SYNPATH_STYLEGAN_PYTHON` to its `bin/python` if you want to force that
interpreter outside the provided Slurm scripts. On Hypatia, the Slurm templates
set `SYNPATH_STYLEGAN_BACKEND=stylegan3`, train the StyleGAN2 architecture via
StyleGAN3's `--cfg stylegan2` code path, and load a CUDA module with
`TORCH_EXTENSIONS_DIR` under scratch so StyleGAN's custom CUDA extensions can
compile and be cached.

Dataset downloads are also explicit setup steps:

```bash
bash scripts/setup/download_pathmnist_plus.sh
bash scripts/setup/download_spider.sh
```

## CLI

The intended CLI shape is:

```bash
synpath plan --config configs/experiments/main.yaml
synpath prepare-data --dataset pathmnist224
synpath embed --dataset pathmnist224 --encoder uni
synpath cluster --dataset pathmnist224
synpath select-subset --dataset pathmnist224
synpath train-generator --dataset pathmnist224 --model stylegan2ada --conditioning true_label
synpath sample --dataset pathmnist224 --model stylegan2ada --conditioning true_label
synpath evaluate-quality --dataset pathmnist224 --model stylegan2ada --conditioning true_label
synpath train-utility --dataset pathmnist224 --source real_plus_synthetic_true_label
synpath report --config configs/experiments/main.yaml
```

Shell wrappers for common DGX runs live under `scripts/run/`.

DGX Docker deployment is documented in `docs/dgx-docker.md`. The short version
is:

```bash
make dgx-env
make docker-build
make dgx-run
make dgx-logs
```

## Outputs

Artifacts are re-entrant and auditable. The default layout is:

```text
{SYNPATH_OUTPUTS_ROOT}/generators/{dataset}/{model}/{conditioning}/seed{seed}/
    resolved_config.yaml
    condition_manifest.csv
    synthetic_manifest.csv
    checkpoints/
    samples/
    metrics/
    logs/

{SYNPATH_OUTPUTS_ROOT}/utility/{dataset}/resnet18/{source}/seed{seed}/
    resolved_config.yaml
    metrics.json
    classification_report.json
    confusion_matrix.csv
    predictions.csv

{SYNPATH_REPORTS_ROOT}/main/
    results.csv
    quality_metrics.csv
    utility_metrics.csv
    figures/
    report.md
```

UNI embeddings and cluster assignments are cached under
`{SYNPATH_CACHE_ROOT}` because they are reused across clustering, quality
metrics, privacy audits, and reports.

Clustering is fitted on the complete training split before any configured
experiment cap is applied. Validation and test patches are never included.

## Design

See:

- `docs/design.md`
- `docs/pipeline-design.md`
- `docs/experiment-plan.md`
- `docs/coding-standards.md`
- `docs/hypatia-slurm.md`
- `docs/implementation-status.md`
