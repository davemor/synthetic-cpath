# DGX Docker Runbook

The DGX-1 deployment uses Docker rather than Slurm. Runtime data, caches,
checkpoints, generated images, reports, Hugging Face cache, and external model
repos should live under:

```bash
/raid/dm236/synpath
```

## Setup

From the repository root on the DGX:

```bash
make dgx-env
```

Edit `/raid/dm236/synpath/dgx.env` and add private values such as
`HF_TOKEN`. The default paths are:

```bash
SYNPATH_DATA_ROOT=/raid/dm236/synpath/data
SYNPATH_CACHE_ROOT=/raid/dm236/synpath/cache
SYNPATH_OUTPUTS_ROOT=/raid/dm236/synpath/outputs
SYNPATH_REPORTS_ROOT=/raid/dm236/synpath/reports
SYNPATH_EXTERNAL_ROOT=/raid/dm236/synpath/external
HF_HOME=/raid/dm236/synpath/huggingface
TORCH_EXTENSIONS_DIR=/raid/dm236/synpath/torch_extensions
```

Build the image:

```bash
make docker-build
```

Install the external StyleGAN repos and dedicated StyleGAN environment inside
the container before the full run:

```bash
make dgx-shell
docker exec -it synpath-shell bash scripts/setup/install_stylegan2ada.sh
docker rm -f synpath-shell
```

## Run

Smoke run:

```bash
make dgx-smoke
docker logs -f synpath-smoke
```

Full run:

```bash
make dgx-run
docker logs -f synpath-main
```

## GPU Use

The container is launched with `--gpus all`. StyleGAN uses all visible GPUs for
training jobs. Other stages currently run as single processes, so the full
pipeline is not perfectly GPU-saturated end to end. The simplest robust
parallelisation is to keep one end-to-end run and let StyleGAN use all eight
V100s. A more aggressive option is to run one container per dataset with
different `CUDA_VISIBLE_DEVICES` masks, then run the report stage after all
dataset containers finish.
