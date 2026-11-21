# Running on Hypatia

This page records the intended Slurm setup for running `synpath` on the
University of St Andrews Hypatia cluster.

The public Hypatia documentation says GPU jobs should use one of the GPU
partitions and must also request GPUs with `--gres`. The documented GPU
partitions are:

- `gpu.L40S`, up to 8 NVIDIA L40S GPUs per node, maximum runtime 4 days
- `gpu.A100`, up to 4 NVIDIA A100 GPUs per node, maximum runtime 4 days
- `gpu.A30`, 2 NVIDIA A30 GPUs, maximum runtime 4 days

Jobs are submitted with `sbatch <job-script>` and monitored with
`squeue -u $USER`.

Reference:

- https://www.st-andrews.ac.uk/high-performance-computing/help-and-contact/using/

## Repository layout

Hypatia-specific Slurm templates live in:

```text
slurm/hypatia/
```

They are intentionally thin wrappers around the normal CLI. The project should
still be runnable locally and on other DGX-like systems without importing
Hypatia assumptions into the Python package.

## First-time setup

On Hypatia, clone the repo and create a project environment in a location with
enough quota:

```bash
git clone <repo-url> synpath
cd synpath
cp .env.example .env
```

The Hypatia Slurm templates default large runtime artifacts to
`$HOME/scratch/synpath`:

```bash
SYNPATH_DATA_ROOT=$HOME/scratch/synpath/data
SYNPATH_CACHE_ROOT=$HOME/scratch/synpath/cache
SYNPATH_OUTPUTS_ROOT=$HOME/scratch/synpath/outputs
SYNPATH_REPORTS_ROOT=$HOME/scratch/synpath/reports
SYNPATH_EXTERNAL_ROOT=$HOME/scratch/synpath/external
TORCH_EXTENSIONS_DIR=$HOME/scratch/synpath/torch_extensions
HF_HOME=$HOME/scratch/synpath/huggingface
```

Override these in `.env` only if you need a different scratch or GPFS location,
for example:

```bash
SYNPATH_DATA_ROOT=/gpfs01/scratch/$USER/synpath/data
SYNPATH_CACHE_ROOT=/gpfs01/scratch/$USER/synpath/cache
SYNPATH_OUTPUTS_ROOT=/gpfs01/scratch/$USER/synpath/outputs
SYNPATH_REPORTS_ROOT=/gpfs01/scratch/$USER/synpath/reports
SYNPATH_EXTERNAL_ROOT=/gpfs01/scratch/$USER/synpath/external
HF_HOME=/gpfs01/scratch/$USER/synpath/huggingface
```

Then install the Python environment and external tools:

```bash
uv python pin 3.12
uv sync --extra gpu
bash scripts/setup/install_stylegan2ada.sh
```

`install_stylegan2ada.sh` creates a separate StyleGAN environment at
`$SYNPATH_EXTERNAL_ROOT/stylegan2ada-env` by default and installs both the
legacy `stylegan2-ada-pytorch` repo and NVIDIA's newer `stylegan3` repo. The
Slurm templates set `SYNPATH_STYLEGAN_BACKEND=stylegan3`, so the experiment
trains `--cfg stylegan2` through the newer StyleGAN3 code path while preserving
the experiment label `stylegan2ada`. The rest of SynPath continues to use the
main `uv` environment.

The Hypatia templates also load `nvidia-cuda/12.1.1` by default and set
`TORCH_EXTENSIONS_DIR` under scratch. These are needed for StyleGAN's custom
CUDA extensions (`bias_act_plugin` and `upfirdn2d_plugin`) to compile and be
reused across jobs. Override `SYNPATH_CUDA_MODULE` in `.env` if Hypatia changes
the preferred CUDA module.

If `uv` is not available on the cluster, install it in your user environment or
replace the Slurm template setup block with the local environment manager used
on Hypatia.

## Smoke run

Use the smoke job to confirm paths, Python environment, CUDA visibility, and
basic CLI execution:

```bash
sbatch slurm/hypatia/smoke.sbatch
```

The smoke job defaults UNI embedding extraction to CPU with
`SYNPATH_EMBED_DEVICE=cpu`. This keeps the smoke run useful even if one GPU
node is in a bad ECC state. Full experiment jobs should normally use the
default `auto` embedding device.

## Main run

The main Slurm template runs the full configured experiment:

```bash
sbatch slurm/hypatia/main.sbatch
```

The default template requests the `gpu.L40S` partition, four GPUs, 32 CPU cores,
and the maximum GPU-partition runtime of four days. Adjust these lines before
submission if needed:

```bash
#SBATCH --partition=gpu.L40S
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
```

For a full 8-GPU L40S node, use:

```bash
#SBATCH --partition=gpu.L40S
#SBATCH --gres=gpu:8
```

For A100 nodes, use at most four GPUs:

```bash
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:4
```

## Single dataset run

Single-dataset runs are useful for debugging or resuming one dataset:

```bash
sbatch --export=ALL,SYNPATH_DATASET=pathmnist224 \
  slurm/hypatia/dataset.sbatch
```

Valid dataset names are the names in `configs/datasets/`.

## Monitoring

```bash
squeue -u $USER
tail -f logs/slurm/<job-name>_<job-id>.log
```

Cancel a job:

```bash
scancel <job-id>
```

## Notes

- Do not run generator training on the login node.
- Keep downloaded datasets, Hugging Face cache, generated images, and
  checkpoints outside the git checkout.
- The Slurm scripts run through `uv run synpath ...`; all heavy work should be
  triggered through the same CLI used locally.
- The full experiment may exceed a single four-day GPU allocation. If that
  happens, split by dataset or model with `dataset.sbatch` and future
  per-arm array jobs.
