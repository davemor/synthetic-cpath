# iCAIRD Cervical Benchmark

Standalone CLI tooling for benchmarking classifiers on an ImageFolder-style iCAIRD Cervical 100K dataset.

Expected image dataset layout:

```text
dataset_root/
  train/<class_name>/*.png
  valid/<class_name>/*.png
  test/<class_name>/*.png
```

Run probes over embeddings:

```bash
icaird-cervical-benchmark train-probes --config examples/config.yml
```

Run image classifiers:

```bash
icaird-cervical-benchmark train-images --config examples/config.yml
```

Run the ResNet-18 image-augmentation experiment:

```bash
icaird-cervical-benchmark make-task-manifest \
  --config configs/resnet18_augmentation_icaird.yml \
  --include images \
  --output /home/dm236/scratch/icaird_cervical_benchmark/manifests/resnet18_augmentation_icaird_tasks.jsonl
CONFIG=/home/dm236/repos/icaird-cervical-benchmark/configs/resnet18_augmentation_icaird.yml \
MANIFEST=/home/dm236/scratch/icaird_cervical_benchmark/manifests/resnet18_augmentation_icaird_tasks.jsonl \
RUN_NAME=resnet18_augmentation_icaird_$(date +%Y%m%d) \
sbatch slurm/resnet18_augmentation_array.sbatch
```

For Kather100K and PCam, build dataset-specific manifests and submit the same array script with the
matching config:

```bash
icaird-cervical-benchmark make-task-manifest \
  --config configs/resnet18_augmentation_kather100k.yml \
  --include images \
  --output /home/dm236/scratch/icaird_cervical_benchmark/manifests/resnet18_augmentation_kather100k_tasks.jsonl

CONFIG=/home/dm236/repos/icaird-cervical-benchmark/configs/resnet18_augmentation_kather100k.yml \
MANIFEST=/home/dm236/scratch/icaird_cervical_benchmark/manifests/resnet18_augmentation_kather100k_tasks.jsonl \
RUN_NAME=resnet18_augmentation_kather100k_$(date +%Y%m%d) \
sbatch slurm/resnet18_augmentation_array.sbatch

icaird-cervical-benchmark make-task-manifest \
  --config configs/resnet18_augmentation_pcam.yml \
  --include images \
  --output /home/dm236/scratch/icaird_cervical_benchmark/manifests/resnet18_augmentation_pcam_tasks.jsonl

CONFIG=/home/dm236/repos/icaird-cervical-benchmark/configs/resnet18_augmentation_pcam.yml \
MANIFEST=/home/dm236/scratch/icaird_cervical_benchmark/manifests/resnet18_augmentation_pcam_tasks.jsonl \
RUN_NAME=resnet18_augmentation_pcam_$(date +%Y%m%d) \
sbatch slurm/resnet18_augmentation_array.sbatch
```

Summarise completed augmentation runs:

```bash
python scripts/summarise_resnet18_augmentation_study.py \
  --run icaird=/home/dm236/scratch/icaird_cervical_benchmark/runs/resnet18_augmentation_icaird_YYYYMMDD \
  --run kather100k=/home/dm236/scratch/icaird_cervical_benchmark/runs/resnet18_augmentation_kather100k_YYYYMMDD \
  --run pcam=/home/dm236/scratch/icaird_cervical_benchmark/runs/resnet18_augmentation_pcam_YYYYMMDD \
  --output-dir /home/dm236/scratch/icaird_cervical_benchmark/thesis_results/resnet18_augmentation_YYYYMMDD
```

Run both and write a combined summary:

```bash
icaird-cervical-benchmark run --config examples/config.yml
```

Run transformer MIL over iCAIRD UNI2-h slide bags:

```bash
icaird-cervical-benchmark train-mil --config configs/icaird_cervical_mil_uni2h_bags.yml
```

Run the initial ViT-DAE-like synthetic UNI2-h embedding experiment on wopr:

```bash
icaird-cervical-benchmark synthetic-prepare --config configs/icaird_synthetic_uni2h_vitdae_like.yml
icaird-cervical-benchmark synthetic-baselines --config configs/icaird_synthetic_uni2h_vitdae_like.yml
icaird-cervical-benchmark synthetic-train --config configs/icaird_synthetic_uni2h_vitdae_like.yml
icaird-cervical-benchmark synthetic-sample --config configs/icaird_synthetic_uni2h_vitdae_like.yml
icaird-cervical-benchmark synthetic-evaluate \
  --config configs/icaird_synthetic_uni2h_vitdae_like.yml \
  --synthetic-dir /nvme/synthetic_embeddings/icaird100k_uni2h_vitdae_like/samples/ddim_n100000
```

Or submit staged Slurm jobs with:

```bash
STAGE=train sbatch slurm/icaird_synthetic_embeddings_uni2h.sbatch
```

For the Kather100K UNI2-h baseline, use:

```bash
icaird-cervical-benchmark synthetic-prepare --config configs/kather100k_synthetic_uni2h_vitdae_independent_class.yml
icaird-cervical-benchmark synthetic-baselines --config configs/kather100k_synthetic_uni2h_vitdae_independent_class.yml
icaird-cervical-benchmark synthetic-train --config configs/kather100k_synthetic_uni2h_vitdae_independent_class.yml
icaird-cervical-benchmark synthetic-sample --config configs/kather100k_synthetic_uni2h_vitdae_independent_class.yml
```

Kather chunked embedding caches with train/test splits are supported directly. When a cache has no
explicit `valid` directory, `synthetic-prepare` creates a deterministic stratified validation holdout
from the training split using `synthetic_embeddings.validation_fraction`.

Each run records metrics, predictions, confusion matrices, classification reports, training/search dynamics, citations, and timing files.
