# synpath design

`synpath` is an end-to-end experiment runner for label-conditioned,
unconditioned, and pseudo-label-conditioned synthetic histopathology generation.

The implementation should stay thin around external model tooling. The project
owns orchestration, manifests, cached features, metrics, reports, and audit
metadata.

## Core concepts

### Dataset

A dataset is represented by a manifest, not by bespoke loader code. All stages
consume the same schema:

```csv
image_path,label,split,dataset,slide_id,patient_id
/data/path/img.png,tumor,train,pathmnist224,,
```

`slide_id` and `patient_id` may be empty for public patch datasets. iCAIRD will
provide these fields later through CSV metadata.

### Conditioning mode

The three supported modes are:

- `true_label`
- `unconditioned`
- `pseudo_label`

Each mode is compiled into a condition manifest:

```csv
image_path,condition,condition_id,prompt
/data/path/img.png,class_0,0,a histopathology patch of class_0
```

StyleGAN2-ADA consumes integer condition IDs. Stable Diffusion LoRA consumes
prompts.

### Experiment arm

An experiment arm is uniquely identified by:

```text
dataset / generator / conditioning / seed
```

Example:

```text
pathmnist224 / stylegan2ada / pseudo_label / seed0
```

## Pipeline

1. Prepare dataset manifests.
2. Cache UNI embeddings for train, validation, and test splits.
3. Select pseudo-labels using KMeans over training split UNI embeddings.
4. Train or fine-tune generators at 256x256.
5. Generate PNG samples at 256x256 and resize/evaluate at 224x224.
6. Run quality metrics for all generated arms.
7. Run ResNet-18 utility experiments for true-label arms only.
8. Aggregate results into CSV files, figures, and `report.md`.

## External models

StyleGAN2-ADA is installed under the external root and called by wrapper code.
Stable Diffusion 1.5 LoRA is run through Hugging Face `diffusers`,
`accelerate`, and `peft`.

The base Stable Diffusion model is:

```text
runwayml/stable-diffusion-v1-5
```

## Artifact layout

Generator outputs:

```text
{SYNPATH_OUTPUTS_ROOT}/generators/{dataset}/{model}/{conditioning}/seed{seed}/
    resolved_config.yaml
    train_manifest.csv
    condition_manifest.csv
    synthetic_manifest.csv
    checkpoints/
    samples/
    metrics/
    logs/
```

Utility outputs:

```text
{SYNPATH_OUTPUTS_ROOT}/utility/{dataset}/resnet18/{source}/seed{seed}/
    resolved_config.yaml
    metrics.json
    classification_report.json
    confusion_matrix.csv
    predictions.csv
```

Cached embeddings:

```text
{SYNPATH_CACHE_ROOT}/embeddings/{dataset}/{split}_uni_224.parquet
```

Cached clusters:

```text
{SYNPATH_CACHE_ROOT}/clusters/{dataset}/
    kmeans_k4_seed0.csv
    kmeans_k8_seed0.csv
    selected_k.yaml
    pseudo_labels.csv
    label_local_cluster_labels.csv
```

## Review checkpoints

Before implementing full training code, review:

- dataset manifest schemas
- path and cache conventions
- conditioning manifest schema
- selected metrics and capped sampling policy
- external tool installation strategy
- expected DGX run commands
