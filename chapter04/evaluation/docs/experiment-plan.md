# Experiment plan

## Claim

Pseudo-label conditioning is a practical substitute for annotation when the
goal is improving synthetic histopathology fidelity and diversity without
human labels.

The study does not claim pseudo-label conditioning improves supervised
pathology classification utility, because pseudo-label generated samples do not
have pathology labels.

## Datasets

Datasets are evaluated independently because each has its own label space.

- PathMNIST+ 224 / Kather100K
- SPIDER skin
- SPIDER colorectal
- SPIDER thorax
- SPIDER breast
- iCAIRD 100K, added later

PathMNIST+ uses official train/validation/test splits. SPIDER uses the central
224x224 patch and receives a stratified validation split from the official
training set.

## Preprocessing

- No stain normalization.
- Train generators at 256x256.
- Evaluate generated patches at 224x224.
- Save generated images as PNG.

## Pseudo-labels

- Extract UNI embeddings for the real training split.
- Cluster with KMeans.
- Candidate `k` values: `[4, 8, 16, 32, 64]`.
- Select one `k` per dataset using silhouette score, Davies-Bouldin score, and
  clustering stability.
- Use the selected pseudo-labels for both StyleGAN2-ADA and Stable Diffusion
  LoRA.

## Generators

For each dataset:

- StyleGAN2-ADA, true-label conditional
- StyleGAN2-ADA, unconditioned
- StyleGAN2-ADA, pseudo-label conditional
- StyleGAN2-ADA, true-label plus global cluster conditional
- StyleGAN2-ADA, true-label plus within-class cluster conditional
- Stable Diffusion 1.5 LoRA, true-label prompted
- Stable Diffusion 1.5 LoRA, unconditioned prompted
- Stable Diffusion 1.5 LoRA, pseudo-label prompted
- Stable Diffusion 1.5 LoRA, true-label plus global cluster prompted
- Stable Diffusion 1.5 LoRA, true-label plus within-class cluster prompted

Prompt templates:

```text
true_label: a histopathology patch of {class_name}
unconditioned: a histopathology patch
pseudo_label: a histopathology patch from morphology cluster {cluster_id}
label_global_cluster: a histopathology patch of {class_name}, morphology cluster {cluster_id}
label_local_cluster: a histopathology patch of {class_name}, morphology cluster {cluster_id}
```

Generate one synthetic training set per arm at `1x` the real training-set size.

## Metrics

All generated arms:

- Inception-FID
- UNI-FID or UNI-KID
- precision/recall in UNI feature space
- UNI cluster coverage
- nearest-neighbor privacy audit in UNI space, with SSIM/LPIPS on suspicious
  nearest pairs

Metric computation uses capped, reproducible sampling:

- maximum real samples: 10,000
- maximum synthetic samples: 10,000
- stratify true-label arms by pathology label
- stratify pseudo-label arms by pseudo-label
- randomly sample unconditioned arms

## Utility

Utility is evaluated only for generated data that retain pathology class
labels. The `label_global_cluster` and `label_local_cluster` modes are eligible
because each generated sample retains its pathology class label.

Classifier:

- ImageNet-pretrained ResNet-18
- 3 classifier seeds
- evaluation on real test set only

Training sources:

- real train only
- true-label synthetic only
- real plus true-label synthetic
- label-plus-global-cluster synthetic only
- real plus label-plus-global-cluster synthetic
- label-plus-local-cluster synthetic only
- real plus label-plus-local-cluster synthetic

Metrics:

- accuracy
- macro F1
- weighted F1
- macro one-vs-rest ROC-AUC
- classification report
- confusion matrix

## Compute budget

Default budget:

- 1 generative seed per dataset/model/conditioning arm
- 3 classifier seeds
- 3 clustering stability seeds
- capped metric samples
- smoke mode for tiny end-to-end validation

Full repeats of generator training should be added only after the first full
study identifies the most important arms.
