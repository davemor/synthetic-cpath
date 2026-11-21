# Literature notes

## UNI embeddings

UNI is a general-purpose self-supervised pathology foundation model trained on
large-scale H&E patch data and evaluated across computational pathology tasks.
It is a defensible embedding space for morphology-aware clustering and
feature-space quality metrics.

Reference:

- Chen et al., "Towards a general-purpose foundation model for computational
  pathology", Nature Medicine, 2024.

## KMeans pseudo-labels

KMeans over visual embeddings is a standard way to create pseudo-labels in
unsupervised representation learning. DeepCluster is the canonical precedent:
features are clustered and the cluster assignments are used as pseudo-labels.

Reference:

- Caron et al., "Deep Clustering for Unsupervised Learning of Visual Features",
  ECCV, 2018.

## CLAM distinction

CLAM supports the broader computational pathology idea that instance-level
pseudo supervision over patch features can improve learning, but it is not an
offline KMeans method.

CLAM uses attention-based multiple instance learning with slide-level labels.
High-attention and low-attention patches are used to construct an auxiliary
instance-level clustering/classification constraint. That differs from
`synpath`, where KMeans pseudo-labels are produced without annotations.

Reference:

- Lu et al., "Data Efficient and Weakly Supervised Computational Pathology on
  Whole Slide Images", Nature Biomedical Engineering, 2021.
