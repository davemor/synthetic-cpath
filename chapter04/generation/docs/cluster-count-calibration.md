# Training-Set Cluster-Count Calibration

This isolated calibration evaluates global K-means partitions for every `k`
from 2 through 50 in DINO ViT-B/8 and UNI2-h feature spaces. Only training
patches are used for feature standardisation, fitting, scoring, and selection.
Validation and test patches are excluded.

The populations are:

- Kather 100K: 89,996 training patches.
- iCAIRD Cervical: the 100,000-patch training split; its 20,000 validation
  patches are excluded.
- Official PCam: 262,144 training patches.
- Osorio PCam comparison: the Kaggle training population after the released
  foreground curation, currently 216,886 patches.

The primary `k` is the offline Kneedle elbow in the inertia curve. The maximum
mean silhouette and minimum Halkidi SD-index are reported independently. They
are not combined into a composite score. Silhouette is reported as mean and
standard deviation over three seeded uniform samples of 10,000 training
patches. The Halkidi score uses `Dis(50)` as alpha.

K-means, inertia, silhouette, and the Halkidi score use globally standardised
training embeddings. A separately labelled Osorio audit uses `s-dbw==0.4.0`
on raw DINO embeddings. The existing released-preprocessing/scikit-learn audit
is imported into the report rather than silently treated as the same feature
geometry.

Run the resumable DGX workflow with:

```bash
make dgx-calibrate-cluster-count
```

Logs are written to:

```text
/raid/dm236/synpath/logs/cluster_k_calibration_2_50/
```

Reports are written to:

```text
/raid/dm236/synpath/reports/cluster_k_calibration_2_50/
```

The cache and selected assignments are isolated under
`cache/cluster_k_calibration_2_50`. Production pseudo-labels are not modified.

## References

- Thorndike, *Who Belongs in the Family?*, Psychometrika, 1953,
  doi:10.1007/BF02289263.
- Satopaa et al., *Finding a Kneedle in a Haystack*, ICDCSW, 2011,
  doi:10.1109/ICDCSW.2011.20.
- Rousseeuw, *Silhouettes: A Graphical Aid to the Interpretation and
  Validation of Cluster Analysis*, JCAM, 1987,
  doi:10.1016/0377-0427(87)90125-7.
- Halkidi, Vazirgiannis, and Batistakis, *Quality Scheme Assessment in the
  Clustering Process*, PKDD, 2000, doi:10.1007/3-540-45372-5_26.
- Osorio et al., *Latent Diffusion Models with Image-Derived Annotations for
  Enhanced AI-Assisted Cancer Diagnosis in Histopathology*, Diagnostics,
  2024, doi:10.3390/diagnostics14131442.
