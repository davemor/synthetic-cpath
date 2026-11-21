# SD-Index Cluster Calibration

This calibration selects a global K-means cluster count independently for
Kather 100K, iCAIRD Cervical 100K, and PCam in two feature spaces:

- DINO ViT-B/8 class-token features, matching Osorio et al.
- UNI2-h features as a histopathology-specific sensitivity analysis

Both tracks use raw model features without PCA or class labels. K-means is
fitted for every `k` from 2 through 50 using K-means++, 10 initializations,
300 maximum iterations, and seed 0. The selected value minimizes

```text
SD(k) = Dis(50) * Scat(k) + Dis(k)
```

The weighting term `Dis(50)` follows Halkidi et al. rather than treating alpha
as one. Fitting and scoring use every patch in each complete training split
before the configured experiment subset is selected. Validation and test
patches are excluded.

The PCam calibration population contains all 262,144 patches from the official
HDF5 training split. Osorio et al. used 216,868 foreground-filtered patches
from the Kaggle release, so the DINO track is literature-aligned but not an
exact data reproduction.

## DGX-1

Run the complete resumable calibration from the repository root:

```bash
make dgx-calibrate-sd-index
```

Individual stages can also be run with:

```bash
bash scripts/docker/run_sd_index_calibration_dgx.sh prepare
bash scripts/docker/run_sd_index_calibration_dgx.sh embed
bash scripts/docker/run_sd_index_calibration_dgx.sh fit
bash scripts/docker/run_sd_index_calibration_dgx.sh report
```

Embedding uses the SynPath PyTorch image. K-means uses the RAPIDS image set by
`SYNPATH_RAPIDS_IMAGE`, with eight pinned workers. Completed feature and `k`
artifacts are validated and skipped when the run resumes.

Logs are written under:

```text
/raid/dm236/synpath/logs/cluster_sd_index_calibration
```

The final report is written under:

```text
/raid/dm236/synpath/reports/cluster_sd_index_calibration
```

The calibration cache is isolated under
`cache/cluster_sd_index_calibration_full_pool`. It does not modify the production
K-means models, assignments, or experiment configurations.

## References

- Osorio et al., *Latent Diffusion Models with Image-Derived Annotations for
  Enhanced AI-Assisted Cancer Diagnosis in Histopathology*, Diagnostics 2024,
  doi:10.3390/diagnostics14131442.
- Halkidi, Vazirgiannis, and Batistakis, *Quality Scheme Assessment in the
  Clustering Process*, PKDD 2000, doi:10.1007/3-540-45372-5_26.
