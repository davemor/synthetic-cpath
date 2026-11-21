# Cluster-conditioned Patch Synthesis

Generation/evaluation/reporting stages selected and preserved separately.

## Selected components

- `generation`
- `evaluation`
- `reporting`

## Original execution/source locations

- `dgx:/raid/dm236/synpath/repo`
- `dgx:/home/dm236/synpath`
- `local:scripts`

## Evidence

- Canonical quality, utility and generated-condition-count CSVs match DGX final_original_experiment by SHA-256.
- The final quality exporter reads per-arm inception_fid.json, uni_precision_recall.json and uni_authenticity.json; the utility exporter reads outputs/utility.
- Regenerated full quality and RQ2 tables match all 45 and 27 numeric rows in the thesis.
- Docker launch scripts explicitly mount the RAID repo for corrected iCAIRD sampling; authenticity/export code is in the home synpath checkout.

## Boundaries and findings

- The current live tree is missing 12 of 45 main authenticity JSONs and the original iCAIRD utility metric JSONs. The final aggregate exports remain available.
- The original live-tree export scripts therefore do not currently rerun end-to-end without restoring those records; this failure is retained in evidence/chapter4-export-checks.json.
- No exact code snapshot/Git commit was recorded for every training run. Current recovered stage trees are pinned by per-file hashes, not asserted to be byte-identical to every historical runtime.
- Keep code identifier pathmnist224 for the dataset labelled Kather in the chapter figures.

Source hashes and original copy locations are recorded in `../SOURCE_MANIFEST.json`. The full tracing report is `../PROVENANCE.md`.

Historical Docker source recovery and live metric drift are documented in ../PROVENANCE.md and ../evidence/docker-source-*.json.
