# Dimensionality reduction in patch embeddings

Hypatia hproj selected; reporting numerically reproduced.

## Selected components

- `hproj`
- `reporting`
- `run-configs`

## Original execution/source locations

- `hypatia:/home/dm236/repos/hproj`
- `local:AllResults/Chapter5-dimensionality-reduction/results_with_icaird/hproj-2026-05-25_16-41-27-bb2a5203-icaird100k/scripts`

## Evidence

- Thesis LR accuracy/ROC-AUC plots match the local report outputs by SHA-256.
- The source result family is hproj-2026-05-25_16-41-27-bb2a5203, with an icaird100k extension.
- Hypatia result files and configurations match the archived run family; completed job logs name gpu04/gpu05.hpc.st-andrews.ac.uk and the same run manifests.
- Regenerated threshold table has identical table rows. Regenerated best-generalisation and robustness tables match all 24 dataset rows each numerically; the thesis applies presentation edits.

## Boundaries and findings

- The current server icaird100k generalisation CSV has additional/changed run contents versus the archived thesis reporting input; use the archived input identified here for report reproduction.
- bb2a5203 is a configuration hash, not a Git commit. The pinned base repository commit is 2759db99c plus the preserved Hypatia working-tree modifications.
- Completed generalisation jobs ran on Hypatia; the blanket DGX hardware statement in the thesis needs reconciliation.

Source hashes and original copy locations are recorded in `../SOURCE_MANIFEST.json`. The full tracing report is `../PROVENANCE.md`.
