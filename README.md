# synthetic-cpath

Source code for David Morrison's PhD thesis, **Synthetic Data Generation in Computational Pathology**, University of St Andrews, August 2026.

Each chapter has its own directory. These are the most definitive recovered source sets identified by tracing the thesis tables and figures back to saved results, configurations and execution scripts. Original generation, later evaluation and reporting stages are kept explicit where they used different code.

## Chapters

| Chapter | Code and documentation | Contents |
|---|---|---|
| 1 — Introduction | [chapter01](chapter01/) | Context; no independent experiment. |
| 2 — Background | [chapter02](chapter02/) | Illustration and explanatory plotting scripts. |
| 3 — Dataset creation and preprocessing | [chapter03](chapter03/) | Histokit, Patcher, patch/MIL benchmarks and result reporting. |
| 4 — Cluster-conditioned Patch Synthesis | [chapter04](chapter04/) | SynPath generation, evaluation, authenticity and quality/utility reporting. |
| 5 — Dimensionality reduction in patch embeddings | [chapter05](chapter05/) | The Hypatia hproj source, saved run configurations and thesis reporting scripts. |
| 6 — Clustering low-dimensional representations | [chapter06](chapter06/) | Clustering, cluster selection, low-dimensional conditioning configurations and reporting. |
| 7 — Conclusion | [chapter07](chapter07/) | Summary; no independent experiment. |

## Appendices

| Appendix | Code and documentation |
|---|---|
| A — Patch Synthesis Results | [appendixA](appendixA/), using [Chapter 4](chapter04/). |
| B — Full Clustering Results | [appendixB](appendixB/), using [Chapter 6](chapter06/). |
| C — Anonymising Pathology Data using Generative Adversarial Networks | [appendixC](appendixC/): the recovered pathgen paper notebooks and package. |
| D — Ethics | [appendixD](appendixD/): no experimental code or private ethics documents. |
| E — WSI Pipe | [appendixE](appendixE/): wsipipe and its iSyntax reader extension. |

## Reproduction and provenance

- [Reproduction instructions](REPRODUCTION.md) describe entry points and required inputs.
- [Provenance report](PROVENANCE.md) records source choices, result matches and remaining historical-version gaps.
- [Project selections](PROJECT_SELECTION.json) and the [source manifest](SOURCE_MANIFEST.json) record the recovered locations and file hashes. Run `python verify_manifest.py` to check the selected files.
- [Validation evidence](evidence/) records report-level reproduction checks. Full GPU training has not been rerun for this recovery.
- [Third-party dependency revisions and changes](dependencies/patches/INDEX.json) preserve upstream references; associated licence files are retained under [dependencies](dependencies/).

Chapter 4 has incomplete historical run metadata and some live metrics have changed since the thesis exports. Chapter 5 includes preserved working-tree changes rather than only an old clean checkout. These limits are documented rather than treated as proof of an exact historical commit for every run.

## Data and credentials

Private slides, annotations, patient-level records, model weights, environments and backup Git histories are excluded. Small aggregate reporting inputs are included. Credentials must be supplied privately. Study-specific slide identifiers in public examples have been replaced, and the pilot preparation helper accepts its five IDs through `--pilot-ids`; see [publication notes](PUBLICATION.md).

## Licences

Existing component licences and copyright notices are retained. This collection does not apply a new blanket licence to all components. Consult the relevant component and upstream licence before reuse.
