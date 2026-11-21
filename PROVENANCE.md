# Result-to-source provenance

## Scope and numbering

The submitted PDF was checked directly. Chapters 5 and 6 are separate chapters even though both are included through `chap5/doc.tex`: `chap5/2-clustering.tex` contains its own chapter command. The GAN and WSI Pipe appendices are C and E. `chap6/doc.tex` is an unused low-dimensional synthesis draft, not the submitted clustering chapter.

## Chapter 3: authoritative preprocessing and benchmark sources

Select Hypatia `/home/dm236/repos/histokit`, `/home/dm236/repos/patcher` and `/home/dm236/repos/icaird-cervical-benchmark`, with the local reporting scripts. The Mac and Hypatia histokit copies have identical Python/shell/configuration source in the preserved snapshots; the server copy establishes the execution context.

Table 3.9 traces to `/sharedscratch/dm236/data/icaird_cervical/fgbg_method_comparison/methods_summary/methods_performance_summary.csv`. Every one of its five rows matches all five reported performance values. Its generating chain is `summarize_fgbg_method_results.py` followed by `summarize_background_segmentation_methods.py`, using the Canny/GrandQC/slide detector tuning and `train_resnet19_patch_classifier.py` implementations.

Tables 3.13 and 3.14 trace to `/sharedscratch/dm236/icaird_cervical_benchmark/thesis_results/bootstrap_1000_20260702/icaird_cervical_thesis_results_bootstrap.csv`. All 20 rows match the reported accuracy, ROC-AUC and confidence intervals. The export records the original prediction paths under runs `icaird120k_annotation_labels_seed0_20260615` and `icaird120k_annotation_label_images_seed0_20260615`; the model code is `benchmark/src/icaird_cervical_benchmark/{probes,images}.py` and the table generator is `reporting/make_icaird_thesis_results_table.py`.

Table 3.15 traces to run `icaird_mil_uni2h_bags_gpu_20260616_1335`, the benchmark's `mil.py`, and `make_mil_bootstrap_table.py`. Its saved local table numerically matches the thesis and records 10,000 bootstraps.

**Discrepancy:** the patch/embedding export records 1,000 resamples, although the thesis says 10,000. This is not a blanket discrepancy for MIL, whose saved export records 10,000.

## Chapter 4: one chapter codebase with explicit execution stages

Use `generation` from DGX `/raid/dm236/synpath/repo`, `evaluation` from DGX `/home/dm236/synpath`, and the selected local `reporting` scripts. A single old checkout does not contain the full final pipeline. The corrected iCAIRD Docker scripts explicitly mount `/raid/dm236/synpath/repo`; authenticity backfill and final export scripts are present in the home checkout. Keep separate package environments for these two synpath stages, because both expose the same Python package name.

The canonical quality, utility and generation-count exports under `/raid/dm236/synpath/reports/final_original_experiment` match the local final-results exports by SHA-256. `export_final_quality_table.py` enumerates generator FID, precision/recall and authenticity JSONs. `export_final_utility_table.py` aggregates per-seed ResNet-18 utility metrics. This identifies `synpath/utility/resnet.py` as the downstream utility implementation; the separate benchmark augmentation follow-up is not the source of these canonical exports.

The normalized-quality metadata records input CSV hashes, real/synthetic embedding and manifest hashes, sample-selection seed and original generator output paths. `build_rq1_normalized_fid_report.py` and `build_full_quality_nufd_table.py` perform the normalization/reporting stages.

The selected renderer reproduces all 45 numeric rows of Appendix A's quality table and all 27 numeric rows of Table 4.10. Multiple chapter figures also match the saved local outputs byte-for-byte. The quality/utility figure scripts use the preserved aggregate input CSVs under `final-results` and `utility-report-sources`.

**Live-data gap:** the current DGX tree contains only 33 of the 45 expected main authenticity JSONs and no original iCAIRD main utility JSONs at the paths used by the exporter. An attempted fresh export fails; its error is retained in `evidence/chapter4-export-checks.json`. The final exports survive, so the thesis report layer can be reproduced from them. A direct measurement comparison also found live-data drift: only 33/45 current FID values, 33/45 precision values and 41/45 recall values match the final export; the saved authenticity backfill export matches all 45 authenticity values. Preserve the final exports as the thesis result version rather than replacing them with the current live metrics. Do not claim that the entire original export chain currently reruns from the live tree.

The code identifier `pathmnist224` is labelled Kather in these figures. Preserve that identifier and the recorded manifests rather than substituting the separate `kather100k` dataset used in Chapter 5.


Historical Docker source was also recovered, without enabling networking or modifying experiment files. `synpath:latest` is pinned to image `sha256:59f33c100f182feb2839c45c06246e61edbd9f0f340bb55710f36e6ee65f1de6` (4 August 2026), and `synpath-authenticity:latest` to `sha256:d3585902cc71d6718340cdad3faa98a322c27de6eb1a76bbfa823d679cf49bfc` (9 August 2026). All selected authenticity metric/backfill Python implementations match the latter image byte-for-byte; 32 Python files match overall. The earlier generation image has 11 Python differences from the current RAID tree, including generator wrappers and utility code. Its original source archive is retained privately rather than silently substituted for the final corrected-generation stage. See the two Docker evidence JSON files.

## Chapter 5: select Hypatia hproj and the archived run's reporting scripts

The authoritative run family is `hproj-2026-05-25_16-41-27-bb2a5203`, plus `-icaird100k`. The local source for final reporting is the archived `AllResults/Chapter5-dimensionality-reduction/results_with_icaird/...-icaird100k/scripts` directory. The selected experiment source is Hypatia `/home/dm236/repos/hproj`, whose base commit is `2759db99c` plus preserved working-tree changes. The clean Mac/DGX copies omit material worker changes; they are not substituted for the executed Hypatia version.

The thesis LR accuracy/ROC-AUC plots exactly match the local report outputs. Original Hypatia run configurations, intrinsic-dimension files, curves and multiple reports match archived files by SHA-256. Completed generalisation job logs record `gpu04.hpc.st-andrews.ac.uk` and `gpu05.hpc.st-andrews.ac.uk`, the matching run ID and task manifest. The launch scripts change directory to `/home/dm236/repos/hproj`.

`make_threshold_tables.py` regenerates Table 5.9 with identical table rows. `summarise_generalisation_for_thesis.py` supplies the threshold-validation summaries; `make_generalisation_summary_package.py` regenerates all 24 numerical rows of Table 5.10 and all 24 rows of Table 5.11. The thesis changes grouping, captions and highlighting, not the tested numerical values.

Use the archived thesis reporting inputs rather than the current server `-icaird100k/generalisation_results.csv`, which has additional/changed contents. `bb2a5203` is a configuration hash, not a Git revision. The separate DGX dimensionality follow-up belongs to the later low-dimensional synthesis work and is not selected as Chapter 5's primary experiment.

**Hardware discrepancy:** completed generalisation jobs establish Hypatia execution, while the chapter's blanket hardware statement names DGX. The thesis statement needs reconciliation; it was not edited here.

## Chapter 6: Hypatia computation, WOPR aggregation, local final reporting

The WOPR script `run_combined_summary_with_spherical.sh` explicitly copies completed spherical tasks from Hypatia and runs `cluster_exp.cli summarise` using `/home/dm236/repos/histology-clustering-experiment` with 32 workers. Fifty-one WOPR summary/trend/table files match the archived Chapter 6 outputs. The selected local `clustering_experiment` computational modules match the WOPR copy; its extra CLI/appendix module provides final thesis reporting.

Both full Appendix B dimension tables regenerate byte-for-byte using `rq1_appendix.py` and the saved `clustering_stability.csv`. `plots/plot_rq1_dimension_metric_families.py` regenerates the exact combined PCA/UMAP plotted-values CSV. PNG bytes differ in the present rendering environment, so the numeric CSV match is the reproduction claim.

The local `res/clustering_results.csv` is a 4,704-row subset containing only k-means and HDBSCAN. It is not the full four-method input. The matching stability file contains 75,144 rows and all four methods; it is the input that reproduces the final plots and appendix tables.

Section 6.6's generation configs were recovered from `/raid/dm236/synpath_balanced_global_clusters/configs`. They specify UNI2 embeddings, PCA, k-means with k=24, and 42 dimensions for iCAIRD versus 50 for `pathmnist224`. `fit_pathmnist_balanced_global_clusters.py` implements the aligned Kather feature/PCA/cluster export. Generation and evaluation reuse the Chapter 4 code with these configs.

`plot_class_global_vs_low_d_global_grid.py` reproduces every value in the 24-row quality-delta export and 72-row utility-delta export behind Figures 6.4 and 6.5. The saved per-seed input avoids dependence on the expired `/private/tmp/balanced_global_results` path in the original script. The server's misleadingly named `main_pathmnist224_balanced_global_cluster/utility_metrics.csv` contains both datasets; the similarly named iCAIRD report file is empty and must not be used instead.

Selection-related source was additionally recovered from `/raid/experiments/lowgen_v2` and the Hypatia calibrated-selection helper. **Unresolved description:** the saved selection metadata describes `balanced_compromise` with Pareto/utopia selection, whereas the thesis describes evenly weighting six measurements. The exact selection invocation/history was not recorded, so this remains an explicit provenance question rather than an invented definitive algorithm history.

## Historical and non-experimental chapters

Chapters 1 and 7 are narrative. Chapter 2 has illustration scripts rather than an independent training experiment. Appendix A uses Chapter 4 and Appendix B uses Chapter 6.

Appendix C's authoritative code is `davemor/pathgen` commit `7dc9feea213a3ec9884274132615b53d7054f75e`, especially the two paper GAN notebooks and three paper classifier notebooks. The saved classifier test accuracies round exactly to the reported 0.78, 0.75 and 0.59, and the GAN notebooks specify the reported 400 epochs/batch size 256. Original saved outputs remain in the private bare clone; publication copies have outputs removed. Precision/recall/F1 were not independently rerun.

Appendix D is ethics documentation. Appendix E uses the recovered DGX `wsipipe` and `wsipipe-isyntax` checkouts, pinned to their recorded commits. It documents a toolkit, not a separate numerical benchmark.

## Meaning of the selected codebases

These directories are the selected, frozen chapter working sets, with result-backed inclusion decisions and file hashes. They eliminate the earlier undifferentiated list of copies. They do not manufacture historical Git commits where working trees were unversioned. Full training, private-data access, every original run state and public-release checks remain distinct from the report-level reproductions completed here. No thesis source or server experiment was modified.
