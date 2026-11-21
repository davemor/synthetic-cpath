# Publication notes

This public source snapshot was prepared on 21 November 2026 from the result-traced chapter codebases. The private preservation archives and original Git histories were not included.

The recovered files had already had known credential literals, saved notebook outputs and embedded image payloads removed. This publication pass also removes study-specific slide identifiers from examples and tests. The private original files remain unchanged.

`chapter03/patcher/scripts/prepare_hypatia_icaird_dataset.py` now requires five explicit `--pilot-ids` instead of embedding the original pilot selection. Supply the IDs from your private study configuration to reproduce that pilot. Dataset-dependent tests derive their expected IDs from the private fixture index rather than recording them in this repository. Standalone filename tests use synthetic example names.

These changes affect configuration/examples and publication packaging. The scientific training and evaluation implementations and recorded aggregate results are preserved. [PUBLICATION_CHANGES.json](PUBLICATION_CHANGES.json) lists the affected files. The source manifest retains the recovered hash for deliberately changed files and verifies the published contents.

The provenance report records limitations in the historical run evidence. Report-level checks do not claim a new end-to-end training reproduction.
