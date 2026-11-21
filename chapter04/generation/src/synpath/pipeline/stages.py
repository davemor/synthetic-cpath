from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from synpath.artifacts import utc_now, write_json
from synpath.manifests import (
    CONDITION_COLUMNS,
    DATASET_COLUMNS,
    read_manifest,
    validate_columns,
    write_manifest,
)
from synpath.paths import SynpathPaths
from synpath.pipeline.jobs import Job, StageResult


class Stage(Protocol):
    job: Job

    def outputs(self) -> list[Path]: ...

    def run(self) -> StageResult: ...


@dataclass
class BaseStage:
    job: Job
    cfg: dict
    paths: SynpathPaths
    resume: bool = False

    def outputs(self) -> list[Path]:
        raise NotImplementedError

    def is_complete(self) -> bool:
        return all(path.exists() for path in self.outputs())

    def skip_or_run(self) -> StageResult | None:
        if self.resume and self.is_complete():
            return StageResult(self.job, self.outputs(), skipped=True)
        return None


class PrepareDataStage(BaseStage):
    def manifest_path(self) -> Path:
        assert self.job.dataset is not None
        return (
            self.paths.data_root / "manifests" / f"{self.job.dataset}_full.csv"
        )

    def outputs(self) -> list[Path]:
        return [self.manifest_path()]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped

        path = self.manifest_path()
        if path.exists():
            validate_columns(path, DATASET_COLUMNS)
            return StageResult(self.job, [path])
        from synpath.data.prepare import build_full_manifest

        return StageResult(
            self.job,
            [build_full_manifest(self.job.dataset, self.paths, self.cfg)],
        )


class EmbedStage(BaseStage):
    def outputs(self) -> list[Path]:
        assert self.job.dataset is not None
        return [
            self.paths.cache_root
            / "embeddings"
            / self.job.dataset
            / "cluster_pool_uni_224.parquet"
        ]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        assert self.job.dataset is not None
        from synpath.embeddings.uni import extract_uni_cluster_pool_embeddings

        output = extract_uni_cluster_pool_embeddings(
            self.job.dataset, self.paths
        )
        return StageResult(self.job, [output])


class ClusterStage(BaseStage):
    def outputs(self) -> list[Path]:
        assert self.job.dataset is not None
        root = self.paths.cache_root / "clusters" / self.job.dataset
        return [
            root / "selected_global_k.yaml",
            root / "pseudo_labels.csv",
            root / "label_local_cluster_labels.csv",
            root / "full_pool_complete.json",
        ]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        assert self.job.dataset is not None
        from synpath.clustering.kmeans import run_clustering

        outputs = run_clustering(self.job.dataset, self.paths, self.cfg)
        return StageResult(self.job, outputs)


class SelectSubsetStage(BaseStage):
    def manifest_path(self) -> Path:
        assert self.job.dataset is not None
        return self.paths.data_root / "manifests" / f"{self.job.dataset}.csv"

    def metadata_path(self) -> Path:
        assert self.job.dataset is not None
        return (
            self.paths.data_root
            / "manifests"
            / f"{self.job.dataset}_selection.json"
        )

    def embedding_path(self) -> Path:
        assert self.job.dataset is not None
        return (
            self.paths.cache_root
            / "embeddings"
            / self.job.dataset
            / "train_uni_224.parquet"
        )

    def outputs(self) -> list[Path]:
        return [
            self.manifest_path(),
            self.metadata_path(),
            self.embedding_path(),
        ]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        assert self.job.dataset is not None

        from synpath.data.prepare import apply_limits, write_selected_manifest
        from synpath.embeddings.uni import select_train_embeddings

        full_path = (
            self.paths.data_root / "manifests" / f"{self.job.dataset}_full.csv"
        )
        full_rows = read_manifest(full_path)
        selected_rows = apply_limits(full_rows, self.cfg)
        write_selected_manifest(
            self.job.dataset,
            full_rows,
            self.paths,
            self.cfg,
        )
        self._validate_cluster_assignments(selected_rows)
        select_train_embeddings(self.job.dataset, self.paths, selected_rows)
        write_json(
            self.metadata_path(),
            {
                "created_at": utc_now(),
                "dataset": self.job.dataset,
                "full_manifest": str(full_path),
                "selected_manifest": str(self.manifest_path()),
                "cluster_assignments": str(
                    self.paths.cache_root
                    / "clusters"
                    / self.job.dataset
                    / "pseudo_labels.csv"
                ),
                "limits": self.cfg.get("limits", {}),
                "full_counts": split_counts(full_rows),
                "selected_counts": split_counts(selected_rows),
                "clustering_population": "complete_train_split",
                "held_out_splits_excluded": True,
            },
        )
        return StageResult(self.job, self.outputs())

    def _validate_cluster_assignments(
        self,
        selected_rows: list[dict[str, object]],
    ) -> None:
        assert self.job.dataset is not None
        selected_paths = {
            str(row["image_path"])
            for row in selected_rows
            if row["split"] == "train"
        }
        root = self.paths.cache_root / "clusters" / self.job.dataset
        for name in ("pseudo_labels.csv", "label_local_cluster_labels.csv"):
            path = root / name
            assignments = {row["image_path"] for row in read_manifest(path)}
            missing = selected_paths - assignments
            if missing:
                raise RuntimeError(
                    f"{path} is missing {len(missing)} selected training images"
                )


def split_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "val", "test")
    }


class BuildConditionsStage(BaseStage):
    def condition_path(self) -> Path:
        assert self.job.dataset is not None
        assert self.job.conditioning is not None
        return (
            self.paths.outputs_root
            / "conditions"
            / self.job.dataset
            / self.job.conditioning
            / "condition_manifest.csv"
        )

    def outputs(self) -> list[Path]:
        return [self.condition_path()]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped

        assert self.job.dataset is not None
        assert self.job.conditioning is not None
        manifest_path = (
            self.paths.data_root / "manifests" / f"{self.job.dataset}.csv"
        )
        validate_columns(manifest_path, DATASET_COLUMNS)
        rows = read_manifest(manifest_path)
        train_rows = [row for row in rows if row.get("split") == "train"]

        if self.job.conditioning == "true_label":
            condition_rows = self._true_label_rows(train_rows)
        elif self.job.conditioning == "unconditioned":
            condition_rows = self._unconditioned_rows(train_rows)
        elif self.job.conditioning == "pseudo_label":
            condition_rows = self._cluster_rows(
                train_rows,
                "pseudo_labels.csv",
            )
        elif self.job.conditioning == "label_global_cluster":
            condition_rows = self._cluster_rows(
                train_rows,
                "pseudo_labels.csv",
                include_label=True,
            )
        elif self.job.conditioning == "label_local_cluster":
            condition_rows = self._cluster_rows(
                train_rows,
                "label_local_cluster_labels.csv",
                include_label=True,
            )
        else:
            raise ValueError(
                f"Unknown conditioning mode: {self.job.conditioning}"
            )

        write_manifest(
            self.condition_path(),
            condition_rows,
            CONDITION_COLUMNS,
        )
        return StageResult(self.job, self.outputs())

    def _true_label_rows(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        labels = sorted({row["label"] for row in rows})
        condition_ids = {label: idx for idx, label in enumerate(labels)}
        return [
            {
                "image_path": row["image_path"],
                "label": row["label"],
                "label_id": row["label_id"],
                "cluster_id": "",
                "condition": row["label"],
                "condition_id": condition_ids[row["label"]],
                "prompt": f"a histopathology patch of {row['label']}",
            }
            for row in rows
        ]

    def _unconditioned_rows(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        return [
            {
                "image_path": row["image_path"],
                "label": row["label"],
                "label_id": row["label_id"],
                "cluster_id": "",
                "condition": "histopathology",
                "condition_id": 0,
                "prompt": "a histopathology patch",
            }
            for row in rows
        ]

    def _cluster_rows(
        self,
        rows: list[dict[str, str]],
        cache_name: str,
        include_label: bool = False,
    ) -> list[dict[str, object]]:
        assert self.job.dataset is not None
        cluster_path = (
            self.paths.cache_root / "clusters" / self.job.dataset / cache_name
        )
        if not cluster_path.exists():
            raise RuntimeError(f"Missing cluster cache: {cluster_path}")

        cluster_rows = read_manifest(cluster_path)
        clusters = {
            row["image_path"]: row["cluster_id"]
            for row in cluster_rows
            if row.get("image_path") and row.get("cluster_id") is not None
        }
        missing = [
            row["image_path"]
            for row in rows
            if row["image_path"] not in clusters
        ]
        if missing:
            raise RuntimeError(
                f"{cluster_path} is missing {len(missing)} training images"
            )

        conditions = []
        for row in rows:
            cluster_id = clusters[row["image_path"]]
            if include_label:
                condition = f"{row['label']}__cluster_{cluster_id}"
            else:
                condition = f"cluster_{cluster_id}"
            conditions.append(condition)
        condition_ids = {
            condition: idx
            for idx, condition in enumerate(sorted(set(conditions)))
        }

        output = []
        for row, condition in zip(rows, conditions, strict=True):
            cluster_id = clusters[row["image_path"]]
            if include_label:
                prompt = (
                    f"a histopathology patch of {row['label']}, "
                    f"morphology cluster {cluster_id}"
                )
            else:
                prompt = (
                    "a histopathology patch from morphology cluster "
                    f"{cluster_id}"
                )
            output.append(
                {
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "label_id": row["label_id"],
                    "cluster_id": cluster_id,
                    "condition": condition,
                    "condition_id": condition_ids[condition],
                    "prompt": prompt,
                }
            )
        return output


class GeneratorStage(BaseStage):
    def outputs(self) -> list[Path]:
        assert self.job.dataset is not None
        assert self.job.model is not None
        assert self.job.conditioning is not None
        assert self.job.seed is not None
        root = (
            self.paths.outputs_root
            / "generators"
            / self.job.dataset
            / self.job.model
            / self.job.conditioning
            / f"seed{self.job.seed}"
        )
        return [root / "resolved_config.yaml", root / "training_complete.json"]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        assert self.job.dataset is not None
        assert self.job.model is not None
        assert self.job.conditioning is not None
        assert self.job.seed is not None
        from synpath.generation.wrappers import train_generator

        outputs = train_generator(
            self.job.dataset,
            self.job.model,
            self.job.conditioning,
            self.job.seed,
            self.paths,
            self.cfg,
        )
        return StageResult(self.job, outputs)


class SampleStage(BaseStage):
    def outputs(self) -> list[Path]:
        assert self.job.dataset is not None
        assert self.job.model is not None
        assert self.job.conditioning is not None
        assert self.job.seed is not None
        root = (
            self.paths.outputs_root
            / "generators"
            / self.job.dataset
            / self.job.model
            / self.job.conditioning
            / f"seed{self.job.seed}"
        )
        return [root / "synthetic_manifest.csv", root / "samples"]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        assert self.job.dataset is not None
        assert self.job.model is not None
        assert self.job.conditioning is not None
        assert self.job.seed is not None
        from synpath.generation.wrappers import sample_generator

        outputs = sample_generator(
            self.job.dataset,
            self.job.model,
            self.job.conditioning,
            self.job.seed,
            self.paths,
            self.cfg,
        )
        return StageResult(self.job, outputs)


class QualityStage(BaseStage):
    def outputs(self) -> list[Path]:
        assert self.job.dataset is not None
        assert self.job.model is not None
        assert self.job.conditioning is not None
        assert self.job.seed is not None
        root = (
            self.paths.outputs_root
            / "generators"
            / self.job.dataset
            / self.job.model
            / self.job.conditioning
            / f"seed{self.job.seed}"
            / "metrics"
        )
        return [root / "quality_metrics.json"]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        assert self.job.dataset is not None
        assert self.job.model is not None
        assert self.job.conditioning is not None
        assert self.job.seed is not None
        from synpath.metrics.quality import evaluate_quality

        outputs = evaluate_quality(
            self.job.dataset,
            self.job.model,
            self.job.conditioning,
            self.job.seed,
            self.paths,
            self.cfg,
        )
        return StageResult(self.job, outputs)


class UtilityStage(BaseStage):
    def outputs(self) -> list[Path]:
        assert self.job.dataset is not None
        assert self.job.source is not None
        assert self.job.seed is not None
        root = (
            self.paths.outputs_root / "utility" / self.job.dataset / "resnet18"
        )
        if self.job.synthetic_model is not None:
            root = root / self.job.synthetic_model
        root = root / self.job.source / f"seed{self.job.seed}"
        return [root / "metrics.json"]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        assert self.job.dataset is not None
        assert self.job.source is not None
        assert self.job.seed is not None
        from synpath.utility.resnet import train_utility_classifier

        outputs = train_utility_classifier(
            self.job.dataset,
            self.job.source,
            self.job.seed,
            self.paths,
            self.cfg,
            synthetic_model=self.job.synthetic_model,
        )
        return StageResult(self.job, outputs)


class ReportStage(BaseStage):
    def outputs(self) -> list[Path]:
        name = self.cfg.get("name", "experiment")
        root = self.paths.reports_root / name
        return [root / "results.csv", root / "report.md"]

    def run(self) -> StageResult:
        skipped = self.skip_or_run()
        if skipped:
            return skipped
        from synpath.reports.aggregate import build_report

        outputs = build_report(self.paths, self.cfg)
        return StageResult(self.job, outputs)


def make_stage(
    job: Job,
    cfg: dict,
    paths: SynpathPaths,
    resume: bool,
) -> Stage:
    mapping = {
        "prepare_data": PrepareDataStage,
        "embed": EmbedStage,
        "cluster": ClusterStage,
        "select_subset": SelectSubsetStage,
        "build_conditions": BuildConditionsStage,
        "train_generator": GeneratorStage,
        "sample": SampleStage,
        "evaluate_quality": QualityStage,
        "train_utility": UtilityStage,
        "report": ReportStage,
    }
    try:
        stage_type = mapping[job.stage]
    except KeyError as exc:
        raise ValueError(f"Unknown stage: {job.stage}") from exc
    return stage_type(job=job, cfg=cfg, paths=paths, resume=resume)
