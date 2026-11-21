from __future__ import annotations

from pathlib import Path

from synpath.config import load_resolved_yaml
from synpath.paths import SynpathPaths
from synpath.pipeline.jobs import Job, StageResult
from synpath.pipeline.stages import make_stage


class PipelineRunner:
    def __init__(
        self,
        config_path: Path,
        resume: bool = False,
        paths: SynpathPaths | None = None,
    ):
        self.config_path = config_path
        self.cfg = load_resolved_yaml(config_path)
        self.resume = resume
        self.paths = SynpathPaths.from_env() if paths is None else paths

    def jobs(self) -> list[Job]:
        stages = self.cfg.get("pipeline", {}).get("stages", [])
        jobs: list[Job] = []
        for stage in stages:
            jobs.extend(self._jobs_for_stage(stage))
        return jobs

    def plan_lines(self) -> list[str]:
        lines = [
            "synpath experiment plan",
            f"config: {self.config_path}",
            f"experiment: {self.cfg.get('name', 'unnamed')}",
            f"data_root: {self.paths.data_root}",
            f"cache_root: {self.paths.cache_root}",
            f"outputs_root: {self.paths.outputs_root}",
            f"reports_root: {self.paths.reports_root}",
            "",
            "jobs:",
        ]
        lines.extend(f"  - {job.key}" for job in self.jobs())
        return lines

    def run(self) -> list[StageResult]:
        results = []
        for job in self.jobs():
            stage = make_stage(job, self.cfg, self.paths, self.resume)
            results.append(stage.run())
        return results

    def run_single(self, job: Job) -> StageResult:
        stage = make_stage(job, self.cfg, self.paths, self.resume)
        return stage.run()

    def _jobs_for_stage(self, stage: str) -> list[Job]:
        datasets = self.cfg.get("datasets", [])
        generators = self.cfg.get("generators", [])
        modes = self.cfg.get("conditioning_modes", [])
        gen_seeds = self.cfg.get("seeds", {}).get("generators", [0])
        cls_seeds = self.cfg.get("seeds", {}).get("classifiers", [0])
        utility_sources = self.cfg.get("utility", {}).get("sources", [])
        utility_models = self.cfg.get("utility", {}).get("synthetic_generators")
        if utility_models is None:
            utility_models = generators

        if stage in {"prepare_data", "embed", "cluster", "select_subset"}:
            return [Job(stage=stage, dataset=dataset) for dataset in datasets]

        if stage == "build_conditions":
            return [
                Job(stage=stage, dataset=dataset, conditioning=mode)
                for dataset in datasets
                for mode in modes
            ]

        if stage in {"train_generator", "sample", "evaluate_quality"}:
            return [
                Job(
                    stage=stage,
                    dataset=dataset,
                    model=model,
                    conditioning=mode,
                    seed=seed,
                )
                for dataset in datasets
                for model in generators
                for mode in modes
                for seed in gen_seeds
            ]

        if stage == "train_utility":
            utility_sources = self._enabled_utility_sources(utility_sources)
            jobs = []
            for dataset in datasets:
                for source in utility_sources:
                    synthetic_models = (
                        [None] if source == "real_only" else utility_models
                    )
                    for synthetic_model in synthetic_models:
                        for seed in cls_seeds:
                            jobs.append(
                                Job(
                                    stage=stage,
                                    dataset=dataset,
                                    source=source,
                                    seed=seed,
                                    synthetic_model=synthetic_model,
                                )
                            )
            return jobs

        if stage == "report":
            return [Job(stage=stage)]

        raise ValueError(f"Unknown stage: {stage}")

    def _enabled_utility_sources(self, sources: list[str]) -> list[str]:
        modes = set(self.cfg.get("conditioning_modes", []))
        enabled = []
        for source in sources:
            if "label_global_cluster" in source:
                if "label_global_cluster" in modes:
                    enabled.append(source)
            elif "label_local_cluster" in source:
                if "label_local_cluster" in modes:
                    enabled.append(source)
            else:
                enabled.append(source)
        return enabled
