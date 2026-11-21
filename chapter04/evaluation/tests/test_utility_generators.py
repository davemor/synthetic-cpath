from synpath.artifacts import write_json
from synpath.paths import SynpathPaths
from synpath.pipeline.runner import PipelineRunner
from synpath.reports.aggregate import collect_utility


def make_paths(tmp_path):
    return SynpathPaths(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        outputs_root=tmp_path / "outputs",
        reports_root=tmp_path / "reports",
        external_root=tmp_path / "external",
    )


def test_utility_jobs_expand_over_synthetic_generators(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
name: demo
datasets:
  - pathmnist224
generators:
  - stylegan2ada
  - sd_lora
conditioning_modes:
  - true_label
seeds:
  classifiers: [0, 1]
utility:
  synthetic_generators:
    - stylegan2ada
    - sd_lora
  sources:
    - real_only
    - synthetic_only_true_label
pipeline:
  stages:
    - train_utility
""",
    )

    jobs = PipelineRunner(config, paths=make_paths(tmp_path)).jobs()

    real_jobs = [job for job in jobs if job.source == "real_only"]
    synthetic_jobs = [job for job in jobs if job.source == "synthetic_only_true_label"]
    assert len(real_jobs) == 2
    assert {job.synthetic_model for job in real_jobs} == {None}
    assert len(synthetic_jobs) == 4
    assert {job.synthetic_model for job in synthetic_jobs} == {
        "stylegan2ada",
        "sd_lora",
    }


def test_collect_utility_records_synthetic_generator(tmp_path):
    paths = make_paths(tmp_path)
    metric_path = (
        paths.outputs_root
        / "utility"
        / "pathmnist224"
        / "resnet18"
        / "stylegan2ada"
        / "synthetic_only_true_label"
        / "seed0"
        / "metrics.json"
    )
    write_json(metric_path, {"accuracy": 0.5})

    rows = collect_utility(paths.outputs_root)

    assert rows.iloc[0]["model"] == "resnet18"
    assert rows.iloc[0]["generator_model"] == "stylegan2ada"
    assert rows.iloc[0]["conditioning"] == "synthetic_only_true_label"
