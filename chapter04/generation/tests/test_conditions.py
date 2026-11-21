from synpath.manifests import DATASET_COLUMNS, read_manifest, write_manifest
from synpath.paths import SynpathPaths
from synpath.pipeline.jobs import Job
from synpath.pipeline.stages import BuildConditionsStage


def make_paths(tmp_path):
    return SynpathPaths(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        outputs_root=tmp_path / "outputs",
        reports_root=tmp_path / "reports",
        external_root=tmp_path / "external",
    )


def write_dataset_manifest(paths):
    rows = [
        {
            "image_path": "/images/a.png",
            "label": "normal",
            "label_id": 0,
            "split": "train",
            "dataset": "demo",
            "slide_id": "",
            "patient_id": "",
        },
        {
            "image_path": "/images/b.png",
            "label": "tumor",
            "label_id": 1,
            "split": "train",
            "dataset": "demo",
            "slide_id": "",
            "patient_id": "",
        },
    ]
    write_manifest(paths.data_root / "manifests" / "demo.csv", rows, DATASET_COLUMNS)


def test_build_true_label_conditions(tmp_path):
    paths = make_paths(tmp_path)
    write_dataset_manifest(paths)
    job = Job(stage="build_conditions", dataset="demo", conditioning="true_label")

    result = BuildConditionsStage(job, {}, paths).run()

    rows = read_manifest(result.outputs[0])
    assert rows[0]["condition"] == "normal"
    assert rows[1]["condition"] == "tumor"
    assert rows[0]["prompt"] == "a histopathology patch of normal"


def test_build_label_local_cluster_conditions(tmp_path):
    paths = make_paths(tmp_path)
    write_dataset_manifest(paths)
    write_manifest(
        paths.cache_root
        / "clusters"
        / "demo"
        / "label_local_cluster_labels.csv",
        [
            {"image_path": "/images/a.png", "cluster_id": 0},
            {"image_path": "/images/b.png", "cluster_id": 3},
        ],
        ["image_path", "cluster_id"],
    )
    job = Job(
        stage="build_conditions",
        dataset="demo",
        conditioning="label_local_cluster",
    )

    result = BuildConditionsStage(job, {}, paths).run()

    rows = read_manifest(result.outputs[0])
    assert rows[0]["condition"] == "normal__cluster_0"
    assert rows[1]["condition"] == "tumor__cluster_3"
    assert rows[1]["prompt"] == (
        "a histopathology patch of tumor, morphology cluster 3"
    )


def test_build_label_global_cluster_conditions(tmp_path):
    paths = make_paths(tmp_path)
    write_dataset_manifest(paths)
    write_manifest(
        paths.cache_root / "clusters" / "demo" / "pseudo_labels.csv",
        [
            {"image_path": "/images/a.png", "cluster_id": 4},
            {"image_path": "/images/b.png", "cluster_id": 4},
        ],
        ["image_path", "cluster_id"],
    )
    job = Job(
        stage="build_conditions",
        dataset="demo",
        conditioning="label_global_cluster",
    )

    result = BuildConditionsStage(job, {}, paths).run()

    rows = read_manifest(result.outputs[0])
    assert rows[0]["condition"] == "normal__cluster_4"
    assert rows[1]["condition"] == "tumor__cluster_4"
