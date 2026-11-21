import json

import pandas as pd

from synpath.manifests import DATASET_COLUMNS, read_manifest, write_manifest
from synpath.paths import SynpathPaths
from synpath.pipeline.jobs import Job
from synpath.pipeline.stages import SelectSubsetStage


def test_subset_is_selected_after_full_pool_clustering(tmp_path):
    paths = SynpathPaths(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        outputs_root=tmp_path / "outputs",
        reports_root=tmp_path / "reports",
        external_root=tmp_path / "external",
    )
    full_rows = [
        {
            "image_path": f"/images/{label}_{index}.png",
            "label": label,
            "label_id": label_id,
            "split": "train",
            "dataset": "demo",
            "slide_id": "",
            "patient_id": "",
        }
        for label_id, label in enumerate(("negative", "positive"))
        for index in range(3)
    ]
    full_path = paths.data_root / "manifests" / "demo_full.csv"
    write_manifest(full_path, full_rows, DATASET_COLUMNS)

    embedding_path = (
        paths.cache_root
        / "embeddings"
        / "demo"
        / "cluster_pool_uni_224.parquet"
    )
    embedding_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {**row, "f0": float(index), "f1": float(index + 1)}
            for index, row in enumerate(full_rows)
        ]
    ).to_parquet(embedding_path, index=False)

    cluster_root = paths.cache_root / "clusters" / "demo"
    assignments = [
        {"image_path": row["image_path"], "cluster_id": index % 2}
        for index, row in enumerate(full_rows)
    ]
    for name in ("pseudo_labels.csv", "label_local_cluster_labels.csv"):
        write_manifest(
            cluster_root / name,
            assignments,
            ["image_path", "cluster_id"],
        )

    stage = SelectSubsetStage(
        job=Job(stage="select_subset", dataset="demo"),
        cfg={
            "limits": {
                "max_train_images": 2,
                "train_sampling": "balanced_labels",
            }
        },
        paths=paths,
    )
    stage.run()

    assert len(read_manifest(full_path)) == 6
    selected = read_manifest(paths.data_root / "manifests" / "demo.csv")
    assert len(selected) == 2
    assert {row["label"] for row in selected} == {"negative", "positive"}
    selected_embeddings = pd.read_parquet(
        paths.cache_root / "embeddings" / "demo" / "train_uni_224.parquet"
    )
    assert selected_embeddings["image_path"].tolist() == [
        row["image_path"] for row in selected
    ]
    metadata = json.loads(
        (paths.data_root / "manifests" / "demo_selection.json").read_text()
    )
    assert metadata["full_counts"]["train"] == 6
    assert metadata["selected_counts"]["train"] == 2
    assert metadata["held_out_splits_excluded"] is True
