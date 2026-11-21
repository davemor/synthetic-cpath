from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from icaird_cervical_benchmark.cluster_pseudolabels import prepare_cluster_pseudolabels
from icaird_cervical_benchmark.config import ClusterPseudoLabelConfig, load_config


def test_cluster_pseudolabel_config_parses_kmeans_parameters(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
data:
  dataset_root: /tmp/example
cluster_pseudolabels:
  enabled: true
  source_embedding_dir: /tmp/source_embeddings
  source_bag_root: /tmp/source_bags
  pca_components: 42
  kmeans_clusters: 24
  kmeans_init: k-means++
  kmeans_n_init: 10
  kmeans_max_iter: 300
  random_state: 0
"""
    )

    config = load_config(config_path)

    assert config.cluster_pseudolabels.pca_components == 42
    assert config.cluster_pseudolabels.kmeans_clusters == 24
    assert config.cluster_pseudolabels.kmeans_init == "k-means++"
    assert config.cluster_pseudolabels.kmeans_n_init == 10
    assert config.cluster_pseudolabels.kmeans_max_iter == 300
    assert config.cluster_pseudolabels.random_state == 0


def test_prepare_cluster_pseudolabels_writes_derived_patch_and_bag_roots(tmp_path: Path) -> None:
    source_embedding_dir = _write_patch_root(tmp_path)
    source_bag_root = _write_bag_root(tmp_path)
    config = ClusterPseudoLabelConfig(
        source_embedding_dir=source_embedding_dir,
        source_bag_root=source_bag_root,
        output_embedding_dir=tmp_path / "derived_embeddings" / "uni2_h",
        output_bag_root=tmp_path / "derived_bags",
        artifact_dir=tmp_path / "artifacts",
        pca_components=2,
        kmeans_clusters=3,
        kmeans_n_init=1,
        transform_batch_size=4,
    )

    outputs = prepare_cluster_pseudolabels(config)
    patch_index = pd.read_parquet(outputs["output_embedding_dir"] / "index.parquet")
    bag_index = pd.read_parquet(
        outputs["output_bag_root"]
        / "embeddings"
        / "uni2_h"
        / "slides"
        / "slide_a.uni2_h.embedding"
        / "index.parquet"
    )
    mapping = (outputs["artifact_dir"] / "cluster_label_mapping.json").read_text()

    assert len(patch_index) == 12
    assert patch_index["sample_id"].tolist() == [f"sample_{idx}" for idx in range(12)]
    assert patch_index["cluster_pseudo_label"].str.match(r"cluster_0[0-2]").all()
    assert bag_index["cluster_pseudo_label"].str.match(r"cluster_0[0-2]").all()
    assert (outputs["output_embedding_dir"] / "features.npz").exists()
    assert (outputs["output_bag_root"] / "dataset_index_ometiff.csv").exists()
    assert "cluster_00" in mapping
    assert "cluster_02" in mapping


def _write_patch_root(tmp_path: Path) -> Path:
    root = tmp_path / "source_embeddings"
    root.mkdir(parents=True)
    features = np.asarray(
        [
            [-4.0, -4.0, 0.0, 0.0],
            [-4.2, -3.8, 0.0, 0.0],
            [-3.8, -4.1, 0.0, 0.0],
            [0.0, 0.0, 4.0, 4.0],
            [0.0, 0.0, 4.2, 3.8],
            [0.0, 0.0, 3.8, 4.1],
            [4.0, 4.0, 0.0, 0.0],
            [4.2, 3.8, 0.0, 0.0],
            [3.8, 4.1, 0.0, 0.0],
            [-3.9, -4.2, 0.0, 0.0],
            [0.0, 0.0, 4.1, 3.9],
            [4.1, 3.9, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    index = pd.DataFrame(
        {
            "sample_id": [f"sample_{idx}" for idx in range(len(features))],
            "split": ["train"] * 9 + ["valid", "valid", "test"],
            "annotation_label_name": ["normal"] * len(features),
        }
    )
    np.savez_compressed(root / "features.npz", features=features)
    index.to_parquet(root / "index.parquet", index=False)
    return root


def _write_bag_root(tmp_path: Path) -> Path:
    root = tmp_path / "source_bags"
    slide_dir = root / "embeddings" / "uni2_h" / "slides" / "slide_a.uni2_h.embedding"
    slide_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id": "slide_a",
                "split": "train",
                "category": "low_grade",
                "subcategory": "cin1",
            }
        ]
    ).to_csv(root / "dataset_index_ometiff.csv", index=False)
    features = np.asarray(
        [
            [-4.1, -4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 4.1],
            [4.0, 4.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(slide_dir / "features.npz", features=features)
    pd.DataFrame(
        {
            "frame_index": [0, 1, 2],
            "annotation_label_name": ["normal", "low_grade", "malignant"],
        }
    ).to_parquet(slide_dir / "index.parquet", index=False)
    return root
