from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from icaird_cervical_benchmark.config import SyntheticEmbeddingConfig, load_config
from icaird_cervical_benchmark.synthetic_embeddings import (
    AdaLNZeroMLPDenoiser,
    EDMLatentDiffusion,
    LatentDiffusion,
    MLPSkipDenoiser,
    evaluate_synthetic_embeddings,
    prepare_synthetic_embedding_run,
    sample_synthetic_embeddings,
    train_synthetic_embedding_model,
    write_synthetic_baselines,
)


CLASS_NAMES = ["normal", "low_grade", "high_grade", "malignant"]


def _toy_embedding_dir(path: Path, dim: int = 4) -> Path:
    rows = []
    features = []
    rng = np.random.default_rng(0)
    for split in ("train", "valid", "test"):
        for class_id, class_name in enumerate(CLASS_NAMES):
            for rep in range(3):
                rows.append(
                    {
                        "sample_id": f"{split}_{class_name}_{rep}",
                        "split": split,
                        "annotation_label_name": class_name,
                    }
                )
                center = np.full(dim, class_id, dtype=np.float32)
                features.append(center + rng.normal(0, 0.05, size=dim).astype(np.float32))
    path.mkdir(parents=True)
    np.savez(path / "features.npz", features=np.vstack(features).astype(np.float32))
    pd.DataFrame(rows).to_parquet(path / "index.parquet")
    return path


def _toy_config(tmp_path: Path) -> SyntheticEmbeddingConfig:
    return SyntheticEmbeddingConfig(
        enabled=True,
        embedding_dir=_toy_embedding_dir(tmp_path / "real"),
        output_dir=tmp_path / "out",
        embedding_dim=4,
        class_order=CLASS_NAMES,
        sample_sizes=[12],
        batch_size=4,
        num_workers=0,
        max_seen_samples=16,
        validation_interval_steps=1,
        validation_batches=1,
        patience_validations=2,
        timesteps=8,
        beta=0.01,
        hidden_dim=16,
        depth=3,
        ddim_steps=4,
        eval_max_per_class=3,
        eval_knn_k=1,
        utility_max_iter=200,
        use_amp=False,
    )


def _toy_chunked_embedding_cache(path: Path, dim: int = 4) -> Path:
    label_map = {"0": "ADI - Adipose", "1": "TUM - Tumor"}
    path.mkdir(parents=True)
    (path / "label_map.json").write_text(json.dumps(label_map))
    rng = np.random.default_rng(1)
    for split, shards, reps_per_class in (("train", 2, 6), ("test", 1, 2)):
        split_dir = path / split
        split_dir.mkdir()
        all_features = []
        all_labels = []
        all_paths = []
        for class_id in range(2):
            for rep in range(reps_per_class):
                all_features.append(
                    np.full(dim, class_id, dtype=np.float32)
                    + rng.normal(0, 0.05, size=dim).astype(np.float32)
                )
                all_labels.append(class_id)
                all_paths.append(
                    {
                        "patch_filename": f"{split}_{class_id}_{rep}.png",
                        "slide_id": f"{split}_{class_id}_{rep}",
                        "class": str(class_id),
                        "split": split,
                    }
                )
        features = np.vstack(all_features).astype(np.float32)
        labels = np.asarray(all_labels, dtype=np.int64)
        for shard_id, indices in enumerate(np.array_split(np.arange(len(labels)), shards)):
            torch.save(torch.from_numpy(features[indices]), split_dir / f"emb_{shard_id:04d}.pt")
            torch.save(torch.from_numpy(labels[indices]), split_dir / f"labels_{shard_id:04d}.pt")
            (split_dir / f"paths_{shard_id:04d}.json").write_text(
                json.dumps([all_paths[idx] for idx in indices.tolist()])
            )
    return path


def test_synthetic_embedding_config_loads(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
data:
  dataset_root: /tmp/images
synthetic_embeddings:
  enabled: true
  embedding_dir: /tmp/embeddings
  output_dir: /tmp/out
  embedding_dim: 1536
  sample_sizes: [10000]
"""
    )

    config = load_config(config_path)

    assert config.synthetic_embeddings.enabled is True
    assert config.synthetic_embeddings.embedding_dim == 1536
    assert config.synthetic_embeddings.sample_sizes == [10000]


def test_prepare_and_baseline_outputs_mirror_npz_parquet_layout(tmp_path):
    config = _toy_config(tmp_path)

    prepare_synthetic_embedding_run(config)
    written = write_synthetic_baselines(config)

    assert (config.output_dir / "metadata" / "train_stats.npz").exists()
    assert len(written) == 2
    for path in written:
        with np.load(path / "features.npz") as data:
            assert data["features"].shape == (12, 4)
        index = pd.read_parquet(path / "index.parquet")
        assert set(index["annotation_label_name"]) == set(CLASS_NAMES)
        assert set(index["split"]) == {"synthetic"}


def test_prepare_supports_chunked_embedding_cache_without_valid_split(tmp_path):
    config = _toy_config(tmp_path)
    config.embedding_dir = _toy_chunked_embedding_cache(tmp_path / "chunked")
    config.output_dir = tmp_path / "chunked_out"
    config.label_column = "class_name"
    config.class_order = ["ADI - Adipose", "TUM - Tumor"]
    config.validation_fraction = 0.25
    config.sample_sizes = [4]

    prepare_synthetic_embedding_run(config)
    written = write_synthetic_baselines(config)

    summary = pd.read_csv(config.output_dir / "metadata" / "dataset_summary.csv")
    assert set(summary["split"]) == {"train", "valid", "test"}
    assert summary.loc[summary["split"].eq("valid"), "count"].sum() == 4
    index = pd.read_parquet(written[0] / "index.parquet")
    assert set(index["class_name"]) == {"ADI - Adipose", "TUM - Tumor"}


def test_evaluate_synthetic_baseline_writes_metrics(tmp_path):
    config = _toy_config(tmp_path)
    synthetic_dir = write_synthetic_baselines(config)[0]

    result_dir = evaluate_synthetic_embeddings(config, synthetic_dir)

    assert (result_dir / "fidelity_metrics.csv").exists()
    assert (result_dir / "utility_metrics.csv").exists()


def test_latent_diffusion_training_loss_and_sampling_shapes():
    model = MLPSkipDenoiser(input_dim=4, n_classes=4, hidden_dim=16, depth=3)
    diffusion = LatentDiffusion(model, timesteps=8, beta=0.01)
    z0 = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 3, 0])

    loss = diffusion.training_loss(z0, labels)
    samples = diffusion.sample(labels, steps=4, device=torch.device("cpu"))

    assert loss.ndim == 0
    assert samples.shape == (5, 4)


def test_edm_latent_diffusion_training_loss_and_sampling_shapes():
    model = MLPSkipDenoiser(
        input_dim=4,
        n_classes=4,
        hidden_dim=16,
        depth=3,
        use_class_condition=True,
        extra_condition_tokens=1,
    )
    diffusion = EDMLatentDiffusion(
        model,
        sigma_min=0.002,
        sigma_max=2.0,
        sigma_data=0.5,
        rho=7.0,
        class_dropout=0.2,
    )
    z0 = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 3, 0])

    loss = diffusion.training_loss(z0, labels)
    samples = diffusion.sample(labels, steps=4, device=torch.device("cpu"), guidance_scale=1.5)

    assert loss.ndim == 0
    assert samples.shape == (5, 4)


def test_adaln_zero_mlp_denoiser_shapes_and_zero_init():
    model = AdaLNZeroMLPDenoiser(
        input_dim=4,
        n_classes=4,
        hidden_dim=16,
        depth=3,
        extra_condition_tokens=1,
    )
    z_t = torch.randn(5, 4)
    timesteps = torch.randn(5)
    labels = torch.tensor([0, 1, 2, 3, 4])

    out = model(z_t, timesteps, labels)

    assert out.shape == (5, 4)
    assert torch.allclose(out, torch.zeros_like(out))


def test_independent_per_class_training_and_sampling(tmp_path):
    config = _toy_config(tmp_path)
    config.model_mode = "independent_per_class"
    config.max_seen_samples = 8
    config.sample_sizes = [8]

    manifest_path = train_synthetic_embedding_model(config)
    written = sample_synthetic_embeddings(config)

    assert manifest_path.name == "independent_class_checkpoints.json"
    assert len(written) == 1
    with np.load(written[0] / "features.npz") as data:
        assert data["features"].shape == (8, 4)
    index = pd.read_parquet(written[0] / "index.parquet")
    assert set(index["generator"]) == {"independent_class_latent_ddim"}


def test_edm_conditional_training_and_sampling(tmp_path):
    config = _toy_config(tmp_path)
    config.model_mode = "edm_conditional"
    config.max_seen_samples = 8
    config.sample_sizes = [8]
    config.edm_sigma_max = 2.0
    config.guidance_scale = 1.2

    checkpoint_path = train_synthetic_embedding_model(config)
    written = sample_synthetic_embeddings(config)

    assert checkpoint_path.name == "best.pt"
    assert len(written) == 1
    with np.load(written[0] / "features.npz") as data:
        assert data["features"].shape == (8, 4)
    index = pd.read_parquet(written[0] / "index.parquet")
    assert set(index["generator"]) == {"edm_conditional"}


def test_classifier_free_edm_conditional_training_and_sampling(tmp_path):
    config = _toy_config(tmp_path)
    config.model_mode = "classifier_free_edm_conditional"
    config.max_seen_samples = 8
    config.sample_sizes = [8]
    config.edm_sigma_max = 2.0
    config.edm_class_dropout = 0.2
    config.guidance_scale = 1.2

    checkpoint_path = train_synthetic_embedding_model(config)
    written = sample_synthetic_embeddings(config)

    assert checkpoint_path.name == "best.pt"
    assert len(written) == 1
    assert written[0].name == "classifier_free_edm_cfg1.2_n8"
    with np.load(written[0] / "features.npz") as data:
        assert data["features"].shape == (8, 4)
    index = pd.read_parquet(written[0] / "index.parquet")
    assert set(index["generator"]) == {"classifier_free_edm_conditional"}


def test_edm_adaln_mlp_conditional_training_and_sampling(tmp_path):
    config = _toy_config(tmp_path)
    config.model_mode = "edm_adaln_mlp_conditional"
    config.max_seen_samples = 8
    config.sample_sizes = [8]
    config.edm_sigma_max = 2.0
    config.edm_class_dropout = 0.2
    config.guidance_scale = 1.2

    checkpoint_path = train_synthetic_embedding_model(config)
    written = sample_synthetic_embeddings(config)

    assert checkpoint_path.name == "best.pt"
    assert len(written) == 1
    assert written[0].name == "edm_adaln_mlp_cfg1.2_n8"
    with np.load(written[0] / "features.npz") as data:
        assert data["features"].shape == (8, 4)
    index = pd.read_parquet(written[0] / "index.parquet")
    assert set(index["generator"]) == {"edm_adaln_mlp_conditional"}
