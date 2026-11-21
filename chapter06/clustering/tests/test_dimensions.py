from pathlib import Path

from cluster_exp.config import ExperimentConfig
from cluster_exp.dimensions import load_projection_targets


def _write_dimensionality_run(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "intrinsic_dims.csv").write_text(
        "dataset,encoder,intrinsic_dim_mle\n"
        "toy,uni,3\n"
    )
    curve = root / "curve"
    curve.mkdir()
    (curve / "thresholds_results.csv").write_text(
        "dataset,encoder,projector,classifier,metric,condition,achieved_dim,baseline\n"
        "toy,uni,pca,logistic-regression,accuracy,0.95,4,1.0\n"
        "toy,uni,pca,logistic-regression,accuracy,0.99,5,1.0\n"
        "toy,uni,pca,logistic-regression,roc_auc,0.95,6,1.0\n"
        "toy,uni,pca,logistic-regression,roc_auc,0.99,6,1.0\n"
    )


def _cfg(tmp_path: Path) -> ExperimentConfig:
    dim_root = tmp_path / "dim"
    _write_dimensionality_run(dim_root)
    return ExperimentConfig.from_dict(
        {
            "paths": {
                "embeddings_root": str(tmp_path / "emb"),
                "dimensionality_run_root": str(dim_root),
                "output_root": str(tmp_path / "out"),
                "run_name": "run",
            },
            "datasets": ["toy"],
            "encoders": ["uni"],
            "seeds": [0, 1],
            "projection": {
                "projectors": ["pca"],
                "threshold_metrics": ["roc_auc"],
                "include_one_dim": False,
            },
            "runtime": {"require_cuml": False, "allow_cpu_fallback": True},
        }
    )


def test_load_projection_targets_deduplicates_per_seed(tmp_path: Path) -> None:
    targets = load_projection_targets(_cfg(tmp_path))
    raw = [target for target in targets if target.projector == "raw"]
    pca = [target for target in targets if target.projector == "pca"]

    assert len(raw) == 2
    assert sorted({target.seed for target in raw}) == [0, 1]
    assert sorted((target.seed, target.dimension) for target in pca) == [
        (0, 3),
        (0, 6),
        (1, 3),
        (1, 6),
    ]


def test_load_projection_targets_explicit_dimension_range(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.projection.explicit_dimension_min = 1
    cfg.projection.explicit_dimension_max = 5
    cfg.projection.include_one_dim = True

    targets = load_projection_targets(cfg)
    pca = [target for target in targets if target.projector == "pca"]

    assert sorted((target.seed, target.dimension) for target in pca) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
    ]
    by_dim = {(target.seed, target.dimension): target.target_name for target in pca}
    assert by_dim[(0, 1)] == "dim_1"
    assert by_dim[(0, 3)] == "intrinsic_mle"
    assert by_dim[(0, 6)] == "roc_auc_99"
