"""ExperimentRunner: orchestrates the full three-phase factorial benchmark."""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Iterator

import cupy as cp
import numpy as np
import torch

from lowgen.data.feature_space import FeatureSpace
from lowgen.data.folds import generate_stratified_folds
from lowgen.data.paths import Paths, ReducerRunPath
from lowgen.eval.fidelity import eval_fidelity
from lowgen.eval.reconstruction import eval_reconstruction_quality
from lowgen.eval.reduction import eval_reduction_quality
from lowgen.eval.utility import eval_utility
from lowgen.generate.base import LatentGenerator
from lowgen.generate.gaussian import FullGaussian, IsotropicGaussian
from lowgen.generate.gmm import BayesianGMMGenerator, GMMGenerator
from lowgen.generate.resample import EmpiricalResampler
from lowgen.reduce.base import Reducer
from lowgen.reduce.pca import PCAReducer
from lowgen.reduce.umap_ import UMAPReducer
from lowgen.util.config import Config, EvalConfig, GeneratorConfig, ReducerConfig
from lowgen.util.seeds import make_seed


class ExperimentRunner:
    def __init__(
        self,
        cfg: Config,
        paths: Paths,
        run_id: str,
        force: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg
        self.paths = paths
        self.run_id = run_id
        self.force = force
        self.logger = logger or logging.getLogger("hproj")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        all_rows: list[dict] = []

        for dataset, encoder in itertools.product(
            self.cfg.datasets, self.cfg.encoders
        ):
            self.logger.info(f"=== dataset={dataset}  encoder={encoder} ===")
            try:
                rows = self._run_dataset_encoder(dataset, encoder)
            except FileNotFoundError as exc:
                self.logger.warning(f"  Skipping — {exc}")
                continue
            all_rows.extend(rows)

        self._write_results_csv(all_rows)
        self.logger.info("Experiment complete.")

    # ------------------------------------------------------------------
    # Per (dataset, encoder) loop
    # ------------------------------------------------------------------

    def _run_dataset_encoder(
        self, dataset: str, encoder: str
    ) -> list[dict]:
        emb_path = self.paths.embedding(dataset, encoder)
        train_space, test_space = emb_path.load_splits()
        cv_folds = generate_stratified_folds(
            train_space, self.cfg.num_folds, seed=self.cfg.seeds.base_seed
        )

        rows: list[dict] = []
        for reducer_cfg in self.cfg.reducers:
            for dim in reducer_cfg.dimensions:
                rows.extend(
                    self._run_reducer_dim(
                        dataset, encoder, reducer_cfg, dim,
                        train_space, test_space, cv_folds
                    )
                )
        return rows

    # ------------------------------------------------------------------
    # Per (reducer, dimension) loop
    # ------------------------------------------------------------------

    def _run_reducer_dim(
        self,
        dataset: str,
        encoder: str,
        reducer_cfg: ReducerConfig,
        dim: int,
        train_space: FeatureSpace,
        test_space: FeatureSpace,
        cv_folds,
    ) -> list[dict]:
        base_seed = self.cfg.seeds.base_seed

        if reducer_cfg.kind == "pca":
            reducers = [PCAReducer(n_components=dim, whiten=reducer_cfg.whiten)]
        elif reducer_cfg.kind == "umap":
            reducers = list(
                self._select_umap_reducer(
                    dim, reducer_cfg, train_space, cv_folds, base_seed, dataset, encoder
                )
            )
        else:
            raise ValueError(f"Unknown reducer kind: {reducer_cfg.kind!r}")

        rows: list[dict] = []
        for reducer in reducers:
            rows.extend(
                self._run_single_reducer(
                    dataset, encoder, reducer, dim,
                    train_space, test_space, cv_folds
                )
            )
        return rows

    # ------------------------------------------------------------------
    # UMAP hyperparameter selection via CV
    # ------------------------------------------------------------------

    def _select_umap_reducer(
        self,
        dim: int,
        reducer_cfg: ReducerConfig,
        train_space: FeatureSpace,
        cv_folds,
        base_seed: int,
        dataset: str,
        encoder: str,
    ) -> Iterator[UMAPReducer]:
        """Yield the best UMAPReducer per (n_neighbors, min_dist) combo.

        Each combo is evaluated by average trustworthiness across CV folds;
        the combo with the highest score is fit on the full training set.
        """
        from lowgen.util.compat import trustworthiness as tw

        best_score = -1.0
        best_nn, best_md = reducer_cfg.n_neighbors[0], reducer_cfg.min_dist[0]

        X_train_np = _to_numpy(train_space.features)

        for nn, md in itertools.product(reducer_cfg.n_neighbors, reducer_cfg.min_dist):
            scores = []
            for fold_idx, (tr_idx, val_idx) in enumerate(cv_folds):
                seed = make_seed(base_seed, dataset, encoder, "umap", dim, nn, md, fold_idx)
                tr_idx_np = _to_numpy(tr_idx)
                val_idx_np = _to_numpy(val_idx)
                X_tr = X_train_np[tr_idx_np]
                X_val = X_train_np[val_idx_np]

                from lowgen.reduce.umap_ import UMAPReducer as UR

                r = UR(
                    n_components=dim,
                    n_neighbors=nn,
                    min_dist=md,
                    metric=reducer_cfg.metric,
                    random_state=seed,
                )
                # fit a temporary FeatureSpace for this fold
                y_tr = _to_numpy(train_space.labels[tr_idx_np])
                fs_tr = FeatureSpace(X_tr, y_tr)
                r.fit(fs_tr)
                y_val = _to_numpy(train_space.labels[val_idx_np])
                fs_val = FeatureSpace(X_val, y_val)
                Z_val = _to_numpy(r.transform(fs_val).features)
                scores.append(
                    float(tw(X_val, Z_val, n_neighbors=self.cfg.eval.knn_k))
                )

            mean_score = float(np.mean(scores))
            self.logger.info(
                f"  UMAP d={dim} nn={nn} md={md}: "
                f"avg trustworthiness={mean_score:.4f}"
            )
            if mean_score > best_score:
                best_score = mean_score
                best_nn, best_md = nn, md

        self.logger.info(
            f"  UMAP d={dim}: selected nn={best_nn} md={best_md} "
            f"(trustworthiness={best_score:.4f})"
        )
        final_seed = make_seed(base_seed, dataset, encoder, "umap_final", dim)
        yield UMAPReducer(
            n_components=dim,
            n_neighbors=best_nn,
            min_dist=best_md,
            metric=reducer_cfg.metric,
            random_state=final_seed,
        )

    # ------------------------------------------------------------------
    # Core per-reducer work (phases 1, 2, 3)
    # ------------------------------------------------------------------

    def _run_single_reducer(
        self,
        dataset: str,
        encoder: str,
        reducer: Reducer,
        dim: int,
        train_space: FeatureSpace,
        test_space: FeatureSpace,
        cv_folds,
    ) -> list[dict]:
        run_path = self.paths.run(self.run_id)
        rr_path = run_path.reducer_run(dataset, encoder, reducer.key())
        rr_path.mkdir()

        reducer_pkl = rr_path.reducer_pkl()
        if reducer_pkl.exists() and not self.force:
            self.logger.info(f"  Loading cached reducer {reducer.key()}")
            reducer = Reducer.load(reducer_pkl)
        else:
            self.logger.info(f"  Fitting reducer {reducer.key()}")
            reducer.fit(train_space)
            reducer.save(reducer_pkl)

        # --- project train/test into low-D ---
        train_z_space = reducer.transform(train_space)
        test_z_space = reducer.transform(test_space)

        X_train_np = _to_numpy(train_space.features)
        X_test_np = _to_numpy(test_space.features)
        Z_train_np = _to_numpy(train_z_space.features)
        Z_test_np = _to_numpy(test_z_space.features)
        y_train_np = _to_numpy(train_space.labels).astype(int)
        y_test_np = _to_numpy(test_space.labels).astype(int)

        # ---- Phase 1: reduction quality ----
        phase1 = self._phase1(
            rr_path, Z_train_np, Z_test_np,
            y_train_np, y_test_np, X_test_np
        )

        # ---- Phase 2: round-trip reconstruction quality ----
        phase2 = self._phase2(rr_path, reducer, test_z_space, X_test_np, y_test_np)

        base_row = dict(
            dataset=dataset,
            encoder=encoder,
            reducer=reducer.key(),
            dim=dim,
        )

        rows: list[dict] = []
        for seed_idx in range(self.cfg.seeds.num_seeds):
            seed = make_seed(
                self.cfg.seeds.base_seed, dataset, encoder, reducer.key(), seed_idx
            )
            rng = np.random.default_rng(seed)

            for gen in self._build_generators(
                Z_train_np, y_train_np,
                rr_path, seed_idx, seed, cv_folds,
                train_z_space, train_space
            ):
                row = dict(**base_row, seed_idx=seed_idx, generator=gen.key())
                row.update({f"p1_{k}": v for k, v in phase1.items()})
                row.update({f"p2_{k}": v for k, v in phase2.items()})

                # ---- Phase 3a: fidelity ----
                fidelity = self._phase3a(
                    rr_path, gen, seed_idx, rng,
                    reducer, X_test_np, y_test_np, train_z_space
                )
                row.update({f"fid_{k}": v for k, v in _flatten(fidelity).items()})

                # ---- Phase 3b: utility ----
                utility = self._phase3b(
                    rr_path, gen, seed_idx, rng,
                    reducer, X_test_np, y_test_np,
                    X_train_np, y_train_np, train_z_space
                )
                row.update({f"util_{k}": v for k, v in _flatten(utility).items()})

                rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _phase1(
        self,
        rr_path: ReducerRunPath,
        Z_train: np.ndarray,
        Z_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_test_orig: np.ndarray,
    ) -> dict:
        p = rr_path.phase1_metrics()
        if p.exists() and not self.force:
            return _load_json(p)
        self.logger.info("    Phase 1: reduction quality")
        metrics = eval_reduction_quality(
            Z_train, Z_test, y_train, y_test, X_test_orig, self.cfg.eval
        )
        _save_json(p, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _phase2(
        self,
        rr_path: ReducerRunPath,
        reducer: Reducer,
        test_z_space: FeatureSpace,
        X_test_orig: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        p = rr_path.phase2_metrics()
        if p.exists() and not self.force:
            return _load_json(p)
        self.logger.info("    Phase 2: reconstruction quality")
        recon_space = reducer.inverse_transform(test_z_space)
        X_recon = _to_numpy(recon_space.features)
        metrics = eval_reconstruction_quality(
            X_test_orig, X_recon, y_test, self.cfg.eval
        )
        _save_json(p, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Generator loop
    # ------------------------------------------------------------------

    def _build_generators(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        rr_path: ReducerRunPath,
        seed_idx: int,
        seed: int,
        cv_folds,
        train_z_space: FeatureSpace,
        train_space: FeatureSpace,
    ) -> list[LatentGenerator]:
        generators = []
        for kind in self.cfg.generator.kinds:
            gen = self._fit_generator(
                kind, Z_train, y_train, rr_path, seed_idx, seed,
                cv_folds, train_z_space, train_space
            )
            generators.append(gen)
        return generators

    def _fit_generator(
        self,
        kind: str,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        rr_path: ReducerRunPath,
        seed_idx: int,
        seed: int,
        cv_folds,
        train_z_space: FeatureSpace,
        train_space: FeatureSpace,
    ) -> LatentGenerator:
        use_cond = self.cfg.generator.conditional

        if kind == "resample":
            gen = EmpiricalResampler()
            gen.fit(Z_train, y_train if use_cond else None)
            return gen

        if kind == "isotropic":
            gen = IsotropicGaussian()
            gen.fit(Z_train, y_train if use_cond else None)
            return gen

        if kind == "gaussian":
            gen = FullGaussian()
            gen.fit(Z_train, y_train if use_cond else None)
            return gen

        if kind in ("gmm", "bgmm"):
            n_comp = self._select_n_components(
                kind, Z_train, y_train, cv_folds, train_z_space, seed
            )
            if kind == "gmm":
                gen = GMMGenerator(n_components=n_comp, random_state=seed)
            else:
                gen = BayesianGMMGenerator(n_components=n_comp, random_state=seed)
            gen.fit(Z_train, y_train if use_cond else None)
            return gen

        raise ValueError(f"Unknown generator kind: {kind!r}")

    # ------------------------------------------------------------------
    # GMM component count selection via BIC on CV folds
    # ------------------------------------------------------------------

    def _select_n_components(
        self,
        kind: str,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds,
        train_z_space: FeatureSpace,
        seed: int,
    ) -> int:
        max_k = self.cfg.generator.max_components
        bic_per_k: dict[int, list[float]] = {k: [] for k in range(1, max_k + 1)}

        for fold_idx, (tr_idx, val_idx) in enumerate(cv_folds):
            tr_np = _to_numpy(tr_idx)
            val_np = _to_numpy(val_idx)
            Z_tr = Z_train[tr_np]
            Z_val = Z_train[val_np]
            for k in range(1, max_k + 1):
                try:
                    if kind == "gmm":
                        m = GMMGenerator(n_components=k, random_state=seed)
                    else:
                        m = BayesianGMMGenerator(n_components=k, random_state=seed)
                    m.fit(Z_tr)
                    bic_per_k[k].append(m.bic(Z_val))
                except Exception:
                    bic_per_k[k].append(float("inf"))

        mean_bic = {k: float(np.mean(v)) for k, v in bic_per_k.items()}
        best_k = min(mean_bic, key=mean_bic.__getitem__)
        self.logger.info(
            f"    {kind.upper()} component selection: best k={best_k} "
            f"(BIC={mean_bic[best_k]:.1f})"
        )
        return best_k

    # ------------------------------------------------------------------
    # Phase 3a: fidelity
    # ------------------------------------------------------------------

    def _phase3a(
        self,
        rr_path: ReducerRunPath,
        gen: LatentGenerator,
        seed_idx: int,
        rng: np.random.Generator,
        reducer: Reducer,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_z_space: FeatureSpace,
    ) -> dict:
        gr_path = rr_path.generator_run(gen.key(), seed_idx)
        p = gr_path.fidelity_metrics()
        if p.exists() and not self.force:
            return _load_json(p)

        gr_path.mkdir()
        Z_synth, y_synth = self._sample_latent(gen, len(X_test), y_test, rng)
        X_synth = self._reconstruct(reducer, Z_synth, train_z_space)
        metrics = eval_fidelity(
            X_test, X_synth, y_test, y_synth,
            subsample=self.cfg.eval.mmd_subsample
        )
        _save_json(p, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Phase 3b: utility
    # ------------------------------------------------------------------

    def _phase3b(
        self,
        rr_path: ReducerRunPath,
        gen: LatentGenerator,
        seed_idx: int,
        rng: np.random.Generator,
        reducer: Reducer,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_real_train: np.ndarray,
        y_real_train: np.ndarray,
        train_z_space: FeatureSpace,
    ) -> dict:
        gr_path = rr_path.generator_run(gen.key(), seed_idx)
        p = gr_path.utility_metrics()
        if p.exists() and not self.force:
            return _load_json(p)

        gr_path.mkdir()
        Z_synth, y_synth = self._sample_latent(gen, len(X_test), y_test, rng)
        X_synth = self._reconstruct(reducer, Z_synth, train_z_space)
        metrics = eval_utility(
            X_synth, y_synth, X_test, y_test,
            X_real_train, y_real_train, self.cfg.eval
        )
        _save_json(p, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_latent(
        self,
        gen: LatentGenerator,
        n_total: int,
        y_ref: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.generator.conditional:
            classes, counts = np.unique(y_ref, return_counts=True)
            n_per_class = max(1, n_total // len(classes))
            Z_synth, y_synth = gen.sample_conditional(n_per_class, rng)
        else:
            Z_synth = gen.sample(n_total, rng)
            y_synth = np.zeros(len(Z_synth), dtype=np.int64)
        return Z_synth, y_synth

    def _reconstruct(
        self,
        reducer: Reducer,
        Z_synth: np.ndarray,
        reference_space: FeatureSpace,
    ) -> np.ndarray:
        # Build a temporary FeatureSpace to hold Z_synth on the right device
        dummy_labels = _to_device(
            np.zeros(len(Z_synth), dtype=np.int64), reference_space.labels
        )
        Z_synth_dev = _to_device(Z_synth, reference_space.features)
        z_space = FeatureSpace(Z_synth_dev, dummy_labels)
        recon = reducer.inverse_transform(z_space)
        return _to_numpy(recon.features)

    # ------------------------------------------------------------------
    # Results aggregation
    # ------------------------------------------------------------------

    def _write_results_csv(self, rows: list[dict]) -> None:
        import pandas as pd

        results_path = self.paths.run(self.run_id).results_csv()
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(results_path, index=False)
            self.logger.info(f"Results written to {results_path}  ({len(df)} rows)")
        else:
            self.logger.warning("No result rows to write.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _to_device(arr: np.ndarray, reference):
    if isinstance(reference, cp.ndarray):
        return cp.asarray(arr)
    return arr


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_serializable(data), f, indent=2)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _flatten(d: dict, prefix: str = "") -> dict:
    """Recursively flatten nested dicts into a single-level dict."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, full_key))
        else:
            out[full_key] = v
    return out
