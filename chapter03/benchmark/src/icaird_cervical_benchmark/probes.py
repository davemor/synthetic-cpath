from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProbeClassifierConfig, ProbeConfig
from .data import FeatureSplit, load_feature_splits, validate_feature_splits
from .metrics import compute_metrics, write_evaluation_artifacts
from .timing import TimerRecorder


def _param_grid(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not params:
        return [{}]
    keys = list(params)
    return [dict(zip(keys, values)) for values in product(*(params[key] for key in keys))]


def _param_candidates(config: ProbeClassifierConfig) -> list[dict[str, Any]]:
    if config.search.method == "random":
        n_iter = config.search.n_iter
        if n_iter is None:
            raise ValueError("Random probe search requires n_iter.")
        return list(
            ParameterSampler(
                config.params,
                n_iter=int(n_iter),
                random_state=int(config.search.random_state),
            )
        )
    return _param_grid(config.params)


def _make_estimator(classifier: str, params: dict[str, Any], seed: int, n_classes: int):
    params = dict(params)
    params.pop("linesearch_max_iter", None)
    if classifier == "logistic-regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=seed,
                solver="lbfgs",
                n_jobs=None,
                **params,
            ),
        )
    if classifier == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(random_state=seed, **params),
        )
    if classifier == "xgboost":
        from xgboost import XGBClassifier

        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        extra = {} if n_classes == 2 else {"num_class": n_classes}
        return XGBClassifier(
            random_state=seed,
            objective=objective,
            eval_metric="logloss",
            tree_method="hist",
            **extra,
            **params,
        )
    raise ValueError(f"Unsupported probe classifier {classifier!r}.")


def _predict_proba(estimator, features: np.ndarray, n_classes: int) -> np.ndarray:
    proba = estimator.predict_proba(features)
    proba = np.asarray(proba)
    if n_classes == 2 and proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return proba


def _concat(train: FeatureSplit, valid: FeatureSplit) -> FeatureSplit:
    return FeatureSplit(
        features=np.vstack([train.features, valid.features]),
        labels=np.concatenate([train.labels, valid.labels]),
        sample_ids=[*train.sample_ids, *valid.sample_ids],
    )


def _encode_feature_splits(splits):
    class_values = sorted(np.unique(splits.train.labels).tolist())
    mapping = {value: idx for idx, value in enumerate(class_values)}

    def encode(split: FeatureSplit) -> FeatureSplit:
        return FeatureSplit(
            features=split.features,
            labels=np.asarray([mapping[value] for value in split.labels]),
            sample_ids=split.sample_ids,
        )

    return (
        type(splits)(train=encode(splits.train), valid=encode(splits.valid), test=encode(splits.test)),
        [str(value) for value in class_values],
    )


def run_probes(config: ProbeConfig, output_dir: Path, seeds: list[int], timer: TimerRecorder) -> list[dict]:
    results = []
    if not config.enabled:
        return results
    for embedding_name, embedding_config in config.embeddings.items():
        with timer.time("load_embedding", embedding=embedding_name):
            splits = load_feature_splits(
                embedding_config.path,
                label_column=embedding_config.label_column,
            )
            dim = validate_feature_splits(splits, embedding_config.expected_dim)
            splits, class_names = _encode_feature_splits(splits)
        n_classes = len(class_names)
        for classifier_name, classifier_config in config.classifiers.items():
            candidates = _param_candidates(classifier_config)
            for seed in seeds:
                run_dir = output_dir / "probes" / embedding_name / classifier_name / f"seed_{seed}"
                started = perf_counter()
                best_params, best_score, dynamics = _select_probe_params(
                    classifier_name,
                    classifier_config,
                    candidates,
                    splits.train,
                    config.cv_folds,
                    seed,
                    n_classes,
                )
                train_valid = _concat(splits.train, splits.valid)
                final_estimator = _make_estimator(classifier_name, best_params, seed, n_classes)
                final_estimator.fit(train_valid.features, train_valid.labels)
                proba = _predict_proba(final_estimator, splits.test.features, n_classes)
                pred = np.argmax(proba, axis=1)
                metrics = write_evaluation_artifacts(
                    run_dir,
                    splits.test.labels,
                    pred,
                    proba,
                    splits.test.sample_ids,
                    class_names,
                )
                elapsed = perf_counter() - started
                timer.add(
                    "probe",
                    elapsed,
                    embedding=embedding_name,
                    classifier=classifier_name,
                    seed=seed,
                )
                _write_probe_dynamics(run_dir, dynamics)
                enriched = {
                    **metrics,
                    "task": "probe",
                    "embedding": embedding_name,
                    "classifier": classifier_name,
                    "model_id": f"{embedding_name}:{classifier_name}",
                    "seed": seed,
                    "feature_dim": dim,
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "cv_score": best_score,
                    "runtime_seconds": elapsed,
                }
                pd.DataFrame([enriched]).to_csv(run_dir / "metrics.csv", index=False)
                results.append(enriched)
    return results


def run_probe_task(
    config: ProbeConfig,
    output_dir: Path,
    embedding_name: str,
    classifier_name: str,
    seed: int,
    timer: TimerRecorder,
) -> dict:
    if embedding_name not in config.embeddings:
        raise ValueError(f"Unknown embedding {embedding_name!r}.")
    if classifier_name not in config.classifiers:
        raise ValueError(f"Unknown probe classifier {classifier_name!r}.")
    embedding_config = config.embeddings[embedding_name]
    classifier_config = config.classifiers[classifier_name]
    with timer.time("load_embedding", embedding=embedding_name):
        splits = load_feature_splits(
            embedding_config.path,
            label_column=embedding_config.label_column,
        )
        dim = validate_feature_splits(splits, embedding_config.expected_dim)
        splits, class_names = _encode_feature_splits(splits)
    n_classes = len(class_names)
    candidates = _param_candidates(classifier_config)
    run_dir = output_dir / "probes" / embedding_name / classifier_name / f"seed_{seed}"
    started = perf_counter()
    best_params, best_score, dynamics = _select_probe_params(
        classifier_name,
        classifier_config,
        candidates,
        splits.train,
        config.cv_folds,
        seed,
        n_classes,
    )
    train_valid = _concat(splits.train, splits.valid)
    final_estimator = _make_estimator(classifier_name, best_params, seed, n_classes)
    final_estimator.fit(train_valid.features, train_valid.labels)
    proba = _predict_proba(final_estimator, splits.test.features, n_classes)
    pred = np.argmax(proba, axis=1)
    metrics = write_evaluation_artifacts(
        run_dir,
        splits.test.labels,
        pred,
        proba,
        splits.test.sample_ids,
        class_names,
    )
    elapsed = perf_counter() - started
    timer.add("probe", elapsed, embedding=embedding_name, classifier=classifier_name, seed=seed)
    _write_probe_dynamics(run_dir, dynamics)
    enriched = {
        **metrics,
        "task": "probe",
        "embedding": embedding_name,
        "classifier": classifier_name,
        "model_id": f"{embedding_name}:{classifier_name}",
        "seed": seed,
        "feature_dim": dim,
        "best_params": json.dumps(best_params, sort_keys=True),
        "cv_score": best_score,
        "runtime_seconds": elapsed,
    }
    pd.DataFrame([enriched]).to_csv(run_dir / "metrics.csv", index=False)
    return enriched


def _select_probe_params(
    classifier_name: str,
    classifier_config: ProbeClassifierConfig,
    candidates: list[dict[str, Any]],
    train: FeatureSplit,
    cv_folds: int,
    seed: int,
    n_classes: int,
) -> tuple[dict[str, Any], float, list[dict]]:
    min_class_count = int(np.min(np.bincount(train.labels.astype(int))))
    folds = min(cv_folds, min_class_count)
    if folds < 2:
        raise ValueError("Need at least two samples per class for probe cross-validation.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params = candidates[0]
    dynamics = []
    for candidate_idx, params in enumerate(candidates):
        fold_scores = []
        candidate_started = perf_counter()
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(train.features, train.labels)):
            estimator = _make_estimator(classifier_name, params, seed + fold_idx, n_classes)
            estimator.fit(train.features[train_idx], train.labels[train_idx])
            proba = _predict_proba(estimator, train.features[valid_idx], n_classes)
            pred = np.argmax(proba, axis=1)
            metrics = compute_metrics(train.labels[valid_idx], pred, proba)
            fold_scores.append(metrics["auc"])
            dynamics.append(
                {
                    "candidate": candidate_idx,
                    "fold": fold_idx,
                    "seed": seed,
                    "classifier": classifier_name,
                    "params": json.dumps(params, sort_keys=True),
                    "accuracy": metrics["accuracy"],
                    "auc": metrics["auc"],
                }
            )
        mean_auc = float(np.mean(fold_scores))
        dynamics.append(
            {
                "candidate": candidate_idx,
                "fold": "mean",
                "seed": seed,
                "classifier": classifier_name,
                "params": json.dumps(params, sort_keys=True),
                "auc": mean_auc,
                "runtime_seconds": perf_counter() - candidate_started,
            }
        )
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = params
    return best_params, best_score, dynamics


def _write_probe_dynamics(run_dir: Path, dynamics: list[dict]) -> None:
    pd.DataFrame(dynamics).to_csv(run_dir / "training_dynamics.csv", index=False)
