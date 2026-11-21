from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FittedGenerator:
    kind: str
    condition_mode: str
    model: Any
    metadata: dict[str, Any]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
        if self.condition_mode == "unconditioned":
            return self.model.sample(n, rng), None
        raise ValueError("Use sample_counts for conditioned generators.")

    def sample_counts(self, counts: dict[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.condition_mode != "true_label":
            raise ValueError("sample_counts is only supported for true_label conditioning.")
        parts_x, parts_y = [], []
        for label, count in counts.items():
            if count <= 0:
                continue
            parts_x.append(self.model.sample_for_label(int(label), int(count), rng))
            parts_y.append(np.full(int(count), int(label), dtype=np.int64))
        return np.concatenate(parts_x).astype(np.float32), np.concatenate(parts_y).astype(np.int64)


class EmpiricalGenerator:
    def __init__(self):
        self.x: np.ndarray | None = None
        self.by_label: dict[int, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray | None) -> "EmpiricalGenerator":
        self.x = x.astype(np.float32, copy=True)
        if y is not None:
            self.by_label = {int(label): self.x[y == label].copy() for label in np.unique(y)}
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, len(self.x), size=n)
        return self.x[idx].astype(np.float32)

    def sample_for_label(self, label: int, n: int, rng: np.random.Generator) -> np.ndarray:
        x = self.by_label[int(label)]
        idx = rng.integers(0, len(x), size=n)
        return x[idx].astype(np.float32)


class DiagonalGaussianGenerator:
    def __init__(self, min_variance: float = 1e-6):
        self.min_variance = min_variance
        self.params: tuple[np.ndarray, np.ndarray] | None = None
        self.by_label: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, x: np.ndarray, y: np.ndarray | None) -> "DiagonalGaussianGenerator":
        self.params = self._fit_one(x)
        if y is not None:
            self.by_label = {int(label): self._fit_one(x[y == label]) for label in np.unique(y)}
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._sample_params(self.params, n, rng)

    def sample_for_label(self, label: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._sample_params(self.by_label[int(label)], n, rng)

    def _fit_one(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0).astype(np.float32)
        var = np.maximum(x.var(axis=0), self.min_variance).astype(np.float32)
        return mean, var

    def _sample_params(
        self, params: tuple[np.ndarray, np.ndarray], n: int, rng: np.random.Generator
    ) -> np.ndarray:
        mean, var = params
        return rng.normal(mean, np.sqrt(var), size=(n, len(mean))).astype(np.float32)


class DiagonalGmmGenerator:
    def __init__(
        self,
        n_components: int,
        random_state: int,
        max_iter: int = 100,
        class_components: dict[int, int] | None = None,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_components = class_components or {}
        self.model = None
        self.by_label = {}

    def fit(self, x: np.ndarray, y: np.ndarray | None) -> "DiagonalGmmGenerator":
        from sklearn.mixture import GaussianMixture

        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=1,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(x)
        if y is not None:
            self.by_label = {}
            for label in np.unique(y):
                x_label = x[y == label]
                k = min(
                    int(self.class_components.get(int(label), self.n_components)),
                    max(1, len(x_label)),
                )
                self.by_label[int(label)] = GaussianMixture(
                    n_components=k,
                    covariance_type="diag",
                    reg_covar=1e-4,
                    n_init=1,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                ).fit(x_label)
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        self.model.random_state = int(rng.integers(0, 2**31 - 1))
        x, _ = self.model.sample(n)
        return x.astype(np.float32)

    def sample_for_label(self, label: int, n: int, rng: np.random.Generator) -> np.ndarray:
        model = self.by_label[int(label)]
        model.random_state = int(rng.integers(0, 2**31 - 1))
        x, _ = model.sample(n)
        return x.astype(np.float32)


def fit_generator(
    kind: str,
    condition_mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_components: int,
    timeout_seconds: int,
    fit_subsample: int = 50000,
    max_iter: int = 100,
    select_k: bool = True,
    candidate_components: list[int] | tuple[int, ...] = (1, 2, 3, 5, 8, 10),
    selection_metric: str = "bic",
    min_samples_per_component: int = 100,
    x_valid: np.ndarray | None = None,
) -> FittedGenerator:
    condition_labels = None if condition_mode == "unconditioned" else y_train
    with time_limit(timeout_seconds):
        if kind == "resample":
            model = EmpiricalGenerator().fit(x_train, condition_labels)
            return FittedGenerator(kind, condition_mode, model, {"kind": kind})
        if kind == "diagonal_gaussian":
            model = DiagonalGaussianGenerator().fit(x_train, condition_labels)
            return FittedGenerator(kind, condition_mode, model, {"kind": kind})
        if kind == "diagonal_gmm":
            x_fit, y_fit = stratified_subsample(
                x_train, condition_labels, fit_subsample, seed + 1
            )
            selection_metadata: dict[str, Any]
            if select_k:
                k, class_components, selection_metadata = select_gmm_components(
                    x_fit,
                    y_fit,
                    candidate_components,
                    selection_metric,
                    min_samples_per_component,
                    seed,
                    max_iter,
                )
            else:
                k = min(int(n_components), max(1, len(x_fit) - 1))
                class_components = None
                selection_metadata = {
                    "enabled": False,
                    "selected_k": int(k),
                    "candidate_components": [int(n_components)],
                    "metric": selection_metric,
                }
            model = DiagonalGmmGenerator(
                k,
                seed,
                max_iter=max_iter,
                class_components=class_components,
            ).fit(x_fit, y_fit)
            return FittedGenerator(
                kind,
                condition_mode,
                model,
                {
                    "kind": kind,
                    "n_components": k,
                    "fit_subsample": None if len(x_train) == len(x_fit) else len(x_fit),
                    "max_iter": int(max_iter),
                    "gmm_select_k": bool(select_k),
                    "gmm_selection": selection_metadata,
                },
            )
    raise ValueError(f"Unsupported generator kind {kind!r}")


def stratified_subsample(
    x: np.ndarray | None,
    y: np.ndarray | None,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if x is None or max_samples <= 0 or len(x) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    if y is None:
        idx = np.sort(rng.choice(len(x), size=max_samples, replace=False))
        return x[idx], None
    parts = []
    labels, counts = np.unique(y, return_counts=True)
    quotas = np.floor(counts / counts.sum() * max_samples).astype(int)
    quotas = np.maximum(1, quotas)
    while quotas.sum() > max_samples:
        quotas[np.argmax(quotas)] -= 1
    while quotas.sum() < max_samples:
        quotas[np.argmax(counts - quotas)] += 1
    for label, quota in zip(labels, quotas):
        label_idx = np.flatnonzero(y == label)
        take = min(int(quota), len(label_idx))
        parts.append(rng.choice(label_idx, size=take, replace=False))
    idx = np.sort(np.concatenate(parts))
    return x[idx], y[idx]


def select_gmm_components(
    x: np.ndarray,
    y: np.ndarray | None,
    candidate_components: list[int] | tuple[int, ...],
    selection_metric: str,
    min_samples_per_component: int,
    seed: int,
    max_iter: int,
) -> tuple[int, dict[int, int] | None, dict[str, Any]]:
    metric = str(selection_metric).lower()
    if metric not in {"bic", "aic"}:
        raise ValueError(f"Unsupported GMM selection metric {selection_metric!r}")
    candidates = sorted({int(k) for k in candidate_components if int(k) > 0})
    if y is None:
        selected, records = _select_one_gmm_k(
            x,
            candidates,
            metric,
            min_samples_per_component,
            seed,
            max_iter,
        )
        return selected, None, {
            "enabled": True,
            "metric": metric,
            "candidate_components": candidates,
            "selected_k": int(selected),
            "records": records,
        }

    selected_by_label: dict[int, int] = {}
    records_by_label: dict[str, list[dict[str, Any]]] = {}
    for label in sorted(int(value) for value in np.unique(y)):
        x_label = x[y == label]
        selected, records = _select_one_gmm_k(
            x_label,
            candidates,
            metric,
            min_samples_per_component,
            seed + label,
            max_iter,
        )
        selected_by_label[label] = int(selected)
        records_by_label[str(label)] = records
    selected_values = list(selected_by_label.values())
    global_k = max(selected_values) if selected_values else 1
    return global_k, selected_by_label, {
        "enabled": True,
        "metric": metric,
        "candidate_components": candidates,
        "selected_k": None,
        "selected_k_by_label": {str(k): int(v) for k, v in selected_by_label.items()},
        "selected_k_min": int(min(selected_values)) if selected_values else None,
        "selected_k_median": (
            float(np.median(selected_values)) if selected_values else None
        ),
        "selected_k_max": int(max(selected_values)) if selected_values else None,
        "records_by_label": records_by_label,
    }


def candidate_components_for_samples(
    n_samples: int,
    candidate_components: list[int] | tuple[int, ...],
    min_samples_per_component: int,
) -> list[int]:
    n_samples = int(n_samples)
    if n_samples <= 0:
        return [1]
    min_samples = max(1, int(min_samples_per_component))
    max_by_density = max(1, n_samples // min_samples)
    max_k = max(1, min(n_samples, max_by_density))
    filtered = sorted({int(k) for k in candidate_components if 0 < int(k) <= max_k})
    return filtered or [1]


def _select_one_gmm_k(
    x: np.ndarray,
    candidate_components: list[int],
    metric: str,
    min_samples_per_component: int,
    seed: int,
    max_iter: int,
) -> tuple[int, list[dict[str, Any]]]:
    from sklearn.mixture import GaussianMixture

    candidates = candidate_components_for_samples(
        len(x), candidate_components, min_samples_per_component
    )
    records: list[dict[str, Any]] = []
    best_k = candidates[0]
    best_value = float("inf")
    for k in candidates:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=1,
            max_iter=int(max_iter),
            random_state=int(seed),
        ).fit(x)
        bic = float(model.bic(x))
        aic = float(model.aic(x))
        value = bic if metric == "bic" else aic
        records.append(
            {
                "k": int(k),
                "bic": bic,
                "aic": aic,
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
            }
        )
        if value < best_value:
            best_value = value
            best_k = int(k)
    return int(best_k), records


@contextmanager
def time_limit(seconds: int):
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"generator exceeded timeout of {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
