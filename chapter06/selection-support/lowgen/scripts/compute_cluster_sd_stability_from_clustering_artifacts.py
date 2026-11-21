from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


GROUP_COLS = [
    "dataset",
    "encoder",
    "projector",
    "dimension",
    "target_name",
    "source_metric",
    "condition",
    "clusterer",
    "cluster_params_json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-subsample", type=int, default=10_000)
    parser.add_argument("--sd-alpha", type=float, default=1.0)
    args = parser.parse_args()

    selected = _read_csv(args.selected)
    for index, row in enumerate(selected, start=1):
        row["_row_number"] = str(index)
    full_by_key = _load_full_rows(args.full_results, {_row_key(row) for row in selected})

    row_metrics: dict[int, dict[str, float]] = {}
    labels_by_row: dict[int, np.ndarray] = {}
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)

    selected_sorted = sorted(
        selected,
        key=lambda row: (
            full_by_key[_row_key(row)]["projection_path"],
            full_by_key[_row_key(row)]["labels_path"],
        ),
    )
    current_projection_path: str | None = None
    current_features: np.ndarray | None = None
    current_metric_indices: np.ndarray | None = None

    for processed, row in enumerate(selected_sorted, start=1):
        if processed == 1 or processed % 25 == 0 or processed == len(selected_sorted):
            print(f"artifact rows: {processed}/{len(selected_sorted)}", flush=True)
        row_number = int(row["_row_number"])
        full = full_by_key[_row_key(row)]
        projection_path = full["projection_path"]
        if not projection_path:
            current_projection_path = None
            current_features = None
            current_metric_indices = None
        elif projection_path != current_projection_path:
            with np.load(projection_path, allow_pickle=True) as data:
                current_features = np.asarray(data["features"], dtype=np.float32)
            current_metric_indices = deterministic_sample_indices(
                len(current_features),
                min(int(args.metric_subsample), len(current_features)),
                0,
            )
            current_projection_path = projection_path
        with np.load(full["labels_path"], allow_pickle=True) as data:
            labels = np.asarray(data["labels"], dtype=np.int32)
        labels_by_row[row_number] = labels
        if current_features is None or current_metric_indices is None:
            sd = _nan_sd()
        else:
            sd = sd_index(
                current_features[current_metric_indices],
                labels[current_metric_indices],
                alpha=float(args.sd_alpha),
            )
        row_metrics[row_number] = sd
        groups[_group_key(row)].append(row_number)

    for group_index, row_numbers in enumerate(groups.values(), start=1):
        if group_index == 1 or group_index % 25 == 0 or group_index == len(groups):
            print(f"stability groups: {group_index}/{len(groups)}", flush=True)
        ari_values = []
        for left_row, right_row in combinations(row_numbers, 2):
            left = labels_by_row[left_row]
            right = labels_by_row[right_row]
            if len(left) == len(right):
                ari_values.append(adjusted_rand_score(left, right))
        mean_ari = float(np.mean(ari_values)) if ari_values else float("nan")
        for row_number in row_numbers:
            row_metrics[row_number]["mean_pairwise_ari"] = mean_ari

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_number", "sd_index", "sd_scat", "sd_dis", "mean_pairwise_ari"],
        )
        writer.writeheader()
        for row_number in range(1, len(selected) + 1):
            values = row_metrics[row_number]
            writer.writerow(
                {
                    "row_number": row_number,
                    "sd_index": values.get("sd_index", float("nan")),
                    "sd_scat": values.get("sd_scat", float("nan")),
                    "sd_dis": values.get("sd_dis", float("nan")),
                    "mean_pairwise_ari": values.get("mean_pairwise_ari", float("nan")),
                }
            )
    print(f"wrote {args.output}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_full_rows(path: Path, keys: set[tuple[str, str, str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    found: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = _row_key(row)
            if key in keys:
                found[key] = row
                if len(found) == len(keys):
                    break
    missing = keys - set(found)
    if missing:
        raise KeyError(f"Missing {len(missing)} selected rows in {path}; first={next(iter(missing))}")
    return found


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row["task_id"]),
        str(row["config_id"]),
        str(row["clusterer"]),
        str(row["cluster_seed"]),
    )


def _group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in GROUP_COLS)


def deterministic_sample_indices(n_samples: int, sample_size: int, seed: int) -> np.ndarray:
    sample_size = min(int(sample_size), int(n_samples))
    rng = np.random.default_rng(int(seed))
    if sample_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False)).astype(np.int64)


def sd_index(features: np.ndarray, labels: np.ndarray, *, alpha: float = 1.0) -> dict[str, float]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    mask = labels != -1
    features = features[mask]
    labels = labels[mask]
    unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if features.shape[0] < 3 or unique_labels.shape[0] < 2:
        return _nan_sd()

    sums = np.zeros((len(unique_labels), features.shape[1]), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    np.add.at(sums, inverse, features)
    np.add.at(sums_sq, inverse, features * features)
    count_matrix = counts.astype(np.float64)[:, None]
    centroids = sums / count_matrix
    variances = np.maximum(sums_sq / count_matrix - centroids * centroids, 0.0)
    cluster_std_norms = np.linalg.norm(np.sqrt(variances), axis=1)

    data_std_norm = float(np.linalg.norm(np.std(features, axis=0)))
    if not np.isfinite(data_std_norm) or data_std_norm <= 0:
        return _nan_sd()

    scat = float(np.mean(cluster_std_norms) / data_std_norm)
    distances = _pairwise_distances(centroids)
    positive = distances[distances > 0]
    if positive.size == 0:
        return _nan_sd(scat=scat)
    d_min = float(np.min(positive))
    d_max = float(np.max(positive))
    row_sums = distances.sum(axis=1)
    if d_min <= 0 or not np.all(row_sums > 0):
        return _nan_sd(scat=scat)
    dis = float((d_max / d_min) * np.sum(1.0 / row_sums))
    return {"sd_index": float(alpha * scat + dis), "sd_scat": scat, "sd_dis": dis}


def _pairwise_distances(features: np.ndarray) -> np.ndarray:
    diffs = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def _nan_sd(*, scat: float = float("nan")) -> dict[str, float]:
    return {"sd_index": float("nan"), "sd_scat": scat, "sd_dis": float("nan")}


def adjusted_rand_score(left: np.ndarray, right: np.ndarray) -> float:
    left_inverse = np.unique(left, return_inverse=True)[1]
    right_inverse = np.unique(right, return_inverse=True)[1]
    n_right = int(right_inverse.max()) + 1
    contingency = np.bincount(
        left_inverse.astype(np.int64) * n_right + right_inverse.astype(np.int64)
    )
    row_sums = np.bincount(left_inverse)
    col_sums = np.bincount(right_inverse)
    sum_comb = _comb2(contingency).sum()
    sum_comb_rows = _comb2(row_sums).sum()
    sum_comb_cols = _comb2(col_sums).sum()
    total_comb = math.comb(len(left), 2)
    if total_comb == 0:
        return 1.0
    expected = sum_comb_rows * sum_comb_cols / total_comb
    maximum = 0.5 * (sum_comb_rows + sum_comb_cols)
    denominator = maximum - expected
    if denominator == 0:
        return 1.0
    return float((sum_comb - expected) / denominator)


def _comb2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return values * (values - 1.0) / 2.0


if __name__ == "__main__":
    main()
