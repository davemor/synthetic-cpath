import hashlib
import json
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

import click
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import as_completed
from tqdm import tqdm

from hproj.classifiers.classifier import ClassifierFactory
from hproj.classifiers.metrics import compute_classification_metrics
from hproj.cli.calibrate_classifier import load_task_results
from hproj.cli.thresholds import compute_thresholds
from hproj.data.feature_space import FeatureSpace
from hproj.data.folds import generate_stratified_folds
from hproj.data.paths import Paths
from hproj.measure.measurement import MeasurementFactory
from hproj.projectors.projector import ProjectorFactory
from hproj.util.atomic import atomic_write_json
from hproj.util.config import ClassifierConfig, Config, ProjectorConfig
from hproj.util.dask import make_dask_client
from hproj.util.logging import setup_logging
from hproj.util.seeds import make_seed
from hproj.util.timing import append_stage_time


@dataclass
class TaskDescription:
    projector_cfg: ProjectorConfig
    classifier_cfg: ClassifierConfig
    dataset_name: str
    n_components: int
    base_seed: int

    def key(self) -> str:
        return (
            f"{self.projector_cfg.name}_{self.classifier_cfg.name}_"
            f"{self.dataset_name}_{self.n_components}_{self.base_seed}"
        )


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _projection_cache_key(
    dataset_name: str,
    projector_name: str,
    projector_params: dict,
    n_components: int,
    base_seed: int,
    fold_index: int,
    sample_count: int,
) -> str:
    payload = {
        "version": 1,
        "dataset": dataset_name,
        "projector_name": projector_name,
        "projector_params": projector_params,
        "n_components": n_components,
        "base_seed": base_seed,
        "fold_index": fold_index,
        "sample_count": sample_count,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()[:16]


def _save_fold_projection(path: Path, train_proj: FeatureSpace, valid_proj: FeatureSpace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("wb") as f:
        np.savez(
            f,
            train_features=cp.asnumpy(train_proj.features),
            train_labels=cp.asnumpy(train_proj.labels),
            valid_features=cp.asnumpy(valid_proj.features),
            valid_labels=cp.asnumpy(valid_proj.labels),
        )
    tmp_path.replace(path)


def _load_fold_projection(path: Path) -> tuple[FeatureSpace, FeatureSpace]:
    with np.load(path) as data:
        train_proj = FeatureSpace(
            cp.asarray(data["train_features"]),
            cp.asarray(data["train_labels"]),
        )
        valid_proj = FeatureSpace(
            cp.asarray(data["valid_features"]),
            cp.asarray(data["valid_labels"]),
        )
    return train_proj, valid_proj


def load_best_params(path, logger, name):
    if path.exists():
        with open(path) as f:
            return json.load(f)

    logger.info(f"Skipping loading hyperparameters for {name}.")
    return {}


def build_projector_configs(cfg, paths, run_id, logger):
    return [
        ProjectorConfig(
            name=name,
            params=load_best_params(
                paths.run(run_id).calibration().projector(name).best_params(),
                logger,
                name,
            ),
        )
        for name in cfg.curve.projectors
    ]


def build_classifier_configs(cfg, paths, run_id, logger):
    return [
        ClassifierConfig(
            name=name,
            params=load_best_params(
                paths.run(run_id).calibration().classifier(name).best_params(),
                logger,
                name,
            ),
        )
        for name in cfg.curve.classifiers
    ]


def plot_metric_with_error_bars(
    df,
    metric: str,
    output_path,
    x_col: str = "n_components",
    group_col: str = "projector",
    y_col: str | None = None,
    err_col: str | None = None,
    title: str | None = None,
):
    y_col = y_col or f"{metric}_seed_mean_dataset_mean"
    err_col = err_col or f"{metric}_seed_mean_dataset_std"
    output_path = Path(output_path)

    required_cols = {x_col, group_col, y_col, err_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(8, 5))

    for group_value, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        ax.errorbar(
            group_df[x_col],
            group_df[y_col],
            yerr=group_df[err_col],
            marker="o",
            capsize=4,
            label=str(group_value),
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {x_col}")
    ax.legend(title=group_col)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def score_curve_task(embeddings, task_desc, folds, measurement_cfgs, projection_cache_dir=None):
    seed = make_seed(
        task_desc.base_seed,
        task_desc.projector_cfg.name,
        task_desc.classifier_cfg.name,
        task_desc.dataset_name,
        task_desc.n_components,
    )

    projector = ProjectorFactory.create(
        task_desc.projector_cfg.name,
        task_desc.n_components,
        seed,
        **task_desc.projector_cfg.params,
    )
    classifier = ClassifierFactory.create(
        task_desc.classifier_cfg.name,
        seed,
        **task_desc.classifier_cfg.params,
    )
    measurements = [
        (cfg.name, MeasurementFactory.create(cfg.name, seed, **cfg.params))
        for cfg in measurement_cfgs
    ]

    mean_scores = score_feature_space(
        embeddings,
        folds,
        classifier,
        measurements,
        projector,
        task_desc=task_desc,
        projection_cache_dir=projection_cache_dir,
    )

    metadata = {
        "classifier": task_desc.classifier_cfg.name,
        "dataset": task_desc.dataset_name,
        "projector": task_desc.projector_cfg.name,
        "n_components": task_desc.n_components,
        "base_seed": task_desc.base_seed,
        "seed": seed,
    }

    return metadata | mean_scores

def score_feature_space(
    embeddings,
    folds,
    classifier,
    measurements,
    projector=None,
    task_desc: TaskDescription | None = None,
    projection_cache_dir: Path | None = None,
):
    fold_scores = defaultdict(list)

    for fold_index, (train, valid) in enumerate(embeddings.get_folds(folds)):
        if projector:
            cache_path = None
            if projection_cache_dir is not None and task_desc is not None:
                cache_key = _projection_cache_key(
                    task_desc.dataset_name,
                    task_desc.projector_cfg.name,
                    task_desc.projector_cfg.params,
                    task_desc.n_components,
                    task_desc.base_seed,
                    fold_index,
                    embeddings.num_samples(),
                )
                cache_path = Path(projection_cache_dir) / f"{cache_key}.npz"

            if cache_path is not None and cache_path.exists():
                train_proj, valid_proj = _load_fold_projection(cache_path)
            else:
                projector.fit(train)
                train_proj = projector.transform(train)
                valid_proj = projector.transform(valid)
                if cache_path is not None:
                    _save_fold_projection(cache_path, train_proj, valid_proj)
        else:
            train_proj = train
            valid_proj = valid

        for name, measurement in measurements:
            fold_scores[name].append(
                float(measurement(train_proj, valid_proj, train, valid))
            )

        classifier.fit(train_proj)
        y_pred, y_scores = classifier.predict_and_score(valid_proj)

        for name, value in compute_classification_metrics(
            valid.labels, y_pred, y_scores
        ).items():
            fold_scores[name].append(float(value))

    mean_scores = {
        name: float(mean(values))
        for name, values in fold_scores.items()
        if values
    }
    
    return mean_scores

def score_unprojected_embeddings(embeddings, base_seed, classifier_cfg, dataset_name, folds, measurement_cfgs):
    seed = make_seed(
        base_seed,
        "noprojector",
        classifier_cfg.name,
        dataset_name,
        f"{embeddings.num_dimensions()}",
    )

    classifier = ClassifierFactory.create(
        classifier_cfg.name,
        seed,
        **classifier_cfg.params,
    )
    measurements = [
        (cfg.name, MeasurementFactory.create(cfg.name, seed, **cfg.params))
        for cfg in measurement_cfgs
    ]

    mean_scores = score_feature_space(embeddings, folds, classifier, measurements)

    metadata = {
        "classifier": classifier_cfg.name,
        "dataset": dataset_name,
        "n_components": embeddings.num_dimensions(),
        "base_seed": base_seed,
        "seed": seed,
    }

    return metadata | mean_scores

def aggregate_group(df, cols_group, spec):
    stats = df.groupby(cols_group, dropna=False)
    stats = stats.agg(**spec).reset_index()
    stats = stats.fillna(0.0)
    return stats


def aggregate_curve_results_across_seeds(results_df):
    group_cols = ["classifier", "dataset", "projector", "n_components"]

    value_cols = [
        col
        for col in results_df.columns
        if col not in {*group_cols, "base_seed", "seed"}
    ]

    spec = {
        f"{col}_seed_mean": (col, "mean")
        for col in value_cols
    } | {
        f"{col}_seed_std": (col, "std")
        for col in value_cols
    }

    return aggregate_group(results_df, group_cols, spec)


def aggregate_across_datasets(df_seed):
    group_cols = ["classifier", "projector", "n_components"]

    value_cols = [
        col
        for col in df_seed.columns
        if col not in {*group_cols, "dataset"}
    ]

    spec = {
        f"{col}_dataset_mean": (col, "mean")
        for col in value_cols
    } | {
        f"{col}_dataset_std": (col, "std")
        for col in value_cols
    }

    return aggregate_group(df_seed, group_cols, spec)


@click.command()
@click.option("--run-id", "-r", type=str, required=True, help="The id of a run to continue.")
@click.option("--force", is_flag=True, help="Recompute existing task outputs.")
@click.option(
    "--dataset-name",
    "dataset_names",
    multiple=True,
    help="Only process a dataset/encoder key, e.g. kather100k_uni. Can be repeated.",
)
@click.option(
    "--projector",
    "projector_names",
    multiple=True,
    help="Only process this projector. Can be repeated.",
)
@click.option(
    "--classifier",
    "classifier_names",
    multiple=True,
    help="Only process this classifier. Can be repeated.",
)
@click.option(
    "--num-shards",
    type=int,
    default=None,
    help="Split selected projector/classifier/dataset keys into this many shards.",
)
@click.option(
    "--shard-index",
    type=int,
    default=None,
    help="Run only this zero-based shard index.",
)
@click.option(
    "--projection-cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Cache projected train/validation folds under this directory.",
)
@click.option(
    "--tasks-only",
    is_flag=True,
    help="Write task JSON checkpoints only; skip curve summaries and plots.",
)
def estimate_curve(
    run_id: str,
    force: bool,
    dataset_names: tuple[str, ...],
    projector_names: tuple[str, ...],
    classifier_names: tuple[str, ...],
    num_shards: int | None,
    shard_index: int | None,
    projection_cache_dir: Path | None,
    tasks_only: bool,
):
    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    curve_paths = run_paths.curve()
    cfg = Config.from_yaml(run_paths.config())

    logger = setup_logging(run_paths.log_file())
    logger.info("Running the curve estimation step of the experiment.")
    logger.info(f"Run id is {run_id}.")
    logger.info(f"Force recomputation is {'enabled' if force else 'disabled'}.")

    all_dataset_names = [
        f"{dataset}_{encoder}"
        for dataset, encoder in product(cfg.datasets, cfg.encoders)
    ]
    selected_dataset_names = all_dataset_names
    if dataset_names:
        requested = list(dataset_names)
        unknown = sorted(set(requested) - set(all_dataset_names))
        if unknown:
            raise click.UsageError(
                "Unknown dataset/encoder key(s): "
                + ", ".join(unknown)
                + ". Valid keys are: "
                + ", ".join(all_dataset_names)
            )
        selected_dataset_names = [
            name for name in all_dataset_names if name in set(requested)
        ]

    selected_projector_names = list(cfg.curve.projectors)
    if projector_names:
        unknown = sorted(set(projector_names) - set(selected_projector_names))
        if unknown:
            raise click.UsageError(
                "Unknown projector(s): "
                + ", ".join(unknown)
                + ". Valid projectors are: "
                + ", ".join(selected_projector_names)
            )
        selected_projector_names = [
            name for name in selected_projector_names if name in set(projector_names)
        ]

    selected_classifier_names = list(cfg.curve.classifiers)
    if classifier_names:
        unknown = sorted(set(classifier_names) - set(selected_classifier_names))
        if unknown:
            raise click.UsageError(
                "Unknown classifier(s): "
                + ", ".join(unknown)
                + ". Valid classifiers are: "
                + ", ".join(selected_classifier_names)
            )
        selected_classifier_names = [
            name for name in selected_classifier_names if name in set(classifier_names)
        ]

    selected_work = list(
        product(selected_classifier_names, selected_projector_names, selected_dataset_names)
    )
    if (num_shards is None) != (shard_index is None):
        raise click.UsageError("--num-shards and --shard-index must be used together.")
    if num_shards is not None:
        if num_shards < 1:
            raise click.UsageError("--num-shards must be >= 1.")
        if shard_index < 0 or shard_index >= num_shards:
            raise click.UsageError("--shard-index must satisfy 0 <= index < num_shards.")
        selected_work = [
            item for idx, item in enumerate(selected_work) if idx % num_shards == shard_index
        ]

    if not selected_work:
        raise click.UsageError("No curve work selected for this shard.")

    selected_work_set = set(selected_work)
    selected_dataset_names = [
        name
        for name in selected_dataset_names
        if any(dataset_name == name for _, _, dataset_name in selected_work_set)
    ]
    selected_projector_names = [
        name
        for name in selected_projector_names
        if any(projector_name == name for _, projector_name, _ in selected_work_set)
    ]
    selected_classifier_names = [
        name
        for name in selected_classifier_names
        if any(classifier_name == name for classifier_name, _, _ in selected_work_set)
    ]

    logger.info(
        "Selected curve work: "
        + ", ".join(
            f"{classifier}/{projector}/{dataset}"
            for classifier, projector, dataset in selected_work
        )
    )
    if projection_cache_dir is not None:
        logger.info(f"Curve projection cache is enabled at {projection_cache_dir}.")
    if tasks_only:
        logger.info("Tasks-only mode is enabled; summary outputs will not be written.")

    train_embeddings = {
        dataset_name: paths.embedding(*dataset_name.rsplit("_", 1)).split("train").load()
        for dataset_name in selected_dataset_names
    }

    if cfg.curve.subsample:
        # Keep the sampled training set stable so projection-cache keys refer to
        # the same fold contents across retries/resubmissions.
        cp.random.seed(42)
        train_embeddings = {
            name: embeddings.stratified_sample(cfg.curve.subsample)
            for name, embeddings in train_embeddings.items()
        }

    dataset_folds = {
        name: generate_stratified_folds(embeddings, n_splits=cfg.num_folds, seed=42)
        for name, embeddings in train_embeddings.items()
    }

    projector_cfgs = build_projector_configs(cfg, paths, run_id, logger)
    classifier_cfgs = build_classifier_configs(cfg, paths, run_id, logger)
    projector_cfgs = [
        projector_cfg
        for projector_cfg in projector_cfgs
        if projector_cfg.name in set(selected_projector_names)
    ]
    classifier_cfgs = [
        classifier_cfg
        for classifier_cfg in classifier_cfgs
        if classifier_cfg.name in set(selected_classifier_names)
    ]

    curve_paths.mkdir()
    curve_paths.summaries().mkdir()
    curve_paths.plots().mkdir()

    client, cluster = make_dask_client()

    try:
        start_time = perf_counter()

        for classifier_cfg in classifier_cfgs:
            classifier_results = []

            for projector_cfg in projector_cfgs:
                output_path = curve_paths.projector_classifier(
                    projector_cfg.name,
                    classifier_cfg.name,
                )
                output_path.mkdir()
                tasks_dir = output_path.tasks()
                tasks_dir.mkdir()

                unprojected_scores_list = []

                for dataset_name, embeddings in train_embeddings.items():
                    if (
                        classifier_cfg.name,
                        projector_cfg.name,
                        dataset_name,
                    ) not in selected_work_set:
                        continue

                    folds = dataset_folds[dataset_name]
                    embeddings_future = client.scatter(embeddings, broadcast=True)
                    desc = f"{classifier_cfg.name} | {projector_cfg.name} | {dataset_name}"

                    if not tasks_only:
                        for base_seed in cfg.seeds.curve:
                            unprojected_scores = score_unprojected_embeddings(
                                embeddings,
                                base_seed,
                                classifier_cfg,
                                dataset_name,
                                folds,
                                cfg.curve.measurements,
                            )
                            unprojected_scores_list.append(unprojected_scores)

                    # optionally - assess at the full number of dimensions (but projected)
                    dimensions = cfg.curve.dimensions
                    if cfg.curve.include_full_dimension:
                        dimensions.append(embeddings.num_dimensions())

                    pending_tasks = {}

                    for n_components in dimensions:
                        for base_seed in cfg.seeds.curve:
                            task_desc = TaskDescription(
                                projector_cfg=projector_cfg,
                                classifier_cfg=classifier_cfg,
                                dataset_name=dataset_name,
                                n_components=n_components,
                                base_seed=base_seed,
                            )
                            task_file = tasks_dir.task(task_desc.key())

                            if task_file.exists() and not force:
                                continue

                            future = client.submit(
                                score_curve_task,
                                embeddings_future,
                                task_desc,
                                folds,
                                cfg.curve.measurements,
                                projection_cache_dir,
                                pure=False,
                            )
                            pending_tasks[future] = task_file

                    logger.info(f"{desc}: {len(pending_tasks)} pending task(s)")

                    for future in tqdm(
                        as_completed(pending_tasks),
                        total=len(pending_tasks),
                        desc=desc,
                    ):
                        atomic_write_json(pending_tasks[future], future.result())

                if tasks_only:
                    logger.info(
                        f"Tasks-only mode: skipping summaries for "
                        f"{projector_cfg.name}/{classifier_cfg.name}."
                    )
                    continue

                unprojected_scores_df = pd.DataFrame(unprojected_scores_list)
                if output_path.unprojected_scores().exists() and dataset_names:
                    previous_unprojected_scores_df = pd.read_csv(
                        output_path.unprojected_scores()
                    )
                    previous_unprojected_scores_df = previous_unprojected_scores_df[
                        ~previous_unprojected_scores_df["dataset"].isin(
                            selected_dataset_names
                        )
                    ]
                    unprojected_scores_df = pd.concat(
                        [previous_unprojected_scores_df, unprojected_scores_df],
                        ignore_index=True,
                    )
                unprojected_scores_df.to_csv(output_path.unprojected_scores())

                results_df = load_task_results(output_path.tasks())
                classifier_results.append(results_df)

                results_df.to_csv(output_path.results(), index=False)
                logger.info(f"Writing results csv to {output_path.results()}")

                seed_summary_df = aggregate_curve_results_across_seeds(results_df)
                seed_summary_df.to_csv(output_path.seed_summary_results(), index=False)
                logger.info(
                    f"Writing summary results csv to {output_path.seed_summary_results()}"
                )

                if cfg.thresholds.metrics:
                    all_thresholds = []
                    for dataset_name in train_embeddings:
                        dataset, encoder = dataset_name.split('_', 1)
                        df_per_dataset = seed_summary_df[seed_summary_df['dataset'] == dataset_name]
                        thresholds = compute_thresholds(df_per_dataset, unprojected_scores_df, cfg.thresholds, projector_cfg.name, classifier_cfg.name, dataset, encoder, dataset_name)
                        all_thresholds.extend(thresholds)
                    thresholds_df = pd.DataFrame(all_thresholds)
                    thresholds_df.to_csv(output_path.thresholds_results(), index=False)
                    logger.info(f"Writing thresholds results csv to {output_path.thresholds_results()}")

                dataset_summary_df = aggregate_across_datasets(seed_summary_df)
                dataset_summary_df.to_csv(output_path.dataset_summary_results(), index=False)
                logger.info(
                    f"Writing summary results csv to {output_path.dataset_summary_results()}"
                )

            if tasks_only:
                continue

            classifier_results_df = pd.concat(classifier_results, ignore_index=True)

            classifier_seed_summary_df = aggregate_curve_results_across_seeds(
                classifier_results_df
            )
            classifier_dataset_summary_df = aggregate_across_datasets(
                classifier_seed_summary_df
            )

            classifier_seed_summary_path = curve_paths.summaries().classifier_seed_summary(
                classifier_cfg.name
            )
            classifier_dataset_summary_path = (
                curve_paths.summaries().classifier_dataset_summary(classifier_cfg.name)
            )

            classifier_seed_summary_df.to_csv(classifier_seed_summary_path, index=False)
            classifier_dataset_summary_df.to_csv(
                classifier_dataset_summary_path, index=False
            )

            logger.info(f"Writing classifier seed summary to {classifier_seed_summary_path}")
            logger.info(
                f"Writing classifier dataset summary to {classifier_dataset_summary_path}"
            )

            accuracy_plot_path = curve_paths.plots().classifier_accuracy(
                classifier_cfg.name
            )
            roc_auc_plot_path = curve_paths.plots().classifier_roc_auc(
                classifier_cfg.name
            )

            plot_metric_with_error_bars(
                df=classifier_dataset_summary_df,
                metric="accuracy",
                output_path=accuracy_plot_path,
                title=f"{classifier_cfg.name}: accuracy vs n_components",
            )
            logger.info(f"Wrote accuracy plot to {accuracy_plot_path}")

            plot_metric_with_error_bars(
                df=classifier_dataset_summary_df,
                metric="roc_auc",
                output_path=roc_auc_plot_path,
                title=f"{classifier_cfg.name}: roc_auc vs n_components",
            )
            logger.info(f"Wrote roc_auc plot to {roc_auc_plot_path}")

        elapsed_time = perf_counter() - start_time
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Run output saved to: {run_paths.root}")
        append_stage_time(run_paths.root, "estimate_curve", elapsed_time)

    finally:
        client.close()
        cluster.close()
