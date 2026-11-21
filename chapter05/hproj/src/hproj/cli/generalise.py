import hashlib
import json
import os
import uuid
from itertools import product
from pathlib import Path
from time import perf_counter

import click
import cupy as cp
import numpy as np
import pandas as pd
from dask.distributed import as_completed
from sklearn.model_selection import ParameterSampler
from tqdm.auto import tqdm

from hproj.classifiers.classifier import ClassifierFactory
from hproj.classifiers.metrics import compute_classification_metrics
from hproj.data.feature_space import FeatureSpace
from hproj.data.folds import generate_stratified_folds
from hproj.data.paths import Paths
from hproj.projectors.projector import ProjectorFactory
from hproj.util.config import Config
from hproj.util.dask import make_dask_client
from hproj.util.hyperparams import make_param_grid
from hproj.util.logging import setup_logging
from hproj.util.seeds import make_seed
from hproj.util.timing import append_stage_time


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _task_id(task: dict) -> str:
    payload = json.dumps(task, sort_keys=True, default=_json_default)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{task['dataset']}_{task['encoder']}_{task['projector_name']}_{task['classifier_name']}_{task['n_components']}_{task['base_seed']}_{digest}"


def _grid_digest(grid: list[dict]) -> str:
    payload = json.dumps(grid, sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _make_classifier_param_grid(clf_cfg) -> list[dict]:
    search = getattr(clf_cfg, "search", None)
    method = getattr(search, "method", "grid")
    if method == "grid":
        return make_param_grid(clf_cfg.params)
    if method == "random":
        n_iter = getattr(search, "n_iter", None)
        if n_iter is None:
            raise click.ClickException(
                f"Random search for classifier {clf_cfg.name} requires n_iter."
            )
        random_state = getattr(search, "random_state", 0)
        return list(
            ParameterSampler(
                clf_cfg.params,
                n_iter=int(n_iter),
                random_state=int(random_state),
            )
        )
    raise click.ClickException(
        f"Unsupported search method {method!r} for classifier {clf_cfg.name}."
    )


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, default=_json_default)
    tmp_path.replace(path)


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default))
            f.write("\n")
    tmp_path.replace(path)


def _manifest_record(index: int, task: dict) -> dict:
    return {
        "index": index,
        "task_id": task["task_id"],
        "task_path": str(task["task_path"]),
        "dataset": task["dataset"],
        "encoder": task["encoder"],
        "n_components": task["n_components"],
        "projector_name": task["projector_name"],
        "projector_params": task["projector_params"],
        "classifier_name": task["classifier_name"],
        "classifier_grid_digest": task["classifier_grid_digest"],
        "base_seed": task["base_seed"],
        "use_projector": task["use_projector"],
    }


def _load_manifest(path: Path, tasks_dir: Path) -> list[dict]:
    work_items = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            task = {
                "dataset": record["dataset"],
                "encoder": record["encoder"],
                "ds_enc": f"{record['dataset']}_{record['encoder']}",
                "n_components": int(record["n_components"]),
                "projector_name": record["projector_name"],
                "projector_params": record["projector_params"],
                "classifier_name": record["classifier_name"],
                "classifier_grid_digest": record.get("classifier_grid_digest", "legacy"),
                "base_seed": int(record["base_seed"]),
                "use_projector": bool(record["use_projector"]),
                "task_id": record["task_id"],
                "task_path": tasks_dir / f"{record['task_id']}.json",
            }
            work_items.append(task)
    return work_items


def _select_task_indexes(
    total_tasks: int,
    num_shards: int,
    shard_index: int,
    task_index: int | None,
    task_batch_size: int,
) -> set[int]:
    if task_index is None:
        return {
            idx for idx in range(total_tasks) if idx % num_shards == shard_index
        }

    start = task_index * task_batch_size
    stop = min(start + task_batch_size, total_tasks)
    if start >= total_tasks:
        return set()
    return set(range(start, stop))


def _projection_cache_key(
    dataset_name: str,
    encoder: str,
    n_components: int,
    projector_name: str,
    projector_params: dict,
    base_seed: int,
    use_projector: bool,
) -> str:
    payload = {
        "version": 1,
        "dataset": dataset_name,
        "encoder": encoder,
        "n_components": n_components,
        "projector_name": projector_name,
        "projector_params": projector_params,
        "base_seed": base_seed,
        "use_projector": use_projector,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()[:16]


def _save_feature_space(path: Path, train_data: FeatureSpace, test_data: FeatureSpace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("wb") as f:
        np.savez(
            f,
            train_features=cp.asnumpy(train_data.features),
            train_labels=cp.asnumpy(train_data.labels),
            test_features=cp.asnumpy(test_data.features),
            test_labels=cp.asnumpy(test_data.labels),
        )
    tmp_path.replace(path)


def _load_feature_space(path: Path) -> tuple[FeatureSpace, FeatureSpace]:
    with np.load(path) as data:
        train = FeatureSpace(
            cp.asarray(data["train_features"]),
            cp.asarray(data["train_labels"]),
        )
        test = FeatureSpace(
            cp.asarray(data["test_features"]),
            cp.asarray(data["test_labels"]),
        )
    return train, test


def _project_train_test(
    train_embeddings: FeatureSpace,
    test_embeddings: FeatureSpace,
    dataset_name: str,
    encoder: str,
    n_components: int,
    projector_name: str,
    projector_params: dict,
    base_seed: int,
    use_projector: bool,
    projection_cache_dir: Path | None,
) -> tuple[FeatureSpace, FeatureSpace, bool]:
    if not use_projector:
        return train_embeddings, test_embeddings, False

    seed = make_seed(
        base_seed,
        dataset_name,
        encoder,
        n_components,
        projector_name,
    )
    cache_path = None
    if projection_cache_dir is not None:
        key = _projection_cache_key(
            dataset_name,
            encoder,
            n_components,
            projector_name,
            projector_params,
            base_seed,
            use_projector,
        )
        cache_path = projection_cache_dir / f"{key}.npz"
        if cache_path.exists():
            train_data, test_data = _load_feature_space(cache_path)
            return train_data, test_data, True

    projector = ProjectorFactory.create(
        projector_name, n_components, seed, **projector_params
    )
    projector.fit(train_embeddings)
    train_data = projector.transform(train_embeddings)
    test_data = projector.transform(test_embeddings)
    if cache_path is not None:
        _save_feature_space(cache_path, train_data, test_data)
    return train_data, test_data, False


def evaluate_generalisation_task(
    train_embeddings,
    test_embeddings,
    folds,
    dataset_name: str,
    encoder: str,
    n_components: int,
    projector_name: str,
    projector_params: dict,
    classifier_name: str,
    classifier_param_grid: list[dict],
    base_seed: int,
    select_metric: str,
    use_projector: bool = True,
    projection_cache_dir: Path | None = None,
):
    """
    Evaluate a projector/classifier combination at a single dimension.

    1. Project (or skip for D) train and test data.
    2. Tune classifier hyperparameters via cross-validation on training folds.
    3. Retrain on full projected training set with best params.
    4. Evaluate on projected test set.
    """
    from sklearn.metrics import confusion_matrix

    # step 1: project data (or use raw embeddings for full dimension D)
    train_data, test_data, projection_cache_hit = _project_train_test(
        train_embeddings,
        test_embeddings,
        dataset_name,
        encoder,
        n_components,
        projector_name,
        projector_params,
        base_seed,
        use_projector,
        projection_cache_dir,
    )

    seed = make_seed(
        base_seed,
        dataset_name,
        encoder,
        n_components,
        projector_name,
        classifier_name,
    )

    # step 2: tune classifier hyperparameters via cross-validation
    best_score = -float("inf")
    best_params = classifier_param_grid[0] if classifier_param_grid else {}

    for params in classifier_param_grid:
        fold_scores = []
        for train_idx, val_idx in folds:
            train_fold = FeatureSpace(train_data.features[train_idx], train_data.labels[train_idx])
            val_fold = FeatureSpace(train_data.features[val_idx], train_data.labels[val_idx])

            clf = ClassifierFactory.create(classifier_name, seed, **params)
            clf.fit(train_fold)
            y_pred, y_scores = clf.predict_and_score(val_fold)
            metrics = compute_classification_metrics(val_fold.labels, y_pred, y_scores)
            fold_scores.append(metrics.get(select_metric, 0.0))

        mean_score = sum(fold_scores) / len(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # step 3: retrain on full projected training set with best params
    final_clf = ClassifierFactory.create(classifier_name, seed, **best_params)
    final_clf.fit(train_data)

    # step 4: evaluate on projected test set
    y_pred, y_scores = final_clf.predict_and_score(test_data)
    metrics = compute_classification_metrics(test_data.labels, y_pred, y_scores)
    cm = confusion_matrix(
        cp.asnumpy(test_data.labels), cp.asnumpy(y_pred)
    ).tolist()

    return {
        "dataset": dataset_name,
        "encoder": encoder,
        "n_components": n_components,
        "projector": projector_name if use_projector else "none",
        "classifier": classifier_name,
        "best_params": str(best_params),
        "cv_score": best_score,
        "base_seed": base_seed,
        "projection_cache_hit": projection_cache_hit,
        **metrics,
        "confusion_matrix": cm,
    }


def evaluate_generalisation_task_timed(*args, **kwargs):
    started = perf_counter()
    result = evaluate_generalisation_task(*args, **kwargs)
    result["task_runtime_seconds"] = perf_counter() - started
    return result


@click.command()
@click.option(
    "--run-id", "-r", type=str, required=True, help="The id of a run to evaluate.",
)
@click.option("--force", is_flag=True, help="Recompute existing generalisation outputs.")
@click.option("--num-shards", type=int, default=1, show_default=True, help="Number of shards for task execution.")
@click.option("--shard-index", type=int, default=0, show_default=True, help="Shard index for task execution.")
@click.option("--task-index", type=int, default=None, help="Contiguous small-batch index to execute.")
@click.option("--task-batch-size", type=int, default=1, show_default=True, help="Number of manifest tasks per --task-index.")
@click.option("--task-manifest", type=click.Path(path_type=Path), default=None, help="Write the generated task manifest to this JSONL path.")
@click.option("--manifest-only", is_flag=True, help="Write the task manifest and exit without running tasks.")
@click.option("--projection-cache-dir", type=click.Path(path_type=Path), default=None, help="Cache projected train/test features under this directory.")
@click.option("--tasks-only", is_flag=True, help="Write per-task outputs only and skip final summary/report outputs.")
def generalise(
    run_id: str,
    force: bool,
    num_shards: int,
    shard_index: int,
    task_index: int | None,
    task_batch_size: int,
    task_manifest: Path | None,
    manifest_only: bool,
    projection_cache_dir: Path | None,
    tasks_only: bool,
):
    if num_shards < 1:
        raise click.ClickException("--num-shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise click.ClickException("--shard-index must be between 0 and --num-shards - 1.")
    if task_index is not None and task_index < 0:
        raise click.ClickException("--task-index must be at least 0.")
    if task_batch_size < 1:
        raise click.ClickException("--task-batch-size must be at least 1.")

    paths = Paths.from_env()
    cfg = Config.from_yaml(paths.run(run_id).config())

    logger = setup_logging(paths.run(run_id).log_file())
    logger.info("Running the generalisation evaluation step.")
    logger.info(f"Run id is {run_id}.")
    logger.info(f"Force recomputation is {'enabled' if force else 'disabled'}.")
    if task_index is not None:
        logger.info(
            f"Selected generalisation task batch {task_index} "
            f"with batch size {task_batch_size}."
        )
    if num_shards > 1:
        logger.info(f"Selected generalisation shard {shard_index} of {num_shards}.")
    if task_manifest is not None:
        logger.info(f"Writing generalisation task manifest to {task_manifest}.")
    if manifest_only:
        logger.info("Manifest-only mode is enabled; no tasks will be executed.")
    if projection_cache_dir is not None:
        logger.info(f"Projection cache is enabled at {projection_cache_dir}.")
    if tasks_only:
        logger.info("Tasks-only mode is enabled; summary outputs will not be written.")

    run_root = paths.run(run_id).root
    output_path = run_root / "generalisation_results.csv"
    tasks_dir = run_root / "generalisation" / "tasks"
    if output_path.exists() and not force:
        logger.info(f"Generalisation output already exists at {output_path}, use --force to overwrite.")
        return
    if force and output_path.exists():
        output_path.unlink()

    # ---- load intrinsic dimensionality (d_ID) from MLE estimates ----
    id_path = paths.run(run_id).intrinsic_dims()
    intrinsic_dims = {}
    if id_path.exists():
        id_df = pd.read_csv(id_path)
        intrinsic_dims = {
            (row["dataset"], row["encoder"]): int(row["intrinsic_dim_mle"])
            for _, row in id_df.iterrows()
        }

    # ---- load threshold dimensions from curve results ----
    thresh_path = paths.run(run_id).curve().root / "thresholds_results.csv"
    thresh_dims = {}
    if thresh_path.exists():
        thresh_df = pd.read_csv(thresh_path)
        for _, row in thresh_df.iterrows():
            if pd.isna(row["achieved_dim"]):
                continue
            k = (
                row["dataset"],
                row["encoder"],
                row["projector"],
                row["classifier"],
            )
            thresh_dims.setdefault(k, set()).add(int(row["achieved_dim"]))

    # ---- determine select metric for classifier cv ----
    select_metric = cfg.calibration.classifier.select or "accuracy"

    # ---- build classifier param grids from generalisation config ----
    classifier_grids = {}
    for clf_cfg in cfg.generalisation.classifiers:
        classifier_grids[clf_cfg.name] = _make_classifier_param_grid(clf_cfg)

    # ---- build projector param grids from generalisation config ----
    projector_grids = {}
    for proj_cfg in cfg.generalisation.projectors:
        projector_grids[proj_cfg.name] = make_param_grid(proj_cfg.params)

    read_manifest = (
        task_manifest is not None
        and task_manifest.exists()
        and not manifest_only
    )
    train_embeddings = {}
    if read_manifest:
        work_items = _load_manifest(task_manifest, tasks_dir)
        logger.info(f"Loaded {len(work_items)} generalisation manifest records.")
    else:
        # ---- load embeddings ----
        train_embeddings = {
            f"{ds}_{enc}": paths.embedding(ds, enc).split("train").load()
            for ds, enc in product(cfg.datasets, cfg.encoders)
        }

        # ---- build explicit task list ----
        work_items = []
        for ds_enc, train_emb in train_embeddings.items():
            dataset, encoder = ds_enc.split("_", 1)
            D = train_emb.num_dimensions()
            d_id = intrinsic_dims.get((dataset, encoder), D)

            for proj_cfg in cfg.generalisation.projectors:
                proj_name = proj_cfg.name
                proj_param_list = projector_grids[proj_name]

                for clf_cfg in cfg.generalisation.classifiers:
                    clf_name = clf_cfg.name
                    clf_param_grid = classifier_grids[clf_name]

                    dims = {1, d_id, D}
                    dims.update(
                        thresh_dims.get((dataset, encoder, proj_name, clf_name), set())
                    )

                    for n_components in sorted(dims):
                        use_projector = n_components != D
                        proj_params_iter = proj_param_list if use_projector else [{}]

                        for proj_params in proj_params_iter:
                            for base_seed in cfg.seeds.evaluation:
                                task = {
                                    "dataset": dataset,
                                    "encoder": encoder,
                                    "ds_enc": ds_enc,
                                    "n_components": int(n_components),
                                    "projector_name": proj_name,
                                    "projector_params": proj_params,
                                    "classifier_name": clf_name,
                                    "classifier_grid_digest": _grid_digest(
                                        clf_param_grid
                                    ),
                                    "base_seed": int(base_seed),
                                    "use_projector": bool(use_projector),
                                }
                                task["task_id"] = _task_id(task)
                                task["task_path"] = tasks_dir / f"{task['task_id']}.json"
                                work_items.append(task)

    all_count = len(work_items)
    if task_manifest is not None and not read_manifest:
        _write_jsonl_atomic(
            task_manifest,
            [_manifest_record(idx, task) for idx, task in enumerate(work_items)],
        )
        logger.info(f"Wrote {all_count} generalisation manifest records.")
    if manifest_only:
        return

    selected_indexes = _select_task_indexes(
        all_count,
        num_shards,
        shard_index,
        task_index,
        task_batch_size,
    )
    selected_items = [
        task
        for idx, task in enumerate(work_items)
        if idx in selected_indexes
    ]
    pending_items = [
        task
        for task in selected_items
        if force or not task["task_path"].exists()
    ]
    logger.info(
        f"Generalisation tasks: total={all_count}, selected={len(selected_items)}, "
        f"pending={len(pending_items)}"
    )
    if tasks_only and not pending_items:
        logger.info("Tasks-only mode: no pending tasks for this selection.")
        return

    needed_ds_enc = sorted({task["ds_enc"] for task in pending_items})
    for ds_enc in needed_ds_enc:
        if ds_enc not in train_embeddings:
            dataset, encoder = ds_enc.split("_", 1)
            train_embeddings[ds_enc] = paths.embedding(dataset, encoder).split("train").load()
    test_embeddings = {
        ds_enc: paths.embedding(*ds_enc.split("_", 1)).split("test").load()
        for ds_enc in needed_ds_enc
    }
    folds_map = {
        key: generate_stratified_folds(train_embeddings[key], cfg.num_folds, seed=0)
        for key in needed_ds_enc
    }

    # ---- set up dask ----
    client, cluster = make_dask_client()

    try:
        start_time = perf_counter()
        futures = []

        for task in pending_items:
            ds_enc = task["ds_enc"]
            train_emb = train_embeddings[ds_enc]
            test_emb = test_embeddings[ds_enc]
            folds = folds_map[ds_enc]
            train_f = client.scatter(train_emb, broadcast=True)
            test_f = client.scatter(test_emb, broadcast=True)
            folds_f = client.scatter(folds, broadcast=True)

            future = client.submit(
                evaluate_generalisation_task_timed,
                train_f,
                test_f,
                folds_f,
                task["dataset"],
                task["encoder"],
                task["n_components"],
                task["projector_name"],
                task["projector_params"],
                task["classifier_name"],
                classifier_grids[task["classifier_name"]],
                task["base_seed"],
                select_metric,
                task["use_projector"],
                projection_cache_dir,
                pure=False,
            )
            future.task_path = task["task_path"]
            futures.append(future)

        # ---- collect results ----
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating generalisation",
        ):
            result = future.result()
            _write_json_atomic(future.task_path, result)
            logger.info(
                f"Generalisation task written to {future.task_path} "
                f"in {result['task_runtime_seconds']:.2f} seconds"
            )

        if tasks_only:
            logger.info("Tasks-only mode: skipping generalisation summary.")
            elapsed = perf_counter() - start_time
            logger.info(f"Total time: {elapsed:.2f} seconds")
            append_stage_time(paths.run(run_id).root, "generalisation_tasks", elapsed)
            return

        expected_paths = [task["task_path"] for task in work_items]
        missing_paths = [path for path in expected_paths if not path.exists()]
        if missing_paths:
            raise click.ClickException(
                f"Cannot write generalisation summary; {len(missing_paths)} task file(s) are missing."
            )

        # ---- save ----
        all_results = []
        for path in expected_paths:
            with path.open() as f:
                all_results.append(json.load(f))
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Generalisation results saved to {output_path}")

        elapsed = perf_counter() - start_time
        logger.info(f"Total time: {elapsed:.2f} seconds")
        append_stage_time(paths.run(run_id).root, "generalisation", elapsed)

    finally:
        try:
            client.close()
        except Exception as exc:
            logger.warning(f"Ignoring Dask client close failure: {exc}")
        try:
            cluster.close()
        except Exception as exc:
            logger.warning(f"Ignoring Dask cluster close failure: {exc}")
