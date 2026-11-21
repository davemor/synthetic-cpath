from time import perf_counter

import click
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from skdim.id import MLE, DANCo, TwoNN

from hproj.data.paths import Paths
from hproj.util.config import Config
from hproj.util.logging import setup_logging
from hproj.util.timing import append_stage_time


def chavez_id(X):
    D = pdist(X)
    mu = np.mean(D)
    var = np.var(D)
    return mu**2 / (2 * var)


@click.command()
@click.option(
    "--run-id", "-r", type=str, required=True, help="The id of a run to continue."
)
@click.option(
    "--force", is_flag=True, help="Overwrite existing intrinsic dimensions output."
)
@click.option(
    "--dataset-name",
    "dataset_names",
    multiple=True,
    help="Restrict estimation to one or more dataset names from the run config.",
)
@click.option(
    "--append",
    is_flag=True,
    help="Append/update selected dataset rows in an existing intrinsic dimensions CSV.",
)
def intrinsic_dims(
    run_id: str, force: bool, dataset_names: tuple[str, ...], append: bool
):
    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    cfg = Config.from_yaml(run_paths.config())

    subsample = cfg.intrinsic_dims.subsample

    logger = setup_logging(run_paths.log_file())
    logger.info("Running intrinsic dimensionality estimation for run.")
    logger.info(f"Run id is {run_id}.")

    output_path = run_paths.intrinsic_dims()
    if output_path.exists() and not force and not append:
        logger.info(
            f"Intrinsic dimensions output already exists at {output_path}, "
            "use --force to overwrite."
        )
        return

    rows = []
    existing_df = None
    selected_datasets = list(dataset_names) if dataset_names else list(cfg.datasets)
    unknown_datasets = sorted(set(selected_datasets) - set(cfg.datasets))
    if unknown_datasets:
        raise click.ClickException(
            f"Unknown dataset(s) for run config: {', '.join(unknown_datasets)}"
        )

    if append and output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_df = existing_df[~existing_df["dataset"].isin(selected_datasets)]

    start_time = perf_counter()

    for dataset in selected_datasets:
        for encoder in cfg.encoders:
            embedding_path = paths.embedding(dataset, encoder).split("train")
            logger.info(
                f"Loading embeddings for {dataset}/{encoder}: {embedding_path.root}"
            )
            feature_space = embedding_path.load()
            features = feature_space.features

            if subsample is not None and len(features) > subsample:
                indices = np.random.default_rng(42).choice(
                    len(features), size=subsample, replace=False
                )
                features = features[indices]

            # skdim MLE expects numpy arrays
            if hasattr(features, "get"):
                features_np = features.get()  # cupy array
            else:
                features_np = np.asarray(features)

            estimator = MLE()
            estimator.fit(features_np)
            estimated_mle = float(estimator.dimension_)

            estimator = DANCo()
            estimator.fit(features_np)
            estimated_danco = float(estimator.dimension_)

            estimator = TwoNN()
            estimator.fit(features_np)
            estimated_twonn = float(estimator.dimension_)

            estimated_chavez = float(chavez_id(features_np))

            num_dimensions = feature_space.num_dimensions()

            rows.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "num_dimensions": int(num_dimensions),
                    "n_samples": len(features_np),
                    "intrinsic_dim_mle": estimated_mle,
                    "intrinsic_dim_danco": estimated_danco,
                    "intrinsic_dim_twonn": estimated_twonn,
                    "intrinsic_dim_chavez": estimated_chavez,
                }
            )

            logger.info(
                f"{dataset}/{encoder}: D={int(num_dimensions)}, "
                f"mle={estimated_mle:.2f}, "
                f"danco={estimated_danco:.2f}, twonn={estimated_twonn:.2f}, "
                f"chavez={estimated_chavez:.2f}, n_samples={len(features_np)}"
            )

    output_df = pd.DataFrame(rows)
    if existing_df is not None:
        output_df = pd.concat([existing_df, output_df], ignore_index=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Intrinsic dimensionality results written to {output_path}")

    elapsed_time = perf_counter() - start_time
    append_stage_time(run_paths.root, "intrinsic_dims", elapsed_time)
