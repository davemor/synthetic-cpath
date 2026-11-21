from pathlib import Path
import shutil

import click

from lowgen.data.paths import Paths
from lowgen.pipeline.runner import ExperimentRunner
from lowgen.util.config import Config
from lowgen.util.logging import setup_logging
from lowgen.util.runs import generate_run_id


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML configuration file.",
    default=None,
)
@click.option(
    "--run-id", "-r", type=str, help="The id of a run to continue.", default=None
)
@click.option("--force", is_flag=True, help="Recompute existing task outputs.")
def synth(config: Path, run_id: str, force: bool):
    # set up the run_id
    assert not (
        config and run_id
    ), "Use --config for a fresh run and --run-id to continue an existing run."
    paths = Paths.from_env()
    if run_id:
        run_path = paths.run(run_id)
        cfg = Config.from_yaml(run_path.config())  # load the config from the run dir
    else:
        run_id = generate_run_id("lowgen", config)  # create a new run id
        run_path = paths.run(run_id)
        run_path.mkdir()  # make sure the output dir exists
        shutil.copy(
            config, run_path.config()
        )  # copy the config into the new runs output directory
        cfg = Config.from_yaml(config)  # load the config from the config path given

    # set up the logger
    logger = setup_logging(paths.run(run_id).log_file())
    logger.info("Running synthesis step of the lowgen pipeline.")
    logger.info(f"Run id is {run_id}.")
    logger.info(f"Force recomputation is {'enabled' if force else 'disabled'}.")

    runner = ExperimentRunner(cfg=cfg, paths=paths, run_id=run_id, force=force, logger=logger)
    runner.run()
