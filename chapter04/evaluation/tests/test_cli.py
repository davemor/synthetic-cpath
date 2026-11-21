from typer.testing import CliRunner

from synpath.cli.main import app


def test_plan_command() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plan", "--config", "configs/experiments/main.yaml"],
    )
    assert result.exit_code == 0
    assert "synpath experiment plan" in result.stdout


def test_run_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run an experiment end-to-end" in result.stdout
