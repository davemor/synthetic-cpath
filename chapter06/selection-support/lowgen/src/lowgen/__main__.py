import click

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

from .dgx.cli import dgx_cli

try:
    from .cli import synth
except ImportError:
    synth = None

load_dotenv()


@click.group(
    help="CLI tool for generating synthetic low-dimensional datasets and evaluating them."
)
def cli():
    pass


if synth is not None:
    cli.add_command(synth)
cli.add_command(dgx_cli)


def main():
    cli(prog_name="hproj")


if __name__ == "__main__":
    main()
