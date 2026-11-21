import typer

from .export import export_cmd
from .embed import embed
from .list import list_pipelines
from .plan import plan
from .preview import preview
from .run import run
from .patchmap import patchmap

app = typer.Typer(help="histokit — histopathology toolkit CLI.")
app.command(name="list")(list_pipelines)
app.command(name="plan")(plan)
app.command(name="run")(run)
app.command(name="preview")(preview)
app.command(name="export")(export_cmd)
app.command(name="embed")(embed)
app.command(name="patchmap")(patchmap)

def main() -> None:
    app()
