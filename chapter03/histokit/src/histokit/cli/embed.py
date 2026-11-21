"""histokit embed - encode patches from saved PatchSet directories."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .helpers import load_dataset


def embed(
    patchsets: Annotated[
        Path,
        typer.Argument(
            help=(
                "Saved PatchSet directory, or a directory containing saved "
                "PatchSet directories."
            ),
        ),
    ],
    index: Annotated[
        Path,
        typer.Option("--index", help="Path to dataset index CSV."),
    ],
    labels: Annotated[
        Path,
        typer.Option("--labels", help="Path to dataset labels JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for embeddings."),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model alias or reference, e.g. resnet18, "
                "hf:facebook/dinov2-small, or "
                "timm:hf-hub:1aurent/vit_small_patch16_224.lunit_dino."
            ),
        ),
    ] = "resnet18",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Number of patches per model batch."),
    ] = 64,
    device: Annotated[
        str,
        typer.Option(
            "--device", help="Device: auto, cpu, cuda, cuda:0, or mps."
        ),
    ] = "auto",
    only_kept: Annotated[
        bool,
        typer.Option(
            "--only-kept/--all-patches",
            help="Embed only patches with keep=True when the column exists.",
        ),
    ] = True,
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            help="Dotenv file containing HF_TOKEN or HUGGING_FACE_HUB_TOKEN.",
        ),
    ] = Path(".env"),
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite", help="Overwrite existing embedding outputs."
        ),
    ] = False,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", help="Optional maximum number of patches."),
    ] = None,
) -> None:
    """Embed patch images from one or more saved PatchSets."""
    from histokit.embeddings.encoder import embed_patchset
    from histokit.embeddings.io import (
        discover_patchset_dirs,
        is_patchset_dir,
        safe_model_name,
        write_embedding_result,
    )
    from histokit.patchset.patchset import PatchSet

    try:
        patchset_dirs = discover_patchset_dirs(patchsets)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    if batch_size < 1:
        typer.echo("--batch-size must be at least 1")
        raise typer.Exit(1)

    single_patchset = is_patchset_dir(patchsets)
    if not single_patchset:
        output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(index, labels)

    typer.echo(f"PatchSets: {len(patchset_dirs)}")
    typer.echo(f"Model:     {model}")
    typer.echo(f"Device:    {device}")
    typer.echo(f"Output:    {output}\n")

    for patchset_dir in patchset_dirs:
        target = output
        if not single_patchset:
            target = output / (
                f"{patchset_dir.stem}.{safe_model_name(model)}.embedding"
            )

        typer.echo(f"Loading {patchset_dir}")
        patchset = PatchSet.load(patchset_dir, dataset)
        n_patches = len(patchset.frame)
        typer.echo(f"Embedding {n_patches} patch row(s) -> {target}")

        try:
            result = embed_patchset(
                patchset,
                model,
                batch_size=batch_size,
                device=device,
                only_kept=only_kept,
                env_file=env_file,
                limit=limit,
            )
            write_embedding_result(result, target, overwrite=overwrite)
        except (ImportError, FileExistsError, ValueError) as exc:
            typer.echo(str(exc))
            raise typer.Exit(1) from exc

        shape = result.features.shape
        typer.echo(f"Saved {shape[0]} x {shape[1]} embeddings\n")

    typer.echo("Done.")
