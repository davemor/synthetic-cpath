# Coding standards

These standards were extracted from nearby local repositories, especially
`hdims`, `histokit`, and related experiment projects.

## Project layout

- Use a `src/` Python package layout.
- Expose a console script through `pyproject.toml`.
- Keep command line workflows first-class; notebooks are for exploration, not
  the primary experiment path.
- Use YAML configs for experiment settings.
- Store environment-specific roots in `.env` and `.env.example`.
- Keep generated data, model checkpoints, reports, and external repositories
  out of git.

## Tooling

- Use `uv` for dependency management and command execution.
- Use `uv_build` as the build backend.
- Use `ruff` for linting and formatting.
- Use `pytest` for tests.
- Keep Makefile targets small and memorable: `fmt`, `tests`, `plan`, `smoke`.

## Code style

- Prefer simple modules with explicit stage responsibilities.
- Keep line length at 80 characters.
- Use typed function signatures for public helpers.
- Prefer dataclasses for small path/config containers.
- Write JSON, YAML, CSV, and Parquet artifacts rather than opaque ad hoc
  formats.
- Make experiment stages re-entrant by checking for expected artifacts and
  recording resolved configs.

## Experiment style

- Use environment variables for data roots so local macOS development and DGX
  runs do not need different code.
- Write manifests for all datasets and generated samples.
- Cache expensive intermediate assets, especially embeddings and cluster
  assignments.
- Include a smoke path that runs the whole pipeline on tiny data before any
  full DGX run.
- Write final outputs as `results.csv`, supporting metric CSVs, figures, and a
  short Markdown report.

## DGX assumptions

- The target training system is Ubuntu with NVIDIA GPUs.
- RAPIDS may be used for acceleration where it reduces runtime without making
  the CPU fallback hard to maintain.
- PyTorch is the primary deep learning runtime.
- External model repositories live outside the Python package and are called
  through wrappers.
