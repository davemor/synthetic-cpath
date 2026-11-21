FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.12
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libgl1 \
    ninja-build \
    openssh-client \
    python3 \
    python3.10-dev \
    python3-venv \
    rsync \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/synpath

COPY pyproject.toml uv.lock README.md LICENSE .python-version ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN uv sync --extra gpu --group dev

COPY docs ./docs
COPY slurm ./slurm
COPY Makefile ./

CMD ["uv", "run", "synpath", "plan", "--config", "configs/experiments/main.yaml"]
