SYNPATH_DOCKER_IMAGE ?= synpath:latest
SYNPATH_DGX_ROOT ?= /raid/dm236/synpath
SYNPATH_DGX_CONTAINER ?= synpath-main
SYNPATH_EXPERIMENT_CONFIG ?= configs/experiments/main.yaml
SYNPATH_AUTHENTICITY_IMAGE ?= synpath-authenticity:latest

.PHONY: fmt tests plan smoke docker-build docker-build-authenticity dgx-env dgx-plan dgx-plan-pathmnist dgx-plan-pathmnist-budget dgx-smoke dgx-run dgx-run-pathmnist dgx-run-pathmnist-budget dgx-authenticity dgx-logs dgx-shell

fmt:
	uv run ruff check --fix
	uv run ruff format

tests:
	uv run pytest tests/

plan:
	uv run synpath plan --config configs/experiments/main.yaml

smoke:
	bash scripts/run/run_smoke.sh

docker-build:
	docker build -t $(SYNPATH_DOCKER_IMAGE) .

docker-build-authenticity:
	docker build -f Dockerfile.authenticity -t $(SYNPATH_AUTHENTICITY_IMAGE) .

dgx-env:
	mkdir -p $(SYNPATH_DGX_ROOT)
	test -f $(SYNPATH_DGX_ROOT)/dgx.env || cp configs/docker/dgx.env.example $(SYNPATH_DGX_ROOT)/dgx.env
	@echo "DGX env: $(SYNPATH_DGX_ROOT)/dgx.env"

dgx-plan:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_EXPERIMENT_CONFIG=$(SYNPATH_EXPERIMENT_CONFIG) SYNPATH_DGX_CONTAINER=synpath-plan bash scripts/docker/run_dgx.sh plan

dgx-plan-pathmnist:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_EXPERIMENT_CONFIG=configs/experiments/main_pathmnist224.yaml SYNPATH_DGX_CONTAINER=synpath-plan-pathmnist bash scripts/docker/run_dgx.sh plan

dgx-plan-pathmnist-budget:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_EXPERIMENT_CONFIG=configs/experiments/main_pathmnist224_budget.yaml SYNPATH_DGX_CONTAINER=synpath-plan-pathmnist-budget bash scripts/docker/run_dgx.sh plan

dgx-smoke:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_DGX_CONTAINER=synpath-smoke bash scripts/docker/run_dgx.sh smoke

dgx-run:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_EXPERIMENT_CONFIG=$(SYNPATH_EXPERIMENT_CONFIG) SYNPATH_DGX_CONTAINER=$(SYNPATH_DGX_CONTAINER) bash scripts/docker/run_dgx.sh main

dgx-run-pathmnist:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_EXPERIMENT_CONFIG=configs/experiments/main_pathmnist224.yaml SYNPATH_DGX_CONTAINER=synpath-pathmnist bash scripts/docker/run_dgx.sh main

dgx-run-pathmnist-budget:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_EXPERIMENT_CONFIG=configs/experiments/main_pathmnist224_budget.yaml SYNPATH_DGX_CONTAINER=synpath-pathmnist-budget bash scripts/docker/run_dgx.sh main

dgx-authenticity:
	docker run --rm --gpus all --ipc=host -v /raid:/raid $(SYNPATH_AUTHENTICITY_IMAGE)

dgx-logs:
	docker logs -f $(SYNPATH_DGX_CONTAINER)

dgx-shell:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_DGX_CONTAINER=synpath-shell bash scripts/docker/run_dgx.sh shell
