SYNPATH_DOCKER_IMAGE ?= synpath:latest
SYNPATH_DGX_ROOT ?= /raid/dm236/synpath
SYNPATH_DGX_CONTAINER ?= synpath-main
SYNPATH_EXPERIMENT_CONFIG ?= configs/experiments/main.yaml
SYNPATH_RAPIDS_IMAGE ?= rapidsai/base:26.06-cuda12-py3.12
SYNPATH_RAPIDS_SDBW_IMAGE ?= synpath-rapids-sdbw:26.06

.PHONY: fmt tests plan smoke docker-build docker-build-rapids-sdbw dgx-env dgx-plan dgx-plan-pathmnist dgx-plan-pathmnist-budget dgx-smoke dgx-run dgx-run-pathmnist dgx-run-pathmnist-budget dgx-quality-replicates dgx-sd-lora-convergence dgx-calibrate-sd-index dgx-calibrate-cluster-count dgx-calibrate-pcam-osorio dgx-calibrate-pcam-osorio-sdbw dgx-calibrate-pcam-kaggle-osorio-sdbw dgx-logs dgx-shell

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

docker-build-rapids-sdbw:
	docker build --build-arg RAPIDS_IMAGE=$(SYNPATH_RAPIDS_IMAGE) -f Dockerfile.rapids-sdbw -t $(SYNPATH_RAPIDS_SDBW_IMAGE) .

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

dgx-quality-replicates:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_DGX_CONTAINER=synpath-quality-replicates bash scripts/docker/run_dgx_quality_replicates.sh

dgx-sd-lora-convergence:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_DGX_CONTAINER=synpath-sd-lora-convergence bash scripts/docker/run_dgx_sd_lora_convergence_audit.sh

dgx-calibrate-sd-index:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_RAPIDS_IMAGE=$(SYNPATH_RAPIDS_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) bash scripts/docker/run_sd_index_calibration_dgx.sh all

dgx-calibrate-cluster-count:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_RAPIDS_IMAGE=$(SYNPATH_RAPIDS_SDBW_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) bash scripts/docker/run_cluster_count_calibration_dgx.sh all

dgx-calibrate-pcam-osorio:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_RAPIDS_IMAGE=$(SYNPATH_RAPIDS_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) bash scripts/docker/run_sd_index_pcam_osorio_dgx.sh all

dgx-calibrate-pcam-osorio-sdbw:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_RAPIDS_IMAGE=$(SYNPATH_RAPIDS_SDBW_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_SD_INDEX_CONFIG=configs/experiments/sd_index_calibration_pcam_osorio_standardized_sdbw.yaml SYNPATH_SD_INDEX_RUN_NAME=cluster_sd_index_calibration_pcam_osorio_standardized_sdbw bash scripts/docker/run_sd_index_pcam_osorio_dgx.sh all

dgx-calibrate-pcam-kaggle-osorio-sdbw:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_RAPIDS_IMAGE=$(SYNPATH_RAPIDS_SDBW_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_SD_INDEX_CONFIG=configs/experiments/sd_index_calibration_pcam_kaggle_osorio_standardized_sdbw.yaml SYNPATH_SD_INDEX_RUN_NAME=cluster_sd_index_calibration_pcam_kaggle_osorio_standardized_sdbw bash scripts/docker/run_sd_index_pcam_osorio_dgx.sh all

dgx-logs:
	docker logs -f $(SYNPATH_DGX_CONTAINER)

dgx-shell:
	SYNPATH_DOCKER_IMAGE=$(SYNPATH_DOCKER_IMAGE) SYNPATH_DGX_ROOT=$(SYNPATH_DGX_ROOT) SYNPATH_DGX_CONTAINER=synpath-shell bash scripts/docker/run_dgx.sh shell
