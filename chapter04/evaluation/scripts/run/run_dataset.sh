#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: bash scripts/run/run_dataset.sh DATASET}"

uv run synpath prepare-data --dataset "${DATASET}"
uv run synpath embed --dataset "${DATASET}" --encoder uni
uv run synpath cluster --dataset "${DATASET}"

for MODE in true_label unconditioned pseudo_label label_global_cluster label_local_cluster; do
    uv run synpath build-conditions \
        --dataset "${DATASET}" \
        --conditioning "${MODE}"
done

echo "Generator, metric, utility, and report stages run through run_all_main.sh."
