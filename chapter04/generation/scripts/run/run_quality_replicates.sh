#!/usr/bin/env bash
set -euo pipefail

read -r -a DATASETS <<<"${SYNPATH_REPLICATE_DATASETS:-pathmnist224 icaird_cervical100k pcam}"
read -r -a MODELS <<<"${SYNPATH_REPLICATE_MODELS:-stylegan2ada sd_lora dcgan}"
read -r -a CONDITIONING_MODES <<<"${SYNPATH_REPLICATE_CONDITIONING:-true_label unconditioned pseudo_label label_global_cluster label_local_cluster}"
read -r -a SEEDS <<<"${SYNPATH_REPLICATE_SEEDS:-0 1 2}"

config_for_dataset() {
    case "$1" in
        pathmnist224)
            echo "configs/experiments/main_pathmnist224_budget.yaml"
            ;;
        icaird_cervical100k)
            echo "configs/experiments/main_icaird_cervical100k_budget.yaml"
            ;;
        pcam)
            echo "configs/experiments/main_pcam_budget.yaml"
            ;;
        *)
            echo "No quality-replicate config for dataset: $1" >&2
            return 1
            ;;
    esac
}

for dataset in "${DATASETS[@]}"; do
    config="$(config_for_dataset "${dataset}")"
    for model in "${MODELS[@]}"; do
        for conditioning in "${CONDITIONING_MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                key="${dataset}/${model}/${conditioning}/seed${seed}"
                echo "quality replicate: ${key}"
                uv run --extra gpu synpath train-generator \
                    "${dataset}" "${model}" "${conditioning}" \
                    --seed "${seed}" --config "${config}" --resume
                uv run --extra gpu synpath sample \
                    "${dataset}" "${model}" "${conditioning}" \
                    --seed "${seed}" --config "${config}" --resume
                uv run --extra gpu synpath evaluate-quality \
                    "${dataset}" "${model}" "${conditioning}" \
                    --seed "${seed}" --config "${config}" --resume
            done
        done
    done
done

uv run --extra gpu synpath report-appendix \
    --output "${SYNPATH_REPORTS_ROOT}/appendix_quality_utility_by_dataset.tex"
