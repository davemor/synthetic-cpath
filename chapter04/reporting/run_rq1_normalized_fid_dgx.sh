#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
CONTAINER="${SYNPATH_RQ1_CONTAINER:-synpath-rq1-normalized-fid}"
RUN_ROOT="${SYNPATH_RUN_ROOT:-/raid/dm236/synpath}"
REPORT_ROOT="${SYNPATH_RQ1_REPORT_ROOT:-${RUN_ROOT}/reporting/rq1-normalized-fid}"
OUTPUT_ROOT="${SYNPATH_RQ1_OUTPUT_ROOT:-${RUN_ROOT}/reports/rq1-normalized-fid}"

QUALITY_CSV="${REPORT_ROOT}/inputs/quality_results_fid_ip_ir_authenticity.csv"
LONG_RESULTS_CSV="${REPORT_ROOT}/inputs/canonical_results.csv"
PROGRAM="${REPORT_ROOT}/scripts/build_rq1_normalized_fid_report.py"

for required in "${QUALITY_CSV}" "${LONG_RESULTS_CSV}" "${PROGRAM}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing staged reporting input: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/matplotlib"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib" \
    -v /raid:/raid \
    -v "${REPORT_ROOT}:/workspace/report:ro" \
    "${IMAGE}" \
    /workspace/synpath/.venv/bin/python \
        /workspace/report/scripts/build_rq1_normalized_fid_report.py \
        --quality-csv /workspace/report/inputs/quality_results_fid_ip_ir_authenticity.csv \
        --long-results-csv /workspace/report/inputs/canonical_results.csv \
        --data-root "${RUN_ROOT}/data" \
        --cache-root "${RUN_ROOT}/cache" \
        --outputs-root "${RUN_ROOT}/outputs" \
        --output-dir "${OUTPUT_ROOT}" \
        --fid-device cuda \
        --fid-batch-size 64 \
        --fid-workers 32

echo "container=${CONTAINER}"
echo "output=${OUTPUT_ROOT}"
echo "progress=docker logs -f ${CONTAINER}"
