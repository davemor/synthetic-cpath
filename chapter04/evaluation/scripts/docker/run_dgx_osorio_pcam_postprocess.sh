#!/usr/bin/env bash
set -euo pipefail

REPO=/home/dm236/synpath
ROOT=/raid/dm236/synpath/outputs/reproductions/osorio_pcam_diffusers_legacy
IMAGE=synpath:latest

for arm in class_only mep; do
  docker run --rm \
    -v /raid/dm236:/raid/dm236 \
    -v "${REPO}:/workspace/synpath:ro" \
    -w /workspace/synpath "${IMAGE}" \
    uv run python scripts/audits/osorio_pcam_postprocess.py prepare --arm "${arm}"
done

launch_shard() {
  local arm="$1" shard="$2" gpu="$3"
  local name="osorio-pcam-sample-${arm}-${shard}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${name}" --gpus all --ipc=host \
    --env-file /raid/dm236/synpath/dgx.env \
    -e CUDA_VISIBLE_DEVICES="${gpu}" \
    -e SYNPATH_RUN_ROOT=/raid/dm236/synpath \
    -e HF_HOME=/raid/dm236/synpath/huggingface \
    -v /raid/dm236:/raid/dm236 \
    -v "${REPO}:/workspace/synpath:ro" \
    -w /workspace/synpath "${IMAGE}" \
    uv run python -m synpath.generation.sd_full_sample \
      --model_dir "${ROOT}/${arm}/model" \
      --condition_manifest "${ROOT}/${arm}/sample_conditions.csv" \
      --output_dir "${ROOT}/${arm}/samples" \
      --num_inference_steps 50 --guidance_scale 7.5 \
      --batch_size 8 --height 512 --width 512 --seed 18021999 \
      --num_shards 4 --shard_index "${shard}"
}

for shard in 0 1 2 3; do
  launch_shard class_only "${shard}" "${shard}"
  launch_shard mep "${shard}" "$((shard + 4))"
done

nohup bash -c '
  set -euo pipefail
  names=()
  for arm in class_only mep; do
    for shard in 0 1 2 3; do
      names+=("osorio-pcam-sample-${arm}-${shard}")
    done
  done
  failed=0
  for name in "${names[@]}"; do
    code=$(docker wait "${name}")
    if [[ "${code}" != 0 ]]; then failed=1; fi
  done
  if [[ "${failed}" != 0 ]]; then
    echo "Sampling failed; evaluation not started" >&2
    exit 1
  fi
  for arm in class_only mep; do
    docker run --rm \
      -v /raid/dm236:/raid/dm236 \
      -v /home/dm236/synpath:/workspace/synpath:ro \
      -w /workspace/synpath synpath:latest \
      uv run python scripts/audits/osorio_pcam_postprocess.py finalize --arm "${arm}"
  done
  for spec in class_only:0 mep:4; do
    arm=${spec%%:*}; gpu=${spec##*:}
    docker run --rm --gpus all --ipc=host \
      --env-file /raid/dm236/synpath/dgx.env \
      -e CUDA_VISIBLE_DEVICES="${gpu}" \
      -e SYNPATH_RUN_ROOT=/raid/dm236/synpath \
      -e SYNPATH_OUTPUTS_ROOT=/raid/dm236/synpath/outputs \
      -e SYNPATH_CACHE_ROOT=/raid/dm236/synpath/cache \
      -e HF_HOME=/raid/dm236/synpath/huggingface \
      -v /raid/dm236:/raid/dm236 \
      -v /home/dm236/synpath:/workspace/synpath:ro \
      -w /workspace/synpath synpath:latest \
      uv run python scripts/audits/osorio_pcam_postprocess.py evaluate --arm "${arm}" \
      >"/raid/dm236/synpath/outputs/reproductions/osorio_pcam_diffusers_legacy/${arm}/evaluation.log" 2>&1 &
  done
  wait
' >"${ROOT}/postprocess_watch.log" 2>&1 &

echo "sampling containers launched"
echo "watcher log: ${ROOT}/postprocess_watch.log"
