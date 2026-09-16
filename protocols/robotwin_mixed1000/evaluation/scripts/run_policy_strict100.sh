#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
OGPO_ROOT=${OGPO_ROOT:-$(cd "${BUNDLE_ROOT}/../../.." && pwd -P)}
: "${PI05_ROOT:?Set PI05_ROOT to the PI0.5/RoboTwin checkout}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new or resumable result directory}"
: "${POLICY_LABEL:?Set POLICY_LABEL to a stable policy name}"

PYTHON_BIN=${PYTHON_BIN:-${PI05_ROOT}/.conda-pi05-openpi-final/bin/python}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${PI05_ROOT}/external/RoboTwin}
PI05_CHECKPOINT_DIR=${PI05_CHECKPOINT_DIR:-${PI05_ROOT}/model_clean50}
RESUME_EXISTING=${RESUME_EXISTING:-0}
ACTOR_CHECKPOINT=${ACTOR_CHECKPOINT:-}

for path in "${PYTHON_BIN}" "${PI05_CHECKPOINT_DIR}/model.safetensors" \
  "${BUNDLE_ROOT}/protocol.json"; do
  test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done
if [[ -n "${ACTOR_CHECKPOINT}" ]]; then
  test -s "${ACTOR_CHECKPOINT}" || { echo "Missing actor checkpoint: ${ACTOR_CHECKPOINT}" >&2; exit 1; }
fi
if [[ -e "${OUTPUT_ROOT}" && "${RESUME_EXISTING}" != 1 ]]; then
  echo "Refusing to overwrite ${OUTPUT_ROOT}; set RESUME_EXISTING=1 to resume" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

export OGPO_ROOT PI05_ROOT ROBOTWIN_ROOT PYTHONUNBUFFERED=1
export PATH=${PI05_ROOT}/.conda-pi05-openpi-final/bin:${PATH}
export PYTHONPATH=${OGPO_ROOT}/src:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/policy/pi05/src:${ROBOTWIN_ROOT}/policy/pi05/packages/openpi-client/src:${PYTHONPATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export ROBOTWIN_ASSET_ID=${ROBOTWIN_ASSET_ID:-aloha_clean50_50task}
export ROBOTWIN_COMPACT_LOG=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

ALLOCATED=${CUDA_VISIBLE_DEVICES:-}
if [[ -z "${ALLOCATED}" ]]; then
  ALLOCATED=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi
IFS=',' read -r -a GPUS <<< "${ALLOCATED}"
GPU_COUNT=${#GPUS[@]}
if (( GPU_COUNT < 1 || GPU_COUNT > 8 )); then
  echo "Expected 1-8 visible GPUs, got CUDA_VISIBLE_DEVICES=${ALLOCATED}" >&2
  exit 1
fi

mapfile -t TASKS < <("${PYTHON_BIN}" -c 'import json,sys; print("\n".join(t["name"] for t in json.load(open(sys.argv[1]))["tasks"]))' "${BUNDLE_ROOT}/protocol.json")
(( ${#TASKS[@]} == 10 )) || { echo "protocol must contain 10 tasks" >&2; exit 1; }

run_shard() {
  local task=$1 manifest=$2 task_root=$3 shard=$4 gpu=$5 episodes=13
  (( shard == 7 )) && episodes=9
  local shard_root=${task_root}/shard_$(printf '%02d' "${shard}")
  mkdir -p "${shard_root}" "${task_root}/logs"
  export CUDA_VISIBLE_DEVICES=${gpu}
  export TORCH_EXTENSIONS_DIR=${TMPDIR:-/tmp}/strict100-ext-${SLURM_JOB_ID:-local}-${task}-${shard}
  export MPLCONFIGDIR=${TMPDIR:-/tmp}/strict100-mpl-${SLURM_JOB_ID:-local}-${task}-${shard}
  mkdir -p "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"
  local log=${task_root}/logs/shard_$(printf '%02d' "${shard}").log
  local args=(
    --task-name "${task}" --task-config demo_clean
    --train-config-name pi05_robotwin2_clean50_full --model-name model_clean50
    --checkpoint-id 20000 --pi05-checkpoint-dir "${PI05_CHECKPOINT_DIR}"
    --episode-manifest "${manifest}" --trust-episode-manifest
    --seed "${shard}" --num-episodes "${episodes}" --pi0-step 50
    --metadata-only --output-dir "${shard_root}"
  )
  [[ -z "${ACTOR_CHECKPOINT}" ]] || args+=(--ogpo-checkpoint "${ACTOR_CHECKPOINT}")
  echo "shard_start=$(date --iso-8601=seconds) task=${task} shard=${shard} gpu=${gpu}" > "${log}"
  cd "${PI05_ROOT}"
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/evaluator/collect_robotwin_dense_rollouts.py" \
    "${args[@]}" >> "${log}" 2>&1
  echo "shard_end=$(date --iso-8601=seconds) task=${task} shard=${shard}" >> "${log}"
}

for task in "${TASKS[@]}"; do
  manifest=${BUNDLE_ROOT}/manifests/${task}.json
  task_root=${OUTPUT_ROOT}/${task}
  summary=${task_root}/strict100_summary.json
  if [[ -s "${summary}" ]]; then
    echo "strict100_skip_complete task=${task}"
    continue
  fi
  mkdir -p "${task_root}"
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/evaluator/live_robotwin_eval_progress.py" \
    --output-root "${task_root}" --total-episodes 100 --shard-count 8 \
    --interval-seconds 10 > "${task_root}/progress_watcher.log" 2>&1 &
  watcher=$!
  echo "strict100_start=$(date --iso-8601=seconds) task=${task} policy=${POLICY_LABEL}"
  for (( first=0; first<8; first+=GPU_COUNT )); do
    pids=()
    for (( slot=0; slot<GPU_COUNT && first+slot<8; slot++ )); do
      shard=$((first + slot))
      run_shard "${task}" "${manifest}" "${task_root}" "${shard}" "${GPUS[${slot}]}" &
      pids+=("$!")
    done
    failures=0
    for pid in "${pids[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
    if (( failures )); then
      kill "${watcher}" 2>/dev/null || true
      echo "strict100_failed task=${task} failures=${failures}" >&2
      exit 1
    fi
  done
  audit_args=(
    audit-task --manifest "${manifest}" --result-root "${task_root}"
    --task "${task}" --label "${POLICY_LABEL}" --output "${summary}"
  )
  [[ -z "${ACTOR_CHECKPOINT}" ]] || audit_args+=(--checkpoint "${ACTOR_CHECKPOINT}")
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/bundle_tools.py" "${audit_args[@]}"
  wait "${watcher}" || true
  echo "strict100_end=$(date --iso-8601=seconds) task=${task}"
done

"${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/bundle_tools.py" summarize-policy \
  --bundle-root "${BUNDLE_ROOT}" --result-root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/ten_task_summary.json"
