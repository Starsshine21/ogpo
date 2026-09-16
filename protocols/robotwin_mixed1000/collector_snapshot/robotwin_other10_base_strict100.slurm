#!/usr/bin/env bash
#SBATCH -A CHANGE_ME
#SBATCH -J rt_other10_base
#SBATCH -o <EVO_RL_ROOT>/dexjoco/outputs/ogpo/logs/rt-other10-base-%j.out
#SBATCH -e <EVO_RL_ROOT>/dexjoco/outputs/ogpo/logs/rt-other10-base-%j.err
#SBATCH -p CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH -t 1-00:00:00

set -euo pipefail

EVO_ROOT=<EVO_RL_ROOT>
ROOT=${EVO_ROOT}/dexjoco
PI05_ROOT=<PI05_ROOT>
ROBOTWIN_ROOT=${PI05_ROOT}/external/RoboTwin
PYTHON=${PI05_ROOT}/.conda-pi05-openpi-final/bin/python
MANIFEST_ROOT=${ROOT}/outputs/robotwin_eval/manifests_other10_base_strict100_v1
OUTPUT_ROOT=${ROOT}/outputs/robotwin_rollouts/other10_pi05_base_dense100_v1
TASKS=(
  adjust_bottle
  pick_diverse_bottles
  move_pillbottle_pad
  move_stapler_pad
  place_mouse_pad
  place_object_stand
  place_empty_cup
  press_stapler
  stamp_seal
  turn_switch
)
TASKS_PER_JOB=${TASKS_PER_JOB:-5}
TASK_GROUP_INDEX=${TASK_GROUP_INDEX:?Set TASK_GROUP_INDEX to 0 or 1}
(( TASK_GROUP_INDEX == 0 || TASK_GROUP_INDEX == 1 )) || {
  echo "TASK_GROUP_INDEX must be 0 or 1" >&2
  exit 1
}
START_INDEX=$((TASK_GROUP_INDEX * TASKS_PER_JOB))
END_INDEX=$((START_INDEX + TASKS_PER_JOB))
(( END_INDEX <= ${#TASKS[@]} )) || END_INDEX=${#TASKS[@]}

for path in "${PYTHON}" "${PI05_ROOT}/model_clean50/model.safetensors"; do
  test -s "${path}"
done

export PYTHONUNBUFFERED=1
export PATH=${PI05_ROOT}/.conda-pi05-openpi-final/bin:${PATH}
export PYTHONPATH=${ROOT}/dexjoco:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/policy/pi05/src:${ROBOTWIN_ROOT}/policy/pi05/packages/openpi-client/src:${PYTHONPATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export ROBOTWIN_ASSET_ID=aloha_clean50_50task ROBOTWIN_COMPACT_LOG=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#GPUS[@]} == 4 )) || { echo "expected four allocated GPUs" >&2; exit 1; }

if [[ "${REPAIR_BOTTLE_MANIFEST:-0}" == 1 ]]; then
  (( TASK_GROUP_INDEX == 0 )) || { echo 'bottle repair is group0 only' >&2; exit 1; }
  REPAIRED_ROOT=${ROOT}/outputs/robotwin_eval/manifests_other10_base_strict100_v2
  mkdir -p "${REPAIRED_ROOT}/preflight"
  pids=()
  for slot in 0 1 2 3; do
    (
      export CUDA_VISIBLE_DEVICES=${GPUS[$slot]}
      export TORCH_EXTENSIONS_DIR=/tmp/rt-bottle-preflight-${SLURM_JOB_ID}-${slot}-ext
      export MPLCONFIGDIR=/tmp/rt-bottle-preflight-${SLURM_JOB_ID}-${slot}-mpl
      "${PYTHON}" "${ROOT}/scripts/repair_other10_bottle_manifest.py" \
        --source-dir "${MANIFEST_ROOT}" --output-dir "${REPAIRED_ROOT}" \
        --data-root "${OUTPUT_ROOT}" --worker-index "$slot"
    ) > "${REPAIRED_ROOT}/preflight/job_${SLURM_JOB_ID}_worker_${slot}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "${PYTHON}" "${ROOT}/scripts/repair_other10_bottle_manifest.py" \
    --source-dir "${MANIFEST_ROOT}" --output-dir "${REPAIRED_ROOT}" --data-root "${OUTPUT_ROOT}"
  MANIFEST_ROOT=${REPAIRED_ROOT}
fi

run_shard() {
  local shard=$1 gpu=$2 episodes=13
  (( shard == 7 )) && episodes=9
  local shard_root=${TASK_ROOT}/shard_$(printf '%02d' "${shard}")
  local log_root=${TASK_ROOT}/logs/job_${SLURM_JOB_ID}
  mkdir -p "${shard_root}" "${log_root}"
  export CUDA_VISIBLE_DEVICES=${gpu}
  export TORCH_EXTENSIONS_DIR=/tmp/rt-other10-${SLURM_JOB_ID}-${TASK_GROUP_INDEX}-${shard}-ext
  export MPLCONFIGDIR=/tmp/rt-other10-${SLURM_JOB_ID}-${TASK_GROUP_INDEX}-${shard}-mpl
  mkdir -p "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"
  cd "${PI05_ROOT}"
  "${PYTHON}" "${ROOT}/scripts/collect_robotwin_dense_rollouts.py" \
    --task-name "${TASK}" --task-config demo_clean \
    --train-config-name pi05_robotwin2_clean50_full --model-name model_clean50 \
    --checkpoint-id 20000 --pi05-checkpoint-dir "${PI05_ROOT}/model_clean50" \
    --episode-manifest "${MANIFEST}" --trust-episode-manifest \
    --seed "${shard}" --num-episodes "${episodes}" --pi0-step 50 \
    --output-dir "${shard_root}" \
    > "${log_root}/shard_$(printf '%02d' "${shard}").log" 2>&1
}

SELECTED_TASKS=("${TASKS[@]:START_INDEX:TASKS_PER_JOB}")
if [[ -n "${DEFER_TASK:-}" ]]; then
  ordered=()
  for task in "${SELECTED_TASKS[@]}"; do
    [[ "${task}" == "${DEFER_TASK}" ]] || ordered+=("${task}")
  done
  (( ${#ordered[@]} + 1 == ${#SELECTED_TASKS[@]} )) || { echo 'DEFER_TASK must belong to this group' >&2; exit 1; }
  SELECTED_TASKS=("${ordered[@]}" "${DEFER_TASK}")
fi
task_failures=0
for TASK in "${SELECTED_TASKS[@]}"; do
  MANIFEST=${MANIFEST_ROOT}/${TASK}.json
  TASK_ROOT=${OUTPUT_ROOT}/${TASK}
  test -s "${MANIFEST}"
  if [[ -s "${TASK_ROOT}/strict100_summary.json" ]]; then
    echo "strict100_skip_complete task=${TASK}"
    continue
  fi
  mkdir -p "${TASK_ROOT}"
  task_failed=0
  echo "strict100_start=$(date --iso-8601=seconds) task=${TASK} policy=pi05_base manifest=${MANIFEST}"
  for first in 0 4; do
    pids=()
    for slot in 0 1 2 3; do
      shard=$((first + slot))
      run_shard "${shard}" "${GPUS[${slot}]}" &
      pids+=("$!")
    done
    failures=0
    for pid in "${pids[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
    if (( failures > 0 )); then
      echo "strict100_failed task=${TASK} wave=${first} failures=${failures}" >&2
      task_failed=1
      break
    fi
  done
  if (( task_failed )); then
    task_failures=$((task_failures + 1))
    continue
  fi

  "${ROOT}/.conda/openpi/bin/python" "${ROOT}/scripts/audit_robotwin_dense_rollouts.py" \
    --root "${TASK_ROOT}" --expected-episodes 100 --gamma 0.999

  "${ROOT}/.conda/openpi/bin/python" - "${MANIFEST}" "${TASK_ROOT}" "${TASK}" <<'PY'
import json
from pathlib import Path
import sys
manifest_path, root, task = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
manifest = json.loads(manifest_path.read_text())
successes = 0
for row in manifest["episodes"]:
    path = root / f"shard_{row['shard']:02d}" / "raw_rollouts" / f"episode_{row['episode_index']:06d}" / "meta.json"
    meta = json.loads(path.read_text())
    if not (path.parent / "frames.npz").is_file() or meta.get("num_frames", 0) <= 0:
        raise RuntimeError(f"{path}: missing dense training trajectory")
    for key in ("seed", "prompt", "policy_noise_seed"):
        if meta.get(key) != row[key]:
            raise RuntimeError(f"{path}: {key} mismatch")
    if meta.get("task_name") != task or meta.get("ogpo_checkpoint") is not None:
        raise RuntimeError(f"{path}: policy/task provenance mismatch")
    if meta.get("train_config") != "pi05_robotwin2_clean50_full" or meta.get("model_name") != "model_clean50":
        raise RuntimeError(f"{path}: base model provenance mismatch")
    if type(meta.get("success")) is not bool:
        raise RuntimeError(f"{path}: success is not boolean")
    successes += int(meta["success"])
summary = {
    "schema_version": 1,
    "audit_passed": True,
    "task_name": task,
    "episodes": 100,
    "manifest": str(manifest_path.resolve()),
    "policy": {"label": "pi05_base", "checkpoint": None, "successes": successes, "failures": 100-successes, "success_rate_percent": float(successes)},
}
(root / "strict100_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  echo "strict100_end=$(date --iso-8601=seconds) task=${TASK}"
done
(( task_failures == 0 ))
