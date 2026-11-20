#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python}"
SRC_ROOT="${SRC_ROOT:-/mnt/sdb/agi_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/sdb/agi_data/data_lerobot_v3/agibotworld}"
VIDEO_MODE="${VIDEO_MODE:-hardlink}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"

# Locally available tasks observed in your current dataset. task_475 is dexhand
# in current any4lerobot and is intentionally omitted from this 16-D gripper run.
TASKS=(
  task_361 task_362 task_365 task_366 task_367 task_368
  task_372 task_374 task_376 task_377 task_378 task_380
  task_384 task_385 task_388 task_389 task_390 task_392
  task_398 task_410 task_414 task_421 task_422 task_424
  task_440 task_446
)

MODE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  MODE_ARGS+=(--force)
elif [[ -f "${OUTPUT_ROOT}/meta/info.json" ]]; then
  MODE_ARGS+=(--resume)
fi

printf 'Python : %s\n' "$(command -v "$PYTHON" || echo "$PYTHON")"
printf 'Source : %s\n' "$SRC_ROOT"
printf 'Output : %s\n' "$OUTPUT_ROOT"
printf 'Tasks  : %d gripper tasks\n' "${#TASKS[@]}"
printf 'Videos : %s\n' "$VIDEO_MODE"

"$PYTHON" "${SCRIPT_DIR}/agibot_h5.py" \
  --src-path "$SRC_ROOT" \
  --output-path "$OUTPUT_ROOT" \
  --repo-id agibotworld \
  --fps 30 \
  --cameras head hand_left hand_right \
  --video-mode "$VIDEO_MODE" \
  --video-backend "$VIDEO_BACKEND" \
  --task-ids "${TASKS[@]}" \
  "${MODE_ARGS[@]}" \
  "$@"
