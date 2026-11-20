#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sdf/dataset/zhouxunzhe/xunzhe/The-Imitator-Game}"
V2_ROOT="${V2_ROOT:-/mnt/sdb/agi_data/data_lerobot/agibotworld}"
OUT="${OUT:-/mnt/sdb/agi_data/data_lerobot_v3/agibotworld}"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/imitator/dataset/agibot/agibot_v2_to_v3.py}"
VIDEO_MODE="${VIDEO_MODE:-hardlink}"

TASKS=(
  task_361 task_362 task_365 task_366 task_367 task_368
  task_372 task_374 task_376 task_377 task_378 task_380
  task_384 task_385 task_388 task_389 task_390 task_392
  task_398 task_410 task_414 task_421 task_422 task_424
  task_440 task_446
)

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=()
if [[ -f "$OUT/meta/info.json" ]]; then
  EXTRA+=(--resume)
else
  EXTRA+=(--force)
fi

python "$SCRIPT" \
  --v2-root "$V2_ROOT" \
  --output-root "$OUT" \
  --repo-id agibotworld \
  --task-ids "${TASKS[@]}" \
  --cameras head hand_left hand_right \
  --video-mode "$VIDEO_MODE" \
  "${EXTRA[@]}"
