#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sdf/dataset/zhouxunzhe/xunzhe/The-Imitator-Game}"
V2_ROOT="${V2_ROOT:-/mnt/sdb/agi_data/data_lerobot/agibotworld}"
OUT="${OUT:-/mnt/sdb/agi_data/data_lerobot_v3/agibotworld_smoke}"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/examples/baselines/lerobot_dataset/agibot/agibot_v2_to_v3.py}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python "$SCRIPT" \
  --v2-root "$V2_ROOT" \
  --output-root "$OUT" \
  --repo-id agibotworld_smoke \
  --task-ids task_398 \
  --max-episodes-per-task 1 \
  --cameras head hand_left hand_right \
  --video-mode hardlink \
  --force \
  --fail-fast
