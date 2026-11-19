#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
bash scripts/setup.sh
bash scripts/setup_tabletop.sh
.venv/bin/python -m resimpair_pickplace.pose_bundle inspect \
  --library assets/precomputed_grasps_v1
