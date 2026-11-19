#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_uv="${UV_BIN:-uv}"
"${task_uv}" pip install --python "${task_root}/.venv/bin/python" \
  --target "${task_root}/runtime/collision_cpu_deps" --no-deps \
  -r "${task_root}/requirements/collision_cpu.txt"
