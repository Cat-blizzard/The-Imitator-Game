#!/usr/bin/env bash
set -euo pipefail
resimpair_scene_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$resimpair_scene_root"
exec "$resimpair_scene_root/.venv/bin/python" -m resimpair_scene "$@"
