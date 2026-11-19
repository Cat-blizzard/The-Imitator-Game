#!/usr/bin/env bash
set -euo pipefail
resimpair_multitask_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$resimpair_multitask_root"
exec "$resimpair_multitask_root/.venv/bin/python" -m resimpair_tasks.multitask_campaign_v1 "$@"
