#!/usr/bin/env bash
set -euo pipefail
resimpair_batch_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$resimpair_batch_root"
case "${1:-}" in
  metric-v3)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.metric_production_batch_v3 "$@"
    ;;
  metric-v2)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.metric_production_batch_v2 "$@"
    ;;
  metric)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.metric_production_batch_v1 "$@"
    ;;
  prepare-appearance-library)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.visual_furniture_bulk_render_v4 prepare "$@"
    ;;
  run-appearance-library)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.visual_furniture_bulk_render_v4 run "$@"
    ;;
  gallery-appearance-library)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.visual_furniture_bulk_render_v4 gallery "$@"
    ;;
  compile-metric-requests)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.metric_task_factory_v2 "$@"
    ;;
  run-metric-physics)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.production_integration_v2 run "$@"
    ;;
  prepare-appearance-bulk)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.visual_furniture_bulk_render_v1 prepare "$@"
    ;;
  run-appearance-bulk)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.visual_furniture_bulk_render_v1 run "$@"
    ;;
  prepare-appearance)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.multitask_appearance_v6 prepare "$@"
    ;;
  run-appearance)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.multitask_appearance_v6 run "$@"
    ;;
  prepare-render)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.batch_render_conditions_v1 prepare "$@"
    ;;
  run-render)
    shift
    exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.batch_render_conditions_v1 run "$@"
    ;;
esac
exec "$resimpair_batch_root/.venv/bin/python" -m resimpair_tasks.batch_pickplace_v1 "$@"
