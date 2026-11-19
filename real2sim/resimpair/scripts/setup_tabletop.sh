#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
command -v uv >/dev/null || { echo 'Install uv first.'; exit 1; }
test -x .venv/bin/python || { echo 'Run scripts/setup.sh first.'; exit 1; }
uv pip install --python .venv/bin/python --target runtime/tabletop_deps \
  --no-deps --only-binary=:all: --require-hashes -r requirements-tabletop.txt
.venv/bin/python -B - <<'PY'
from resimpair_tabletop import geometry
from trimesh.ray.ray_pyembree import RayMeshIntersector
import embreex
import trimesh
mesh = trimesh.creation.box()
assert RayMeshIntersector(mesh).intersects_any([[0,0,2]],[[0,0,-1]])[0]
print('Tabletop Embree ray backend ready:', embreex.__version__)
PY
