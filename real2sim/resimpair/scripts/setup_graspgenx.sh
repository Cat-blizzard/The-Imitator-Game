#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
uv_bin="${UV_BIN:-uv}"
upstream_dir="$repo_dir/runtime/upstream/GraspGenX"
upstream_commit="b9429097728cb1c430dd78b92edf17ba318aad03"
mkdir -p runtime/upstream runtime/envs
if [[ ! -d "$upstream_dir/.git" ]]; then
  git clone https://github.com/NVlabs/GraspGenX.git "$upstream_dir"
  git -C "$upstream_dir" checkout --detach "$upstream_commit"
fi
[[ "$(git -C "$upstream_dir" rev-parse HEAD)" == "$upstream_commit" ]]
if [[ ! -x runtime/envs/graspgenx/bin/python ]]; then
  "$uv_bin" venv --python 3.12 runtime/envs/graspgenx
fi
"$uv_bin" pip install --python runtime/envs/graspgenx/bin/python --prerelease=allow \
  --overrides requirements/graspgenx_gpu_overrides.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -e "$upstream_dir"
# Pin the public LFS assets before the package's auto-download hook runs.
fetch_pinned() {
  local source_url="$1" target_dir="$2" revision="$3"
  if [[ ! -d "$target_dir/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 timeout 600 git clone "$source_url" "$target_dir"
    git -C "$target_dir" checkout --detach "$revision"
    timeout 600 git -C "$target_dir" lfs pull
  fi
  [[ "$(git -C "$target_dir" rev-parse HEAD)" == "$revision" ]]
}
mkdir -p "$upstream_dir/ext"
fetch_pinned https://huggingface.co/datasets/adithyamurali/gripper_descriptions \
  "$upstream_dir/ext/gripper_descriptions" 19a03c00d19aeaf052d0f6801f0041982d676e8a
fetch_pinned https://huggingface.co/adithyamurali/GraspGenXModel \
  "$upstream_dir/ext/graspgenx_checkpoints" 7c834043c11a11417e31d6d5ea9355801e40a2c1
# This imports metadata only; it does not initialize a model or GPU.
runtime/envs/graspgenx/bin/python -c 'import graspgenx; print(graspgenx.get_checkpoints_root())'
[[ "$(git -C "$upstream_dir/ext/gripper_descriptions" rev-parse HEAD)" == "19a03c00d19aeaf052d0f6801f0041982d676e8a" ]]
[[ "$(git -C "$upstream_dir/ext/graspgenx_checkpoints" rev-parse HEAD)" == "7c834043c11a11417e31d6d5ea9355801e40a2c1" ]]
printf '%s\n' 'GraspGenX inference environment ready. Run through the pickplace GPU supervisor.'
