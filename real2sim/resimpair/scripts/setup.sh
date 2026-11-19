#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
chmod 700 private
chmod 600 private/.env.local
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install --user uv'; exit 1; }
command -v ffmpeg >/dev/null || { echo 'Install system ffmpeg and ffprobe first.'; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 --relocatable .venv
fi
uv pip sync --python .venv/bin/python requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/bin/python -B resimpair.py doctor
