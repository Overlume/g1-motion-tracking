#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p environment/bin environment/uv-cache environment/python environment/logs
export UV_CACHE_DIR="$ROOT/environment/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/environment/python"
export UV_LINK_MODE=copy
if [ ! -x environment/bin/uv ]; then
  curl -LsSf -o /tmp/uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf /tmp/uv.tar.gz -C /tmp
  cp /tmp/uv-x86_64-unknown-linux-gnu/uv environment/bin/uv
fi
environment/bin/uv --version
if [ ! -x environment/.venv/bin/python ]; then
  environment/bin/uv venv --python 3.13 environment/.venv
fi
environment/.venv/bin/python -V
environment/bin/uv pip install --python environment/.venv/bin/python -e repos/mjlab_research
environment/.venv/bin/python -c "import torch, mujoco, warp, mjlab; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('mujoco', mujoco.__version__); print('mjlab ok')"
