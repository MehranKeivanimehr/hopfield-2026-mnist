#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HOME}/research/cmkc"
PROJECT_REPO="${PROJECT_ROOT}/project"
UPSTREAM_ROOT="${PROJECT_ROOT}/upstream"
DATA_ROOT="${PROJECT_ROOT}/data"
LOG_ROOT="${PROJECT_ROOT}/logs"

mkdir -p "${PROJECT_ROOT}" "${UPSTREAM_ROOT}" "${DATA_ROOT}" "${LOG_ROOT}"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if ! conda env list | grep -q "^cmkc-vl"; then
    conda create -y -n cmkc-vl python=3.10
  fi
  conda activate cmkc-vl
else
  python3 -m venv "${PROJECT_ROOT}/envs/cmkc-vl"
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/envs/cmkc-vl/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install gdown huggingface_hub

if [ ! -d "${UPSTREAM_ROOT}/VQACL" ]; then
  git clone --depth 1 https://github.com/zhangxi1997/VQACL.git "${UPSTREAM_ROOT}/VQACL"
fi
if [ ! -d "${UPSTREAM_ROOT}/QUAD" ]; then
  git clone --depth 1 https://github.com/IemProg/QUAD.git "${UPSTREAM_ROOT}/QUAD"
fi
if [ ! -d "${UPSTREAM_ROOT}/CoIN" ]; then
  git clone --depth 1 https://github.com/zackschen/CoIN.git "${UPSTREAM_ROOT}/CoIN"
fi

echo "Bootstrap complete."
echo "Project root: ${PROJECT_ROOT}"
echo "Project repo: ${PROJECT_REPO}"
echo "Upstream root: ${UPSTREAM_ROOT}"
echo "Data root: ${DATA_ROOT}"
