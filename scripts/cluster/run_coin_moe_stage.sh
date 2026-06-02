#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HOME}/research/cmkc"
COIN_ROOT="${PROJECT_ROOT}/upstream/CoIN"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cmkc-vl || true
fi

cd "${COIN_ROOT}"

python -m pip install -e .
python -m pip install -e ".[train]"

echo "CoIN install complete."
echo "Next step: populate CoIN images and instruction files described in dataset.md."
echo "Then launch the official MoELoRA task-order scripts under scripts/LLaVA/Train_MOE."
