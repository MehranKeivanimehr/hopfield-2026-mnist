#!/bin/bash
#SBATCH --job-name=cmkc-nextqa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/research/cmkc"
QUAD_ROOT="${PROJECT_ROOT}/upstream/QUAD"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cmkc-vl || true
fi

cd "${QUAD_ROOT}"
python -m pip install -r requirements.txt
python download_backbones.py
cd VL-T5/nextqa
bash scripts/ours.sh
