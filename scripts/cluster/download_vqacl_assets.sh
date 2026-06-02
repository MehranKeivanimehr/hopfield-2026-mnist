#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HOME}/research/cmkc"
UPSTREAM_ROOT="${PROJECT_ROOT}/upstream"
DATA_ROOT="${PROJECT_ROOT}/data"
QUAD_ROOT="${UPSTREAM_ROOT}/QUAD"

mkdir -p "${DATA_ROOT}/downloads" "${DATA_ROOT}/datasets"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cmkc-vl || true
fi

python -m pip install --upgrade gdown

# Official links surfaced in the VQACL and QUAD repositories.
gdown --fuzzy "https://drive.google.com/file/d/11gx7AxyeMP1KVuzHErIfNKCLeBWGq3pE/view?usp=share_link" -O "${DATA_ROOT}/downloads/vqav2_partition_q"
gdown --fuzzy "https://drive.google.com/file/d/1lwWL_PhNKactFEqQF8IVx-HeJEuboe8D/view?usp=share_link" -O "${DATA_ROOT}/downloads/nextqa_partition_q"
gdown --fuzzy "https://drive.google.com/file/d/1rS5X_t_VSDF4uP3HL1gPQ0ZgWIEuglgk/view?usp=share_link" -O "${DATA_ROOT}/downloads/nextqa_vid_feat.zip"
gdown --folder "https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing" -O "${DATA_ROOT}/downloads/COCO"

mkdir -p "${QUAD_ROOT}/datasets/vqa" "${QUAD_ROOT}/datasets/nextqa"

echo "Downloaded VQACL assets into ${DATA_ROOT}/downloads."
echo "Manually verify the archive/file types, then place them as follows:"
echo "  ${QUAD_ROOT}/datasets/vqa/Partition_Q"
echo "  ${QUAD_ROOT}/datasets/nextqa/Partition_Q"
echo "  ${QUAD_ROOT}/datasets/nextqa/video_features"
echo "  ${QUAD_ROOT}/datasets/COCO"
