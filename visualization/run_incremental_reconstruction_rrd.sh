#!/bin/bash
# 3D Reconstruction with StreamVGGT
# Output format: RRD (Rerun) file

# === Configuration ===
INPUT_DIR="/path/to/scene/color"          # Input image folder
OUTPUT_DIR="/path/to/output"              # Output directory
CKPT_PATH="/path/to/checkpoints.pth"      # StreamVGGT checkpoint
STEP=30                                    # Frame sampling stride
MAX_FRAMES=50                              # Maximum frames to process
CONF_THRES=50.0                            # Confidence threshold (0-100)

# === Run ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_incremental_reconstruction_rrd.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --ckpt "${CKPT_PATH}" \
    --step ${STEP} \
    --max_frames ${MAX_FRAMES} \
    --conf_thres ${CONF_THRES} \
    --mode "Depthmap and Camera Branch" \
    --scannet_fix
