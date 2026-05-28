#!/bin/bash
# Incremental 3D Reconstruction with StreamVGGT
# Output format: GLB point cloud files

# === Configuration ===
INPUT_DIR="/path/to/scene/color"          # Input image folder
OUTPUT_DIR="/path/to/output"              # Output directory
CKPT_PATH="/path/to/checkpoints.pth"      # StreamVGGT checkpoint
STEP=30                                    # Frame sampling stride
MAX_FRAMES=50                              # Maximum frames to process
INCREMENTAL_STEP=1                         # Save GLB every N frames (0 = final only)
CONF_THRES=50.0                            # Confidence threshold (0-100)

# === Run ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_incremental_reconstruction.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --ckpt "${CKPT_PATH}" \
    --step ${STEP} \
    --max_frames ${MAX_FRAMES} \
    --incremental_step ${INCREMENTAL_STEP} \
    --conf_thres ${CONF_THRES} \
    --mode "Depthmap and Camera Branch" \
    --scannet_fix
