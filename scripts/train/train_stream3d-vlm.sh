#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# GPU Configuration
# ======================
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)         

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(python3 -c "
import socket
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port
print(find_free_port())
")

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
GEOMETRY_ENCODER_TYPE="streamvggt"
GEOMETRY_ENCODER_PATH="lch01/StreamVGGT"
OUTPUT_DIR="./results_checkpoint/Stream3D-VLM-4B"
# OUTPUT_DIR="/results_checkpoint/Stream3D-VLM-8B"
CACHE_DIR="./cache"
mkdir -p $OUTPUT_DIR

# ======================
# Dataset Configuration
# ======================
DATASETS="ca1m-ego_motion_estimation,scannet-ego_motion_estimation,scannet-environment_measurement,scannet-object_attributes,scannet-object_camera_relationship,scannet-object_chronology,scannetpp-ego_motion_estimation,scannetpp-environment_measurement,scannetpp-object_camera_relationship,scannetpp-object_chronology" 
# ======================
# Training Parameters
# ======================
export NCCL_NVLS_ENABLE=0
feature_fusion_method="cross_attention"
train_or_eval_mode="train"
LR=1e-5
total_batch_size=32
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))
num_train_epochs=1
stream_loss_weight=2.0
llm_loss_weight=1.0

torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 128000 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs $num_train_epochs \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 500 \
            --save_steps 5000 \
            --save_total_limit 10 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "none" \
            --use_geometry_encoder True \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --geometry_encoder_train_or_eval_mode $train_or_eval_mode \
            --feature_fusion_method $feature_fusion_method \
            --stream_loss_weight $stream_loss_weight \
            --llm_loss_weight $llm_loss_weight \
            2>&1 | tee ${OUTPUT_DIR}/train.log