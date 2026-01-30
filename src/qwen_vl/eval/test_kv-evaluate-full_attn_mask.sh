# 设置更长的 NCCL 超时时间
export NCCL_TIMEOUT=7200  # 2 小时


NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  

MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
# MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts

# ✅ 使用 Python 自动查找可用端口
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
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

torchrun --nproc_per_node=$NPROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/eval/final_eval/test_kv-model_evaluate-full_attn_mask.py \
    --model_path  /ours_checkpoints \
    --data_path ./evaluation_samples.json \
    --image_root ./Stream3D-Bench \
    --output_path ./output_results_${TIMESTAMP}.json