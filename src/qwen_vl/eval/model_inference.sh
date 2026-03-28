# Set a longer NCCL timeout
export NCCL_TIMEOUT=7200  # 2 hours


NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  

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
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

torchrun --nproc_per_node=$NPROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/eval/model_evaluate.py \
    --model_path  /model_checkpoint \
    --data_path ./examples/Stream3D-Bench_example.json \
    --image_root ./examples/images \
    --output_path ./output_results/evaluation_${TIMESTAMP}.json