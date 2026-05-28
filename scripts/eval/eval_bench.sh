# Use a longer NCCL timeout for long evaluations.
export NCCL_TIMEOUT=7200  # 2 hours


NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  

MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training

# Use Python to find a free port automatically.
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


python -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/evaluation/evaluate.py \
    --model_path  JonnyYu828/Stream3D-VLM-4B \
    --data_path  ./benchmark/stream3d_bench.json  \
    --image_root ./datasets/ \
    --output_path ./output_logs/Stream3D-VLM-4B/evaluate_results${TIMESTAMP}.json