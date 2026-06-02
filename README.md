<h1 align="center">
  <img src="assets/logo.png" height="48" alt="Stream3D-VLM Logo" align="absmiddle">
  &nbsp;Stream3D-VLM: Online 3D Spatial Understanding with Incremental Geometry Priors
</h1>

<p align="center">
    <a href="https://hanxunyu.github.io/" target="_blank">Hanxun Yu<sup>1,2*</sup></a>,
    <a href="https://openreview.net/profile?id=%7EXuan_Qu1" target="_blank">Xuan Qu<sup>1,2*</sup></a>,
    <a href="https://www.kelei.site/" target="_blank">Lei Ke<sup>2</sup></a>,
    <a href="https://cyrilsterling.github.io/" target="_blank">Boqiang Zhang<sup>2</sup></a>,
    <a href="https://w-ted.github.io/" target="_blank">Yuxin Wang<sup>2,3</sup></a>,
    <a href="https://person.zju.edu.cn/en/jkzhu" target="_blank">Jianke Zhu<sup>1,4</sup></a>,
    <a href="https://dongyu888.github.io/" target="_blank">Dong Yu<sup>2</sup></a>
    <br>
    <sup>1</sup>Zhejiang University,
    <sup>2</sup>Tencent Hunyuan,
    <sup>3</sup>HKUST,
    <sup>4</sup>Shenzhen Loop Area Institute
</p>

<div align="center">
    <a href='xxxxx' target="_blank"><img src='https://img.shields.io/badge/arXiv-XXXX-b31b1b?logo=arxiv&logoColor=red'></a>  
    <a href='https://stream3d-vlm.github.io/' target="_blank"><img src='https://img.shields.io/badge/Project-Home%20Page-Green?logo=safari&logoColor=white'></a>  
    <a href='https://huggingface.co/JonnyYu828/Stream3D-VLM-4B' target="_blank">
        <img src='https://img.shields.io/badge/%F0%9F%93%A6%EF%B8%8F%20Hugging%20Face-Model-orange'>
    </a>
    <a href='https://huggingface.co/datasets/JonnyYu828/Stream3D-1M-Dataset' target="_blank">
    <img src='https://img.shields.io/badge/%F0%9F%93%9A%20Hugging%20Face-Dataset-blue'>
</a>
<a href='https://huggingface.co/datasets/JonnyYu828/Stream3D-Bench' target="_blank">
    <img src='https://img.shields.io/badge/%F0%9F%8F%86%20Hugging%20Face-Benchmark-blueviolet'>
</a>
</div>


https://github.com/user-attachments/assets/41888813-b040-4305-9188-ceb010ad84d1



## 🔍 Overview

<div align="left">
<img src="assets/pipeline.png" width="99%" alt="model">
</div>

**Stream3D-VLM** is an online 3D vision-language model that supports real-time spatial understanding and interaction directly from streaming video. By incrementally integrating geometry priors and employing geometry-adaptive voxel compression, our approach enables efficient and continuous 3D scene comprehension without requiring offline processing or complete scene observations.


## 📰 News

- [2026-03-27] 🔥 We release [Stream3D-1M Dataset](https://huggingface.co/datasets/JonnyYu828/Stream3D-1M-Dataset) and [Stream3D-Bench](https://huggingface.co/datasets/JonnyYu828/Stream3D-Bench) in Hugging Face 🤗.
- [2026-03-27] 🔥 We release the checkpoint of [Stream3D-VLM-4B](https://huggingface.co/JonnyYu828/Stream3D-VLM-4B) in Hugging Face 🤗.
- [2026-03-27] 🔥 We release the training and inference code.
- [2026-03-27] 🔥 We release the [paper](xxxxx) of Stream3D-VLM.


## 🛠️ Installation

```
git clone https://github.com/hanxunyu/Stream3D-VLM.git
cd Stream3D-VLM
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

conda create -n stream3d-llm python=3.10 -y
conda activate stream3d-llm
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```
## 📊 Datasets and Benchmark
<div align="left">
<img src="assets/data_generation.png" width="99%" alt="data_generation">
</div>

**Illustration of our data generation pipeline.** Guided by a comprehensive task taxonomy spanning five cognitive competencies and three temporal interaction modes, the pipeline leverages detailed metadata from RGB-D video streams and a hybrid generation strategy to construct a large-scale spatio-temporal 3D QA dataset and the Stream3D-Bench for evaluating online 3D spatial understanding.

- Due to licensing restrictions, we are unable to directly release the source media. Instead, we provide annotations for the [Stream3D-1M Dataset](https://huggingface.co/datasets/JonnyYu828/Stream3D-1M-Dataset) and [Stream3D-Bench](https://huggingface.co/datasets/JonnyYu828/Stream3D-Bench). Please refer to [dataset_description.md](./dataset_description.md) for detailed dataset information.

## 📦️ Pretrained Models
We provide the pretrained model [Stream3D-VLM-4B](https://huggingface.co/JonnyYu828/Stream3D-VLM-4B) in Hugging Face 🤗. 

## 🤖 Inference Examples
Run our example inference script:
```bash
bash scripts/eval/eval_examples.sh
```

To evaluate on [Stream3D-Bench](https://huggingface.co/datasets/JonnyYu828/Stream3D-Bench), set `--data_path` to the benchmark annotation file and `--image_root` to the directory containing the source media, then run:
```bash
bash scripts/eval/eval_bench.sh
```

After inference, set `input_file` in the corresponding metric script to the generated result file and compute the metrics:
```bash
# NA and MCA metrics
python scripts/comput_metrics/compute_score-NA+MCA.py

# OEA metric
# Set `api_key` and `base_url` before running.
python scripts/comput_metrics/compute_score-OEA.py

# Answer-Timing Accuracy (ATA)
python scripts/comput_metrics/compute_answer-timing-accuracy.py

# Response latency (TTFT and end-to-end latency)
python scripts/comput_metrics/compute_response-latency.py
```
<!-- > *Due to further refinements and updates to the dataset and benchmark, the model’s actual performance may differ slightly from the results reported in the paper.* -->
## 🚀 Training

Configure the datasets in `src/qwen_vl/data/__init__.py`, set `DATASETS` in the training script, and run:
```bash
bash scripts/train/train_stream3d-vlm.sh
```

## 🎥 Visualization

### 🌐 Incremental 3D Reconstruction

We use [StreamVGGT](https://github.com/wzzheng/StreamVGGT) to generate point clouds that visualize the streaming 3D perception process. Two output formats are supported:

- **GLB** — incremental step-by-step point clouds viewable in any 3D viewer
- **RRD** — [Rerun](https://rerun.io/) recording with camera poses and point clouds

#### Setup

```bash
# For GLB output:
pip install roma trimesh scipy matplotlib

# For RRD output:
pip install roma rerun-sdk

# Checkpoint will be auto-downloaded from HuggingFace on first run,
# or specify a local path via --ckpt.
```

#### Usage

**GLB (incremental point clouds):**
```bash
bash visualization/run_incremental_reconstruction.sh
```

**RRD (Rerun recording):**
```bash
bash visualization/run_incremental_reconstruction_rrd.sh
```

Edit the scripts to configure input/output paths and parameters (e.g., `INPUT_DIR`, `CKPT_PATH`, `STEP`).

#### Output

- **GLB**: `incremental_steps/step_XXXX.glb` (each incremental step) and `scene_final_conf{thres}.glb` (final full reconstruction)
- **RRD**: `scene_final_conf{thres}.rrd` (final reconstruction with camera poses and point clouds), viewable via:
  ```bash
  rerun output/scene_final_conf50.0.rrd
  ```

### 🕘 Demo 1-Backward Tracing (Memory)

https://github.com/user-attachments/assets/2b479d23-cfba-4e4e-a226-e2d94b024731


### 👁️ Demo 2-Realtime Perception (Observation)

https://github.com/user-attachments/assets/eceea976-8004-4db2-b977-7dee083e8aca


### ⏩ Demo 3-Forward Response (Monitoring)

https://github.com/user-attachments/assets/c1f2083f-cc4a-4ef7-ab09-72edfa351f65


## 👏 Acknowledgements
We are grateful for the open-source contributions of other projects:
- [StreamVGGT](https://github.com/wzzheng/StreamVGGT)
- [VG-LLM](https://github.com/LaVi-Lab/VG-LLM)
- [VLM-3R](https://github.com/VITA-Group/VLM-3R)

## 📑 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🖊️ Citation

```BibTeX

```
