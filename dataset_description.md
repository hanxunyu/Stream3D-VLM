# Stream3D Dataset

The Stream3D Dataset contains over 1 million online spatio-temporal 3D QA pairs across 5.2k indoor videos, spanning 29 task types that require streaming video spatial understanding.

We release the following:

- **[Stream3D-1M Dataset](https://huggingface.co/datasets/JonnyYu828/Stream3D-1M-Dataset)** — over 1 million QA pairs from 5.2k videos for instruction tuning.
- **[Stream3D-Bench](https://huggingface.co/datasets/JonnyYu828/Stream3D-Bench)** — a high-quality benchmark of 10k samples spanning 518 videos for evaluating online 3D spatial understanding.

---

## 🧭 Task Taxonomy

The dataset is organized along two orthogonal dimensions:

### 🧠 Cognitive Competencies

| Category | Description | Example Tasks |
|----------|-------------|---------------|
| **Ego-Motion Estimation** | Captures the agent's own motion | path length, rotation angles, displacement |
| **Environment Measurement** | Quantifies scene-level properties | room area, inter-object distances |
| **Object–Camera Relationship** | Spatial relations between the agent and objects | camera-to-object distance, relative direction |
| **Object Chronology** | Tracks objects over time | object counting, appearance order, first/last seen time |
| **Object Attributes** | Recognizes object-level features | category, color, spatial location |

### ⏱️ Temporal Interaction Modes

| Mode | Description |
|------|-------------|
| **Backward (Memory)** | Retrieves information from past frames invisible in the current view, probing long-term memory. |
| **Realtime (Observation)** | Grounds responses in the current frame, emphasizing immediate spatial perception. |
| **Forward (Monitoring)** | Requires continuous monitoring of incoming frames and responding only when future conditions are satisfied. |

The full taxonomy covers **29 task types** formed by combining the above competencies and modes.

---

## 🗂️ Directory Structure

```
.
├── Stream3D-1M-Dataset/
│   ├── ca1m_ego_motion_estimation_train.json
│   ├── scannet_ego_motion_estimation_train.json
│   ├── scannet_environment_measurement_train.json
│   ├── scannet_object_attributes_train.json
│   ├── scannet_object_camera_relationship_train.json
│   ├── scannet_object_chronology_train.json
│   ├── scannetpp_ego_motion_estimation_train.json
│   ├── scannetpp_environment_measurement_train.json
│   ├── scannetpp_object_camera_relationship_train.json
│   └── scannetpp_object_chronology_train.json
└── Stream3D-Bench/
    └── stream3d_bench_10k.json
```

### 🏷️ File Naming Convention

Training files follow the pattern: `{source}_{task_category}_train.json`

- **Source**: `scannet`, `scannetpp`, or `ca1m` (corresponding to ScanNet, ScanNet++, and ARKitScenes respectively)
- **Task category**: `ego_motion_estimation`, `environment_measurement`, `object_attributes`, `object_camera_relationship`, or `object_chronology`

---

## 🧾 Data Format

Each JSON file contains an array of QA samples. Below is the schema description.

### 🔑 Fields (Training & Benchmark)

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | string | Unique identifier for the QA sample |
| `scene_id` | string | Scene identifier from the source dataset |
| `dataset_type` | string | Source dataset: `"scannet"`, `"scannetpp"`, or `"ca1m"` |
| `question_type` | string | Specific task type (e.g., `"camera_displacement_forward"`, `"object_recognition_realtime"`) |
| `question_mode` | string | Temporal interaction mode: `"realtime"`, `"backward"`, or `"forward"` |
| `question` | string | The question text |
| `question_frame` | int | Frame index at which the question is posed |
| `question_time` | float | Timestamp (seconds) at which the question is posed |
| `answers` | array | List of answer objects (see below) |
| `test_type` | string | Answer evaluation type: `"open"`, `"numerical"`, or `"choice"` |
| `question_for_test` | string | Formatted question used during evaluation (may include answer format instructions) |
| `answer_for_test` | string | Ground-truth answer for evaluation |
| `answer_for_test_frame` | int | Frame index at which the ground-truth answer is given |
| `answer_for_test_time` | float | Timestamp at which the ground-truth answer is given |
| `sampling_fps` | float | Sampling frame rate used for frame extraction |
| `native_fps` | float | Native video frame rate of the source video |

#### ✅ `answers` Array Schema

Each element in the `answers` array is an object with:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | The answer text |
| `frame` | int | Frame index at which this answer is given |
| `time` | float | Timestamp (seconds) at which this answer is given |
| `type` | string (optional) | For forward-mode multi-step answers: `"immediate"` (initial acknowledgement) or `"final"` (actual answer when condition is met) |

---

## 🧪 Examples

### 🚶 Ego-Motion Estimation — Backward (Numerical)

```json
{
    "question_id": "scene0169_01_4_camera_path_backward",
    "scene_id": "scene0169_01",
    "dataset_type": "scannet",
    "question_type": "camera_path_backward",
    "question_mode": "backward",
    "question": "Can you tell me the total ground covered by the camera over the preceding 10 seconds?",
    "question_frame": 420,
    "question_time": 14.0,
    "answers": [
        {
            "answer": "Calculation shows the camera moved a total distance of 3.0 meters during the preceding 10 seconds.",
            "frame": 420,
            "time": 14.0
        }
    ],
    "test_type": "numerical",
    "question_for_test": "Can you tell me the total ground covered by the camera over the preceding 10 seconds? (Answer with a single number in meters)",
    "answer_for_test": "3.0",
    "answer_for_test_frame": 420,
    "answer_for_test_time": 14.0,
    "sampling_fps": 1.0,
    "native_fps": 30.0
}
```

### 🔎 Object Finding — Forward (Open-Ended)

```json
{
    "question_id": "scene0389_00_9_object_finding_forward",
    "scene_id": "scene0389_00",
    "dataset_type": "scannet",
    "question_type": "object_finding_forward",
    "question_mode": "forward",
    "question": "I am controlling the camera to explore. Locate the mini fridge for me. Once visible, tell me its distance and clock position.",
    "question_frame": 750,
    "question_time": 25.0,
    "answers": [
        {
            "answer": "Understood. I am scanning for the mini fridge as you move.",
            "frame": 750,
            "time": 25.0,
            "type": "immediate"
        },
        {
            "answer": "The mini fridge is now visible. It is located 1.4 meters away, at 12 o'clock.",
            "frame": 990,
            "time": 33.0,
            "type": "final"
        }
    ],
    "test_type": "open",
    "question_for_test": "I am controlling the camera to explore. Locate the mini fridge for me. Once visible, tell me its distance and clock position.",
    "answer_for_test": "The mini fridge is now visible. It is located 1.4 meters away, at 12 o'clock.",
    "answer_for_test_frame": 990,
    "answer_for_test_time": 33.0,
    "sampling_fps": 1.0,
    "native_fps": 30.0
}
```

### 📏 Object–Camera Relative Distance — Realtime (Multiple Choice)

```json
{
    "question_id": "scene0552_00_0_object2camera_reldistance_realtime",
    "scene_id": "scene0552_00",
    "dataset_type": "scannet",
    "question_type": "object2camera_reldistance_realtime",
    "question_mode": "realtime",
    "question": "Look at these items: (A) toaster oven, (B) soap dispenser, (C) microwave, (D) sink. Which one has the longest distance to my current position?",
    "question_frame": 360,
    "question_time": 12.0,
    "answers": [
        {
            "answer": "soap dispenser.",
            "frame": 360,
            "time": 12.0
        }
    ],
    "test_type": "choice",
    "question_for_test": "Look at these items: (A) toaster oven, (B) soap dispenser, (C) microwave, (D) sink. Which one has the longest distance to my current position?\nPlease answer with the option letter only (e.g., A).",
    "answer_for_test": "B",
    "answer_for_test_frame": 360,
    "answer_for_test_time": 12.0,
    "sampling_fps": 1.0,
    "native_fps": 30.0
}
```

---

## 📊 Evaluation Metrics

Stream3D-Bench employs task-specific evaluation metrics:

| Answer Type | Metric | Description |
|-------------|--------|-------------|
| Numerical | Mean Relative Accuracy | Measures accuracy relative to ground-truth numerical values |
| Multiple-Choice | Exact Match | Checks if the predicted option matches the correct answer |
| Open-Ended | LLM-as-a-Judge | Uses GPT-4o to assess answer correctness |

Additionally, we introduce **Answer-Timing Accuracy (ATA)** to measure temporal response precision in streaming tasks:

$$S(t_{\text{pred}}) = \mathbb{1}(t_{\text{pred}} \geq t_{\text{gt}}) \cdot \exp\left(-\beta \cdot (t_{\text{pred}} - t_{\text{gt}})\right)$$

where $t_{\text{pred}}$ is the predicted response time, $t_{\text{gt}}$ is the earliest answerable ground-truth timestamp, and $\beta = 0.5$ is the delay penalty factor. ATA is computed by averaging the timing score over all samples.

---

## 🧬 Data Sources

The dataset is constructed from the three widely used 3D datasets:

| Source Dataset | Identifier in JSON | Description |
|----------------|-------------------|-------------|
| [ScanNet](http://www.scan-net.org/) | `scannet` | Indoor RGB-D scans with instance segmentation |
| [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) | `scannetpp` | High-fidelity indoor scans with dense annotations |
| [ARKitScenes](https://github.com/apple/ARKitScenes) / [CA-1M](https://github.com/apple/ml-cubifyanything) | `ca1m` | Large-scale real-world indoor scenes captured with ARKit |

### ❓ Why `ca1m` instead of `arkitscenes`?

[CA-1M (CubifyAnything-1M)](https://github.com/apple/ml-cubifyanything) shares the same underlying raw sensor captures (iPad Pro RGB-D videos) as ARKitScenes, but additionally provides ground-truth camera poses registered to FARO laser scans. We download from CA-1M to leverage these high-quality GT poses, which are essential for our Ego-Motion Estimation tasks. Therefore, while our paper refers to the data source as "ARKitScenes" (since the scenes originate from that collection), the file naming uses `ca1m` to reflect the actual download source and its superior pose annotations.

