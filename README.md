#### 环境安装和配置：

conda create -n stream3d-llm python=3.10 -y

conda activate stream3d-llm

pip install -r requirements.txt

pip install flash-attn==2.7.4.post1 --no-build-isolation

cd xxx/Stream3D-VLM

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

#### inference相关：

推理的代码：[test_kv-model_evaluate-full_attn_mask.py](src/qwen_vl/eval/test_kv-model_evaluate-full_attn_mask.py)

推理的脚本：[test_kv-evaluate-full_attn_mask.sh](src/qwen_vl/eval/test_kv-evaluate-full_attn_mask.sh) 

--model_path即模型所在路径（已上传到huggingface的私有仓库，8B版本）：JonnyYu828/Stream3D-VLM

--data_path即annotation (scannet+scannetpp)： [evaluation_samples.json](evaluation_samples.json) 

--image_root即场景图片所在的路径（已上传到huggingface的私有仓库）：JonnyYu828/Streaming3D-Bench

--output_path：自定义输出结果的json路径，包含 问题的提问时间(question_time)、模型预测的回答时间(prediction_answer_time)、GT回答时间(gt_answer_time)、模型输出的答案(prediction)、GT答案的信息(ground_truth)
