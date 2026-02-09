<p align="center">
<h1 align="center"><strong>Stream3D-VLM: Online 3D Spatial Understanding
with Incremental Geometry Priors</strong></h1>
<!-- <h3 align="center">Arxiv 2025</h3> -->

<p align="center">
    <a href="https://hanxunyu.github.io/" target="_blank">Hanxun Yu<sup>1,2*</sup></a>,
    <a href="https://github.com/Select-ing" target="_blank">Xuan Qu<sup>1,2*</sup></a>,
    <a href="https://www.kelei.site/" target="_blank">Lei Ke<sup>2</sup></a>,
    <a href="https://cyrilsterling.github.io/" target="_blank">Boqiang Zhang<sup>2</sup></a>,
    <a href="https://w-ted.github.io/" target="_blank">Yuxin Wang<sup>2,3</sup></a>,
    <a href="https://person.zju.edu.cn/jkzhu" target="_blank">Jianke Zhu<sup>1&dagger;</sup></a>,
    <a href="https://dongyu888.github.io/" target="_blank">Dong Yu<sup>2</sup></a>
    <br>
    <sup>1</sup>ZJU,
    <sup>2</sup>Tencent AI Lab,
    <sup>3</sup>HKUST
</p>

<div align="center">
    <a href='https://arxiv.org/abs/2512.16561' target="_blank"><img src='https://img.shields.io/badge/arXiv-2512.16561-b31b1b.svg'></a>  
    <a href='' target="_blank"><img src='https://img.shields.io/badge/Project-Home%20Page-Green'></a>  
    <a href='' target="_blank">
        <img src='https://img.shields.io/badge/Hugging%20Face-Dataset%20%26%20Benchmark-blue'>
    </a>
    <a href='' target="_blank">
        <img src='https://img.shields.io/badge/Hugging%20Face-Models-orange'>
    </a>
</div>

https://github.com/user-attachments/assets/a2788b03-6eb1-4e18-ad0a-904e6a408992


## 🔍 Overview

<div align="left">
<img src="assets/pipeline.png" width="99%" alt="Inst3D-LLM">
</div>

**Stream3D-VLM** is an online 3D vision-language model that supports real-time spatial understanding and interaction directly from streaming video. By incrementally integrating geometry priors and employing geometry-adaptive token compression, our approach enables efficient and continuous 3D scene comprehension without requiring offline processing or complete scene observations.


## 📰 News

- **`2025/12/19`**: We released this repo with the pre-trained model and inference code.


## 🛠️ Installation

```
git clone --recursive https://github.com/W-Ted/N3D-VLM.git
cd N3D-VLM

conda env create -n n3d_vlm python=3.11 -y
conda activate n3d_vlm
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 📦️ Pre-trained models
We provide the pre-trained models [yuxinhk/N3D-VLM](https://huggingface.co/yuxinhk/N3D-VLM) in Hugging Face 🤗. 

## 🚀 Training

## 🤖 Inference 
We provide three examples for inference of N3D-VLM. You could check the source files in `data` directory, where `*.jpg` are the source images and `*.npz` are the monocular point clouds obtained by using [MoGe2](https://github.com/microsoft/moge). 
```
# inference 
python demo.py
```

### Demo 1


https://github.com/user-attachments/assets/e86306f2-152d-4337-a8d2-d165a26ce305


### Demo 2


https://github.com/user-attachments/assets/1bc0ee64-7a15-4592-941d-1037a26fb108


### Demo 3


https://github.com/user-attachments/assets/ba7ece12-4288-411d-9964-c676b78c6d5c


After running the code above, the inference results will be saved in the `outputs` directory, including generated answers in `*.json` format, and 3D grounding results in `*.rrd` format. 
The rrd files can be visualized by using [Rerun](https://rerun.io):
```
rerun outputs/demo1.rrd
```

If you want to do the 3D Detection only, please check the example as below. 
```
# inference 
python detection.py
# visualization
rerun outputs/test1.rrd
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Citation

```BibTeX
@article{wang2025n3d,
    title={N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models},
    author={Wang, Yuxin and Ke, Lei and Zhang, Boqiang and Qu, Tianyuan and Yu, Hanxun and Huang, Zhenpeng and Yu, Meng and Xu, Dan and Yu, Dong},
    journal={arXiv preprint arXiv:2512.16561},
    year={2025}
}
```



