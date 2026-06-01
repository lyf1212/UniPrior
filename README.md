## Towards Generalized Representations for Low-Light Understanding: When Signal Constancy Meets Semantic Enrichment [**CVPR'26**]

Author: [Yifan Li](https://lyf1212.github.io/), [Haofeng Huang](https://huangerbai.github.io/), [Wenhan Yang](https://flyywh.github.io/), and [Jiaying Liu](http://39.96.165.147/people/liujiaying.html).

<div align="center">
    <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Towards_Generalized_Representations_for_Low-Light_Understanding_When_Signal_Constancy_Meets_CVPR_2026_paper.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Paper-orange" alt="paper"></a>
    <a href="https://lyf1212.github.io/lyf1212bachelor.github.io/UniPrior/" target="_blank">
    <img src="https://img.shields.io/badge/Project Page-blue" alt="Project Page"/></a>
</div>
</h2>

This is the official PyTorch code for our paper: Towards Generalized Representations for Low-Light Understanding: When Signal Constancy Meets Semantic Enrichment.

## 👓 Our **Key Features** includes:
- a **unified prior** based low-light adaptation framework that integrates the
general semantic prior embedded in VFMs and physical illumination-invariant priors.

- a **contrastive regularization** scheme based on the
VFM-driven spatial intra-feature correlation to enrich representation.

- a **test-time adaptation** paradigm for extreme hard cases, which injects signal prior and enriched semantics into a single-input low-light enhancement process.

 
## Setup
### 🖥️ Environment Preparation
We highly recommend to create a new conda environment and run:
```
pip install -r requirements.txt
```
We have tested our model on CUDA11.8. 
We provide some basic instructions in `requirements.txt`.
You can download correct version of `torch`, `torchvision` from [this website](https://pytorch.org/get-started/previous-versions/).

### 🔥 Checkpoint download
You can download our pretrained ckpt in [Google Drive](https://drive.google.com/drive/folders/1h3mq_4JII-QoIdZy0RlmtXWr9kh9mhc-?usp=drive_link).

Put the ckpts in folder `pretrained_ckpt`.

Complete `tools/path_register.py` with your personal dinov2 checkpoint (vitb-14) and our provided prior weights (`prior_conv.pth`).


### 💾 Dataset Preparation
We use [CoDaN](https://github.com/Attila94/CIConv) to train and test classification model.

Complete `CODAN_ROOT` in `train.py` to use this dataset.

We will add more instructions about segmentation and face detection in few months. Stay tuned!

## 🚀 Test
```
cd UniPrior/classification

bash test.sh
```
The result is tested to be `Nighttime accuracy: 74.60%` in RTX 4090 GPU within a few seconds.


### 🏃 Train
```
cd UniPrior/classification

bash train.sh
```


### ✅ TODO

Sorry for the inconvience, but the original implementation is complex and takes a lot of time to make them clean.
We will release the codes within a few months.
- [x] Release classification train and test code.
- [ ] Release segmentation and face detection code.
- [ ] Release TTA code.

-------



If you have any questions, you can submit an Issue or contact liyifan02@stu.pku.edu.cn.

If you find our code useful, please consider citing our paper.

```
@InProceedings{uniprior,
  title={Towards Generalized Representations for Low-Light Understanding: When Signal Constancy Meets Semantic Enrichment}, 
  author={Yifan Li, Haofeng Huang, Wenhan Yang, and Jiaying Liu},
  booktitle={CVPR}, 
  year={2026},
}
```

-------
