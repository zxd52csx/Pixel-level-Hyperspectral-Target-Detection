# Pixel-level Hyperspectral Target Detection 

The implementation of three deep-learning-based pixel-level hyperspectral target detection models. 

## Introduction

[Example of pixel-level hyperspectral target detection (Cross-Scene Detection)](./docs/exp.jpg)

Pixel-level hyperspectral target detection (HTD) aims to identify targets of interests in hyperspectral images (HSIs) based on their known prior spectral signatures (also known as prior spectra). An example of HTD is shown above. 

Due to complex imaging conditions, there are differences between prior spectral and test target spectra, which is known as  spectral variability and limits the performances of HTD models. What is more, the quantity of prior spectra is usually quite limited and the background training samples are generally unavailable. To solve the above problems, we have proposed three deep-learning-based HTD models:

1. Siamese Fully Connected-based Target Detector (SFCTD) [1]
2. Implicit Contrastive Learning-based Target Detector (ICLTD) [2]
3. Self-supervised Deep Clustering-based Target Detector (SSDCTD) [3]

This repository contains the example code of the proposed three HTD models. 

The proposed and comparison HTD methods are evaluated with two kinds of detection pipeline: 

1. Prior spectra and test spectra are collected in the same HSI, denoted as Within-Scene Detection (WSD)
2. Prior spectra and test spectra are collected in different HSIs, denoted as Cross-Scene Detection (CSD)

## Requirements

The code in the repository is written in Python, and the deep learning models are built using PyTorch v1.11. 

## Get started

Get started three model by clicking the above link:

[Get started SFCTD](./docs/SFCTD.md)

[Get started ICLTD](./docs/ICLTD.md)

[Get started SSDCTD](./docs/SSDCTD.md)

## Dataset

WSD is conducted on Airport-Beach-Urban (ABU) datasets, collected by Xudong Kang. We adjust the data structure of ABU-dataset. To realize CSD, we collected multi-temporal Airport-Beach-Urban (MT-ABU) datasets using AVIRIS data.  

The ABU and MT-ABU dataset is available as: 

link: https://pan.baidu.com/s/1WWnD1u8uWDltGg1pDw5lNQ extracting code: 270e

This dataset is provided for **research purposes only** and may **not** be used for commercial purposes. 

## Cite this work

if you find this work help, please cite our work.

SFCTD:

```
@Article{rs14051260,
AUTHOR = {Zhang, Xiaodian and Gao, Kun and Wang, Junwei and Hu, Zibo and Wang, Hong and Wang, Pengyu},
TITLE = {Siamese Network Ensembles for Hyperspectral Target Detection with Pseudo Data Generation},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {5},
ARTICLE-NUMBER = {1260},
}
```

ICLTD:

```
@Article{rs16040718,
AUTHOR = {Zhang, Xiaodian and Gao, Kun and Wang, Junwei and Wang, Pengyu and Hu, Zibo and Yang, Zhijia and Zhao, Xiaobin and Li, Wei},
TITLE = {Target Detection Adapting to Spectral Variability in Multi-Temporal Hyperspectral Images Using Implicit Contrastive Learning},
JOURNAL = {Remote Sensing},
VOLUME = {16},
YEAR = {2024},
NUMBER = {4},
ARTICLE-NUMBER = {718},
}
```

SSDCTD:

```
@article{ZHANG2023103405,
author = {Xiaodian Zhang and Kun Gao and Junwei Wang and Zibo Hu and Hong Wang and Pengyu Wang and Xiaobin Zhao and Wei Li},
title = {Self-supervised learning with deep clustering for target detection in hyperspectral images with insufficient spectral variation prior},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {122},
pages = {103405},
year = {2023},
}
```

## Contact us

Xiaodian Zhang (email: xdz@bit.edu.cn / 2414182986@qq.com)

corresponding author: Kun Gao (email: gaokun@bit.edu.cn)









