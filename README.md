# 🔥 FLARE23 Solution

This repository contains our solution for the FLARE23 challenge, based on nnU-Netv2.

## 🔍 Overview

Our approach employs a two-stage pseudo-labeling method to tackle the issue of partial labels for organs and tumors in the FLARE23 dataset. For more details, see the pipeline diagram below:

![Pipeline Diagram](./assets/pipeline.png)

Additionally, we present a speed comparison for inference, both before and after optimizations on nnU-Netv2. The following diagram illustrates the improvements:

<img src="./assets/efficiency.png" alt="Efficiency Comparison" width="600"/>

## ⚙️ Environment Setup

To set up the environment, follow these steps:

```
conda create -n FLARE23_blackbean
conda activate FLARE23_blackbean
```
Then make sure to install PyTorch 2 compatible with your CUDA version.
```
pip install -e .
```

## ⬇️ Download Checkpoints

Download the `checkpoint_final.pth` file from [BaiduNetDisk](https://pan.baidu.com/s/1Nt_ZD2lyp4mS9UA5Xeajuw?pwd=jip3). Place it in the `./model/fold_all/` directory.

## 🚀 Inference

1. Place your input images in the `./inputs` directory.
2. Run the prediction script:

```
sh predict.sh
```

This will generate the output in the `./outputs` directory.
