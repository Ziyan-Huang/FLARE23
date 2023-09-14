# FLARE23 Solution

This repository contains our solution for the FLARE23 challenge, based on nnU-Netv2.

## Environment Setup

To set up the environment, follow these steps:

```
conda create -n FLARE23_blackbean
conda activate FLARE23_blackbean
```
Then make sure to install PyTorch 2 compatible with your CUDA version.
```
pip install -e .
```

## Download Checkpoints

Download the `checkpoint_final.pth` file from [BaiduNetDisk](https://pan.baidu.com/s/1Nt_ZD2lyp4mS9UA5Xeajuw?pwd=jip3). Place it in the `./model/fold_all/` directory.

## Inference

1. Place your input images in the `./inputs` directory.
2. Run the prediction script:

```
sh predict.sh
```

This will generate the output in the `./outputs` directory.
