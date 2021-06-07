# VI-PointConv-pytorch

This repository contains the official implementation of [The Devils in the Point Clouds: Studying the Robustness of Point Cloud Convolutions](https://arxiv.org/abs/2101.07832) in PyTorch.

## A. Environment Setup

1. Use Anaconda

```
conda env create -f environment.yml
```

## B. Prepare Data

1, Please refer https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py to process the raw data

2, Open './dataGeneration/scannet_data_generateNorms.py', please specify the paths for the raw data ({train_files, val_files, test_files}) from the previous step, and the corresponding saving paths {save_train_path, save_val_path, save_test_path}. Then run 

```
python scannet_data_generateNorms.py
```

## C. Training & Evaluation

1. Training

All hyperparameters are set within **./config.yaml**, please read through and change them correspondingly

```
python train_ScanNet.py
```

2. Evaluation

```
python test_ScanNet.py
```

