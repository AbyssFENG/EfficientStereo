<!-- PROJECT LOGO -->
<h1 align="center">EfficientStereo: Real time stereo matching based on lightweight feature extraction and disp dimensional convolution</h1>
<h1 align="center">EfficientStereo: 基于轻量级特征提取和深度维度卷积的实时立体匹配</h1>
  
## Highlighted features
- 使用非常简单的特征提取，GWC构建代价体积，使用沙漏结构进行聚合，在沙漏中插入深度维度卷积模块，使用softmax对disp进行视差回归。
  
- Using very simple feature extraction, GWC constructs the cost volume, uses the hourglass structure for aggregation, inserts the depth dimension convolution module in the hourglass, and uses softmax to
predict the final disparity map.

## Getting Started
你可以直接运行train.py:

You can directly run train.py:
```
python train.py
```
可以直接从Models文件夹里面修改网络结构。

You can modify the network structure from the Models folder.
## Requirements
```
necessary:
-pytorch
-numpy
-chardet
-PIL

optional:
-fvcore
-tensorboard
-torchinfo
```

**Note**: 
This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.

此代码仅用于学术目的，不能将此代码用于任何可能被视为商业用途的事情。
