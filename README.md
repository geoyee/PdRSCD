# PdRSCD

[![Python 3.7](https://img.shields.io/badge/python-3.7+-yellow.svg)](https://www.python.org/downloads/release/python-370/) [![Paddle 2.1.0](https://img.shields.io/badge/Paddle-2.1.0-red.svg)](https://www.python.org/downloads/release/python-370/) [![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE) ![GitHub Repo stars](https://img.shields.io/github/stars/geoyee/PdRSCD)

PdRSCD（PaddlePaddle Remote Sensing Change Detection）是一个基于飞桨PaddlePaddle的遥感变化检测的项目，pypi包名为ppcd。可以像PaddleSeg一样快速完成变化检测任务。目前项目仍在开发中。

## 在线项目实例

1. [【ppcd快速入门】经典LEVIR数据集变化检测](https://aistudio.baidu.com/aistudio/projectdetail/2117261)
2. [【ppcd快速入门】大图滑框变化检测与拼接]()
3. [【ppcd快速入门】多光谱遥感影像变化检测（待更）]()
4. [【ppcd快速入门】多标签遥感图像变化检测（待更）]()
5. [【ppcd快速入门】分类标签遥感变化检测（待更）]()

## 特点

1. 适应两期图像的读取和增强，支持jpg、tmp、tif和npy等格式，支持多光谱/波段
2. 有更多有特色的数据增强
3. 适应分割图标签、分类标签以及多标签（分割+变化标签）
4. 网络多返回、多标签和多损失之间的组合
5. 适应单通道预测图及双通道预测图的输出（argmax与threshold）
6. 支持大图滑框/随机采样训练和滑框预测与拼接
7. 支持保存为带地理坐标的tif

## 代码结构

PdRSCD的主要代码在ppcd中，文件夹组织如下。可以根据自己的任务修改和添加下面的代码。

```
ppcd
  ├── core  # 包含训练和预测的代码
  ├── datasets  # 包含创建数据列表和定义数据集的代码
  ├── losses  # 包含损失函数的代码
  ├── metrics  # 包含指标评价的代码
  ├── models  # 包含网络模型、特殊层、层初始化等代码
  ├── traditions  # 包含一些传统计算方法的代码
  ├── transforms  # 包含数据增强的代码
  ├── utils  # 包含其他代码，如计时等
  └── tools  # 包含工具代码，如分块、图像查看器等
```

## 现有资产与自定义

1. [自定义数据集](ppcd/datasets/README.md)
2. [模型库与自定义模型](ppcd/models/README.md)
3. [损失函数与自定义损失函数](ppcd/losses/README.md)
4. [数据增强与自定义数据增强](ppcd/transforms/README.md)
5. [传统处理与自定义传统处理](ppcd/traditions/README.md)

## 使用入门

- 可以通过pip使用官方原直接进行安装，目前版本0.1.3。

```shell
pip install ppcd==0.1.3 -i https://pypi.org/simple
```

- 也可以通过克隆PdRSCD到项目中，并添加到环境变量。

```shell
# 克隆项目
# git clone https://github.com/geoyee/PdRSCD.git  # github可能较慢
git clone https://gitee.com/Geoyee/pd-rscd.git
    
import sys
sys.path.append('pd-rscd')  # 加载环境变量
```

## 说明

1. 当前更新后需要在PaddlePaddle2.1.0上运行，否则会卡在DataLoader上。除此之外DataLoader可能还存在问题，例如在一个CPU项目上卡住了，不知道原因，建议在2.1.0及以上版本的GPU设备上运行（至少AI Studio的GPU肯定是没问题的）。
2. 由于GDAL无法直接通过pip安装，所以如果需要使用GDAL的地方目前需要自行安装GDAL。

## 后续重点

1. 支持$N(N>=1)$张输入数据，将分割、多期变化检测与多源数据任务结合。
2. 支持ArcGIS shp的输入输出。

## Git链接

- github：https://github.com/geoyee/PdRSCD.git
- gitee：https://gitee.com/Geoyee/pd-rscd.git

## 交流与反馈

Email：Geoyee@yeah.net
