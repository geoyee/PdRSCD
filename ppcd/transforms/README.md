# 数据增强与自定义数据增强

## 一、数据增强

### 1.1 通用数据增强

数据增强适配了遥感图像多通道的特点，并结合了一些专门的增强方法，相关代码可以去往ppcd.transforms.transforms下查看和修改。其中使用参数均在代码中有所说明。

| 数据增强          | 说明                                     |
| ----------------- | ---------------------------------------- |
| Resize            | 对图像进行大小改变                       |
| Normalize         | 对图像进行标准化                         |
| RandomFlip        | 随机对图像进行翻转                       |
| RandomRotate      | 随机对图像进行旋转                       |
| RandomEnlarge     | 随机对图像进行放大然后裁剪回原大小       |
| RandomNarrow      | 随即对图像进行缩小然后填充回原大小       |
| RandomBlur        | 随机对图像添加高斯模糊                   |
| RandomSharpening  | 随机对图像进行锐化                       |
| RandomColor       | 随机改变图像的对比度和亮度               |
| RandomStrip       | 随机对图像添加条纹噪声                   |
| RandomFog         | 随机对图像加雾效果                       |
| RandomSplicing    | 随机对图像进行拼接不匀色改变             |
| RandomRemoveBand  | 随机移除图像部分波段                     |
| NDVI              | 计算图像的归一化植被指数并叠加在新的通道 |
| NDWI              | 计算图像的归一化水体指数并叠加在新的通道 |
| NDBI              | 计算图像的归一化建筑指数并叠加在新的通道 |
| ExchangeTime      | 将两个时段的图像进行交换                 |
| HistogramMatching | 将第二时段的直方图规定到第一时段         |

其中数据增强支持多通道读入（tif/img/npy/npz/jpg/png）、单/双时段增强、多标签增强。

### 2. RGB图像预处理

对一些RGB图像，有一些预处理方法可能会取得一定的效果，相关代码可以去往ppcd.transforms.enhance下查看和修改。目前这一块还未进行使用的定义。

| 预处理方法          | 说明           |
| ------------------- | -------------- |
| percent_linear      | 线性拉伸       |
| bilateral_filtering | 双边滤波       |
| de_haze             | 暗通道先验去雾 |

## 二、自定义损失函数

同可以参考[PaddleSeg的自定义损失](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/design/create/add_new_model.md)，在ppcd中形式如下：

```python
import paddle.nn as nn

class NewTrans(nn.Layer):
    def __init__(self, param1):
        pass
    
    def __call__(self, self, A_img, B_img, label=None):
        return (A_img, B_img, label)
```

