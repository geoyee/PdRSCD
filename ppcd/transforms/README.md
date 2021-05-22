## 现有资产

### 1. 数据增强

数据增强适配了遥感图像多通道的特点，并结合了一些专门的增强方法，相关代码可以去往ppcd.transforms.transforms下查看和修改。其中使用参数均在代码中有所说明。

| 数据增强         | 说明                                     |
| ---------------- | ---------------------------------------- |
| Resize           | 对图像进行大小改变                       |
| Normalize        | 对图像进行标准化                         |
| RandomFlip       | 随机对图像进行翻转                       |
| RandomRotate     | 随机对图像进行旋转                       |
| RandomEnlarge    | 随机对图像进行放大然后裁剪回原大小       |
| RandomNarrow     | 随即对图像进行缩小然后填充回原大小       |
| RandomBlur       | 随机对图像添加高斯模糊                   |
| RandomSharpening | 随机对图像进行锐化                       |
| RandomColor      | 随机改变图像的对比度和亮度               |
| RandomStrip      | 随机对图像添加条纹噪声                   |
| RandomFog        | 随机对图像加雾效果                       |
| RandomSplicing   | 随机对图像进行拼接不匀色改变             |
| RandomRemoveBand | 随机移除图像部分波段                     |
| NDVI             | 计算图像的归一化植被指数并叠加在新的通道 |
| NDWI             | 计算图像的归一化水体指数并叠加在新的通道 |
| NDBI             | 计算图像的归一化建筑指数并叠加在新的通道 |
| ExchangeTime     | 将两个时段的图像进行交换                 |

其中数据增强支持多通道读入（tif/img/npy/npz/jpg/png）、单/双时段增强、多标签增强。