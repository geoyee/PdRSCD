# 模型库与自定义模型

## 一、模型库

相关模型可以去往ppcd.models下查看，包括参数等等。如果需要使用自建的模型，可在ppcd.models下新建.py文件，完成模型搭建，并在对应的init文件夹下导入自己的模型。有关模型的建立需要注意两点：

- 返回的结果为一个list，就算只有一个结果也需要用“[]”包起来；
- BCELoss中使用了sigmoid函数，在网络的最后不需要添加sigmoid。

| 模型      | 使用                   | 输入图像列表张数 | 返回列表长度 |
| --------- | ---------------------- | ---------------- | ------------ |
| Fast-SCNN | ppcd.models.FastSCNN() | $N(N\ge1)$       | 1 / 2        |
| UNet      | ppcd.models.UNet()     | $N(N\ge1)$       | 1            |
| SNUNet-CD | ppcd.models.SNUNet()   | 2                | 1 / 5        |
| DSIFN     | ppcd.models.DSIFN()    | 2                | 1            |
| STANet    | ppcd.models.STANet()   | 2                | 1            |
| *CDMI-Net | ppcd.models.CDMINet()  | 2                | 2            |
| *DTCDSCD  | ppcd.models.CDNet34    | 2                | 3            |

其中上述模型中前5个模型得到的结果均为变化检测图；后2个模型比较特殊，数据组织和训练方式也有所差别，第6个模型以分类的方式进行训练，得到的结果为特征图和分类结果，需要使用阈值等得到变化检测图；第7个模型得到的结果为变化检测图以及两个时段的分割图。

- **注意**：*号注释的两个模型尚未进行验证，不一定能成功进行训练（主要是没那种数据）。所有模型均未与源代码对齐，结果不代表源代码结果。模型仅供参考，最好的用法是自建模型然后在这个流程中进行训练和预测。

## 二、自定义模型

目前的模型与分割模型的定义相似，只是在``forward``中需要list作为输入。分割模型的自定义可以参考[PaddleSeg自定义模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/design/create/add_new_model.md)，在ppcd中构造如下：

```python
import paddle.nn as nn


class NewNet(nn.Layer):
    def __init__(self, param1, param2, param3):
        pass
    
    def forward(self, imgs):
        # 这里的imgs是一个列表，输入图像为[img_1, img_2, img_3, ……]
        return [out]
```

