# 损失函数与自定义损失函数

## 一、损失函数

目前的损失函数多参考PaddleSeg，相关代码可以去往ppcd.losses下查看，如需使用自建的损失函数，请参考PaddleSeg自建组件的[说明](https://gitee.com/paddlepaddle/PaddleSeg/blob/release/v2.0/docs/add_new_model.md)。包括自建模型也可参考。

| 损失函数    | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| BCELoss     | 图像二分类交叉熵                                             |
| DiceLoss    | 处理正负样本不均衡                                           |
| MixedLoss   | 可混合使用上面两个损失                                       |
| TripletLoss | 用于三元组损失计算                                           |
| BCLoss      | 用于STANet中的距离度量                                       |
| ConstLoss   | 返回常数损失，用于网络返回部分不需要计算损失的处理           |
| LabelL1Loss | 分类标签的损失（用于CDMI-Net等使用场景分类完成变化检测的任务） |

## 二、自定义损失函数

参考Paddle的[自定义损失函数](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/quick_start/high_level_api/high_level_api.html#loss)，与PaddleSeg中分割的损失通用。ppcd中定义如下：

```python
import paddle.nn as nn

class NewLoss(nn.Layer):
    def __init__(self, param1):
        pass
    
    def forward(self, x):
        pass
```



## 三、如何使用损失

在ppcd中，损失的计算有三种方式：

1. 单输出+单损失+单标签：计算输出和标签之间的损失；
2. 多输出+多损失+单标签：分别计算每个输出和唯一标签之间对应的损失；
3. 多输出+多损失+多标签：分别计算每个输出与其对应标签之间的对应损失。

所以在使用中一定要注意网络输出的顺序以及loss、标签的顺序。
