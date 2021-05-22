## 现有资产

### 1. 损失函数

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

