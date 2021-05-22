## 现有资产

### 1. 模型库

相关模型可以去往ppcd.models下查看，包括参数等等。如果需要使用自建的模型，可在ppcd.models下新建.py文件，完成模型搭建，并在对应的init文件夹下导入自己的模型。有关模型的建立需要注意两点：

- 返回的结果为一个list，就算只有一个结果也需要用“[]”包起来；
- BCELoss中使用了sigmoid函数，在网络的最后不需要添加sigmoid。

| 模型      | 使用                   | 返回列表长度 |
| --------- | ---------------------- | ------------ |
| Fast-SCNN | ppcd.models.FastSCNN() | 1 / 2        |
| UNet      | ppcd.models.UNet()     | 1            |
| SNUNet-CD | ppcd.models.SNUNet()   | 1 / 5        |
| DSIFN     | ppcd.models.DSIFN()    | 1            |
| *STANet   | ppcd.models.STANet()   | 1            |
| *CDMI-Net | ppcd.models.CDMINet()  | 2            |
| *DTCDSCD  | ppcd.models.CDNet34    | 3            |

其中上述模型中前5个模型得到的结果均为变化检测图；后2个模型比较特殊，数据组织和训练方式也有所差别，第6个模型以分类的方式进行训练，得到的结果为特征图和分类结果，需要使用阈值等得到变化检测图；第7个模型得到的结果为变化检测图以及两个时段的分割图。

- **注意**：*号注释的三个模型尚未进行验证，不一定能成功进行训练。所有模型均为与源代码对齐，结果不代表源代码结果。模型仅供参考，最好的用法是自建模型然后在这个流程中进行训练和预测。

### 2. 损失函数

目前的损失函数多参考PaddleSeg，相关代码可以去往ppcd.losses下查看，如需使用自建的损失函数，请参考PaddleSeg自建组件的[说明](https://gitee.com/paddlepaddle/PaddleSeg/blob/release/v2.0/docs/add_new_model.md)。包括自建模型也可参考。

| 损失函数     | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| BCELoss      | 图像二分类交叉熵                                             |
| DiceLoss     | 处理正负样本不均衡                                           |
| MixedLoss    | 可混合使用上面两个损失                                       |
| TripletLoss  | 用于三元组损失计算                                           |
| BCLoss       | 用于STANet中的距离度量                                       |
| ConstLoss    | 返回常数损失，用于网络返回部分不需要计算损失的处理           |
| LabelBCELoss | 分类标签的二分类交叉熵（用于CDMI-Net等使用场景分类完成变化检测的任务） |