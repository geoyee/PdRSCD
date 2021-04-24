# PdRSCD

PdRSCD（PaddlePaddle Remote Sensing Change Detection）是一个基于飞桨PaddlePaddle，仿造PaddleSeg制作的一个用于遥感变化检测的工具。其中主要的区别在于数据读取器和数据增强可以接收和同时处理两期多通道影像。可以基于此使用工具中自带的网络或搭建自己的网络快速进行变化检测任务的训练和预测。

## 现有资产

### 1. 模型库

相关模型可以去往ppcd.models下查看，包括参数等等。如果需要使用自建的模型，可在ppcd.models下新建.py文件，完成模型搭建，并在对应的init文件夹下导入自己的模型。有关模型的建立需要注意两点：

- 返回的结果为一个list，就算只有一个结果也需要用“[]”包起来；
- BCELoss中使用了sigmoid函数，在网络的最后不需要添加sigmoid。

| 模型               | 使用                   | 返回列表长度 |
| ------------------ | ---------------------- | ------------ |
| Fast-SCNN          | ppcd.models.FastSCNN() | 1 / 2        |
| UNet               | ppcd.models.UNet()     | 1            |
| SNUNet-CD          | ppcd.models.SNUNet()   | 1 / 5        |
| DSIFN              | ppcd.models.DSIFN()    | 1            |
| STANet             | ppcd.models.STANet()   | 1            |
| CDMI-Net（建设中） | ppcd.models.CDMINet()  | ？           |

### 2. 损失函数

目前的损失函数多参考PaddleSeg，相关代码可以去往ppcd.losses下查看，如需使用自建的损失函数，请参考PaddleSeg自建组件的[说明](https://gitee.com/paddlepaddle/PaddleSeg/blob/release/v2.0/docs/add_new_model.md)。包括自建模型也可参考。

| 损失函数    | 说明                   |
| ----------- | ---------------------- |
| BCELoss     | 二分类交叉熵           |
| DiceLoss    | 处理正负样本不均衡     |
| MixedLoss   | 可混合使用上面两个损失 |
| TripletLoss | 用于三元组损失计算     |
| BCLoss      | 用于STANet中的距离度量 |

### 3. 数据增强

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

## 代码结构

PdRSCD的主要代码在ppcd中，组织如下。可以根据自己的任务修改和添加下面的代码。

```
ppcd
  ├── core  # 包含训练和预测的代码
  ├── datasets  # 包含创建数据列表和定义数据集的代码
  ├── losses  # 包含损失函数的代码
  ├── metrics  # 包含指标评价的代码
  ├── models  # 包含网络模型、特殊层、层初始化等代码
  └── transforms  # 包含数据增强的代码
```

## 使用入门

1. 因为目前没有程序包，所以需要首先克隆到项目中，并添加到环境变量。

```python
# 克隆项目（终端操作）
# ! git clone https://github.com/geoyee/PdRSCD.git  # github可能较慢
! git clone https://gitee.com/Geoyee/pd-rscd.git
    
import sys
sys.path.append('pd-rscd')  # 加载环境变量
```

2. 准备数据集，如果数据集是如下格式，可以通过create_list进行创建。

```
dataset
   ├── train  # 训练数据
   |     ├── A  # 时段一
   |     ├── B  # 时段二
   |     └── label  # 变化标签
   ├── val  # 评估数据
   |     ├── A
   |     ├── B
   |     └── label
   └── infer  # 预测数据
         ├── A
         └── B
```

```python
from ppcd.datasets import create_list

datas_path = "datas"  # 数据路径
train_list_path = create_list(datas_path, mode='train')  # 训练数据
val_list_path = create_list(datas_path, mode='val')  # 评估数据
infer_list_path = create_list(datas_path, mode='infer')  # 预测数据
```

3. 生成数据列表后，即可生成对应的数据集，并根据需要使用数据增强。这里只需要从ppcd.datasets以及ppcd.transforms导入数据集和增强方法即可。

```python
from ppcd.datasets import CDataset
import ppcd.transforms as T  # 多种transforms方法可以查看ppcd.transforms中

# 数据增强为一个list
train_transforms = [T.RandomFlip(), T.Resize(512)]
val_transforms = [T.Resize(512)]
# 使用数据列表构造对应的数据，预测数据记得把is_infer设置为True，这样数据读取每次只返回两张图片（没有label）
train_data = CDataset('Dataset/train_list.txt', transforms=train_transforms)
val_data = CDataset('Dataset/val_list.txt', transforms=val_transforms)
infer_data = CDataset('Dataset/infer_list.txt', transforms=val_transforms, is_infer=True)
```

4. 接下来就可以进行训练的准备了。从ppcd.models导入网络，从ppcd.losses导入损失函数，从paddle的API中导入优化器和学习率调整即可。

```python
from ppcd.models import UNet
from ppcd.losses import BCELoss, DiceLoss, MixedLoss  # 这里说明下混合损失怎么构造
import paddle

model = UNet()
# loss的使用方法和PaddleSeg相似，可以对照查看，唯一不同多了一个"decay"
losses = {}
losses['types'] = [MixedLoss([BCELoss(), DiceLoss()], [1, 1])]  # 混合使用BCE和Dice两个损失，各自的权重都为1
losses['coef'] = [1]  # 这是代表MixedLoss的权重为1，如果多个损失的话需要多个权重
losses['decay'] = [0.99]  # 这个表示权重每个epoch的衰减，这里表示每轮之后MixedLoss的权重衰减为原来的0.99
# 学习率和优化器调整的基操
lr = paddle.optimizer.lr.PolynomialDecay(3e-4, 200, end_lr=3e-7)
opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
```

5. 最后一句代码即可进行训练，只需要使用ppcd.core中的Train，填入相应的参数即可。

```python
from ppcd.core import Train

Train(
    model=model,  # 网络
    epoch=50,  # 训练轮数
    batch_size=2,  # 批大小
    train_data=train_data,  # 训练数据
    eval_data=val_data,  # 评估数据
    optimizer=opt,  # 优化器
    losses=losses,  # 损失函数
    pre_params_path="model_output/your_model.pdparams",  # 预训练的权重
    save_model_path="model_output",  # 保存log和模型参数的地址
    save_epoch=10,  # 多少轮评估保存一次模型
    log_batch=10  # 多少批保存一次log
    threshold=1  # 如果输出的仅为单通道的结果，则需要阈值进行变化判定，若输出结果大于等于两个通道，则该参数无效
)
```

6. 训练完后同样使用ppcd.core中的Infer即可完成预测。目前预测的结果的图像名只能获取到输入数据的图像名中的数字，所以为了方便对照，最好数据中的图像的文件名使用数字。

```python
from ppcd.core import Infer

Infer(
    model=model,  # 网络
    infer_data=infer_data,  # 预测数据
    params_path="model_output/your_model.pdparams",  # 模型参数
    save_img_path="output_infer"  # 保存预测结果的路径
    threshold=1  # 如果输出的仅为单通道的结果，则需要阈值进行变化判定，若输出结果大于等于两个通道，则该参数无效
)
```

## Git链接

- github：https://github.com/geoyee/PdRSCD.git
- gitee：https://gitee.com/Geoyee/pd-rscd.git

## 交流与反馈

Email：Geoyee@yeah.net

说明：本项目正在龟速建设中，由于能力和时间有限，诸多BUG亟待修改，很多模型损失都还没有，已经有的模型也不保证原论文的精度，只是尽量集成在一起。如有需要可在此基础上进行增改，如有好的意见建议和反馈欢迎交流。