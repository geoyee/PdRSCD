# PdRSCD

PdRSCD（PaddlePaddle Remote Sensing Change Detection）是一个基于飞桨PaddlePaddle，仿造PaddleSeg制作的一个用于遥感变化检测的工具。其中主要的区别在于数据读取器和数据增强可以接收和同时处理两期多通道影像。可以基于此使用工具中自带的网络或搭建自己的网络快速进行变化检测任务的训练和预测。

## 现有资产

1. [模型库](ppcd/models/README.md)
2. [损失函数](ppcd/losses/README.md)
3. [数据增强方法](ppcd/transforms/README.md)
4. [传统处理方法](ppcd/traditions/README.md)

## 代码结构

PdRSCD的主要代码在ppcd中，组织如下。可以根据自己的任务修改和添加下面的代码。

```
ppcd
  ├── core  # 包含训练和预测的代码
  ├── datasets  # 包含创建数据列表和定义数据集的代码
  ├── losses  # 包含损失函数的代码
  ├── metrics  # 包含指标评价的代码
  ├── models  # 包含网络模型、特殊层、层初始化等代码
  ├── traditions  # 包含一些传统计算方法的代码
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

2. 准备数据集，如果数据集是如下格式，可以通过create_list进行创建；当有多个标签（如DTCDSCD等），只需要将组织数据是将多个标签组织成label_1、label_2……这样的格式即可。关于正负样本分类的场景数据集训练变化网络（如CDMI-Net等），数据集如是下边的格式，可以通过split_create_list_class进行创建。

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
         
dataset
   ├── P  # 正样本数据
   |   ├── A  # 时段一
   |   └── B  # 时段二
   └── N  # 负样本数据
       ├── A
       └── B
```

```python
from ppcd.datasets import create_list, split_create_list_class

datas_path = "datas"  # 数据路径
train_list_path = create_list(datas_path, mode='train')  # 训练数据
val_list_path = create_list(datas_path, mode='val')  # 评估数据
infer_list_path = create_list(datas_path, mode='infer')  # 预测数据
# train_list_path = create_list(datas_path, mode='train', labels_num=?)  # 多标签数据，?代表标签数
# train_list_path, val_list_path, infer_list_path = split_create_list_class('testDataset')  # 分类数据
```

3. 生成数据列表后，即可生成对应的数据集，并根据需要使用数据增强。这里只需要从ppcd.datasets以及ppcd.transforms导入数据集和增强方法即可。如果需要使用分类的数据集，只需要将is_class设置为True即可。

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
# 使用分类数据
# infer_data = CDataset('Dataset/infer_list.txt', transforms=val_transforms, is_infer=True, is_class=True)
```

4. 接下来就可以进行训练的准备了。从ppcd.models导入网络，从ppcd.losses导入损失函数，从paddle的API中导入优化器和学习率调整即可。对于标签的损失，使用LabelBCELoss计算损失，对于返回的数据不参与损失计算的，使用ConstLoss将value设置为0即可。

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

## TODO

- [ ] 添加带地理坐标的划窗预测与拼接
- [ ] 添加ETA，便于估算训练时间
- [ ] 添加F1、OA等评估指标
- [ ] 添加pipy打包，可通过pip install

## Git链接

- github：https://github.com/geoyee/PdRSCD.git
- gitee：https://gitee.com/Geoyee/pd-rscd.git

## 交流与反馈

Email：Geoyee@yeah.net
