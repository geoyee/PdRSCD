# 自定义数据集

## 一、数据集组织

### 1.1 已经划分好的数据

准备数据集，如果数据集是如下格式，可以通过create_list进行创建；当有多个标签（如DTCDSCD等），只需要将组织数据是将多个标签组织成label_1、label_2……这样的格式即可。关于正负样本分类的场景数据集训练变化网络（如CDMI-Net等），数据集如是下边的格式，可以通过split_create_list_class进行创建。

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

### 1.2 大图数据

可以参考入门项目（[【ppcd快速入门】大图滑框变化检测与拼接](https://aistudio.baidu.com/aistudio/projectdetail/2121793)）的数据使用方式。只需要有两期图像和标签，可以通过`split_eval`进行划分。在使用时将`Dataset`中的`big_map`设置为`True`即可。

