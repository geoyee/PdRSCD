import paddle
from ppcd.datasets import create_list, CDataset
import ppcd.transforms as T
from ppcd.models import FastSCNN
from ppcd.losses import BCELoss
from paddle.optimizer import Adam
from ppcd.core import Train, Infer

# 数据
dataset_path = "E:/dataFiles/gitee/pd-rscd/test_dataset"
val_path = create_list(dataset_path, mode='val')
transforms = [T.RandomBlur(), T.RandomColor(), T.RandomEnlarge(), T.RandomFlip(), \
              T.RandomFog(), T.RandomNarrow(), T.RandomRotate(), T.RandomSharpening(), \
              T.RandomSplicing(), T.RandomStrip(), T.Resize(128), T.Normalize([127.5] * 3, [127.5] * 3)]
val_data = CDataset(val_path, transforms=transforms)

# 网络
model = FastSCNN(enable_auxiliary_loss=False)
bce_loss = {}
bce_loss['types'] = [BCELoss()]
bce_loss['coef'] = [1]
opt = Adam(learning_rate=3e-4, parameters=model.parameters(), weight_decay=0.0001)

# 训练
Train(model=model,
      epoch=10,
      batch_size=8,
      train_data=val_data,
      eval_data=val_data,
      optimizer=opt,
      losses=bce_loss,
      save_model_path="E:/dataFiles/gitee/pd-rscd/test_model_save",
      log_batch=1
  )

# 预测
Infer(model=model,
      infer_data=val_data,
      params_path="E:/dataFiles/gitee/pd-rscd/test_model_save/epoch_9.pdparams",
      save_img_path="E:/dataFiles/gitee/pd-rscd/test_image_save"
  )