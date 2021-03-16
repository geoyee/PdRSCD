from ppcd.datasets import create_list, CDataset
from paddle.io import DataLoader
import ppcd.transforms as T
from ppcd.models import UNet, FastSCNN
from ppcd.losses import BCELoss
from paddle.optimizer import Adam

# 数据读取器测试
dataset_path = "E:/dataFiles/gitee/pd-rscd/test_dataset"
val_path = create_list(dataset_path, mode='val')
transforms = [T.RandomBlur(), T.RandomColor(), T.RandomEnlarge(), T.RandomFlip(), \
              T.RandomFog(), T.RandomNarrow(), T.RandomRotate(), T.RandomSharpening(), \
              T.RandomSplicing(), T.RandomStrip(), T.Resize(512), T.Normalize([127.5] * 3, [127.5] * 3), \
              T.RandomRemoveBand(kill_bands=[1])]
val_data = CDataset(val_path, transforms=transforms)
val_loader = DataLoader(val_data, batch_size=8, shuffle=True)

# 网络
model = FastSCNN()  # UNet()
model.train()
bce_loss = BCELoss()
opt = Adam(parameters=model.parameters())
epoch = 5

# 训练
for epoch_id in range(epoch):
    for batch_id, (img1, img2, lab) in enumerate(val_loader()):
        pred = model(img1, img2)
        loss = bce_loss(pred, lab)
        loss.backward()
        opt.step()
        opt.clear_grad()
        print("[Train] epoch: {}, batch: {}, loss: {}".format(epoch_id, batch_id, loss.numpy()))