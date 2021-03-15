import numpy as np
import paddle
from ppcd.models import UNet, FastSCNN
# from paddle.io import Dataset, DataLoader
from ppcd.datasets import CDataset
import ppcd.transforms as T


# # 网络输入输出测试
# model = FastSCNN()  # UNet()
# ima1 = np.array(np.random.randn(3*128*128)).reshape((1,3,128,128)).astype('float32')
# ima2 = np.array(np.random.randn(3*128*128)).reshape((1,3,128,128)).astype('float32')
# ima1 = paddle.to_tensor(ima1)
# ima2 = paddle.to_tensor(ima2)
# c = model(ima1, ima2)
# print(c[0].shape)

# # 数据读取器测试
# class MyData(Dataset):
#     def __init__(self, transforms=None):
#         self.transforms = transforms
#     def __getitem__(self, index):
#         np.random.seed(index)
#         img1 = np.array(np.random.randn(3*128*128)).reshape((3,128,128)).astype('float32')
#         img2 = np.array(np.random.randn(3*128*128)).reshape((3,128,128)).astype('float32')
#         lab = np.array(np.random.randn(128*128)).reshape((1,128,128)).astype('int64')
#         if self.transforms is not None:
#             img1, img2, lab = self.transforms(img1, img2, lab)
#         return (img1, img2, lab)
#     def __len__(self):
#         return 1024

# mydata = MyData(None)
# mydataLoader = DataLoader(mydata, batch_size=64, shuffle=True)

# for img1, img2, lab in mydataLoader():
#     print(img1.shape, img2.shape, lab.shape)