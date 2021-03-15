import paddle
from paddle.io import Dataset
from ppcd.transforms import Compose

class CDataset(Dataset):
    def __init__(self, data_list_path, transforms=None, separator=' '):
        self.transforms = Compose(transforms)
        self.datas = []
        with open(data_list_path, 'r') as f:
            fdatas = f.readlines()
        for fdata in fdatas:
            fdata = fdata.split(separator)
            self.datas.append([fdata[0], fdata[1], fdata[2].strip()])
        self.lens = len(self.datas)
    def __getitem__(self, index):
        A_path, B_path, lab_path = self.datas[index]
        A_img, B_img, lab = self.transforms(A_path, B_path, lab_path)
        return A_img, B_img, lab
    def __len__(self):
        return self.lens