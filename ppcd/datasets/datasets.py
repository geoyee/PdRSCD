import os
import numpy as np
import paddle
from paddle.io import Dataset
from ppcd.transforms import Compose


def create_list(dataset_path, mode='train'):
    save_path = os.path.join(dataset_path, (mode + '_list.txt'))
    with open(save_path, 'w') as f:
        A_path = os.path.join(os.path.join(dataset_path, mode), 'A')
        A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
        A_imgs_name.sort()
        for A_img_name in A_imgs_name:
            A_img = os.path.join(A_path, A_img_name)
            B_img = os.path.join(A_path.replace('A', 'B'), A_img_name)
            if mode != 'infer':
                label_img = os.path.join(A_path.replace('A', 'label'), A_img_name)
                f.write(A_img + ' ' + B_img + ' ' + label_img + '\n')  # 写入list.txt
            else:
                f.write(A_img + ' ' + B_img + '\n')
    print(mode + '_data_list generated')
    return save_path


class CDataset(Dataset):
    def __init__(self, data_list_path, transforms=None, separator=' ', npd_shape='HWC', is_255=True, is_infer=False):
        self.transforms = Compose(transforms=transforms, npd_shape=npd_shape)
        self.datas = []
        self.is_255 = is_255
        self.is_infer = is_infer
        with open(data_list_path, 'r') as f:
            fdatas = f.readlines()
        for fdata in fdatas:
            fdata = fdata.split(separator)
            if is_infer:
                self.datas.append([fdata[0], fdata[1].strip()])
            else:
                self.datas.append([fdata[0], fdata[1], fdata[2].strip()])
        self.lens = len(self.datas)
    def __getitem__(self, index):
        if self.is_infer:
            A_path, B_path = self.datas[index]
            A_img, B_img = self.transforms(A_path, B_path, None)
        else:
            A_path, B_path, lab_path = self.datas[index]
            A_img, B_img, lab = self.transforms(A_path, B_path, lab_path)
        A_img = paddle.to_tensor(A_img.transpose((2, 0, 1)))
        B_img = paddle.to_tensor(B_img.transpose((2, 0, 1)))
        if self.is_infer:
            return A_img, B_img
        else:
            if self.is_255:
                lab = lab.clip(max=1)
            lab = paddle.to_tensor(lab[np.newaxis, :, :], dtype='int64')
            return A_img, B_img, lab
    def __len__(self):
        return self.lens