import os
import re
import random
import numpy as np
import paddle
from paddle.io import Dataset
from ppcd.transforms import Compose


def create_list(dataset_path, mode='train', shuffle=False):
    save_path = os.path.join(dataset_path, (mode + '_list.txt'))
    with open(save_path, 'w') as f:
        A_path = os.path.join(os.path.join(dataset_path, mode), 'A')
        A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
        if shuffle:
            random.shuffle(A_imgs_name)
        else:
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


# 以场景分类数据组织方式来实现变化检测
def split_create_list_class(dataset_path, split_rate=[8, 1, 1], shuffle=True):
    train_save_path = os.path.join(dataset_path, 'train_list.txt')
    eval_save_path = os.path.join(dataset_path, 'val_list.txt')
    test_save_path = os.path.join(dataset_path, 'test_list.txt')
    with open(train_save_path, 'w') as tf:
        with open(eval_save_path, 'w') as ef:
            with open(test_save_path, 'w') as sf:
                PA_path = os.path.join(os.path.join(dataset_path, 'P'), 'A')
                NA_path = os.path.join(os.path.join(dataset_path, 'N'), 'A')
                PA_imgs_name = os.listdir(PA_path)
                A_imgs_name = [os.path.join(PA_path, PA_img_name) for PA_img_name in PA_imgs_name]
                NA_imgs_name = os.listdir(NA_path)
                A_imgs_name.extend([os.path.join(NA_path, NA_img_name) for NA_img_name in NA_imgs_name])
                if shuffle:
                    random.shuffle(A_imgs_name)
                else:
                    A_imgs_name.sort()
                for idx, A_img_name in enumerate(A_imgs_name):
                    A_img = A_img_name
                    B_img = A_img_name.replace('A', 'B')
                    if idx % 10 <  split_rate[0]:
                        tf.write(A_img + ' ' + B_img + ' ' + str(int(A_img.split('/')[-3] == 'P')) + '\n')
                    elif idx % 10 >= (np.sum(split_rate) - 1):
                        sf.write(A_img + ' ' + B_img + ' ' + '\n')
                    else:
                        ef.write(A_img + ' ' + B_img + ' ' + str(int(A_img.split('/')[-3] == 'P')) + '\n')
    print('data_list generated')
    return train_save_path, eval_save_path, test_save_path


class CDataset(Dataset):
    def __init__(self, data_list_path, transforms=None, separator=' ', npd_shape='HWC', is_255=True, is_infer=False, is_class=False):
        self.transforms = Compose(transforms=transforms, npd_shape=npd_shape, is_255=is_255)
        self.datas = []
        self.is_infer = is_infer
        self.is_class = is_class
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
        if self.is_class:
            if self.is_infer:
                A_path, B_path = self.datas[index]
            else:
                A_path, B_path, lab = self.datas[index]
            A_img, B_img = self.transforms(A_path, B_path, None)
        else:
            if self.is_infer:
                A_path, B_path = self.datas[index]
                A_img, B_img = self.transforms(A_path, B_path, None)
            else:
                A_path, B_path, lab_path = self.datas[index]
                A_img, B_img, lab = self.transforms(A_path, B_path, lab_path)
        A_img = paddle.to_tensor(A_img.transpose((2, 0, 1)))
        B_img = paddle.to_tensor(B_img.transpose((2, 0, 1)))
        if self.is_class:
            if self.is_infer:
                name = paddle.to_tensor(int(re.sub('\D', '', A_path)))
                return A_img, B_img, name
            else:
                return A_img, B_img, paddle.to_tensor(int(lab), dtype='float32')
        else:
            if self.is_infer:
                name = paddle.to_tensor(int(re.sub('\D', '', A_path)))
                return A_img, B_img, name
            else:
                lab = paddle.to_tensor(lab[np.newaxis, :, :], dtype='int64')
                return A_img, B_img, lab

    def __len__(self):
        return self.lens