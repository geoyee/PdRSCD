import os
import random
import numpy as np
import paddle
from math import ceil
from PIL import Image
from paddle.io import Dataset
from ppcd.transforms import Compose
from ppcd.tools import random_out, slide_out, open_tif


def create_list(dataset_path, mode='train', shuffle=False, label_end='.png', labels_num=1):
    # labels_num表示有多少标签，默认为1
    save_path = os.path.join(dataset_path, (mode + '_list.txt'))
    with open(save_path, 'w') as f:
        A_path = os.path.join(os.path.join(dataset_path, mode), 'A')
        A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
        if shuffle:
            random.shuffle(A_imgs_name)
        else:
            A_imgs_name.sort()
        image_end = os.path.splitext(A_imgs_name[0])[-1]
        for A_img_name in A_imgs_name:
            A_img = os.path.join(A_path, A_img_name)
            B_img = os.path.join(A_path.replace('A', 'B'), A_img_name)
            if mode != 'infer':
                if labels_num == 1:
                    label_img = os.path.join(A_path.replace('A', 'label'), A_img_name.replace(image_end, label_end))
                    f.write(A_img + ' ' + B_img + ' ' + label_img + '\n')  # 写入list.txt
                else:
                    f.write(A_img + ' ' + B_img + ' ')  # 写入list.txt
                    for i in range(labels_num):
                        label_img_i = os.path.join(
                            A_path.replace('A', ('label_' + str(i + 1))), A_img_name.replace(image_end, label_end)
                        )
                        if i == 0:
                            f.write(label_img_i)
                        else:
                            f.write('?' + label_img_i)
                    f.write('\n')
            else:
                f.write(A_img + ' ' + B_img + '\n')
    print(mode + '_data_list generated')
    return save_path


# 以场景分类数据组织方式来实现变化检测
def split_create_list_class(dataset_path, split_rate=[8, 1, 1], shuffle=True):
    train_save_path = os.path.join(dataset_path, 'train_list.txt')
    eval_save_path = os.path.join(dataset_path, 'val_list.txt')
    test_save_path = os.path.join(dataset_path, 'infer_list.txt')
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
    def __init__(self, data_list_path, data_format='HWC', separator=' ', \
                 transforms=None, classes_num=2, is_infer=False, shuffle=False):
        '''
        说明：
            data_format针对的是npy和npz的数据，因为TIF读取默认为CHW会自动转为HWC，JPG/PNG的读取默认就是HWC
        '''
        self.transforms = Compose(transforms=transforms, \
                                  data_format=data_format, classes_num=classes_num)
        self.datas = []
        self.is_infer = is_infer
        self.classes_num = classes_num
        with open(data_list_path, 'r') as f:
            fdatas = f.readlines()
        for fdata in fdatas:
            fdata = fdata.split(separator)
            if is_infer:
                self.datas.append([fdata[0], fdata[1].strip()])
            else:
                # 如果是多标签，标签间用?隔开
                self.datas.append([fdata[0], fdata[1], fdata[2].strip().split('?')])
        self.lens = len(self.datas)
        if shuffle == True:
            random.shuffle(self.datas)

    def refresh_data(self):
        random.shuffle(self.datas)

    def __getitem__(self, index):
        if self.classes_num == 1:
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
                A_path, B_path, labs_path = self.datas[index]
                A_img, B_img, labs = self.transforms(A_path, B_path, labs_path)
        A_img = A_img.transpose((2, 0, 1))
        B_img = B_img.transpose((2, 0, 1))
        name, _ = os.path.splitext(os.path.split(A_path)[1])
        if self.classes_num == 1:
            if self.is_infer:
                return A_img, B_img, name
            else:
                return A_img, B_img, np.array(float(lab[0]))
        else:
            if self.is_infer:
                return A_img, B_img, name
            else:
                for i in range(len(labs)):
                    labs[i] = paddle.to_tensor(labs[i][np.newaxis, :, :], dtype='int64')
                return A_img, B_img, labs

    def __len__(self):
        return self.lens


# 大范围的遥感数据
class BDataset(Dataset):
    def __init__(self, t1_source, t2_source, lab_source=None, c_size=[512, 512], \
                 transforms=None, classes_num=2, out_mode='random', is_tif=True, geoinfo=None):
        '''
            t1、t2以及lab (str/ndarray)
        '''
        self.classes_num = classes_num
        self.transforms = Compose(transforms=transforms, classes_num=classes_num)
        if isinstance(t1_source, str) and isinstance(t2_source, str):
            if is_tif == False:
                self.t1 = np.asarray(Image.open(t1_source))
                self.t2 = np.asarray(Image.open(t2_source))
                self.lab = np.asarray(Image.open(lab_source)) if lab_source is not None else None
                self.geoinfo = None
            else:
                self.t1, self.geoinfo = open_tif(t1_source, to_np=True)
                self.t2, _ = open_tif(t2_source, to_np=True)
                self.lab, _ = open_tif(lab_source, to_np=True) if lab_source is not None else None
        else:  # 直接传入图像
            self.t1 = t1_source
            self.t2 = t2_source
            self.lab = lab_source if lab_source is not None else None
            self.geoinfo = geoinfo if geoinfo is not None else None
        self.raw_size = [self.t1.shape[0], self.t1.shape[1]]  # 原始大小
        self.c_size = c_size
        self.is_tif = True if geoinfo is not None else is_tif
        self.is_infer = True if lab_source is None else False
        self.out_mode = 'slide' if self.is_infer == True else out_mode
        self.lens = ceil(self.t1.shape[0] / c_size[0]) * ceil(self.t1.shape[1] / c_size[1])

    def refresh_data(self):
        pass

    def __getitem__(self, index):
        # 数据分配
        if self.lab is None:
            imgs = [self.t1, self.t2]
        else:
            imgs = [self.t1, self.t2, self.lab]
        if self.out_mode == 'slide':
            H, W = self.raw_size
            row = ceil(H / self.c_size[0])
            col = ceil(W / self.c_size[1])
            # print('dataset: row, col:', row, col)
            # 计算索引
            idr = index // col
            idc = index % col
            # 全部索引完毕
            if idr == row:
                return None
            idx = [idr, idc]
            # print('row, col, idx:', row, col, idx)
            res = slide_out(imgs, row, col, idx, self.c_size)
        else:
            res = random_out(imgs, self.c_size[0], self.c_size[1])
        t1, t2 = res[:2]
        lab = res[-1] if len(res) == 3 else None
        # 数据增强
        if self.is_infer:
            A_img, B_img = self.transforms(t1, t2, None)
        else:
            A_img, B_img, labs = self.transforms(t1, t2, lab)
        A_img = A_img.transpose((2, 0, 1)).astype('float32')
        B_img = B_img.transpose((2, 0, 1)).astype('float32')
        if self.is_infer == False:
            for i in range(len(labs)):
                labs[i] = paddle.to_tensor(labs[i][np.newaxis, :, :], dtype='int64')
            return A_img, B_img, labs
        else:
            return A_img, B_img

    def __len__(self):
        return self.lens


# 数据集
class Dataset(object):
    def __init__(self, big_map=False):  # big_map为True的话就是对应大图预测
        self.data_mode = CDataset if big_map == False else BDataset

    def __call__(self, *args, **kwargs):
        return self.data_mode(*args, **kwargs)


# 数据读取器
class DataLoader(object):
    def __init__(self, cdataset, batch_size, shuffle=False, is_val=False):
        self.cdataset = cdataset
        if shuffle:
            self.cdataset.refresh_data()
        self.batch_size = batch_size
        self.is_val = is_val
        if self.is_val:
            self.index = iter(range(ceil(len(self.cdataset) / self.batch_size)))
        else:
            self.index = iter(range(len(self.cdataset) // self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            index = next(self.index)
            t1s = []
            t2s = []
            ques = []
            start = index * self.batch_size
            end = (index + 1) * self.batch_size
            if end > len(self.cdataset):
                end = len(self.cdataset)
            for i in range(start, end, 1):
                t1, t2, que = self.cdataset[i]
                t1s.append(t1)
                t2s.append(t2)
                ques.append(que)
            if isinstance(ques[0], list):
                tmp = np.array(ques).transpose((1, 0, 2, 3, 4))
                ques = [paddle.to_tensor(tmp[i, :, : ,:, :]) for i in range(tmp.shape[0])]
            else:
                if not isinstance(ques[0], str):
                    ques = paddle.to_tensor(ques)
            return (paddle.to_tensor(t1s), paddle.to_tensor(t2s), ques)
        except StopIteration:
            pass