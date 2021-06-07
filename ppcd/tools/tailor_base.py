import sys
import random
import numpy as np


def random_out(bimgs, oh, ow):
    '''
        根据输入的图像[H, W, C]和随机输出的大小随机输出块
        oh/ow (int/list)
    '''
    seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)  # 刷新种子
    H, W, _ = bimgs[0].shape
    if isinstance(oh, list) and isinstance(ow, list):
        oh = rng.randint(oh[0], oh[1])
        ow = rng.randint(ow[0], ow[1])
    elif not (isinstance(oh, int) or isinstance(oh, list)) and \
         not (isinstance(ow, int) or isinstance(ow, list)):
        raise ValueError('oh and ow must be int or list!')
    h_range = H - oh
    w_range = W - ow
    x = rng.randint(0, w_range)
    y = rng.randint(0, h_range)
    # print(x, y, oh, ow)
    result = []
    for i in range(len(bimgs)):
        if len(bimgs[i].shape) == 2:
            result.append(bimgs[i][x:(x + ow), y:(y + oh)])
        else:
            result.append(bimgs[i][x:(x + ow), y:(y + oh), :])
    return result


def split_out(bimgs, row, col, index):
    '''
        根据输入的图像[H, W, C]和行列数以及索引输出对应图像块
        index (list)
    '''
    H, W, _ = bimgs[0].shape
    if not isinstance(index, list):
        raise ValueError('index must be list!')
    # 扩展不够的
    h_add = row - (H % row)
    w_add = col - (W % col)
    if h_add != row or w_add != col:
        for bimg in bimgs:
            bimg = np.pad(bimg, ((0, h_add), (0, w_add), (0, 0)), 'constant')
        H, W, _ = bimgs[0].shape
    cell_h = H // row
    cell_w = W // col
    result = []
    for i in range(len(bimgs)):
        if len(bimgs[i].shape) == 2:
            result.append(bimgs[i][(index[0] * cell_h):((index[0] + 1) * cell_h), \
                                   (index[1] * cell_w):((index[1] + 1) * cell_w)])
        else:
            result.append(bimgs[i][(index[0] * cell_h):((index[0] + 1) * cell_h), \
                                   (index[1] * cell_w):((index[1] + 1) * cell_w), :])
    return result


def split_eval(bimgs, rate=0.8, direction='H'):
    '''
        将图像划分为两个部分，训练集和测试集
        TODO：保存文件
        TODO：是否删除源文件
    '''
    if rate <=0 or rate >= 1:
        raise ValueError('the value of rate must be between 0 and 1!')
    H, W, _ = bimgs[0].shape
    train_imgs = []
    val_imgs = []
    for i in range(len(bimgs)):
        if direction == 'H':
            if len(bimgs[i].shape) == 2:
                train_imgs.append(bimgs[i][:int(H * rate), :])
                val_imgs.append(bimgs[i][int(H * rate):, :,])
            else:
                train_imgs.append(bimgs[i][:int(H * rate), :, :])
                val_imgs.append(bimgs[i][int(H * rate):, :, :])
        elif direction == 'V':
            if len(bimgs[i].shape) == 2:
                train_imgs.append(bimgs[i][:, :int(W * rate)])
                val_imgs.append(bimgs[i][:, int(W * rate):])
            else:
                train_imgs.append(bimgs[i][:, :int(W * rate), :])
                val_imgs.append(bimgs[i][:, int(W * rate):, :])
        else:
            raise ValueError('direction must be \'H\' or \'V\'!')
    return train_imgs, val_imgs