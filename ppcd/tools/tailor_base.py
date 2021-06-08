import sys
import random
import numpy as np
from math import ceil


def random_out(bimgs, oh, ow):
    '''
        根据输入的图像[H, W, C]和随机输出的大小随机输出块
        oh/ow (int/list)
    '''
    seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)  # 刷新种子
    H, W = bimgs[0].shape[:2]
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


def slide_out(bimgs, row, col, index, c_size=None):
    '''
        根据输入的图像[H, W, C]和行列数以及索引输出对应图像块
        index (list)
    '''
    H, W = bimgs[0].shape[:2]
    if not isinstance(index, list):
        raise ValueError('index must be list!')
    if c_size is None:
        c_size = [ceil(H / row), ceil(W / col)]
    # 扩展不够的
    h_new = row * c_size[0]
    w_new = col * c_size[1]
    tmps = []
    if h_new != H or w_new != W:
        for i in range(len(bimgs)):
            bimg = bimgs[i]
            if len(bimg.shape) == 2:
                tmp = np.zeros((h_new, w_new))
                tmp[:bimg.shape[0], :bimg.shape[1]] = bimg
            else:
                tmp = np.zeros((h_new, w_new, bimg.shape[-1]))
                tmp[:bimg.shape[0], :bimg.shape[1], :] = bimg
            tmps.append(tmp)
        H, W = tmps[0].shape[:2]
    else:
        tmps = bimgs
    cell_h = c_size[0]
    cell_w = c_size[1]
    result = []
    for i in range(len(tmps)):
        if len(tmps[i].shape) == 2:
            result.append(tmps[i][(index[0] * cell_h):((index[0] + 1) * cell_h), \
                                   (index[1] * cell_w):((index[1] + 1) * cell_w)])
        else:
            result.append(tmps[i][(index[0] * cell_h):((index[0] + 1) * cell_h), \
                                   (index[1] * cell_w):((index[1] + 1) * cell_w), :])
    return result


def split_eval(bimgs, rate=0.8, direction='H'):
    '''
        将图像划分为两个部分，训练集和测试集
    '''
    if rate <=0 or rate >= 1:
        raise ValueError('the value of rate must be between 0 and 1!')
    H, W = bimgs[0].shape[:2]
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