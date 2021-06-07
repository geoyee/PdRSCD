import random
import numpy as np


def random_out(bimgs, oh, ow):
    '''
        根据输入的图像[H, W, C]和随机输出的大小随机输出块
        oh/ow (int/list)
    '''
    H, W, _ = bimgs[0].shape
    if isinstance(oh, list) and isinstance(ow, list):
        oh = random.choice(oh)
        ow = random.choice(ow)
    elif not (isinstance(oh, int) or isinstance(oh, list)) and \
         not (isinstance(ow, int) or isinstance(ow, list)):
        raise ValueError('oh and ow must be int or list!')
    h_range = [0, (H - oh)]
    w_range = [0, (W - ow)]
    x = random.choice(w_range)
    y = random.choice(h_range)
    result = []
    for i in len(bimgs):
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
    for i in len(bimgs):
        result.append(bimgs[i][(index[0] * cell_h):((index[0] + 1) * cell_h), \
                               (index[1] * cell_w):((index[1] + 1) * cell_w), :])
    return result


def split_eval(bimg, rate=0.8, direction='H'):
    if rate <=0 or rate >= 1:
        raise ValueError('the value of rate must be between 0 and 1!')
    H, W, _ = bimg.shape
    if direction == 'H':
        train_img = bimg[:int(H * rate), :, :]
        val_img = bimg[int(H * rate):, :, :]
    elif direction == 'V':
        train_img = bimg[:, :int(W * rate), :]
        val_img = bimg[:, int(W * rate):, :]
    else:
        raise ValueError('direction must be \'H\' or \'V\'!')
    return train_img, val_img