import random
import numpy as np


def random_out(bimg, oh, ow):
    '''
        根据输入的图像和随机输出的大小随机输出块
        oh/ow (int/list)
    '''
    H, W, _ = bimg.shape
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
    return bimg[x:(x + ow), y:(y + oh), :]


def split_out(bimg, row, col, index):
    '''
        根据输入的图像和行列数以及索引输出对应图像块
        index (list)
    '''
    H, W, _ = bimg.shape
    if not isinstance(index, list):
        raise ValueError('index must be list!')
    cell_h = H // row
    cell_w = W // col
    return bimg[(index[0] * cell_h):((index[0] + 1) * cell_h), \
                (index[1] * cell_w):((index[1] + 1) * cell_w), :]