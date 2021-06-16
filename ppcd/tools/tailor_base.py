import sys
import random
import numpy as np
from math import ceil


def random_out(bimgs, oh, ow):
    '''
        根据输入的图像[H, W, C]和随机输出的大小随机输出块
        oh/ow (int)
    '''
    seed = random.randrange(sys.maxsize)
    random.seed(seed)  # 刷新种子
    H, W = bimgs[0].shape[:2]
    # print('H, W, oh, ow:', H, W, oh, ow)
    if not isinstance(oh, int) and not isinstance(ow, int):
        raise ValueError('oh and ow must be int!')
    h_range = H - oh
    w_range = W - ow
    x = random.randint(0, h_range)
    y = random.randint(0, w_range)
    # print('x, y:', x, y)
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


def split_eval(bimgs, rate=0.8, direction='H', geoinfo=None):
    '''
        将图像划分为两个部分，训练集和测试集（默认第一个返回的是训练集，位置在左或上）
    '''
    if rate <=0 or rate >= 1:
        raise ValueError('the value of rate must be between 0 and 1!')
    if geoinfo is not None:
        minx, xres, xskew, maxy, yskew, yres = geoinfo['geotrans']
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
    if geoinfo is None:
        return train_imgs, val_imgs
    else:
        train_geoinfo = geoinfo.copy()
        train_geoinfo['xsize'] = train_imgs[0].shape[1]
        train_geoinfo['ysize'] = train_imgs[0].shape[0]
        # 左上角位置不变
        val_geoinfo = geoinfo.copy()
        val_geoinfo['xsize'] = val_imgs[0].shape[1]
        val_geoinfo['ysize'] = val_imgs[0].shape[0]
        val_geoinfo['geotrans'] = (
            minx + int(W * rate * xres), xres, xskew, maxy, yskew, yres) if direction == 'V' \
            else (minx, xres, xskew, maxy + int(H * rate * yres), yskew, yres)
        return train_imgs, val_imgs, train_geoinfo, val_geoinfo