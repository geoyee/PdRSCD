import imghdr
import random
import gdal
import math
import os
import cv2
import numpy as np
from PIL import Image


# 根据图像类型读取图像
def read_img(img_path, npd_shape):
    img_format = imghdr.what(img_path)
    _, ext = os.path.splitext(img_path)
    # 读取数据
    if img_format == 'tiff' or ext == '.img':
        img_data = gdal.Open(img_path).ReadAsArray()
        return img_data.transpose((1, 2, 0))  # 多波段图像默认是[C, H, W]
    elif ext == '.npy' or ext == '.npz':
        npy_data = np.load(img_path)
        if npd_shape == "HWC":
            return npy_data
        else:
            return npy_data.transpose((1, 2, 0))
    elif img_format == 'jpeg':
        jpg_data = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return jpg_data
    # 读取标签
    elif img_format == 'png':
        img_lab = np.asarray(Image.open(img_path))
        if len(img_lab.shape) == 2:
            img_lab = img_lab.reshape((img_lab.shape[0], img_lab.shape[1], 1))
        return img_lab
    else:
        raise Exception('Not support {} image format!'.format(ext))


# 显示图像
def show_img(img):
    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
    c = img.shape[-1]
    im_show = []
    for i in range(c):
        i_max = np.max(np.max(img[:, :, i]))
        i_min = np.min(np.min(img[:, :, i]))
        i_show = (img[:, :, i] - i_min) / (i_max - i_min + 1e-12)
        i_show *= 255.
        i_show = np.uint8(i_show)
        im_show.append(i_show)
    im_show = np.array(im_show).transpose((1, 2, 0))
    return im_show


# 标准化
def normalize(img, min_value, max_value, mean, std, band_num):
    range_value = [max_value[i] - min_value[i] for i in range(band_num)]
    img[:, :, :band_num] = (img[:, :, :band_num].astype(np.float32, copy=False) - min_value) / range_value
    img[:, :, :band_num] -= mean
    img[:, :, :band_num] /= std
    return img


# 图像翻转
def mode_flip(img, mode):
    if len(img.shape) == 3:
        if mode == 'Horizontal' or mode == 'Both':
            img = img[:, ::-1, :]
        if mode == 'Vertical' or mode == 'Both':
            img = img[::-1, :, :]
    elif len(img.shape) == 2:
        if mode == 'Horizontal' or mode == 'Both':
            img = img[:, ::-1]
        if mode == 'Vertical' or mode == 'Both':
            img = img[::-1, :]
    return img


# 旋转图像
def rotate_img(img, ang, ig_pix=None):
    height, width = img.shape[:2]
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), ang, 1)
    if ig_pix is not None:
        ig_pix = [ig_pix] * 3
    img = cv2.warpAffine(img, matRotate, (width, height), borderValue=ig_pix)
    return img


# 随机裁剪放大图像
def enlarge_img(img, x, y, h_clip, w_clip):
    h, w = img.shape[:2]
    clip_img = img[y:y+h_clip, x:x+w_clip]
    img = cv2.resize(
        clip_img,
        (h, w),
        interpolation=cv2.INTER_NEAREST)
    return img


# 缩小并填充回原图像大小
def narrow_img(img, x_rate, y_rate, ig_pix=None):
    h, w = img.shape[:2]
    rsz_img = cv2.resize(
        img,
        None,
        None,
        fx = x_rate,
        fy = y_rate,
        interpolation=cv2.INTER_NEAREST)
    w_pad = math.floor(w * (1 - x_rate) / 2)
    h_pad = math.floor(h * (1 - y_rate) / 2)
    if ig_pix is not None:
        ig_pix = [ig_pix] * 3
    img = cv2.copyMakeBorder(rsz_img, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=ig_pix)
    return img


# 随机条带
def random_strip(img, strip_num, mode, band_num):
    h, w = img.shape[:2]
    num = h if mode == 'Horizontal' else w
    strips = []
    i = 0
    while (i < strip_num):
        rdx = random.randint(0, num-1)
        if rdx not in strips:
            strips.append(rdx)
            i += 1
    if mode == 'Horizontal':
        for j in strips:
            img[j, :, :band_num] = 0
    else:
        for j in strips:
            img[:, j, :band_num] = 0
    return img


# 图像加雾
def add_fog(img, f_rag, band_num):
    mask = img.copy()
    mask[:, :, :] = 175  # 雾的颜色
    img[:, :, :band_num] = cv2.addWeighted(img[:, :, :band_num], \
                           round(random.uniform(f_rag[0], f_rag[1]), 2), \
                           mask[:, :, :band_num], 1, 0)  # 参数可调雾的浓度
    return img


# 一种波段计算
def band_comput(img, b1, b2):
    img = img.astype('float32')
    out = (img[:, :, b1] - img[:, :, b2]) / (img[:, :, b1] + img[:, :, b2])
    out = out.reshape([out.shape[0], out.shape[1], 1])
    img = np.concatenate((img, out), axis=-1)
    return img


# 随机拼接不匀色效果
def random_splicing(img, mode, band_num):
    h, w = img.shape[:2]
    alpha = random.uniform(0.8, 1.2)
    num = h if mode == 'Horizontal' else w
    rdx = random.randint(1, num-1)
    if mode == 'Horizontal':
        img[0:rdx, :, :band_num] *= alpha
    else:
        img[:, 0:rdx, :band_num] *= alpha
    return img