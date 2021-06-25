import imghdr
import random
import math
import os
import cv2
import numpy as np
from PIL import Image


# 根据图像类型读取图像
def read_img(img_path, data_format, is_lab, classes_num=2):
    img_format = imghdr.what(img_path)
    _, ext = os.path.splitext(img_path)
    ipt_gdal = False
    try:
        try:
            from osgeo import gdal
        except ImportError:
            import gdal
        ipt_gdal = True
    finally:
        if img_format == 'tiff' or ext == '.img':
            if ipt_gdal == True:
                img_data = gdal.Open(img_path).ReadAsArray()
                return img_data.transpose((1, 2, 0)).astype('float32')  # 多波段图像默认是CHW
            else:
                raise Exception('Unable to open TIF/IMG image without GDAL!')
        elif ext == '.npy' or ext == '.npz':
            npy_data = np.load(img_path)
            if data_format == "HWC":
                return npy_data.astype('float32')
            else:
                return npy_data.transpose((1, 2, 0)).astype('float32')
        elif img_format == 'jpeg' or img_format == 'png' or img_format == 'bmp':
            if is_lab:
                jp_data = np.asarray(Image.open(img_path))
                if classes_num == 2:
                    jp_data = jp_data.clip(max=1)
            else:
                jp_data = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            return jp_data.astype('float32')
        else:
            raise Exception('Not support {} image format!'.format(ext))


# 标准化
def normalize(img, min_value, max_value, mean, std, band_num):
    range_value = [max_value[i] - min_value[i] for i in range(band_num)]
    cimg = img.copy().astype(np.float32)
    cimg[:, :, :band_num] = (img[:, :, :band_num].astype(np.float32, copy=False) - min_value) / range_value
    cimg[:, :, :band_num] -= mean
    cimg[:, :, :band_num] /= std
    return cimg


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
    img = img.astype('float32')
    height, width = img.shape[:2]
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), ang, 1)
    if ig_pix is not None:
        ig_pix = [ig_pix]
    # print(img)
    img = cv2.warpAffine(img, matRotate, (width, height), flags=cv2.INTER_NEAREST, borderValue=ig_pix)
    if img.shape == 2:
        img = img.reshape(height, width, 1)
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


# 直方图均衡化
def histogram_equalization(img, band_num):
    for b in range(band_num):
        img[:, :, b] = cv2.equalizeHist(img[:, :, b])
    return img


# 直方图规定化
def histogram_matching(t2, t1, band_num, bit_num=8):
    def_t2 = t2.copy()
    bmax = 2 ** bit_num
    for b in range(band_num):
        hist1, _ = np.histogram(t2[:, :, b].ravel(), bmax, [0, bmax])
        hist2, _ = np.histogram(t1[:, :, b].ravel(), bmax, [0, bmax])
        # 获得累计直方图
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()
        # 归一化处理
        cdf1_hist = hist1.cumsum() / cdf1.max()
        cdf2_hist = hist2.cumsum() / cdf2.max()
        # diff_cdf里是每2个灰度值比率间的差值
        diff_cdf = [[0 for i in range(bmax)] for j in range(bmax)]
        for i in range(bmax):
            for j in range(bmax):
                diff_cdf[i][j] = abs(cdf1_hist[i] - cdf2_hist[j])
        # 灰度级与目标灰度级的对应表
        lut = [0 for i in range(bmax)]
        for i in range(bmax):
            squ_min = diff_cdf[i][0]
            index = 0
            for j in range(bmax):
                if squ_min > diff_cdf[i][j]:
                    squ_min = diff_cdf[i][j]
                    index = j
            lut[i] = ([i, index])
        h = int(t1.shape[0])
        w = int(t1.shape[1])
        # 对原图像进行灰度值的映射
        for i in range(h):
            for j in range(w):
                def_t2[i, j, b] = lut[int(t2[i, j, b])][1]
    return def_t2