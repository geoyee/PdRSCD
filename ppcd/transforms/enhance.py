import cv2
import numpy as np


def percent_linear(image, perc_rate=2, uint_b=16):
    '''
        线性拉伸，输入图像为[H,W,C]，类型为uint8或uint16
    '''
    H, W, C = image.shape
    def gray_process(gray, maxout=255, minout=0):
        truncated_down = np.percentile(gray, perc_rate)
        truncated_up = np.percentile(gray, (100 - perc_rate))
        gray_new = (gray - truncated_down) / (truncated_up - truncated_down) * \
                   (maxout - minout) + minout
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return gray_new
    result = None
    for i_c in range(C):
        if i_c == 0:
            result = gray_process(image[:, :, i_c])[:, :, np.newaxis]
        else:
            result = np.concatenate([result, gray_process(image[:, :, i_c])[:, :, np.newaxis]], \
                                    axis=-1)
    if uint_b == 16:
        return np.uint16(result.reshape([H, W, C]))
    elif uint_b == 8:
        return np.uint8(result.reshape([H, W, C]))
    else:
        return np.float32(result.reshape([H, W, C]))


def bilateral_filtering(image):
    '''
        双边滤波
    '''
    dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)
    return dst


def de_haze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    '''
        暗通道先验去雾
    '''
    def zmMinFilterGray(src, r=7):
        return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

    def guidedfilter(I, p, r, eps):
        height, width = I.shape
        m_I = cv2.boxFilter(I, -1, (r, r))
        m_p = cv2.boxFilter(p, -1, (r, r))
        m_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p
        m_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = m_II - m_I * m_I
        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I
        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b

    def Defog(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
        '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        V1 = np.min(m, 2)  # 获得暗通道图像
        Dark_Channel = zmMinFilterGray(V1, 7)
        V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
        bins = 2000
        ht = np.histogram(V1, bins)  # 计算大气光照A
        d = np.cumsum(ht[0]) / float(V1.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
        V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制
        return V1, A
    if np.max(m) > 1:
        m = m / 255.
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)  # 获得遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - Mask_img) / (1-Mask_img / A)  # 颜色校订
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校订,默认不进行该操做
    return (Y * 255).astype('uint8')