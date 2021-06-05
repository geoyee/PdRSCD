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


# TODO:双边滤波


# TODO:去雾