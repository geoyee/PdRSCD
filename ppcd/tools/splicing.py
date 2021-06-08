import numpy as np


def splicing_list(imgs, row, col, raw_size):
    '''
        将slide的out进行拼接，raw_size保证恢复到原状
    '''
    h, w = imgs[0].shape[:2]
    if len(imgs[0].shape) == 2:
        result = np.zeros((h * row, w * col), dtype=np.uint8)
    else:
        result = np.zeros((h * row, w * col, imgs[0].shape[-1]), dtype=np.uint8)
    k = 0
    for i_r in range(row):
        for i_c in range(col):
            if len(imgs[k]) == 2:
                result[(i_r * h):((i_r + 1) * h), (i_c * w):((i_c + 1) * w)] = imgs[k]
            else:
                result[(i_r * h):((i_r + 1) * h), (i_c * w):((i_c + 1) * w), :] = imgs[k]
            k += 1
            # print('r, c, k:', i_r, i_c, k)
    if len(result.shape) == 2:
        return result[0:raw_size[0], 0:raw_size[1]]
    else:
        return result[0:raw_size[0], 0:raw_size[1], :]