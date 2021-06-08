import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights


def splicing_list(imgs, row, col, raw_size):
    '''
        将slide的out进行拼接，raw_size保证恢复到原状
    '''
    h, w = imgs[:2].shape
    result = np.zeros((h * row, w * col))
    k = 0
    for i_r in range(row - 1):
        for i_c in range(col - 1):
            if len(imgs[k]) == 2:
                result[(i_r * row):((i_r + 1) * row), (i_c * col):((i_c + 1) * col)]
            else:
                result[(i_r * row):((i_r + 1) * row), (i_c * col):((i_c + 1) * col), :]
    if len(result.shape) == 2:
        return result[0:raw_size[0], 0:raw_size[1]]
    else:
        return result[0:raw_size[0], 0:raw_size[1], :]