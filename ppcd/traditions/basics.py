import numpy as np
import cv2


def BaseCompute(T1, T2, out_mode='single', mode='s'):
    # mode: s = subtraction, d = division
    def compute_diff(gt1, gt2, mode):
        if mode == 's':
            return gt2 - gt1
        else:
            return gt2 / (gt1 + 1e-12)

    if len(T1.shape) != 2:
        H, W, input_channels = T1.shape
    else:
        out_mode = 'single'
        H, W = T1.shape
        input_channels = 1
    if out_mode != 'single' and out_mode != 'keep':
        raise ValueError("out_mode must be 'single' or 'keep'!")
    if mode != 's' and mode != 'd':
        raise ValueError("out_mode must be 's'(subtraction) or 'd'(division)!")
    if out_mode == 'single':
        if input_channels == 3:
            T1 = cv2.cvtColor(T1, cv2.COLOR_RGB2GRAY).astype('float32')
            T2 = cv2.cvtColor(T2, cv2.COLOR_RGB2GRAY).astype('float32')
        elif input_channels != 1:
            T1 = np.mean(T2, axis=2)
            T2 = np.mean(T2, axis=2)
        else:
            T1 = T1
            T2 = T2
        return compute_diff(T1, T2, mode)
    else:
        tmps = []
        for i in range(input_channels):
            tmps.append(compute_diff(T1[:, :, i].astype('float32'), T2[:, :, i].astype('float32'), mode))
        return np.array(tmps).reshape([H, W, input_channels])