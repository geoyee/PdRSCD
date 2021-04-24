import numpy as np
import cv2


def CVA(T1, T2, out_mode='single'):
    '''
        变化向量检测方法CVA是一种分类前检测变化的检测方法，通过两个时相间的向量变化表示图像变化。
        Args:
            T1 (nd.array): 时刻一的图像（H, W, C）
            T2 (nd.array): 时刻二的图像
            out_mode (str, 可选): 返回图像的通道数，只能为'single'或'keep'，默认为'single'
        Return：
            Icva (nd.array): 计算得到的向量图像（单通道）
    '''
    def compute_cva(gt1, gt2):
        # 做差求强度变化
        diff_str = gt2 - gt1
        # 计算向量方向变化
        delta_V = np.sqrt(np.sum(diff_str ** 2))
        diff_dir = diff_str / (np.abs(delta_V) + 1e-12)
        # 归一化
        return (diff_dir - np.min(diff_dir)) / (np.max(diff_dir) - np.min(diff_dir) + 1e-12)

    if len(T1.shape) != 2:
        H, W, input_channels = T1.shape
    else:
        out_mode = 'single'
        H, W = T1.shape
        input_channels = 1
    if out_mode != 'single' and out_mode != 'keep':
        raise ValueError("out_mode must be 'single' or 'keep'!")
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
        return compute_cva(T1, T2)
    else:
        tmps = []
        for i in range(input_channels):
            tmps.append(compute_cva(T1[:, :, i].astype('float32'), T2[:, :, i].astype('float32')))
        return np.array(tmps).reshape([H, W, input_channels])