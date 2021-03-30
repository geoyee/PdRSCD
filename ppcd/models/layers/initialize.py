import paddle.nn as nn


def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)