import paddle.nn as nn


def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):
    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)