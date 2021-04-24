import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CAM(nn.Layer):
    # Channels Attention Module
    def __init__(self, in_channels, ratio=8):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.mlp = nn.Sequential(
            nn.Conv2D(in_channels, in_channels//ratio, 1),
            nn.ReLU(),
            nn.Conv2D(in_channels//ratio, in_channels, 1)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = F.sigmoid(avg_out + max_out)
        return out


class SAM(nn.Layer):
    # Spatial Attention Module
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Conv2D(2, 1, 7, padding=3)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class BAM(nn.Layer):
    """ 
        Basic self-attention module
    """
    def __init__(self, in_channels, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.key_channel = in_channels //8
        self.activation = activation
        self.ds = ds
        self.pool = nn.AvgPool2D(self.ds)
        self.query_conv = nn.Conv2D(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2D(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.ParameterList([paddle.create_parameter(shape=[1], dtype='float32', default_initializer=nn.initializer.Constant(value=0))])
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, input):
        """
            inputs :
                x : input feature maps(B C W H)
            returns :
                out : self attention value + input feature
                attention: B N N (N is Width*Height)
        """
        x = self.pool(input)
        N, C, H, W = x.shape
        proj_query = self.query_conv(x).reshape([N, -1, H * W]).transpose((0, 2, 1))
        proj_key = self.key_conv(x).reshape([N, -1, H * W])
        energy = paddle.bmm(proj_query, proj_key)
        energy = (self.key_channel ** -.5) * energy
        attention = self.softmax(energy - paddle.max(energy, axis=-1, keepdim=True))  # 防止溢出
        proj_value = self.value_conv(x).reshape([N, -1, H * W])
        out = paddle.bmm(proj_value, attention.transpose((0, 2, 1)))
        out = out.reshape([N, C, H, W])
        out = F.interpolate(out, [H * self.ds, W * self.ds])
        out = out + input
        return out