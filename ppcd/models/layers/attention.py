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