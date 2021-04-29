import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppcd.models.backbone import base_unet
from ppcd.models.layers import GatedAttentionLayer


class CDMINet(nn.Layer):
    """
    The CDMI-Net implementation based on PaddlePaddle.
    The original article refers to
    Zhang, Min and Shi, et, al. "Deep Multiple Instance Learning for Landslide Mapping"
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9142246).
    Args:
        in_channels (int, optional): Number of an image's channel.  Default: 3.
        feature_channels (int, optional): Number of an feature's channel.  Default: 64.
        attention_channels (int, optional): Number of an attention's channel.  Default: 128.
        # out_channels : 1.
    """
    def __init__(self, in_channels=3, feature_channels=64, attention_channels=128):
        super(CDMINet, self).__init__()
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.attention_channels = attention_channels
        self.unet = base_unet(in_channels=self.in_channels)
        self.attention = GatedAttentionLayer(self.feature_channels, self.attention_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        N, _, xH, xW = x1.shape
        H1 = self.unet(x1)
        H2 = self.unet(x2)
        DI = paddle.abs(H1 - H2)
        H = DI.transpose((0, 2, 3, 1))
        H = H.reshape([N, -1, self.feature_channels])
        A = self.attention(H)
        A_2 = A.reshape([N, 1, xH, xW])
        A = A.transpose((0, 2, 1))
        A = F.softmax(A, axis=2)
        A_3 = A.reshape([N, 1, -1])
        H_3 = H.reshape([N, -1, self.feature_channels])
        M = paddle.bmm(A_3, H_3)
        Y_prob = self.classifier(M).reshape([N, 1])
        # Y_hat = paddle.greater_equal(Y_prob, paddle.to_tensor(0.5)).astype('float32')  # y >= 0.5 ? 1 : 0
        return [A_2, Y_prob]