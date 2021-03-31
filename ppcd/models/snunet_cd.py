import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppcd.models.layers import SyncBatchNorm, kaiming_normal_init, CAM


class ConvolutionBlock(nn.Layer):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(hidden_channels)
        self.conv2 = nn.Conv2D(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = F.relu(x + identity)
        return output


class UpSample(nn.Layer):
    def __init__(self, in_channels, bilinear=False):
        super(UpSample, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.Conv2DTranspose(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Layer):
    def __init__(self, in_channels, filters):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv00 = ConvolutionBlock(in_channels, filters[0], filters[0])
        self.conv10 = ConvolutionBlock(filters[0], filters[1], filters[1])
        self.conv20 = ConvolutionBlock(filters[1], filters[2], filters[2])
        self.conv30 = ConvolutionBlock(filters[2], filters[3], filters[3])
        self.conv40 = ConvolutionBlock(filters[3], filters[4], filters[4])

    def forward(self, x):
        x_00 = self.conv00(x)
        x_10 = self.conv10(self.pool(x_00))
        x_20 = self.conv20(self.pool(x_10))
        x_30 = self.conv30(self.pool(x_20))
        x_40 = self.conv40(self.pool(x_30))
        return x_00, x_10, x_20, x_30, x_40


class Decoder(nn.Layer):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        # upsample number 1
        self.conv01 = ConvolutionBlock(filters[0]*2+filters[1], filters[0], filters[0])
        self.conv11 = ConvolutionBlock(filters[1]*2+filters[2], filters[1], filters[1])
        self.conv21 = ConvolutionBlock(filters[2]*2+filters[3], filters[2], filters[2])
        self.conv31 = ConvolutionBlock(filters[3]*2+filters[4], filters[3], filters[3])
        self.up10 = UpSample(filters[1])
        self.up20 = UpSample(filters[2])
        self.up30 = UpSample(filters[3])
        self.up40 = UpSample(filters[4])
        # upsample number 2
        self.conv02 = ConvolutionBlock(filters[0]*3+filters[1], filters[0], filters[0])
        self.conv12 = ConvolutionBlock(filters[1]*3+filters[2], filters[1], filters[1])
        self.conv22 = ConvolutionBlock(filters[2]*3+filters[3], filters[2], filters[2])
        self.up11 = UpSample(filters[1])
        self.up21 = UpSample(filters[2])
        self.up31 = UpSample(filters[3])
        # upsample number 3
        self.conv03 = ConvolutionBlock(filters[0]*4+filters[1], filters[0], filters[0])
        self.conv13 = ConvolutionBlock(filters[1]*4+filters[2], filters[1], filters[1])
        self.up12 = UpSample(filters[1])
        self.up22 = UpSample(filters[2])
        # upsample number 4
        self.conv04 = ConvolutionBlock(filters[0]*5+filters[1], filters[0], filters[0])
        self.up13 = UpSample(filters[1])

    def forward(self, x_cont):
        x_00, x_10, x_20, x_30, x_40 = x_cont
        # upsample number 1
        x_01 = self.conv01(paddle.concat([x_00, self.up10(paddle.split(x_10, 2, axis=1)[-1])], axis=1))
        x_11 = self.conv11(paddle.concat([x_10, self.up20(paddle.split(x_20, 2, axis=1)[-1])], axis=1))
        x_21 = self.conv21(paddle.concat([x_20, self.up30(paddle.split(x_30, 2, axis=1)[-1])], axis=1))
        x_31 = self.conv31(paddle.concat([x_30, self.up40(x_40)], axis=1))
        # upsample number 2
        x_02 = self.conv02(paddle.concat([x_00, x_01, self.up11(x_11)], axis=1))
        x_12 = self.conv12(paddle.concat([x_10, x_11, self.up21(x_21)], axis=1))
        x_22 = self.conv22(paddle.concat([x_20, x_21, self.up31(x_31)], axis=1))
        # upsample number 3
        x_03 = self.conv03(paddle.concat([x_00, x_01, x_02, self.up12(x_12)], axis=1))
        x_13 = self.conv13(paddle.concat([x_10, x_11, x_12, self.up22(x_22)], axis=1))
        # upsample number 4
        x_04 = self.conv04(paddle.concat([x_00, x_01, x_02, x_03, self.up13(x_13)], axis=1))
        return x_01, x_02, x_03, x_04


class SNUNet(nn.Layer):
    """
    The SNUNet implementation based on PaddlePaddle.
    The original article refers to
    Fang, Sheng , et al. "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images"
    (https://ieeexplore.ieee.org/document/9355573).
    Args:
        in_channels (int, optional): Number of an image's channel.  Default: 3.
        out_channels (int, optional): The unique number of target classes.  Default: 2.
        is_ECAM (bool, optional): Use Channel Attention Module or not.  Default: True.
    """
    def __init__(self, in_channels=3, out_channels=2, is_ECAM=True):
        super(SNUNet, self).__init__()
        self.is_ECAM = is_ECAM
        map_num = 32
        filters = [map_num, map_num*2, map_num*4, map_num*8, map_num*16]
        self.encoder = Encoder(in_channels, filters)
        self.decoder = Decoder(filters)
        if is_ECAM:
            self.cam1 = CAM(filters[0], ratio=16//4)
            self.cam2 = CAM(filters[0]*4, ratio=16)
            self.conv_final = nn.Conv2D(filters[0]*4, out_channels, kernel_size=1)
        else:
            self.final1 = nn.Conv2D(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2D(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2D(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2D(filters[0], out_channels, kernel_size=1)
            self.conv_final = nn.Conv2D(out_channels*4, out_channels, kernel_size=1)
        # kaiming_normal_init
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                kaiming_normal_init(sublayer.weight)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                kaiming_normal_init(sublayer.weight)

    def forward(self, xA, xB):
        xA_0, xA_1, xA_2, xA_3, _ = self.encoder(xA)
        xB_0, xB_1, xB_2, xB_3, xB_4 = self.encoder(xB)
        x_cont = [
            paddle.concat([xA_0, xB_0], axis=1),
            paddle.concat([xA_1, xB_1], axis=1),
            paddle.concat([xA_2, xB_2], axis=1),
            paddle.concat([xA_3, xB_3], axis=1),
            xB_4
        ]
        x_01, x_02, x_03, x_04 = self.decoder(x_cont)
        if self.is_ECAM:
            output = paddle.concat([x_01, x_02, x_03, x_04], axis=1)
            intra = paddle.sum(paddle.stack((x_01, x_02, x_03, x_04)), axis=0)
            cam = self.cam1(intra)
            cam_exp = paddle.concat([cam]*4, axis=1)  # expand
            output = self.cam2(output) * (output + cam_exp)
            output = self.conv_final(output)
            return [output]
        else:
            output1 = self.final1(x_01)
            output2 = self.final2(x_02)
            output3 = self.final3(x_03)
            output4 = self.final4(x_04)
            output = self.conv_final(paddle.concat([output1, output2, output3, output4], axis=1))
            return [output, output1, output2, output3, output4]