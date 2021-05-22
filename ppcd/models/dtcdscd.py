import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from paddle.vision.models import ResNet
import math
from ppcd.models.layers import constant_init, normal_init


class SELayer(nn.Layer):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        assert reduction >= 16
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _, _ = x.shape
        y = self.avg_pool(x).reshape((N, C))
        y = self.fc(y).reshape((N, C, 1, 1))
        return x * paddle.expand_as(y, x)


class Dblock(nn.Layer):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2D(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2D(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2D(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2D(channel, channel, kernel_size=3, dilation=8, padding=8)
        # init
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D) or isinstance(sublayer, nn.Conv2DTranspose):
                if sublayer.bias is not None:
                    constant_init(sublayer.bias, value=0)

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x))
        dilate2_out = F.relu(self.dilate2(dilate1_out))
        dilate3_out = F.relu(self.dilate3(dilate2_out))
        dilate4_out = F.relu(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SEBasicBlock(nn.Layer):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)

        return out


class DecoderBlock(nn.Layer):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, in_channels//4, 1)
        self.norm1 = nn.BatchNorm(in_channels//4)
        self.scse = SCSEBlock(in_channels//4)
        self.deconv2 = nn.Conv2DTranspose(in_channels//4, in_channels//4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm(in_channels//4)
        self.conv3 = nn.Conv2D(in_channels//4, n_filters, 1)
        self.norm3 = nn.BatchNorm(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        y = self.scse(x)
        x = x + y
        x = self.deconv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        return x


class SCSEBlock(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.channel_excitation = nn.Sequential(
            nn.Conv2D(channel, int(channel//reduction), kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(int(channel//reduction), channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.spatial_se = nn.Sequential(
            nn.Conv2D(channel, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # N, C, _, _ = x.shape
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = x * chn_se
        spa_se = self.spatial_se(x)
        spa_se = x * spa_se
        return (chn_se + spa_se)


class CDNet(nn.Layer):
    def __init__(self, in_channels, block, layers, num_classes=2):
        super(CDNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.in_planes = 64
        self.firstconv = nn.Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.firstbn = nn.BatchNorm(64)
        self.firstmaxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # encode
        self.encoder1 = self._make_layer(block, 64, layers[0])
        self.encoder2 = self._make_layer(block, 128, layers[1], stride=2)
        self.encoder3 = self._make_layer(block, 256, layers[2], stride=2)
        self.encoder4 = self._make_layer(block, 512, layers[3], stride=2)
        # decode
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # --
        self.dblock_master = Dblock(512)
        self.dblock = Dblock(512)
        self.decoder4_master = DecoderBlock(filters[3], filters[2])
        self.decoder3_master = DecoderBlock(filters[2], filters[1])
        self.decoder2_master = DecoderBlock(filters[1], filters[0])
        self.decoder1_master = DecoderBlock(filters[0], filters[0])
        # final
        self.finaldeconv1_master = nn.Conv2DTranspose(filters[0], 32, 4, 2, 1)
        self.finalconv2_master = nn.Conv2D(32, 32, 3, padding=1)
        self.finalconv3_master = nn.Conv2D(32, num_classes, 3, padding=1)
        self.finaldeconv1 = nn.Conv2DTranspose(filters[0], 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2D(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2D(32, num_classes, 3, padding=1)
        # init
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                n = sublayer._kernel_size[0] * sublayer._kernel_size[1] * sublayer._out_channels
                normal_init(sublayer.weight, mean=0, std=math.sqrt(2. / n))
            elif isinstance(sublayer, nn.BatchNorm):
                constant_init(sublayer.weight, value=0)
                constant_init(sublayer.bias, value=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.in_planes, planes*block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        # Encoder_1
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = F.relu(x)
        x = self.firstmaxpool(x)
        e1_x = self.encoder1(x)
        e2_x = self.encoder2(e1_x)
        e3_x = self.encoder3(e2_x)
        e4_x = self.encoder4(e3_x)
        # Center_1
        e4_x_center = self.dblock(e4_x)
        # Decoder_1
        d4_x = self.decoder4(e4_x_center) + e3_x
        d3_x = self.decoder3(d4_x) + e2_x
        d2_x = self.decoder2(d3_x) + e1_x
        d1_x = self.decoder1(d2_x)
        out1 = self.finaldeconv1(d1_x)
        out1 = F.relu(out1)
        out1 = self.finalconv2(out1)
        out1 = F.relu(out1)
        out1 = self.finalconv3(out1)
        # Encoder_2
        y = self.firstconv(y)
        y = self.firstbn(y)
        y = F.relu(y)
        y = self.firstmaxpool(y)
        e1_y = self.encoder1(y)
        e2_y = self.encoder2(e1_y)
        e3_y = self.encoder3(e2_y)
        e4_y = self.encoder4(e3_y)
        # Center_2
        e4_y_center = self.dblock(e4_y)
        # Decoder_2
        d4_y = self.decoder4(e4_y_center) + e3_y
        d3_y = self.decoder3(d4_y) + e2_y
        d2_y = self.decoder2(d3_y) + e1_y
        d1_y = self.decoder1(d2_y)
        out2 = self.finaldeconv1(d1_y)
        out2 = F.relu(out2)
        out2 = self.finalconv2(out2)
        out2 = F.relu(out2)
        out2 = self.finalconv3(out2)
        # center_master
        e4 = self.dblock_master(e4_x - e4_y)
        # decoder_master
        d4 = self.decoder4_master(e4) + e3_x - e3_y
        d3 = self.decoder3_master(d4) + e2_x - e2_y
        d2 = self.decoder2_master(d3) + e1_x - e1_y
        d1 = self.decoder1_master(d2)
        out = self.finaldeconv1_master(d1)
        out = F.relu(out)
        out = self.finalconv2_master(out)
        out = F.relu(out)
        out = self.finalconv3_master(out)
        return [out, out1, out2]


def CDNet34(in_channels=3, **kwargs):
    model = CDNet(in_channels, SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model