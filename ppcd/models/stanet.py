import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppcd.models.backbone import resnet18
from ppcd.models.layers import BAM
from ppcd.models.layers import normal_init, constant_init, kaiming_normal_init


class STANet(nn.Layer):
    """
    The STANet implementation based on PaddlePaddle.
    The original article refers to
    Chen H , Shi Z. "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"
    (https://www.researchgate.net/publication/341586750_A_Spatial-Temporal_Attention-Based_Method_and_a_New_Dataset_for_Remote_Sensing_Image_Change_Detection).
    Args:
        in_channels (int, optional): Number of an image's channel.  Default: 3.
        # out_channels : 1.
    """
    def __init__(self, in_channels=3, att_mode='BAM'):
        super(STANet, self).__init__()
        f_c = 64
        if att_mode != 'BAM' and att_mode != 'PAM':
            raise ValueError('att_mode must be BAM or PAM')
        self.netF = backbone3(f_c=f_c,freeze_bn=False, in_channels=in_channels)
        self.netA = CDSA(in_channels=f_c, ds=1, mode=att_mode)
        self.pairwise_distance = nn.PairwiseDistance(keepdim=True)

    def forward(self, t1, t2):
        feat_t1 = self.netF(t1)
        feat_t2 = self.netF(t2)
        feat_t1, feat_t2 = self.netA(feat_t1, feat_t2)
        dist = self.pairwise_distance(feat_t1, feat_t2)  # 特征距离
        dist = F.interpolate(dist, size=t1.shape[2:], mode='bilinear', align_corners=True)
        return [dist]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        normal_init(m.weight, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        normal_init(m.weight, mean=1.0, std=0.02)
        constant_init(m.bias, value=0)


class CDSA(nn.Layer):
    """
        self attention module for change detection
    """
    def __init__(self, in_channels, ds=1, mode='BAM'):
        super(CDSA, self).__init__()
        if mode == 'BAM':
            self.Self_Att = BAM(in_channels, ds=ds)
        elif self.mode == 'PAM':
            self.Self_Att = PAM(in_channels=in_channels, out_channels=in_channels, sizes=[1,2,4,8], ds=ds)
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = paddle.concat([x1, x2], 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]


class backbone3(nn.Layer):
    def __init__(self, f_c = 64, freeze_bn=False, in_channels=3):
        super(backbone3, self).__init__()
        BatchNorm = nn.BatchNorm2D
        self.backbone = resnet18(pretrained=True, in_channels=in_channels)
        self.decoder = Decoder(f_c, BatchNorm)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, f2, f3, f4 = self.backbone(input)
        x = self.decoder(x, f2, f3, f4)
        return x

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class DR(nn.Layer):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2D(self.in_d, self.out_d, 1)
        self.bn1 = nn.BatchNorm2D(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Layer):
    def __init__(self, fc, BatchNorm=nn.BatchNorm2D):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(64, 96)
        self.dr3 = DR(128, 96)
        self.dr4 = DR(256, 96)
        self.dr5 = DR(512, 96)
        self.last_conv = nn.Sequential(nn.Conv2D(384, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2D(256, self.fc, kernel_size=1, stride=1, padding=0),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )
        self._init_weight()

    def forward(self, x,low_level_feat2, low_level_feat3, low_level_feat4):
        # x1 = self.dr1(low_level_feat1)
        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        # x2 = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = paddle.concat([x, x2, x3, x4], axis=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_init(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)