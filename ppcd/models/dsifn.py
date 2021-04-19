import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import vgg16
from ppcd.models.layers import CAM, SAM


class Vgg16Base(nn.Layer):
    # Vgg16 feature extraction backbone
    def __init__(self, in_channels=3):
        super(Vgg16Base, self).__init__()
        features = vgg16(pretrained=True).sublayers()[0].sublayers()
        if in_channels != 3:
            features[0] = nn.Conv2D(in_channels, 64, kernel_size=[3, 3], padding=1, data_format='NCHW')
        self.features = nn.LayerList(features)
        self.features.eval()

    def forward(self, x):
        results = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in {3, 8, 15, 22, 29}:
                results.append(x)
        return results


class CPBD(nn.Sequential):
    def __init__(self, in_channels, out_channels):
    	super(CPBD, self).__init__(
        nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        nn.BatchNorm(out_channels),
        nn.Dropout(p=0.6),
    )


class DSIFN(nn.Layer):
    """
    The DSIFN implementation based on PaddlePaddle.

    The original article refers to
    Cz, A , et al. "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images"
    (https://www.sciencedirect.com/science/article/abs/pii/S0924271620301532).

    Args:
        in_channels (int, optional): The channel number of input image.  default:3.
        num_classes (int, optional): The unique number of target classes.  default:2.
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.backbone = Vgg16Base(in_channels=in_channels)
        self.sa1 = SAM()
        self.sa2 = SAM()
        self.sa3 = SAM()
        self.sa4 = SAM()
        self.sa5 = SAM()
        # branch1
        self.ca1 = CAM(in_channels=1024, ratio=8)
        self.bn_ca1 = nn.BatchNorm(1024)
        self.o1_conv1 = CPBD(1024, 512)
        self.o1_conv2 = CPBD(512, 512)
        self.bn_sa1 = nn.BatchNorm(512)
        self.o1_conv3 = nn.Conv2D(512, num_classes, 1)
        self.trans_conv1 = nn.Conv2DTranspose(512, 512, kernel_size=2, stride=2)
        # branch 2
        self.ca2 = CAM(in_channels=1536, ratio=8)
        self.bn_ca2 = nn.BatchNorm(1536)
        self.o2_conv1 = CPBD(1536, 512)
        self.o2_conv2 = CPBD(512, 256)
        self.o2_conv3 = CPBD(256, 256)
        self.bn_sa2 = nn.BatchNorm(256)
        self.o2_conv4 = nn.Conv2D(256, num_classes, 1)
        self.trans_conv2 = nn.Conv2DTranspose(256, 256, kernel_size=2, stride=2)
        # branch 3
        self.ca3 = CAM(in_channels=768, ratio=8)
        self.o3_conv1 = CPBD(768, 256)
        self.o3_conv2 = CPBD(256, 128)
        self.o3_conv3 = CPBD(128, 128)
        self.bn_sa3 = nn.BatchNorm(128)
        self.o3_conv4 = nn.Conv2D(128, num_classes, 1)
        self.trans_conv3 = nn.Conv2DTranspose(128, 128, kernel_size=2, stride=2)
        # branch 4
        self.ca4 = CAM(in_channels=384, ratio=8)
        self.o4_conv1 = CPBD(384, 128)
        self.o4_conv2 = CPBD(128, 64)
        self.o4_conv3 = CPBD(64, 64)
        self.bn_sa4 = nn.BatchNorm(64)
        self.o4_conv4 = nn.Conv2D(64, num_classes, 1)
        self.trans_conv4 = nn.Conv2DTranspose(64, 64, kernel_size=2, stride=2)
        # branch 5
        self.ca5 = CAM(in_channels=192, ratio=8)
        self.o5_conv1 = CPBD(192, 64)
        self.o5_conv2 = CPBD(64, 32)
        self.o5_conv3 = CPBD(32, 16)
        self.bn_sa5 = nn.BatchNorm(16)
        self.o5_conv4 = nn.Conv2D(16, num_classes, 1)

    def forward(self, t1_input, t2_input):
        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = self.backbone(t1_input)
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29 = self.backbone(t2_input)
        x = paddle.concat([t1_f_l29, t2_f_l29], axis=1)
        x = self.ca1(x) * x
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)
        branch_1_out = self.o1_conv3(x)
        x = self.trans_conv1(x)
        x = paddle.concat([x, t1_f_l22, t2_f_l22], axis=1)
        x = self.ca2(x) * x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) * x
        x = self.bn_sa2(x)
        branch_2_out = self.o2_conv4(x)
        x = self.trans_conv2(x)
        x = paddle.concat([x, t1_f_l15, t2_f_l15], axis=1)
        x = self.ca3(x) * x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) * x
        x = self.bn_sa3(x)
        branch_3_out = self.o3_conv4(x)
        x = self.trans_conv3(x)
        x = paddle.concat([x, t1_f_l8, t2_f_l8], axis=1)
        x = self.ca4(x) * x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) * x
        x = self.bn_sa4(x)
        branch_4_out =self.o4_conv4(x)
        x = self.trans_conv4(x)
        x = paddle.concat([x, t1_f_l3, t2_f_l3], axis=1)
        x = self.ca5(x) * x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) * x
        x = self.bn_sa5(x)
        branch_5_out = self.o5_conv4(x)
        return [branch_5_out, branch_4_out, branch_3_out, branch_2_out, branch_1_out]