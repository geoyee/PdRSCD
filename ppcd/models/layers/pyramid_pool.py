import paddle
from paddle import nn
import paddle.nn.functional as F
from ppcd.models import layers


class PPModule(nn.Layer):
    """
    Pyramid pooling module originally in PSPNet.
    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """
    def __init__(self, in_channels, out_channels, bin_sizes, dim_reduction,
                 align_corners):
        super().__init__()
        self.bin_sizes = bin_sizes
        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)
        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])
        self.conv_bn_relu2 = layers.ConvBNReLU(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1)
        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        """
        Create one pooling layer.
        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.
        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.
        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            size (int): The out size of the pooled layer.
        Returns:
            conv (Tensor): A tensor after Pyramid Pooling Module.
        """
        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)
        
    def forward(self, input):
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = paddle.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)
        return out