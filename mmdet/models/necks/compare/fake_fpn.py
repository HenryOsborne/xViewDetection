import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from mmdet.models.necks.TransNeck import TransEncoder
from mmdet.models.necks.compare.se_net import SELayer
from mmdet.models.builder import NECKS


@NECKS.register_module()
class FakeFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FakeFPN, self).__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(0, 4):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

    def forward(self, inputs):

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))

        return laterals

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


if __name__ == '__main__':
    x = [torch.randn(1, 96, 200, 200),
         torch.randn(1, 192, 100, 100),
         torch.randn(1, 384, 50, 50),
         torch.randn(1, 768, 25, 25)]

    m = FakeFPN(in_channels=[96, 192, 384, 768], out_channels=256)
    y = m(x)
    pass
