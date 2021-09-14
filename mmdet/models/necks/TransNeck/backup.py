import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from mmdet.models.necks.TransNeck import TransEncoder, CBAM
from mmdet.models.builder import NECKS


class MyNeckBackup(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 depth=2,
                 num_heads=3,
                 upsample_cfg=dict(mode='nearest')):
        super(MyNeckBackup, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg

        self.conv_c2 = ConvModule(in_channels[0], in_channels[0], 1, inplace=False)
        self.conv_c3 = ConvModule(in_channels[1], in_channels[1], 1, inplace=False)
        self.conv_c4 = ConvModule(in_channels[2], in_channels[2], 1, inplace=False)
        self.conv_c5 = ConvModule(in_channels[3], in_channels[3], 1, inplace=False)

        self.last_layer1 = TransEncoder(in_channels[-1], in_channels[-1], depth=depth, num_heads=num_heads)
        self.last_layer2 = TransEncoder(in_channels[-1], in_channels[-1], depth=depth, num_heads=num_heads)
        self.last_layer3 = TransEncoder(in_channels[-1], in_channels[-1], depth=depth, num_heads=num_heads)
        self.out_layer = TransEncoder(in_channels[0], in_channels[0], depth=depth, num_heads=num_heads)

        self.cbam1 = CBAM(in_channels[-2])
        self.cbam2 = CBAM(in_channels[-3])

        self.lateral_convs = nn.ModuleList()
        for i in range(len(in_channels) - 1, 0, -1):
            l_conv = ConvModule(
                in_channels[i],
                in_channels[i - 1],
                1,
                inplace=False)
            self.lateral_convs.append(l_conv)

        self.fpn_conv_p2 = ConvModule(in_channels[0], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p3 = ConvModule(in_channels[1], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p4 = ConvModule(in_channels[2], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p5 = ConvModule(in_channels[3], out_channels, 3, padding=1, inplace=False)

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        out = []
        c2, c3, c4, c5 = inputs

        c2 = self.conv_c2(c2)
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        c5 = self.conv_c5(c5)

        p5 = self.last_layer1(c5)
        p5 = self.last_layer2(p5)
        p5 = self.last_layer3(p5)

        p4 = F.interpolate(self.lateral_convs[0](p5), size=c4.size()[2:], **self.upsample_cfg) + c4
        p4 = self.cbam1(p4)
        p3 = F.interpolate(self.lateral_convs[1](p4), size=c3.size()[2:], **self.upsample_cfg) + c3
        p3 = self.cbam2(p3)
        p2 = F.interpolate(self.lateral_convs[2](p3), size=c2.size()[2:], **self.upsample_cfg) + c2
        p2 = self.out_layer(p2)

        p2 = self.fpn_conv_p2(p2)
        p3 = self.fpn_conv_p3(p3)
        p4 = self.fpn_conv_p4(p4)
        p5 = self.fpn_conv_p5(p5)

        #####################################################################
        out.append(p2)
        out.append(p3)
        out.append(p4)
        out.append(p5)
        out.append(F.max_pool2d(out[-1], 1, stride=2))
        #####################################################################

        return tuple(out)


# 96, 192, 384, 768
if __name__ == '__main__':
    x = [torch.randn(1, 96, 200, 200),
         torch.randn(1, 192, 100, 100),
         torch.randn(1, 384, 50, 50),
         torch.randn(1, 768, 25, 25)]
    neck = MyNeck(in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5)
    y = neck(x)
    print(y)
