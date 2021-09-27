import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from mmdet.models.necks.TransNeck import TransEncoder, CBAM
from mmdet.models.builder import NECKS


@NECKS.register_module()
class MyNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=[96, 192, 384, 768],
                 use_path_augment=False,
                 depth=2,
                 num_heads=3,
                 upsample_cfg=dict(mode='nearest')):
        super(MyNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_cfg = upsample_cfg
        self.use_path_augment = use_path_augment

        assert len(in_channels) == len(mid_channels)

        # input conv
        self.conv_c2 = ConvModule(in_channels[0], mid_channels[0], 1, inplace=False)
        self.conv_c3 = ConvModule(in_channels[1], mid_channels[1], 1, inplace=False)
        self.conv_c4 = ConvModule(in_channels[2], mid_channels[2], 1, inplace=False)
        self.conv_c5 = ConvModule(in_channels[3], mid_channels[3], 1, inplace=False)

        # c5 encode
        self.last_layer1 = TransEncoder(mid_channels[-1], mid_channels[-1], depth=depth, num_heads=num_heads)
        self.last_layer2 = TransEncoder(mid_channels[-1], mid_channels[-1], depth=depth, num_heads=num_heads)
        self.last_layer3 = TransEncoder(mid_channels[-1], mid_channels[-1], depth=depth, num_heads=num_heads)

        self.out_layer_p2 = TransEncoder(mid_channels[0], mid_channels[0], depth=depth, num_heads=num_heads)
        if self.use_path_augment:
            self.out_layer_p3 = TransEncoder(mid_channels[0], mid_channels[1], depth=depth, num_heads=num_heads)
            self.out_layer_p4_1 = TransEncoder(mid_channels[1], mid_channels[2], depth=depth, num_heads=num_heads)
            self.out_layer_p4_2 = TransEncoder(mid_channels[2], mid_channels[2], depth=depth, num_heads=num_heads)
            self.out_layer_p5_1 = TransEncoder(mid_channels[2], mid_channels[3], depth=depth, num_heads=num_heads)
            self.out_layer_p5_2 = TransEncoder(mid_channels[3], mid_channels[3], depth=depth, num_heads=num_heads)
            self.out_layer_p5_3 = TransEncoder(mid_channels[3], mid_channels[3], depth=depth, num_heads=num_heads)

        self.cbam1 = CBAM(mid_channels[2])
        self.cbam2 = CBAM(mid_channels[1])
        if self.use_path_augment:
            self.cbam3 = CBAM(mid_channels[0])
            self.cbam4 = CBAM(mid_channels[1])
            self.cbam5 = CBAM(mid_channels[2])

        if self.use_path_augment:
            self.conv_pa_n2 = ConvModule(mid_channels[0], mid_channels[0], kernel_size=3, stride=2, padding=1)
            self.conv_pa_n3 = ConvModule(mid_channels[1], mid_channels[1], kernel_size=3, stride=2, padding=1)
            self.conv_pa_n4 = ConvModule(mid_channels[2], mid_channels[2], kernel_size=3, stride=2, padding=1)

        self.lateral_convs = nn.ModuleList()
        for i in range(len(mid_channels) - 1, 0, -1):
            l_conv = ConvModule(mid_channels[i], mid_channels[i - 1], 1, inplace=False)
            self.lateral_convs.append(l_conv)

        self.fpn_conv_p2 = ConvModule(mid_channels[0], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p3 = ConvModule(mid_channels[1], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p4 = ConvModule(mid_channels[2], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p5 = ConvModule(mid_channels[3], out_channels, 3, padding=1, inplace=False)

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

        if self.use_path_augment:
            p5 = self.lateral_convs[0](p5)
            p4 = F.interpolate(p5, size=c4.size()[2:], **self.upsample_cfg) + c4
            p4 = self.cbam1(p4)
            p4 = self.lateral_convs[1](p4)
            p3 = F.interpolate(p4, size=c3.size()[2:], **self.upsample_cfg) + c3
            p3 = self.cbam2(p3)
            p3 = self.lateral_convs[2](p3)
            p2 = F.interpolate(p3, size=c2.size()[2:], **self.upsample_cfg) + c2
            p2 = self.out_layer_p2(p2)

            n2 = self.cbam3(p2)
            n2 = self.conv_pa_n2(n2)
            p3 = n2 + p3
            p3 = self.out_layer_p3(p3)

            n3 = self.cbam4(p3)
            n3 = self.conv_pa_n3(n3)
            p4 = n3 + p4
            p4 = self.out_layer_p4_1(p4)
            p4 = self.out_layer_p4_2(p4)

            n4 = self.cbam5(p4)
            n4 = self.conv_pa_n4(n4)
            p5 = n4 + p5
            p5 = self.out_layer_p5_1(p5)
            p5 = self.out_layer_p5_2(p5)
            p5 = self.out_layer_p5_3(p5)
        else:
            p4 = F.interpolate(self.lateral_convs[0](p5), size=c4.size()[2:], **self.upsample_cfg) + c4
            p4 = self.cbam1(p4)
            p3 = F.interpolate(self.lateral_convs[1](p4), size=c3.size()[2:], **self.upsample_cfg) + c3
            p3 = self.cbam2(p3)
            p2 = F.interpolate(self.lateral_convs[2](p3), size=c2.size()[2:], **self.upsample_cfg) + c2
            p2 = self.out_layer_p2(p2)

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


if __name__ == '__main__':
    x = [torch.randn(1, 96, 200, 200),
         torch.randn(1, 192, 100, 100),
         torch.randn(1, 384, 50, 50),
         torch.randn(1, 768, 25, 25)]
    neck = MyNeck(in_channels=[96, 192, 384, 768], out_channels=256, use_path_augment=True)
    y = neck(x)
    print(y)
    # z = [torch.randn(1, 256, 200, 200),
    #      torch.randn(1, 512, 100, 100),
    #      torch.randn(1, 1024, 50, 50),
    #      torch.randn(1, 2048, 25, 25)]
    # neck = MyNeck(in_channels=[256, 512, 1024, 2048], out_channels=256)
    # y = neck(z)
    # print(y)
