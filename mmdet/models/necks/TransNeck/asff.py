import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.necks.TransNeck import TransEncoder, CBAM
from mmcv.cnn import ConvModule, xavier_init
from mmdet.models.builder import NECKS


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self,
                 level,
                 use_additional_level,
                 rfb=False,
                 dim=[768, 384, 192, 96]):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = dim
        self.inter_dim = self.dim[self.level]
        self.use_additional_level = use_additional_level
        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        if level == 0:
            self.stride_level_1 = ConvModule(384, self.inter_dim, 3, 2, 1)
            self.stride_level_2 = ConvModule(192, self.inter_dim, 3, 2, 1)
            if self.use_additional_level:
                self.stride_level_3 = ConvModule(96, self.inter_dim, 3, 2, 1)
            self.expand = ConvModule(self.inter_dim, self.dim[self.level], 3, 1, 1)
        elif level == 1:
            self.compress_level_0 = ConvModule(768, self.inter_dim, 1, 1)
            self.stride_level_2 = ConvModule(192, self.inter_dim, 3, 2, 1)
            self.stride_level_3 = ConvModule(96, self.inter_dim, 3, 2, 1)
            self.expand = ConvModule(self.inter_dim, self.dim[self.level], 3, 1, 1)
        elif level == 2:
            self.compress_level_0 = ConvModule(768, self.inter_dim, 1, 1)
            self.compress_level_1 = ConvModule(384, self.inter_dim, 1, 1)
            self.stride_level_3 = ConvModule(96, self.inter_dim, 3, 2, 1)
            self.expand = ConvModule(self.inter_dim, self.dim[self.level], 3, 1, 1)
        elif level == 3:
            self.compress_level_1 = ConvModule(384, self.inter_dim, 1, 1)
            self.compress_level_2 = ConvModule(192, self.inter_dim, 1, 1)
            if self.use_additional_level:
                self.compress_level_0 = ConvModule(768, self.inter_dim, 1, 1)
            self.expand = ConvModule(self.inter_dim, self.dim[self.level], 3, 1, 1)
        else:
            raise ValueError('Wrong Level Number')

        self.weight_level_0 = ConvModule(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = ConvModule(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = ConvModule(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = ConvModule(self.inter_dim, compress_c, 1, 1)

        if self.use_additional_level or self.level == 1 or self.level == 2:
            self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, p5, p4, p3, p2):
        if self.level == 0:
            level_0_resized = p5
            level_1_resized = self.stride_level_1(p4)
            level_2_downsampled_inter = F.max_pool2d(p3, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            if self.use_additional_level:
                level_3_downsampled_inter = F.max_pool2d(p2, 3, stride=4, padding=1)  # 4x downsample
                level_3_resized = self.stride_level_3(level_3_downsampled_inter)
            else:
                level_3_resized = None
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(p5)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = p4
            level_2_resized = self.stride_level_2(p3)
            level_3_downsampled_inter = F.max_pool2d(p2, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(p5)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = self.compress_level_1(p4)
            level_1_resized = F.interpolate(level_1_resized, scale_factor=2, mode='nearest')
            level_2_resized = p3
            level_3_resized = self.stride_level_3(p2)
        else:
            if self.use_additional_level:
                level_0_downsampled_inter = self.compress_level_0(p5)
                level_0_resized = F.interpolate(level_0_downsampled_inter, scale_factor=8, mode='nearest')
            else:
                level_0_resized = None
            level_1_downsampled_inter = self.compress_level_1(p4)
            level_1_resized = F.interpolate(level_1_downsampled_inter, scale_factor=4, mode='nearest')
            level_2_downsampled_inter = self.compress_level_2(p3)
            level_2_resized = F.interpolate(level_2_downsampled_inter, scale_factor=2, mode='nearest')
            level_3_resized = p2

        if self.use_additional_level or self.level == 1 or self.level == 2:
            level_0_weight_v = self.weight_level_0(level_0_resized)
            level_1_weight_v = self.weight_level_1(level_1_resized)
            level_2_weight_v = self.weight_level_2(level_2_resized)
            level_3_weight_v = self.weight_level_3(level_3_resized)
            levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            levels_weight = F.softmax(levels_weight, dim=1)
            fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                                level_1_resized * levels_weight[:, 1:2, :, :] + \
                                level_2_resized * levels_weight[:, 2:3, :, :] + \
                                level_3_resized * levels_weight[:, 3:, :, :]
            out = self.expand(fused_out_reduced)

        elif not self.use_additional_level and self.level == 0:
            level_0_weight_v = self.weight_level_0(level_0_resized)
            level_1_weight_v = self.weight_level_1(level_1_resized)
            level_2_weight_v = self.weight_level_2(level_2_resized)
            levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                                level_1_resized * levels_weight[:, 1:2, :, :] + \
                                level_2_resized * levels_weight[:, 2:, :, :]
            out = self.expand(fused_out_reduced)

        elif not self.use_additional_level and self.level == 3:
            level_1_weight_v = self.weight_level_1(level_1_resized)
            level_2_weight_v = self.weight_level_2(level_2_resized)
            level_3_weight_v = self.weight_level_3(level_3_resized)
            levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
            levels_weight = self.weight_levels(levels_weight_v)
            fused_out_reduced = level_1_resized * levels_weight[:, 0:1, :, :] + \
                                level_2_resized * levels_weight[:, 1:2, :, :] + \
                                level_3_resized * levels_weight[:, 2:, :, :]
            out = self.expand(fused_out_reduced)
        else:
            raise NotImplementedError

        return out


@NECKS.register_module()
class ASFFFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_additional_level=True,
                 mid_channels=[96, 192, 384, 768],
                 depth=2,
                 num_heads=3,
                 upsample_cfg=dict(mode='nearest')
                 ):
        super(ASFFFPN, self).__init__()
        self.upsample_cfg = upsample_cfg

        # input conv
        self.conv_c2 = ConvModule(in_channels[0], mid_channels[0], 1, inplace=False)
        self.conv_c3 = ConvModule(in_channels[1], mid_channels[1], 1, inplace=False)
        self.conv_c4 = ConvModule(in_channels[2], mid_channels[2], 1, inplace=False)
        self.conv_c5 = ConvModule(in_channels[3], mid_channels[3], 1, inplace=False)

        self.lateral_convs = nn.ModuleList()
        for i in range(len(mid_channels) - 1, 0, -1):
            l_conv = ConvModule(mid_channels[i], mid_channels[i - 1], 1, inplace=False)
            self.lateral_convs.append(l_conv)

        self.p5_fusion = ASFF(0, use_additional_level)
        self.p4_fusion = ASFF(1, use_additional_level)
        self.p3_fusion = ASFF(2, use_additional_level)
        self.p2_fusion = ASFF(3, use_additional_level)

        self.fpn_conv_p2 = ConvModule(mid_channels[0], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p3 = ConvModule(mid_channels[1], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p4 = ConvModule(mid_channels[2], out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p5 = ConvModule(mid_channels[3], out_channels, 3, padding=1, inplace=False)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        out = []

        c2, c3, c4, c5 = inputs

        c2 = self.conv_c2(c2)
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        p5 = self.conv_c5(c5)

        p4 = F.interpolate(self.lateral_convs[0](p5), size=c4.size()[2:], **self.upsample_cfg) + c4
        p3 = F.interpolate(self.lateral_convs[1](p4), size=c3.size()[2:], **self.upsample_cfg) + c3
        p2 = F.interpolate(self.lateral_convs[2](p3), size=c2.size()[2:], **self.upsample_cfg) + c2

        p5 = self.p5_fusion(p5, p4, p3, p2)
        p4 = self.p4_fusion(p5, p4, p3, p2)
        p3 = self.p3_fusion(p5, p4, p3, p2)
        p2 = self.p2_fusion(p5, p4, p3, p2)

        p2 = self.fpn_conv_p2(p2)
        p3 = self.fpn_conv_p3(p3)
        p4 = self.fpn_conv_p4(p4)
        p5 = self.fpn_conv_p5(p5)

        out.append(p2)
        out.append(p3)
        out.append(p4)
        out.append(p5)
        out.append(F.max_pool2d(out[-1], 1, stride=2))
        return out


if __name__ == '__main__':
    z = [torch.randn(1, 256, 200, 200),
         torch.randn(1, 512, 100, 100),
         torch.randn(1, 1024, 50, 50),
         torch.randn(1, 2048, 25, 25)]
    model = ASFFFPN(in_channels=[256, 512, 1024, 2048], out_channels=256, use_additional_level=True)
    y = model(z)
    pass
