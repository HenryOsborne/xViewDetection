import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
from mmdet.models.builder import NECKS

from mmdet.models.necks.TransNeck.dcn import DCN


def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels)


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def c2_xavier_fill(module: nn.Module) -> None:
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        c2_xavier_fill(self.conv_atten)
        c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.dcpack_L2 = DCN(out_nc, out_nc, kernel_size=3, stride=1, padding=1, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        return feat_align + feat_arm


@NECKS.register_module()
class FAPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(FAPN, self).__init__()

        # input conv
        self.conv_c2 = ConvModule(in_channels[0], out_channels, 1, inplace=False)
        self.conv_c3 = ConvModule(in_channels[1], out_channels, 1, inplace=False)
        self.conv_c4 = ConvModule(in_channels[2], out_channels, 1, inplace=False)
        self.conv_c5 = ConvModule(in_channels[3], out_channels, 1, inplace=False)

        self.gen_p4 = FeatureAlign_V2(256, 256)
        self.gen_p3 = FeatureAlign_V2(256, 256)
        self.gen_p2 = FeatureAlign_V2(256, 256)

        self.fpn_conv_p2 = ConvModule(out_channels, out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p3 = ConvModule(out_channels, out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p4 = ConvModule(out_channels, out_channels, 3, padding=1, inplace=False)
        self.fpn_conv_p5 = ConvModule(out_channels, out_channels, 3, padding=1, inplace=False)

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

        p4 = self.gen_p4(c4, p5)
        p3 = self.gen_p3(c3, p4)
        p2 = self.gen_p2(c2, p3)

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
    x = [torch.randn(1, 256, 200, 200),
         torch.randn(1, 512, 100, 100),
         torch.randn(1, 1024, 50, 50),
         torch.randn(1, 2048, 25, 25)]

    neck = FAPN([256, 512, 1024, 2048], 256)

    y = neck(x)
    print(y)

    pass
