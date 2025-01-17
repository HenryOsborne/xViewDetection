import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import kaiming_init
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from mmdet.models.builder import NECKS
import torch
import mmcv
import numpy as np
import cv2


class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)
    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                dilation=(dilation, dilation),
                padding=(padding, padding),
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=(stride, stride),
                              padding=(padding, padding), dilation=(dilation, dilation), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CAM(nn.Module):
    def __init__(self, inplanes, reduction_ratio=1, fpn_lvl=4):
        super(CAM, self).__init__()
        self.fpn_lvl = fpn_lvl
        self.dila_conv = nn.Sequential(nn.Conv2d(inplanes * fpn_lvl // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       ASPP(inplanes // reduction_ratio, inplanes // (4 * reduction_ratio)),
                                       nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.BatchNorm2d(inplanes // reduction_ratio),
                                       nn.ReLU(inplace=False)
                                       )
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.upsample_cfg = dict(mode='nearest')
        self.down_conv = nn.ModuleList()
        self.att_conv = nn.ModuleList()
        for i in range(self.fpn_lvl):
            # --------------------------------------------------------------------------------------------------
            # self.att_conv.append(nn.Conv2d(inplanes // reduction_ratio,
            #                                1,
            #                                kernel_size=(3, 3),
            #                                stride=(1, 1),  # 2 ** i
            #                                padding=(1, 1)))
            self.att_conv.append(ChannelPool())
            self.att_conv.append(BasicConv(2, 1, (3, 3), stride=1, padding=(3 - 1) // 2, relu=False))
            # --------------------------------------------------------------------------------------------------
            if i == 0:
                down_stride = 1
            else:
                down_stride = 2
            self.down_conv.append(
                nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio, kernel_size=(3, 3),
                          stride=(down_stride, down_stride),
                          padding=(1, 1)))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):

        prev_shape = x[0].shape[2:]
        multi_feats = [x[0]]

        for i in range(1, len(x)):
            pyr_feats_2x = F.interpolate(x[i], size=prev_shape, **self.upsample_cfg)
            multi_feats.append(pyr_feats_2x)

        multi_feats = torch.cat(multi_feats, 1)
        lvl_fea = self.dila_conv(multi_feats)

        multi_atts = []

        for i in range(self.fpn_lvl):
            lvl_fea = self.down_conv[i](lvl_fea)
            # ---------------------------------------------------
            # lvl_att = self.att_conv[i](lvl_fea)
            lvl_att = self.att_conv[2 * i](lvl_fea)
            lvl_att = self.att_conv[2 * i + 1](lvl_att)
            # ---------------------------------------------------
            multi_atts.append(self.sigmoid(lvl_att))

        # visualization

        # for i in range(self.fpn_lvl):  # self.fpn_lvl
        #     att = (multi_atts[i].detach().cpu().numpy()[0])
        #     att /= np.max(att)
        #     att = np.power(att, 0.8)
        #     att = att * 255
        #     att = att.astype(np.uint8).transpose(1, 2, 0)
        #     att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        #     mmcv.imshow(att)
        #     cv2.waitKey(0)

        return multi_atts


@NECKS.register_module()
class SSFPNModified(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 pool_ratios=[0.1, 0.2, 0.3],
                 residual_feature_augmentation=False,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dual_head=False,
                 upsample_cfg=dict(mode='nearest')):
        super(SSFPNModified, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.use_dual_head = dual_head
        self.use_rfa = residual_feature_augmentation
        self.CAM = CAM(out_channels)
        # self.grads = {}
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        # --------------------------------------------------------------------------------------------------------------
        # add lateral conv for features generated by rato-invariant scale adaptive pooling
        if self.use_rfa:
            self.adaptive_pool_output_ratio = pool_ratios
            self.high_lateral_conv = nn.ModuleList()
            self.high_lateral_conv.extend(
                [nn.Conv2d(out_channels, out_channels, (1, 1)) for k in
                 range(len(self.adaptive_pool_output_ratio))])
            self.high_lateral_conv_attention = nn.Sequential(
                nn.Conv2d(out_channels * (len(self.adaptive_pool_output_ratio)), out_channels, (1, 1)),
                nn.ReLU(),
                nn.Conv2d(out_channels, len(self.adaptive_pool_output_ratio), (3, 3), padding=(1, 1)))
        # --------------------------------------------------------------------------------------------------------------

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

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

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # --------------------------------------------------------------------------------------------------------------
        if self.use_rfa:
            # Residual Feature Augmentation
            h, w = laterals[-1].size(2), laterals[-1].size(3)
            # Ratio Invariant Adaptive Pooling
            AdapPool_Features = [
                F.interpolate(self.high_lateral_conv[j](F.adaptive_avg_pool2d(laterals[-1], output_size=(
                    max(1, int(h * self.adaptive_pool_output_ratio[j])),
                    max(1, int(w * self.adaptive_pool_output_ratio[j]))))),
                              size=(h, w), mode='bilinear', align_corners=True) for j in
                range(len(self.adaptive_pool_output_ratio))]
            Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
            fusion_weights = self.high_lateral_conv_attention(Concat_AdapPool_Features)
            fusion_weights = torch.sigmoid(fusion_weights)
            adap_pool_fusion = 0
            for i in range(len(self.adaptive_pool_output_ratio)):
                adap_pool_fusion += torch.unsqueeze(fusion_weights[:, i, :, :], dim=1) * AdapPool_Features[i]
        else:
            adap_pool_fusion = torch.zeros_like(laterals[-1])
        # --------------------------------------------------------------------------------------------------------------

        # build attention map

        att_list = self.CAM(laterals)
        laterals = [(1 + att_list[i]) * laterals[i] for i in range(len(laterals))]  #
        if self.use_dual_head:
            raw_laternals = laterals.copy()

        # --------------------------------------------------------------------------------------------------------------
        if self.use_rfa:
            laterals[-1] = torch.sigmoid(laterals[-1] + adap_pool_fusion)
        # --------------------------------------------------------------------------------------------------------------

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]

                # get intersection of Adjacent attention maps
                att_2x = F.interpolate(att_list[i], size=prev_shape, **self.upsample_cfg)
                att_insec = att_list[i - 1] * att_2x

                # get ROI of current attention map
                select_gate = att_insec

                laterals[i - 1] = laterals[i - 1] + select_gate * F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        # build outputs

        outs = [
            (1 + att_list[i]) * self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        if self.use_dual_head:
            return tuple(outs), tuple(att_list), tuple(raw_laternals)
        else:
            return tuple(outs), tuple(att_list)


if __name__ == '__main__':
    z = [torch.randn(1, 256, 200, 200),
         torch.randn(1, 512, 100, 100),
         torch.randn(1, 1024, 50, 50),
         torch.randn(1, 2048, 25, 25)]
    neck = SSFPNModified(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
    y = neck(z)
    pass
