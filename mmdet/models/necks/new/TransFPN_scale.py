import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import kaiming_init
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from mmdet.models.builder import NECKS
import torch
from mmdet.models.necks.TransNeck import TransEncoder, CBAM
from mmdet.models.backbones.swin_transformer import BasicLayer


class SwinEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channels,
                 kernel_size,
                 dilation,
                 padding,
                 depth,
                 num_heads,
                 dim=96):
        super(SwinEncoder, self).__init__()
        self.dim = dim
        self.input_proj = nn.Conv2d(in_channel, dim,
                                    kernel_size=(kernel_size, kernel_size),
                                    dilation=(dilation, dilation),
                                    padding=(padding, padding))
        self.encoder = BasicLayer(dim=dim, depth=depth, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Conv2d(dim, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        H, W = x.size(-2), x.size(-1)
        x = self.input_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x_out, H, W, x, Wh, Ww = self.encoder(x, H, W)

        x_out = self.norm(x_out)
        x_out = x_out.view(-1, H, W, self.dim).permute(0, 3, 1, 2).contiguous()
        out = self.output_proj(x_out)

        return out


class SwinEncoder2(nn.Module):
    def __init__(self,
                 in_channel,
                 dim,
                 depth,
                 num_heads):
        super(SwinEncoder2, self).__init__()
        self.dim = dim
        self.input_proj = nn.Conv2d(in_channel, dim, kernel_size=(1, 1))
        self.encoder = BasicLayer(dim=dim, depth=depth, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Conv2d(dim, in_channel, kernel_size=(1, 1))

    def forward(self, x):
        H, W = x.size(-2), x.size(-1)
        x = self.input_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x_out, H, W, x, Wh, Ww = self.encoder(x, H, W)

        x_out = self.norm(x_out)
        x_out = x_out.view(-1, H, W, self.dim).permute(0, 3, 1, 2).contiguous()
        out = self.output_proj(x_out)

        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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
            # encoder = SwinEncoder(in_channels, out_channels, kernel_size, dilation, padding, 2, 3)
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

        # ------------------------------------------------------------------
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels)
        )
        # ------------------------------------------------------------------

        self.swin_att = SwinEncoder2(256, 192, 2, 3)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        # -----------------------------------------------
        # avg_x = self.gap(x)
        # -------------------- switch -------------------
        # tensor_flatten = x.view(x.size(0), x.size(1), -1)
        # s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        # avg_x = (s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()).unsqueeze(-1)
        # -------------------- switch -------------------
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        channel_att_raw = self.mlp(avg_pool)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_att_raw2 = self.mlp(max_pool)
        channel_att_sum = channel_att_raw + channel_att_raw2
        avg_x = channel_att_sum.unsqueeze(-1).unsqueeze(-1)
        # -----------------------------------------------
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        # ------------------------------------------------
        out = self.swin_att(out)
        # ------------------------------------------------
        return out


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
            self.att_conv.append(nn.Conv2d(inplanes // reduction_ratio,
                                           1,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),  # 2 ** i
                                           padding=(1, 1)))
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
            lvl_att = self.att_conv[i](lvl_fea)
            # TODO:channel attention
            multi_atts.append(self.sigmoid(lvl_att))

        # visualization

        # for i in range(self.fpn_lvl):  # self.fpn_lvl
        #     att = (multi_atts[i].detach().cpu().numpy()[0])
        #     # att /= np.max(att)
        #     #att = np.power(att, 0.8)
        #     att = att * 255
        #     att = att.astype(np.uint8).transpose(1, 2, 0)
        #    # att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        #     mmcv.imshow(att)
        #     cv2.waitKey(0)

        return multi_atts


@NECKS.register_module()
class TransFPNScale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 depth=2,
                 num_heads=3):
        super(TransFPNScale, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
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

        # self.last_layer1 = TransEncoder(in_channels[-1], in_channels[-1], depth=depth, num_heads=num_heads)
        # self.last_layer2 = TransEncoder(in_channels[-1], in_channels[-1], depth=depth, num_heads=num_heads)
        # self.last_layer3 = TransEncoder(in_channels[-1], in_channels[-1], depth=depth, num_heads=num_heads)
        #
        # self.cbam = nn.ModuleList()
        # for i in range(len(self.in_channels) - 1, 0, -1):
        #     self.cbam.append(CBAM(out_channels, pool_types=['lse']))

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

        # ------------------------------------------------------------------
        # c5 = self.last_layer1(inputs[-1])
        # c5 = self.last_layer2(c5)
        # c5 = self.last_layer3(c5)
        # inputs = list(inputs)
        # inputs[-1] = c5
        # ------------------------------------------------------------------

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build attention map

        att_list = self.CAM(laterals)
        # ------------------------------------------------------------------
        laterals = [(1 + att_list[i]) * laterals[i] for i in range(len(laterals))]
        # ------------------------------------------------------------------

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
                laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                # ------------------------------------------------------------------
                # laterals[i - 1] = self.cbam[i - 1](laterals[i - 1])
                # ------------------------------------------------------------------

        # build outputs
        # ------------------------------------------------------------------
        outs = [
            (1 + att_list[i]) * self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]
        # ------------------------------------------------------------------

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

        return tuple(outs), tuple(att_list)


if __name__ == '__main__':
    x = [torch.randn(1, 96, 200, 200),
         torch.randn(1, 192, 100, 100),
         torch.randn(1, 384, 50, 50),
         torch.randn(1, 768, 25, 25)]
    neck = TransFPNScale(in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5)
    y = neck(x)
    print(y)
