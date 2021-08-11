import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from ..builder import NECKS
from .. import builder
import os


class fpn_module_global(nn.Module):
    def __init__(self, ):
        super(fpn_module_global, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.smooth1_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth2_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth4_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        self.toplayer_ext = nn.Conv2d(2048 * 2, 256, kernel_size=(1, 1), stride=(1, 1),
                                      padding=(0, 0))  # Reduce channels
        # Lateral layers
        self.latlayer1_ext = nn.Conv2d(1024 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer2_ext = nn.Conv2d(512 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer3_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        ###########################kernel_size=(3, 3)->kernel_size=(1, 1)#######################################
        # Smooth layers
        self.smooth1_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth2_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth3_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth4_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth1_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth2_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth3_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth4_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # self.smooth = nn.Conv2d(128 * 4 * 2, 128 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #########################################################################################################

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        up_x = F.interpolate(x, size=(H, W), **self._up_kwargs)

        return up_x + y

    def forward(self, c2, c3, c4, c5, c2_ext=None, c3_ext=None, c4_ext=None, c5_ext=None, ps0_ext=None, ps1_ext=None,
                ps2_ext=None, mode=None):

        # Top-down
        if c5_ext is None:
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p2 = self._upsample_add(p3, self.latlayer3(c2))
        else:

            p5 = self.toplayer_ext(torch.cat((c5, c5_ext), dim=1))
            p4 = self._upsample_add(p5, self.latlayer1_ext(torch.cat((c4, c4_ext), dim=1)))
            p3 = self._upsample_add(p4, self.latlayer2_ext(torch.cat((c3, c3_ext), dim=1)))
            p2 = self._upsample_add(p3, self.latlayer3_ext(torch.cat((c2, c2_ext), dim=1)))
        ps0 = [p5, p4, p3, p2]

        # Smooth
        if ps0_ext is None:
            p5 = self.smooth1_1(p5)
            p4 = self.smooth2_1(p4)
            p3 = self.smooth3_1(p3)
            p2 = self.smooth4_1(p2)
        else:
            p5 = self.smooth1_1_ext(torch.cat((p5, ps0_ext[0]), dim=1))
            p4 = self.smooth2_1_ext(torch.cat((p4, ps0_ext[1]), dim=1))
            p3 = self.smooth3_1_ext(torch.cat((p3, ps0_ext[2]), dim=1))
            p2 = self.smooth4_1_ext(torch.cat((p2, ps0_ext[3]), dim=1))
        ps1 = [p5, p4, p3, p2]

        if ps1_ext is None:
            p5 = self.smooth1_2(p5)
            p4 = self.smooth2_2(p4)
            p3 = self.smooth3_2(p3)
            p2 = self.smooth4_2(p2)
        else:
            p5 = self.smooth1_2_ext(torch.cat((p5, ps1_ext[0]), dim=1))
            p4 = self.smooth2_2_ext(torch.cat((p4, ps1_ext[1]), dim=1))
            p3 = self.smooth3_2_ext(torch.cat((p3, ps1_ext[2]), dim=1))
            p2 = self.smooth4_2_ext(torch.cat((p2, ps1_ext[3]), dim=1))
        ps2 = [p5, p4, p3, p2]

        if mode == 1 or mode == 3:
            p6 = F.max_pool2d(ps2[0], (1, 1), (2, 2))
            feat = [ps2[3], ps2[2], ps2[1], ps2[0], p6]
            return feat

        return ps0, ps1, ps2


class fpn_module_local(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_local, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        # Top layer
        self.fold = 2
        self.local_flod = 1
        self.toplayer = nn.Conv2d(2048 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1),
                                  padding=(0, 0))  # Reduce channels
        # Lateral layers [C]
        self.latlayer1 = nn.Conv2d(1024 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer2 = nn.Conv2d(512 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer3 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # Smooth layers
        # ps0
        ###################################kernel_size=(3, 3)->kernel_size=(1, 1)  padding=(1, 1)->padding=(0, 0)#######################
        self.smooth1_1 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth2_1 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth3_1 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.smooth4_1 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # ps1
        self.smooth1_2 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))  # 128
        self.smooth2_2 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))  # 128
        self.smooth3_2 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))  # 128
        self.smooth4_2 = nn.Conv2d(256 * self.fold, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))  # 128
        ################################################################################################################################


        # ps2 is concatenation
        # Classify layers
        self.smooth = nn.Conv2d(128 * 4 * self.fold * 2, 128 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.classify = nn.Conv2d(128 * 4, numClass, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        ##############################local-only#####################################################
        self.toplayer_local = nn.Conv2d(2048 * self.local_flod, 256, kernel_size=(1, 1), stride=(1, 1),
                                        padding=(0, 0))  # Reduce channels
        # Lateral layers [C]
        self.latlayer1_local = nn.Conv2d(1024 * self.local_flod, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer2_local = nn.Conv2d(512 * self.local_flod, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.latlayer3_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # Smooth layers
        # ps0
        self.smooth1_1_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth2_1_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth3_1_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth4_1_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # ps1
        self.smooth1_2_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))  # 128
        self.smooth2_2_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))  # 128
        self.smooth3_2_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))  # 128
        self.smooth4_2_local = nn.Conv2d(256 * self.local_flod, 256, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))  # 128
        #################################################################################################

    def _concatenate(self, p5, p4, p3, p2):
        # print("p2.size().. " , p2.size())
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        # print("p5 size ", p5.size())
        # print("p4 size ", p4.size())
        # print("p3 size ", p3.size())
        # print("p2 size ", p2.size())
        # print(" torch.cat([p5, p4, p3, p2], dim=1) size", torch.cat([p5, p4, p3, p2], dim=1).size())
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    # c2_l, c3_l, c4_l, c5_l,
    # c2_ext,  self._crop_global(self.c2_g, top_lefts, ratio),  return [crop]
    # c3_ext,  self._crop_global(self.c3_g, top_lefts, ratio),  return [crop]
    # c4_ext,  self._crop_global(self.c4_g, top_lefts, ratio),
    # c5_ext,  self._crop_global(self.c5_g, top_lefts, ratio),  return [crop]   c5_ext[0]=crop
    # ps0_ext, [self._crop_global(f, top_lefts, ratio) for f in self.ps0_g], ps0_g=[p5,p4,p3,p2]
    # ps1_ext, [self._crop_global(f, top_lefts, ratio) for f in self.ps1_g],
    # ps2_ext, [self._crop_global(f, top_lefts, ratio) for f in self.ps2_g]

    def forward(self, c2, c3, c4, c5, c2_ext, c3_ext, c4_ext, c5_ext, ps0_ext, ps1_ext, ps2_ext, mode=None):
        # Top-down
        if mode == 4:
            p5 = self.toplayer_local(c5)
            p4 = self._upsample_add(p5, self.latlayer1_local(c4))
            p3 = self._upsample_add(p4, self.latlayer2_local(c3))
            p2 = self._upsample_add(p3, self.latlayer3_local(c2))
            #no smooth
            ps2 = [p5, p4, p3, p2]
            return ps2

        else:
            p5 = self.toplayer(
                torch.cat([c5] + [F.interpolate(c5_ext[0], size=c5.size()[2:], **self._up_kwargs)], dim=1))
            p4 = self._upsample_add(p5, self.latlayer1(
                torch.cat([c4] + [F.interpolate(c4_ext[0], size=c4.size()[2:], **self._up_kwargs)], dim=1)))
            p3 = self._upsample_add(p4, self.latlayer2(
                torch.cat([c3] + [F.interpolate(c3_ext[0], size=c3.size()[2:], **self._up_kwargs)], dim=1)))
            p2 = self._upsample_add(p3, self.latlayer3(
                torch.cat([c2] + [F.interpolate(c2_ext[0], size=c2.size()[2:], **self._up_kwargs)], dim=1)))

        ps0 = [p5, p4, p3, p2]




        # Smooth
        if mode == 4:
            p5 = self.smooth1_1_local(p5)
            p4 = self.smooth2_1_local(p4)
            p3 = self.smooth3_1_local(p3)
            p2 = self.smooth4_1_local(p2)
        else:
            p5 = self.smooth1_1(
                torch.cat([p5] + [F.interpolate(ps0_ext[0][0], size=p5.size()[2:], **self._up_kwargs)], dim=1))
            p4 = self.smooth2_1(
                torch.cat([p4] + [F.interpolate(ps0_ext[1][0], size=p4.size()[2:], **self._up_kwargs)], dim=1))
            p3 = self.smooth3_1(
                torch.cat([p3] + [F.interpolate(ps0_ext[2][0], size=p3.size()[2:], **self._up_kwargs)], dim=1))
            p2 = self.smooth4_1(
                torch.cat([p2] + [F.interpolate(ps0_ext[3][0], size=p2.size()[2:], **self._up_kwargs)], dim=1))
        ps1 = [p5, p4, p3, p2]
        if mode == 4:
            p5 = self.smooth1_2_local(p5)
            p4 = self.smooth2_2_local(p4)
            p3 = self.smooth3_2_local(p3)
            p2 = self.smooth4_2_local(p2)
        else:
            p5 = self.smooth1_2(
                torch.cat([p5] + [F.interpolate(ps1_ext[0][0], size=p5.size()[2:], **self._up_kwargs)], dim=1))
            p4 = self.smooth2_2(
                torch.cat([p4] + [F.interpolate(ps1_ext[1][0], size=p4.size()[2:], **self._up_kwargs)], dim=1))
            p3 = self.smooth3_2(
                torch.cat([p3] + [F.interpolate(ps1_ext[2][0], size=p3.size()[2:], **self._up_kwargs)], dim=1))
            p2 = self.smooth4_2(
                torch.cat([p2] + [F.interpolate(ps1_ext[3][0], size=p2.size()[2:], **self._up_kwargs)], dim=1))
        ps2 = [p5, p4, p3, p2]

        if mode == 4:
            return ps2
        # Classify
        # use ps2_ext
        p5 = self.smooth1_2(
            torch.cat([p5] + [F.interpolate(ps2_ext[0][0], size=p5.size()[2:], **self._up_kwargs)], dim=1))
        p4 = self.smooth2_2(
            torch.cat([p4] + [F.interpolate(ps2_ext[1][0], size=p4.size()[2:], **self._up_kwargs)], dim=1))
        p3 = self.smooth3_2(
            torch.cat([p3] + [F.interpolate(ps2_ext[2][0], size=p3.size()[2:], **self._up_kwargs)], dim=1))
        p2 = self.smooth4_2(
            torch.cat([p2] + [F.interpolate(ps2_ext[3][0], size=p2.size()[2:], **self._up_kwargs)], dim=1))

        if mode == 2:
            # print("crop global to local ")
            ps3 = [p5, p4, p3, p2]
            return ps3
        elif mode == 3:
            if c2_ext is None:
                self.fold = 1
                ps3 = self._concatenate(p5, p4, p3, p2)
            else:
                ps3 = self._concatenate(
                    torch.cat([p5] + [F.interpolate(ps2_ext[0][0], size=p5.size()[2:], **self._up_kwargs)], dim=1),
                    torch.cat([p4] + [F.interpolate(ps2_ext[1][0], size=p4.size()[2:], **self._up_kwargs)], dim=1),
                    torch.cat([p3] + [F.interpolate(ps2_ext[2][0], size=p3.size()[2:], **self._up_kwargs)], dim=1),
                    torch.cat([p2] + [F.interpolate(ps2_ext[3][0], size=p2.size()[2:], **self._up_kwargs)], dim=1)
                )

            ps3 = self.smooth(ps3)
            # output = self.classify(ps3)
            return ps0, ps1, ps2, ps3  # output


@NECKS.register_module
class GlNetNeck_3_1(nn.Module):#global_fpn smooth1_1_ext kernel_size(1,1)  local_fpn  smooth1_1.....smooth4_2...kernel_size(1,1)
    def __init__(self, numClass, mode1_work_dir=None):
        super(GlNetNeck_3_1, self).__init__()
        self.mode1_work_dir = mode1_work_dir
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        backbone_config = {'type': 'ResNet', 'depth': 50, 'num_stages': 4,
                           'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'style': 'pytorch'}
        fixed_config = {'type': 'ResNet', 'depth': 50, 'num_stages': 4,
                        'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'style': 'pytorch'}
        self.resnet_global = builder.build_backbone(fixed_config)
        self.resnet_local = builder.build_backbone(backbone_config)

        # self.resnet_local.init_weights()

        # fpn module
        self.fpn_global = fpn_module_global()
        self.fpn_local = fpn_module_local(numClass)

        self.c2_g = None
        self.c3_g = None
        self.c4_g = None
        self.c5_g = None
        self.output_g = None
        self.ps0_g = None
        self.ps1_g = None
        self.ps2_g = None
        self.ps3_g = None

        self.c2_l = []
        self.c3_l = []
        self.c4_l = []
        self.c5_l = []
        self.ps00_l = []
        self.ps01_l = []
        self.ps02_l = []
        self.ps03_l = []
        self.ps10_l = []
        self.ps11_l = []
        self.ps12_l = []
        self.ps13_l = []
        self.ps20_l = []
        self.ps21_l = []
        self.ps22_l = []
        self.ps23_l = []
        self.ps0_l = None
        self.ps1_l = None
        self.ps2_l = None
        self.ps3_l = []  # ; self.output_l = []

        self.c2_b = None
        self.c3_b = None
        self.c4_b = None
        self.c5_b = None
        self.ps00_b = None
        self.ps01_b = None
        self.ps02_b = None
        self.ps03_b = None
        self.ps10_b = None
        self.ps11_b = None
        self.ps12_b = None
        self.ps13_b = None
        self.ps20_b = None
        self.ps21_b = None
        self.ps22_b = None
        self.ps23_b = None
        self.ps3_b = []  # ; self.output_b = []

        self.patch_n = 0

        self.i_p = 0

    def init_weights(self, mode):
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #constant_init()
                xavier_init(m, distribution='uniform')
        """
        if mode == 1 or mode == 3:
            self.resnet_global.init_weights('torchvision://resnet50')  # 'torchvision://resnet50'
        elif mode == 2 or mode == 4:
            self.resnet_local.init_weights('torchvision://resnet50')  # 'torchvision://resnet50'

        ''' 
        state = self.state_dict()
        trained_partial = torch.load("../work_dir/jack/mode1_anchor=8_ga/epoch_50.pth")
        trained_partial_item = trained_partial.get("state_dict")  # 'neck.resnet_global.conv1.weight'

        from collections import OrderedDict
        new_state_dict = OrderedDict()  # 新建一个model
        for k, v in trained_partial_item.items():
            if 'neck' in k:
                name = k[5:]

                new_state_dict[name] = v
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in state and "global" in k}

        state.update(pretrained_dict)
        self.load_state_dict(state)
        '''

        # self.resnet_local.init_weights('torchvision://resnet50')
        # init fpn

        for m in self.fpn_global.children():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)

        for m in self.fpn_local.children():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)

    def clear_cache(self):
        self.c2_g = None
        self.c3_g = None
        self.c4_g = None
        self.c5_g = None
        self.output_g = None
        self.ps0_g = None
        self.ps1_g = None
        self.ps2_g = None
        self.ps3_g = None

        self.c2_l = []
        self.c3_l = []
        self.c4_l = []
        self.c5_l = []
        self.ps00_l = []
        self.ps01_l = []
        self.ps02_l = []
        self.ps03_l = []
        self.ps10_l = []
        self.ps11_l = []
        self.ps12_l = []
        self.ps13_l = []
        self.ps20_l = []
        self.ps21_l = []
        self.ps22_l = []
        self.ps23_l = []
        self.ps0_l = None
        self.ps1_l = None
        self.ps2_l = None
        self.ps3_l = []
        self.output_l = []

        self.c2_b = None
        self.c3_b = None
        self.c4_b = None
        self.c5_b = None
        self.ps00_b = None
        self.ps01_b = None
        self.ps02_b = None
        self.ps03_b = None
        self.ps10_b = None
        self.ps11_b = None
        self.ps12_b = None
        self.ps13_b = None
        self.ps20_b = None
        self.ps21_b = None
        self.ps22_b = None
        self.ps23_b = None
        self.ps3_b = []
        self.output_b = []

        self.patch_n = 0

    def _sample_grid(self, fm, bbox, sampleSize):
        """
        :param fm: tensor(b,c,h,w) the global feature map
        :param bbox: list [b* nparray(x1, y1, x2, y2)] the (x1,y1) is the left_top of bbox, (x2, y2) is the right_bottom of bbox
        there are in range [0, 1]. x is corresponding to width dimension and y is corresponding to height dimension
        :param sampleSize: (oH, oW) the point to sample in height dimension and width dimension
        :return: tensor(b, c, oH, oW) sampled tensor
        """
        b, c, h, w = fm.shape
        b_bbox = len(bbox)
        bbox = [x * 2 - 1 for x in bbox]  # range transform
        if b != b_bbox and b == 1:
            fm = torch.cat([fm, ] * b_bbox, dim=0)
        grid = np.zeros((b_bbox,) + sampleSize + (2,), dtype=np.float32)
        gridMap = np.array(
            [[(cnt_w / (sampleSize[1] - 1), cnt_h / (sampleSize[0] - 1)) for cnt_w in range(sampleSize[1])] for cnt_h in
             range(sampleSize[0])])
        for cnt_b in range(b_bbox):
            grid[cnt_b, :, :, 0] = bbox[cnt_b][0] + (bbox[cnt_b][2] - bbox[cnt_b][0]) * gridMap[:, :, 0]
            grid[cnt_b, :, :, 1] = bbox[cnt_b][1] + (bbox[cnt_b][3] - bbox[cnt_b][1]) * gridMap[:, :, 1]
        grid = torch.from_numpy(grid).cuda()
        return F.grid_sample(fm, grid)

    def _crop_global(self, f_global, top_lefts, ratio):
        """
        top_lefts: [(top, left)] * b
        """
        _, c, H, W = f_global.size()
        b = len(top_lefts)
        h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))

        # bbox = [ np.array([left, top, left + ratio, top + ratio]) for (top, left) in top_lefts ]
        # crop = self._sample_grid(f_global, bbox, (H, W))

        crop = []
        for i in range(b):
            top, left = int(np.round(top_lefts[i][0] * H)), int(np.round(top_lefts[i][1] * W))
            # # global's sub-region & upsample
            # f_global_patch = F.interpolate(f_global[0:1, :, top:top+h, left:left+w], size=(h, w), mode='bilinear')
            f_global_patch = f_global[0:1, :, top:top + h, left:left + w]
            crop.append(f_global_patch[0])
        crop = torch.stack(crop, dim=0)  # stack into mini-batch
        return [crop]  # return as a list for easy to torch.cat

    def _merge_local(self, f_local, merge, f_global, top_lefts, oped, ratio, template):
        """
        merge feature maps from local patches, and finally to a whole image's feature map (on cuda)
        f_local: a sub_batch_size of patch's feature map
        oped: [start, end)
        """
        b, _, _, _ = f_local.size()
        _, c, H, W = f_global.size()  # match global feature size
        if merge is None:
            merge = torch.zeros((1, c, H, W)).cuda()
        h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))
        for i in range(b):
            index = oped[0] + i
            top, left = int(np.round(H * top_lefts[index][0])), int(np.round(W * top_lefts[index][1]))
            merge[:, :, top:top + h, left:left + w] += F.interpolate(f_local[i:i + 1], size=(h, w), **self._up_kwargs)
        if oped[1] >= len(top_lefts):
            template = F.interpolate(template, size=(H, W), **self._up_kwargs)
            template = template.expand_as(merge)
            # template = Variable(template).cuda()
            merge /= template
        return merge

    def ensemble(self, f_local, f_global):
        return self.ensemble_conv(torch.cat((f_local, f_global), dim=1))

    def collect_local_fm(self, image_global, patches, ratio, top_lefts, oped, batch_size, global_model=None,
                         template=None, n_patch_all=None):
        """
        patches: 1 patch
        top_lefts: all top-left
        oped: [start, end)
        """
        with torch.no_grad():
            patches = patches.unsqueeze(0)
            if self.patch_n == 0:
                self.c2_g, self.c3_g, self.c4_g, self.c5_g = global_model.module.resnet_global.forward(image_global)

                self.ps0_g, self.ps1_g, self.ps2_g = global_model.module.fpn_global.forward(
                    self.c2_g, self.c3_g, self.c4_g, self.c5_g, mode=2)

            self.resnet_local.eval()
            self.fpn_local.eval()
            c2, c3, c4, c5 = self.resnet_local.forward(patches)
            ps0, ps1, ps2, ps3 = self.fpn_local.forward(
                c2, c3, c4, c5,
                self._crop_global(self.c2_g, top_lefts[oped[0]:oped[1]], ratio),
                c3_ext=self._crop_global(self.c3_g, top_lefts[oped[0]:oped[1]], ratio),
                c4_ext=self._crop_global(self.c4_g, top_lefts[oped[0]:oped[1]], ratio),
                c5_ext=self._crop_global(self.c5_g, top_lefts[oped[0]:oped[1]], ratio),
                ps0_ext=[self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps0_g],
                ps1_ext=[self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps1_g],
                ps2_ext=[self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps2_g],
                mode=3)

            self.patch_n += 1  # patches.size()[0]
            self.patch_n %= n_patch_all

            # output = F.interpolate(output, patches.size()[2:], mode='nearest')

            self.c2_b = self._merge_local(c2, self.c2_b, self.c2_g, top_lefts, oped, ratio, template)
            self.c3_b = self._merge_local(c3, self.c3_b, self.c3_g, top_lefts, oped, ratio, template)
            self.c4_b = self._merge_local(c4, self.c4_b, self.c4_g, top_lefts, oped, ratio, template)
            self.c5_b = self._merge_local(c5, self.c5_b, self.c5_g, top_lefts, oped, ratio, template)

            self.ps00_b = self._merge_local(ps0[0], self.ps00_b, self.ps0_g[0], top_lefts, oped, ratio, template)
            self.ps01_b = self._merge_local(ps0[1], self.ps01_b, self.ps0_g[1], top_lefts, oped, ratio, template)
            self.ps02_b = self._merge_local(ps0[2], self.ps02_b, self.ps0_g[2], top_lefts, oped, ratio, template)
            self.ps03_b = self._merge_local(ps0[3], self.ps03_b, self.ps0_g[3], top_lefts, oped, ratio, template)
            self.ps10_b = self._merge_local(ps1[0], self.ps10_b, self.ps1_g[0], top_lefts, oped, ratio, template)
            self.ps11_b = self._merge_local(ps1[1], self.ps11_b, self.ps1_g[1], top_lefts, oped, ratio, template)
            self.ps12_b = self._merge_local(ps1[2], self.ps12_b, self.ps1_g[2], top_lefts, oped, ratio, template)
            self.ps13_b = self._merge_local(ps1[3], self.ps13_b, self.ps1_g[3], top_lefts, oped, ratio, template)
            self.ps20_b = self._merge_local(ps2[0], self.ps20_b, self.ps2_g[0], top_lefts, oped, ratio, template)
            self.ps21_b = self._merge_local(ps2[1], self.ps21_b, self.ps2_g[1], top_lefts, oped, ratio, template)
            self.ps22_b = self._merge_local(ps2[2], self.ps22_b, self.ps2_g[2], top_lefts, oped, ratio, template)
            self.ps23_b = self._merge_local(ps2[3], self.ps23_b, self.ps2_g[3], top_lefts, oped, ratio, template)

            self.i_p += 1

            self.ps3_b.append(ps3.cpu())
            # self.output_b.append(output.cpu()) # each output is 1, 7, h, w

            if self.patch_n == 0:
                # merged all patches into an image
                self.c2_l.append(self.c2_b)
                self.c3_l.append(self.c3_b)
                self.c4_l.append(self.c4_b)
                self.c5_l.append(self.c5_b)
                self.ps00_l.append(self.ps00_b)
                self.ps01_l.append(self.ps01_b)
                self.ps02_l.append(self.ps02_b)
                self.ps03_l.append(self.ps03_b)
                self.ps10_l.append(self.ps10_b)
                self.ps11_l.append(self.ps11_b)
                self.ps12_l.append(self.ps12_b)
                self.ps13_l.append(self.ps13_b)
                self.ps20_l.append(self.ps20_b)
                self.ps21_l.append(self.ps21_b)
                self.ps22_l.append(self.ps22_b)
                self.ps23_l.append(self.ps23_b)

                # collected all ps3 and output of patches as a (b) tensor, append into list
                self.ps3_l.append(torch.cat(self.ps3_b, dim=0))  # a list of tensors
                # self.output_l.append(torch.cat(self.output_b, dim=0)) # a list of 36, 7, h, w tensors

                self.c2_b = None
                self.c3_b = None
                self.c4_b = None
                self.c5_b = None
                self.ps00_b = None
                self.ps01_b = None
                self.ps02_b = None
                self.ps03_b = None
                self.ps10_b = None
                self.ps11_b = None
                self.ps12_b = None
                self.ps13_b = None
                self.ps20_b = None
                self.ps21_b = None
                self.ps22_b = None
                self.ps23_b = None
                self.ps3_b = []  # ; self.output_b = []
            if len(self.c2_l) == batch_size:
                self.c2_l = torch.cat(self.c2_l, dim=0)  # .cuda()
                self.c3_l = torch.cat(self.c3_l, dim=0)  # .cuda()
                self.c4_l = torch.cat(self.c4_l, dim=0)  # .cuda()
                self.c5_l = torch.cat(self.c5_l, dim=0)  # .cuda()
                self.ps00_l = torch.cat(self.ps00_l, dim=0)  # .cuda()
                self.ps01_l = torch.cat(self.ps01_l, dim=0)  # .cuda()
                self.ps02_l = torch.cat(self.ps02_l, dim=0)  # .cuda()
                self.ps03_l = torch.cat(self.ps03_l, dim=0)  # .cuda()
                self.ps10_l = torch.cat(self.ps10_l, dim=0)  # .cuda()
                self.ps11_l = torch.cat(self.ps11_l, dim=0)  # .cuda()
                self.ps12_l = torch.cat(self.ps12_l, dim=0)  # .cuda()
                self.ps13_l = torch.cat(self.ps13_l, dim=0)  # .cuda()
                self.ps20_l = torch.cat(self.ps20_l, dim=0)  # .cuda()
                self.ps21_l = torch.cat(self.ps21_l, dim=0)  # .cuda()
                self.ps22_l = torch.cat(self.ps22_l, dim=0)  # .cuda()
                self.ps23_l = torch.cat(self.ps23_l, dim=0)  # .cuda()
                self.ps0_l = [self.ps00_l, self.ps01_l, self.ps02_l, self.ps03_l]
                self.ps1_l = [self.ps10_l, self.ps11_l, self.ps12_l, self.ps13_l]
                self.ps2_l = [self.ps20_l, self.ps21_l, self.ps22_l, self.ps23_l]

    def forward(self, image_global, patches, top_lefts, ratio, templates=None, mode=None, global_model=None,
                n_patch=None, i_patch=None):
        if mode == 1:
            # train global model
            c2_g, c3_g, c4_g, c5_g = self.resnet_global.forward(image_global)
            feat = self.fpn_global.forward(c2_g, c3_g, c4_g, c5_g, mode=1)
            return feat
        elif mode == 2 or mode == 4:
            with torch.no_grad():
                if self.patch_n == 0:
                    self.c2_g, self.c3_g, self.c4_g, self.c5_g = self.resnet_global.forward(image_global)
                    # # output, ps0, ps1, ps2, ps3  #self.output_g,
                    self.ps0_g, self.ps1_g, self.ps2_g = self.fpn_global.forward(self.c2_g, self.c3_g, self.c4_g,
                                                                                 self.c5_g)
                self.patch_n += patches.size()[0]
                self.patch_n %= n_patch
            c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(patches)
            # ps3 = [p5, p4, p3, p2]
            ps3_l = self.fpn_local.forward(c2_l, c3_l, c4_l, c5_l,
                                           self._crop_global(self.c2_g, top_lefts,
                                                             ratio),
                                           self._crop_global(self.c3_g, top_lefts,
                                                             ratio),
                                           self._crop_global(self.c4_g, top_lefts,
                                                             ratio),
                                           self._crop_global(self.c5_g, top_lefts,
                                                             ratio),
                                           [self._crop_global(f, top_lefts, ratio) for f
                                            in self.ps0_g],
                                           [self._crop_global(f, top_lefts, ratio) for f
                                            in self.ps1_g],
                                           [self._crop_global(f, top_lefts, ratio) for f
                                            in self.ps2_g],
                                           mode=mode)
            p6 = F.max_pool2d(ps3_l[0], (1, 1), (2, 2))
            result = [ps3_l[3], ps3_l[2], ps3_l[1], ps3_l[0], p6]
            return result

        elif mode == 3:
            assert 'mode1' or 'global' in self.mode1_work_dir, 'please check config file->model->neck->mode1_work_dir'
            weight_path = os.path.join(self.mode1_work_dir, 'epoch_20.pth')
            assert os.path.isfile(weight_path), 'please run mode1 first'

            global_fixed = GlNetNeck_3_1(2)
            global_fixed = nn.DataParallel(global_fixed)
            global_fixed = global_fixed.cuda()
            state = global_fixed.state_dict()  # 'module.resnet_global.conv1.weight'

            trained_partial = torch.load(weight_path)
            trained_partial_item = trained_partial.get("state_dict")  # 'neck.resnet_global.conv1.weight'
            from collections import OrderedDict
            new_state_dict = OrderedDict()  # 新建一个model
            for k, v in trained_partial_item.items():
                if 'neck' in k:
                    name = "module" + k[4:]

                    new_state_dict[name] = v
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in state and "global" in k}
            state.update(pretrained_dict)
            global_fixed.load_state_dict(state)
            global_fixed.eval()
            i_patch = 0
            while i_patch < len(top_lefts[0]):
                self.collect_local_fm(image_global,
                                      patches[0][i_patch], ratio[0], top_lefts[0],
                                      [i_patch, i_patch + 1], 1,
                                      global_model=global_fixed,
                                      template=templates[0],
                                      n_patch_all=len(top_lefts[0]))

                i_patch += 1

            c2_g, c3_g, c4_g, c5_g = self.resnet_global.forward(image_global)

            # draw_feat_map(self, i_patch)

            # output_g, ps0_g, ps1_g, ps2_g, ps3_g
            feat = self.fpn_global.forward(c2_g, c3_g, c4_g, c5_g,
                                           c2_ext=self.c2_l, c3_ext=self.c3_l,
                                           c4_ext=self.c4_l, c5_ext=self.c5_l,
                                           ps0_ext=self.ps0_l, ps1_ext=self.ps1_l,
                                           ps2_ext=self.ps2_l, mode=3)

            self.clear_cache()
            return feat

        else:
            raise ValueError('wrong mode:{}'.format(mode))


def draw_feat_map(self, i_patch):
    from PIL import Image
    from torchvision import transforms
    feature = self.ps2_l[0][0][0].cpu()
    feature = 1.0 / (1 + np.exp(-1 * feature))
    feature = np.round(feature * 255)
    img = transforms.ToPILImage()(feature).convert('RGB')
    img = img.resize((800, 800), Image.ANTIALIAS)
    img.save('./feature/' + str(i_patch) + 'c2.jpg')
