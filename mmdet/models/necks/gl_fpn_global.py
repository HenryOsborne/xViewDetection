import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import NECKS
from mmcv.cnn import xavier_init


@NECKS.register_module()
class GLFpnGlobal(nn.Module):
    def __init__(self, ):
        super(GLFpnGlobal, self).__init__()
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
        # Smooth layers
        self.smooth1_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth2_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth3_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth4_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth1_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth2_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth3_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.smooth4_2_ext = nn.Conv2d(256 * 2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.smooth = nn.Conv2d(128 * 4 * 2, 128 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

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

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
