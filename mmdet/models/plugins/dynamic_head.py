import torch.nn as nn
from collections import OrderedDict
from mmdet.models.plugins import ConcatFeatureMap, ScaleAwareLayer, SpatialAwareLayer, TaskAwareLayer
import torch
import torch.nn.functional as F


class DyHead_Block(nn.Module):
    def __init__(self, L, S, C):
        super(DyHead_Block, self).__init__()
        # Saving all dimension sizes of F
        self.L_size = L
        self.S_size = S
        self.C_size = C

        # Inititalizing all attention layers
        self.scale_attention = ScaleAwareLayer(s_size=self.S_size)
        self.spatial_attention = SpatialAwareLayer(L_size=self.L_size)
        self.task_attention = TaskAwareLayer(num_channels=self.C_size)

    def forward(self, F_tensor):
        scale_output = self.scale_attention(F_tensor)
        spacial_output = self.spatial_attention(scale_output)
        task_output = self.task_attention(spacial_output)

        return task_output


# def DyHead(num_blocks, L, S, C):
#     blocks = [('Block_{}'.format(i + 1), DyHead_Block(L, S, C)) for i in range(num_blocks)]
#
#     return nn.Sequential(OrderedDict(blocks))


class DynamicHead(nn.Module):
    def __init__(self, num_blocks, L, S, C):
        super(DynamicHead, self).__init__()
        blocks = [('Block_{}'.format(i + 1), DyHead_Block(L, S, C)) for i in range(num_blocks)]
        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.concat_layer = ConcatFeatureMap()

    def forward(self, fpn_output):
        if len(fpn_output) > 4:
            fpn_output = fpn_output[:4]
        concat_levels, median_height = self.concat_layer(fpn_output)
        dynamic_output = self.blocks(concat_levels)
        B, L, _, C = dynamic_output.size()
        output = dynamic_output.transpose(2, 3).reshape(B, L, C, median_height, median_height)

        output = output.split(split_size=1, dim=1)
        output = [o.squeeze(1).contiguous() for o in output]
        output.append(F.max_pool2d(output[-1], 1, stride=2))

        return output


if __name__ == '__main__':
    z = [torch.randn(1, 256, 200, 200),
         torch.randn(1, 256, 100, 100),
         torch.randn(1, 256, 50, 50),
         torch.randn(1, 256, 25, 25)]
    head = DynamicHead(6, 4, 5625, 256)
    y = head(z)
    pass
