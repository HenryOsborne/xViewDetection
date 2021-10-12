import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConcatFeatureMap(nn.Module):
    def __init__(self):
        super(ConcatFeatureMap, self).__init__()

    def forward(self, fpn_output):
        # Calculating median height to upsample or desample each fpn levels
        heights = []
        level_tensors = fpn_output
        for i in fpn_output:
            heights.append(i.shape[2])
        median_height = int(np.median(heights))

        # Upsample and Desampling tensors to median height and width
        for i in range(len(level_tensors)):
            level = level_tensors[i]
            # If level height is greater than median, then downsample with interpolate
            if level.shape[2] > median_height:
                level = F.interpolate(level, size=(median_height, median_height), mode='nearest')
            # If level height is less than median, then upsample
            else:
                level = F.interpolate(level, size=(median_height, median_height), mode='nearest')
            level_tensors[i] = level

        # Concating all levels with dimensions (batch_size, levels, C, H, W)
        concat_levels = torch.stack(level_tensors, dim=1)

        # Reshaping tensor from (batch_size, levels, C, H, W) to (batch_size, levels, HxW=S, C)
        concat_levels = concat_levels.flatten(start_dim=3).transpose(dim0=2, dim1=3)
        return concat_levels, median_height


if __name__ == '__main__':
    concat = ConcatFeatureMap()
    z = [torch.randn(1, 256, 200, 200),
         torch.randn(1, 256, 100, 100),
         torch.randn(1, 256, 50, 50),
         torch.randn(1, 256, 25, 25)]
    y, _ = concat(z)

    pass
