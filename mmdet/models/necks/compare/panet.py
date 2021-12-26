from torch import nn
from typing import Any, Tuple, Sequence, Optional, Iterable
import torch.nn.functional as F
from functools import partial
from mmdet.models.builder import NECKS
from mmcv.cnn import ConvModule, xavier_init


class ModulizedFunction(nn.Module):
    """Convert a function to an nn.Module."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = partial(fn, *args, **kwargs)

    def forward(self, x):
        return self.fn(x)


class Interpolate(ModulizedFunction):
    def __init__(self, mode='bilinear', align_corners=False, **kwargs):
        super().__init__(
            F.interpolate, mode='bilinear', align_corners=False, **kwargs)


class Sum(nn.Module):
    def forward(self, inps):
        return sum(inps)


class Reverse(nn.Module):
    def forward(self, inps):
        return inps[::-1]


class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''

    def forward(self, xs: Any) -> tuple:
        # if multiple inputs, pass the 1st input through the 1st module,
        # the 2nd input through the 2nd module, and so on.
        if isinstance(xs, (list, tuple)):
            return tuple(m(x) for m, x in zip(self, xs))
        # if single input, pass it through all modules
        return tuple(m(xs) for m in self)


class SequentialMultiInputMultiOutput(nn.Sequential):
    """
    Takes in either
    (1) an (n+1)-tuple of the form
      (last_out, 1st input, 2nd input, ..., nth input), or
    (2) an n-tuple of the form
      (1st input, 2nd input, ..., nth input),
    where n is the length of this sequential.

    If (2), the first layer in this sequential should be able to accept
    a single input. All others are expected to accept a 2-tuple of inputs.

    Returns an n-tuple of all outputs of the form:
    (1st out, 2nd out, ..., nth out).

    In other words: the ith layer in this sequential takes in as inputs the
    ith input and the output of the last layer i.e. the (i-1)th layer.
    For the 1st layer, the "output of the last layer" is last_out.

                       last_out
                      (optional)
                          │
                          │
                          V
    1st input ───────[1st layer]───────> 1st out
                          │
                          │
                          V
    2nd input ───────[2nd layer]───────> 2nd out
                          │
                          │
                          V
        .                 .                  .
        .                 .                  .
        .                 .                  .
                          │
                          │
                          V
    nth input ───────[nth layer]───────> nth out

    """

    def forward(self, xs: tuple) -> tuple:
        outs = [None] * len(self)

        if len(xs) == len(self) + 1:
            last_out = xs[0]
            layer_inputs = xs[1:]
            layers = self
            start_idx = 0
        elif len(xs) == len(self):
            last_out = self[0](xs[0])
            layer_inputs = xs[1:]
            layers = self[1:]
            outs[0] = last_out
            start_idx = 1
        else:
            raise ValueError('Invalid input format.')

        for i, (layer, x) in enumerate(zip(layers, layer_inputs), start_idx):
            last_out = layer((x, last_out))
            outs[i] = last_out

        return tuple(outs)


class FPN(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Feature Pyramid Networks for Object Detection" by Lin et al.,
    https://arxiv.com/abs/1612.03144.

    Takes in an n-tuple of feature maps in reverse order
    (1st feature map, 2nd feature map, ..., nth feature map), where
    the 1st feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    n outputs, such that the height and width of the ith output is equal to
    that of the corresponding input feature map and the number of channels
    is equal to out_channels.

    Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)

    Architecture diagram:

    nth feat. map ────────[nth in_conv]──────────┐────────[nth out_conv]────> nth out
                                                 │
                                             [upsample]
                                                 │
                                                 V
    (n-1)th feat. map ────[(n-1)th in_conv]────>(+)────[(n-1)th out_conv]────> (n-1)th out
                                                 │
                                             [upsample]
                                                 │
                                                 V
            .                     .                           .                    .
            .                     .                           .                    .
            .                     .                           .                    .
                                                 │
                                             [upsample]
                                                 │
                                                 V
    1st feat. map ────────[1st in_conv]────────>(+)────────[1st out_conv]────> 1st out

    """

    def __init__(self,
                 in_feats_shapes: Sequence[Tuple[int, ...]],
                 hidden_channels: int = 256,
                 out_channels: int = 2):
        """Constructor.

        Args:
            in_feats_shapes (Sequence[Tuple[int, ...]]): Shapes of the feature
                maps that will be fed into the network. These are expected to
                be tuples of the form (., C, H, ...).
            hidden_channels (int, optional): The number of channels to which
                all feature maps are convereted before being added together.
                Defaults to 256.
            out_channels (int, optional): Number of output channels. This will
                normally be the number of classes. Defaults to 2.
        """
        # reverse so that the deepest (i.e. produced by the deepest layer in
        # the backbone network) feature map is first.
        in_feats_shapes = in_feats_shapes[::-1]
        in_feats_channels = [s[1] for s in in_feats_shapes]

        # 1x1 conv to make the channels of all feature maps the same
        in_convs = Parallel([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            for in_channels in in_feats_channels
        ])

        # yapf: disable
        def resize_and_add(to_size):
            return nn.Sequential(
                Parallel([nn.Identity(), Interpolate(size=to_size)]),
                Sum()
            )

        top_down_layer = SequentialMultiInputMultiOutput(
            nn.Identity(),
            *[resize_and_add(shape[-2:]) for shape in in_feats_shapes[1:]]
        )

        out_convs = Parallel([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_feats_shapes
        ])
        layers = [
            Reverse(),
            in_convs,
            top_down_layer,
            out_convs,
            Reverse()
        ]
        # yapf: enable
        super().__init__(*layers)


class PANetFPN(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Path Aggregation Network for Instance Segmentation" by Liu et al.,
    https://arxiv.com/abs/1803.01534. This architecture adds a bottom-up path
    after the top-down path in a normal FPN. It can be thought of as a normal
    FPN followed by a flipped FPN.

    Takes in an n-tuple of feature maps in reverse order
    (1st feature map, 2nd feature map, ..., nth feature map), where
    the 1st feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    n outputs, such that the height and width of the ith output is equal to
    that of the corresponding input feature map and the number of channels
    is equal to out_channels.

    Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)

    Architecture diagram:

            (1st feature map, 2nd feature map, ..., nth feature map)
                                    │
                                [1st FPN]
                                    │
                                    V
                                    │
                        [Reverse the order of outputs]
                                    │
                                    V
                                    │
                                [2nd FPN]
                                    │
                                    V
                                    │
                        [Reverse the order of outputs]
                                    │
                                    │
                                    V
                       (1st out, 2nd out, ..., nth out)

    """

    def __init__(self, fpn1: nn.Module, fpn2: nn.Module):
        # yapf: disable
        layers = [
            fpn1,
            Reverse(),
            fpn2,
            Reverse(),
        ]
        # yapf: enable
        super().__init__(*layers)


@NECKS.register_module()
class PANet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(PANet, self).__init__()
        fpn1 = FPN(
            in_channels,
            hidden_channels=out_channels,
            out_channels=out_channels)

        feat_shapes = [(n, out_channels, h, w) for (n, c, h, w) in in_channels]

        fpn2 = FPN(
            feat_shapes[::-1],
            hidden_channels=out_channels,
            out_channels=out_channels)

        self.fpn = nn.Sequential(PANetFPN(fpn1, fpn2))

    def forward(self, x):
        outs = list(self.fpn(x))
        outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


if __name__ == '__main__':
    feat_shapes = [(1, 96, 200, 200), (1, 192, 100, 100), (1, 384, 50, 50), (1, 768, 25, 25)]
    fpn_channels = 256

    m = PANet(feat_shapes, fpn_channels)

    import torch

    x = [torch.randn(1, 96, 200, 200),
         torch.randn(1, 192, 100, 100),
         torch.randn(1, 384, 50, 50),
         torch.randn(1, 768, 25, 25)]
    y = m(x)
    print(y)
    pass
