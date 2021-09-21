import torch
import torchvision.ops
from torch import nn
from torch.nn.modules.utils import _pair


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=(kernel_size, kernel_size),
                                     stride=(stride, stride),
                                     padding=(self.padding, self.padding),
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=(kernel_size, kernel_size),
                                        stride=(stride, stride),
                                        padding=(self.padding, self.padding),
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(kernel_size, kernel_size),
                                      stride=(stride, stride),
                                      padding=(self.padding, self.padding),
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=(self.padding, self.padding),
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class DCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 extra_offset_mask=False):
        super(DCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.extra_offset_mask = extra_offset_mask
        channels_ = 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(input[1])
            input = input[0]
        else:
            out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # each has self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] channels
        # o1, o2 = o1.data.cpu().numpy(), o2.data.cpu().numpy()
        # o = o1[0]   # first image in the batch
        # print(o1[0])
        # return
        # k = 0
        # img_h, img_w = 128, 256
        # img_r, img_c, inter = 3, 3, 5
        # img_size = ((img_w + inter) * img_c, img_r * (img_h + inter))
        # to_image = Image.new('RGB', img_size, 'white')
        # print()
        # for j in range(o.shape[0]):     # for group
        #     if j % 9 == 0 and j != 0:  # different kernel
        #         dst_file = os.path.join(main_path, 'offset_{}to{}.png'.format(j-9, j))
        #         # save first
        #         to_image.save(dst_file)
        #         # new image
        #         img_size = ((img_w + inter) * img_c, img_r * (img_h + inter))
        #         to_image = Image.new('RGB', img_size, 'white')
        #     feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        #     for x, y in range
        #     feature_img = Image.fromarray(cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB))
        #     index_r, index_c = j // img_c, j % img_c
        #     to_image.paste(feature_img, (index_c * (img_w + inter), index_r * (img_h + inter)))
        offset = torch.cat((o1, o2), dim=1)  # x, y [0-8]: the first group,
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=input,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride,
                                          )

        return x
