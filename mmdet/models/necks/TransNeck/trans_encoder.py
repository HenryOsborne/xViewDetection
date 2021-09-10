import torch
import torch.nn as nn

from mmdet.models.backbones.swin_transformer import BasicLayer


class TransEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 dim,
                 depth,
                 num_heads):
        super(TransEncoder, self).__init__()
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


if __name__ == '__main__':
    x = torch.randn(1, 768, 25, 25)
    encoder = TransEncoder(768, 768, 2, 3)
    y = encoder(x)
    print(y)
    pass
