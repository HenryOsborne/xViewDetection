import torch
from ptflops import get_model_complexity_info
from torchvision.models.resnet import resnet50
import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--size', type=int, default=200)

    args = parser.parse_args()
    return args


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        y = self.conv1(x)
        return y


if __name__ == '__main__':
    model = Conv()
    model.eval()
    args = parse_args()
    print(args.size)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (512, args.size, args.size), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
