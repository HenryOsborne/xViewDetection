import torch
from ptflops import get_model_complexity_info
from torchvision.models.resnet import resnet50
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        y = self.conv1(x)
        return y


model = Conv()
model.eval()
size = 200

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (512, size, size), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
