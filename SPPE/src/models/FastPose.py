import torch.nn as nn
from torch.autograd import Variable
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
from layers.SE_Resnet import SEResnet
from layers.DUC import DUC
# from opt import opt


def createModel():
    return FastPose()


class FastPose(nn.Module):
    DIM = 128

    def __init__(self, nClasses=33):
        super(FastPose, self).__init__()

        self.preact = SEResnet('resnet101')

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.DIM, nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        duc_out = self.duc2(out)

        final_out = self.conv_out(duc_out)
        return duc_out #duc_out
