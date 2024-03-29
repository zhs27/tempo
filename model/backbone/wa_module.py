import torch
import torch.nn as nn
from .DWT import DWT_2D


class wa_module(nn.Module):
    '''
    This module is used in networks that require a shortcut.
    X --> output, LL(shortcut)
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='haar'):
        super(wa_module, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)
        output = LL

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        return output, LL