
#region Imports

import torch
import torch.nn as nn
import numpy as np

import sys
import pathlib

from model_helpers.fno_layers import *
from MLP import *

#endregion

#region FNO Class(es)

class FNO1D(nn.Module):

    def __init__(self, cfg_p, cfg_q, cfg_fourierblock):
        super(FNO1D, self).__init__()

        # Initialize P Network (elevator)
        self.p_network = MLP(cfg_mlp= cfg_p)

        # Initialize Q Network (reducer)
        self.q_network = MLP(cfg_mlp= cfg_q)

        # Initialize Fourier Block
        self.fb_pout = cfg_p['output_dim']
        self.fb_qin = cfg_q['input_dim']
        self.fb_hidden_dim = cfg_fourierblock['hidden_dim']
        self.fb_modes = cfg_fourierblock['modes']
        self.fb_kernel = cfg_fourierblock['kernel']
        self.fb_activation = cfg_fourierblock.get('activation', nn.Tanh)()

        assert len(self.fb_kernel) == len(self.fb_modes), 'Fourier Block kernel and modes lists have to be same len.'
        assert len(self.fb_hidden_dim) + 1 == len(self.fb_kernel), 'Hidden list has to be 1 less than kernel list in ken.'

        # Build Fourier Block
        self.fb = [FourierLayer1DLocal(in_channels= self.fb_pout, 
                                       out_channels= self.fb_hidden_dim[0], 
                                       modes= self.fb_modes[0], 
                                       kernel= self.fb_kernel[0]), self.fb_activation]

        for i in range(len(self.fb_hidden_dim)):

            self.fb.append(FourierLayer1DLocal(in_channels= self.fb_hidden_dim[i],
                                               out_channels= self.fb_hidden_dim[i+1],
                                               modes= self.fb_modes[i+1],
                                               kernel= self.fb_kernel[i+1]))
            self.fb.append(self.fb_activation)

        self.fb.append(FourierLayer1DLocal(in_channels= self.fb_hidden_dim[-1],
                                           out_channels= self.fb_qin,
                                           modes= self.fb_modes[-1], 
                                           kernel= self.fb_kernel[-1]))
        self.fb.append(self.fb_activation)

        self.fourierblock = nn.Sequential[*self.fb]

    def forward(self, input):

        input = self.p_network(input)
        input = self.fb(input)

        return self.q_network(input)

#endregion
