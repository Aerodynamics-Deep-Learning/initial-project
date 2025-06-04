"""
Here we define a Multi-Layer Perceptron (MLP or DNN) using PyTorch.
This is incredibly crude and just a starting point.
"""

#region Imports

import torch
import torch.nn as nn
import numpy as np

#endregion

#region MLP Class

"""
Here, we define a simple DNN class
"""

class DNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers, activation='relu'):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        