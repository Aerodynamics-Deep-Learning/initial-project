"""
Here, we define the Neural Implict Flow (NIF) models. Basing on the work done by Sahowu Pan and Yaser Afshar, as like in the GitHub repo: https://github.com/pswpswpsw/nif
This currently shares a lot of resemblance with the GitHub repo, we do not claim any originality here. Further models will be added later.
"""
import json

import torch
import torch.nn as nn

import numpy as np

#region NIF Definitions

class NIF(nn.Module):

    def __init__(self, cfg_shape_net, cfg_param_net):

        super(NIF, self).__init__()

        # Initialize shape network parameters
        self.cfg_shape_net = cfg_shape_net
        self.shape_i_dim = cfg_shape_net['input_dim']
        self.shape_o_dim = cfg_shape_net['output_dim']
        self.shape_hidden_units = cfg_shape_net['hidden_units']
        self.shape_num_hidden_layers = cfg_shape_net['num_hidden_layers']
        self.shape_activation = cfg_shape_net.get('shape_activation', nn.Tanh)()

        # Initialize paremeter network parameters
        self.cfg_param_net = cfg_param_net
        self.param_i_dim = cfg_param_net['input_dim']
        self.param_hidden_units = cfg_param_net['hidden_units']
        self.param_num_hidden_layers = cfg_param_net['num_hidden_layers']
        self.param_activation = cfg_param_net.get('param_activation', nn.Tanh)()

        self.param_latent_dim = sum(self.shape_hidden_units) + self.shape_o_dim # Define length of latent dim of param net


        # -----Build param network-----

        """
        Initializes the parameter net. The latent space of parameter net will feed into the shape net's last layer's weights.
        """

        # Add input layer
        self.p_layers = [nn.Linear(in_features=self.param_i_dim, out_features=self.param_hidden_units[0]), self.param_activation]

        # Add hidden layers
        for i in range(len(self.param_hidden_units)-1):
            self.p_layers.append(nn.Linear(in_features=self.param_hidden_units[i], out_features=self.param_hidden_units[i+1]), self.param_activation)

        # Add output (latent space) layer
        self.p_layers.append(nn.Linear(in_features=self.param_hidden_units[-1], out_features=self.param_latent_dim), self.param_activation)

        self.parameter_network = nn.Sequential(*self.p_layers)


        # -----Build shape network-----

        """
        Initializes the shape net. The latent space of parameter net will feed into the shape net's last layer's weights.
        
        """

        self.shape_network = nn.ModuleList()

        # Add input layer
        self.shape_network.append(nn.Linear(in_features=self.shape_i_dim, out_features=self.shape_hidden_units[0]))

        # Add hidden layers
        for i in range(len(self.shape_hidden_units)-1):
            self.shape_network.append(nn.Linear(in_features=self.shape_hidden_units[i], out_features=self.shape_hidden_units[i+1]))

        # Add output layer
        self.shape_network.append(nn.Linear(in_features=self.shape_hidden_units[-1], out_features=self.shape_o_dim))

    def _call_shape_network(self, shape_input, latent_space):

        """
        This is a little untraditional... Hwere we call each layer normally, but at the end of each layer before applying 
        an activation function, we pointwise multiply with some part of the latent space vector obtained from param net.
        Then we obviously input this into an activation function.
        """

        start_offset = 0 # to make sure we start slicing from start and then have a moving slicer

        self.total_units = self.shape_hidden_units + [self.shape_o_dim] # add the output dim since we have as many outputs as the hidden units + output layer

        for layer, out_dim in zip(self.shape_network, self.total_units):

            out = layer(shape_input) # take the result from layer

            latent_slice = latent_space[:, start_offset:start_offset+out_dim] # slice a part of latent space
            out = out * latent_slice # point wise multiply the out and latent space

            out = self.shape_activation(out) # put into the activation function
            start_offset += out_dim # offset the slicing start

        return out # return the output once done

    def forward(self, inputs):

        shape_input = inputs[:, :self.param_i_dim-1] #-1 because dim and slice have -1 offset we, input the dims starting from 1 not like indices where they start from 0
        param_input = inputs[:, self.param_i_dim-1:]

        latent_space = self.parameter_network(param_input) # take the latent space from parameter network

        return self._call_shape_network(shape_input, latent_space) # call the shape network

        
#endregion

"""
For the training process, with the blessing of PyTorch Autograd, we can just forward propogate and then call for a backward prop
and update quite easily. So that is not of the biggest concern, yet.
"""