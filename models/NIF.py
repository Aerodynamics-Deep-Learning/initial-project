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
        self.shape_activation = cfg_shape_net.get(nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.SiLU)

        # Initialize paremeter network parameters
        self.cfg_param_net = cfg_param_net
        self.param_i_dim = cfg_param_net['input_dim']
        self.param_hidden_units = cfg_param_net['hidden_units']
        self.param_num_hidden_layers = cfg_param_net['num_hidden_layers'],
        self.param_activation = cfg_param_net.get(nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.SiLU)

        # Initialize parameter network regularization
        self.param_reg_jacobian = cfg_param_net.get('reg_jacobian', False)
        self.param_reg_l1 = cfg_param_net.get('reg_l1', False)
        self.param_reg_l2 = cfg_param_net.get('reg_l2', False)

        self.param_latent_dim = sum(self.shape_hidden_units)

        # -----Build param network-----

        self.param_net_list = nn.ModuleList()

        self.param_net_list.append()

    def _initialie_parameter_network(self):
        
        """
        Initializes the parameter net. The latent space of parameter net will feed into the shape net's last layer's weights.

        Args: 
            self: The NIF model instance.
        
        Returns:
            nn.Sequential: The initialized parameter network.
        """



        # Add input layer
        p_input_layer = nn.Sequential(nn.Linear(in_features=self.parameter_i_dim, out_features=self.parameter_hidden_units[0]), self.parameter_activation)

        # Add hidden layers

        for i in range(len(self.parameter_hidden_units)-1):

            if i == 0:
                # First hidden layer
                p_hidden_layers = nn.Sequential(nn.Linear(in_features=self.parameter_hidden_units[i], out_features=self.parameter_hidden_units[i+1]), self.parameter_activation)
            else:
                # Subsequent hidden layers
                p_hidden_layers = nn.Sequential(p_hidden_layers, nn.Linear(in_features=self.parameter_hidden_units[i], out_features=self.parameter_hidden_units[i+1]), self.parameter_activation)


        # Add output (latent space) layer
        p_latent_layer = nn.Linear(in_features=self.parameter_hidden_units[-1], out_features=self.parameter_latent_dim)

        # Combine all layers into a single module
        parameter_network = nn.Sequential(p_input_layer, p_hidden_layers, p_latent_layer)

        return parameter_network
    
    def _initialize_shape_network(self):

        """
        Initializes the shape net. The latent space of parameter net will feed into the shape net's last layer's weights.
        
        Args:
        
        """

        shape_network = nn.ModuleList()

        # Add input layer
        s_input_layer = nn.Linear(in_features=self.shape_i_dim, out_features=self.shape_hidden_units[0])

        # Add hidden layers
        for i in range(len(self.shape_hidden_units)-1):

            s_hidden_layers = nn.Linear(in_features=self.shape_hidden_units[i], out_features=self.shape_hidden_units[i+1])

        # Add output layer
        s_output_layer = nn.Linear(in_features=self.shape_hidden_units[-1], out_features=self.shape_o_dim)

        # Combine all layers into a single module
        shape_network = nn.ModuleList(s_input_layer, s_hidden_layers, s_output_layer) # We create a module list because we want to influence each output of the layer manually

        return shape_network
    
    @staticmethod
    def _call_parameter_network(self, inputs):

        """
        Calls the parameter network with the given inputs.

        Args:
            inputs (torch.Tensor): Input tensor for the parameter network.

        Returns:
            torch.Tensor: Output tensor from the parameter network.
        """

        # Forward pass through the parameter network
        latent_space = self.parameter_network(inputs)

        return latent_space
    
    @staticmethod
    def _call_shape_network(self, inputs, latent_space):

        """
        This is a little untraditional...
        """

        out = inputs
        offset = 0

        for i, lin in enumerate(self.shape_network):
            out = lin(out)

            H_i = self.shape_hidden_units
            latent_slice = latent_space[:, offset:offset+i] # THIS HAS TO BE BETTER DEFINED

            out = out * latent_slice

            out = self.shape_activation(out)

            offset += H_i

        return out

    def forward(self, inputs):

        shape_input = inputs[:, :self.param_i_dim]
        param_input = inputs[:, self.param_i_dim:]

        latent_space = self._call_parameter_network(self, param_input)

        output = self._call_shape_network(self, shape_input, latent_space)

        output

        
#endregion

"""
For the training process, with the blessing of PyTorch Autograd, we can just forward propogate and then call for a backward prop
and update quite easily. So that is not of the biggest concern, yet.
"""