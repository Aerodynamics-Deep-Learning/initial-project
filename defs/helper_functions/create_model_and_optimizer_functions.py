import torch

import sys
import pathlib
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from defs.models.NIF import *
from defs.models.MLP import *

def create_NIF_Pointwise_and_optimizer(cfg_shape_net, cfg_param_net, lr= 0.001):
    model = NIF_Pointwise(cfg_shape_net, cfg_param_net)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

def create_NIF_PartialPaper_and_optimizer(cfg_shape_net, cfg_param_net, lr= 0.001):
    model = NIF_PartialPaper(cfg_shape_net, cfg_param_net)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

def create_NIF_PaperFourier_and_optimizer(cfg_shape_net, cfg_param_net, lr= 0.001):
    model = NIF_PaperFourier(cfg_shape_net, cfg_param_net)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

def create_DNN_and_optimizer(cfg, lr= 0.001):
    model = DNN(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer