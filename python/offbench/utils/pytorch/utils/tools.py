import math
import torch
import torch.nn as nn



def variance_scaling_init_linear(net:nn.Linear,scale:float=1.0,with_bias=True) -> nn.Linear:
    """
    Initializes the weights and biases of a linear layer
    using the variance scaling initialization method.

    Args:
        net (nn.Linear): The linear layer to initialize.
        scale (float): The scaling factor for the initialization. Defaults to 1.0.
    """
    fan = (net.weight.size(-2)+net.weight.size(-1))/2
    init_w = math.sqrt(scale/fan)
    net.weight.data.uniform_(-init_w,init_w)
    if with_bias: net.bias.data.fill_(0.0)
    return net



def round_to_precision(raw_action: torch.Tensor, precision: float) -> torch.Tensor:
    """
    Rounds the raw actions to the given precision.

    Args:
        raw_action (torch.Tensor): The raw actions to round.
        precision (float): The precision to round to.
    """
    return torch.round(raw_action / precision) * precision
