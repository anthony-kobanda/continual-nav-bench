from offbench.utils.imports import instantiate_class

import torch
import torch.nn as nn

from .tools import variance_scaling_init_linear
from omegaconf import DictConfig



class Linear(nn.Module):

    def __init__(
        self,
        input_size:int,
        output_size:int,
        use_biases:bool=True,
        use_layer_norm:bool=True,
        activation_cfg:DictConfig={"classname":"torch.nn.ReLU"},
        dropout:float=0.0,
        init_scaling:float=0.1) -> None:

        super().__init__()
        self.linear = nn.Linear(input_size,output_size,use_biases)
        self.layer_norm = nn.LayerNorm(output_size) if use_layer_norm else nn.Identity()
        self.activation:nn.Module = instantiate_class(activation_cfg)
        self.dropout = nn.Dropout(dropout)

        variance_scaling_init_linear(self.linear,init_scaling,with_bias=use_biases)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        o = self.linear(x)
        o = self.layer_norm(o)
        o = self.activation(o)
        o = self.dropout(o)
        return o
    
    def train(self,*args,**kwargs) -> "Linear":
        self.linear.train()
        self.layer_norm.train()
        self.activation.train()
        self.dropout.train()
        for param in self.parameters(): param.requires_grad = True
        for name, param in self.named_parameters(): param.requires_grad = True
        return self
    
    def eval(self) -> "Linear":
        self.linear.eval()
        self.layer_norm.eval()
        self.activation.eval()
        self.dropout.eval()
        for param in self.parameters(): param.requires_grad = False
        for name, param in self.named_parameters(): param.requires_grad = False
        return self



class LinearBlock(nn.Module):

    def __init__(
        self,
        size:int,
        use_biases:bool=True,
        use_layer_norm:bool=True,
        activation_cfg:DictConfig={"classname":"torch.nn.ReLU"},
        dropout:float=0.0,
        init_scaling:float=0.1) -> None:
        
        super().__init__()

        self.linear_1 = nn.Linear(size,size,use_biases)
        self.layer_norm_1 = nn.LayerNorm(size) if use_layer_norm else nn.Identity()
        self.activation_1:nn.Module = instantiate_class(activation_cfg)
        self.dropout_1 = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(size,size,use_biases)
        self.layer_norm_2 = nn.LayerNorm(size) if use_layer_norm else nn.Identity()
        self.activation_2:nn.Module = instantiate_class(activation_cfg)
        self.dropout_2 = nn.Dropout(dropout)

        variance_scaling_init_linear(self.linear_1,init_scaling,with_bias=use_biases)
        variance_scaling_init_linear(self.linear_2,init_scaling,with_bias=use_biases)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        o = self.linear_1(x)
        o = self.layer_norm_1(o)
        o = self.activation_1(o)
        o = self.dropout_1(o)
        o = self.linear_2(o)
        o = self.layer_norm_2(o)
        o = o + x
        o = self.activation_2(o)
        o = self.dropout_2(o)
        return o
    
    def train(self,*args,**kwargs) -> "LinearBlock":
        self.linear_1.train()
        self.layer_norm_1.train()
        self.activation_1.train()
        self.dropout_1.train()
        self.linear_2.train()
        self.layer_norm_2.train()
        self.activation_2.train()
        self.dropout_2.train()
        for param in self.parameters(): param.requires_grad = True
        for name, param in self.named_parameters(): param.requires_grad = True
        return self
    
    def eval(self) -> "LinearBlock":
        self.linear_1.eval()
        self.layer_norm_1.eval()
        self.activation_1.eval()
        self.dropout_1.eval()
        self.linear_2.eval()
        self.layer_norm_2.eval()
        self.activation_2.eval()
        self.dropout_2.eval()
        for param in self.parameters(): param.requires_grad = False
        for name, param in self.named_parameters(): param.requires_grad = False
        return self
