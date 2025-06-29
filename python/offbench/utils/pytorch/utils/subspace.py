from offbench.utils.imports import instantiate_class

import torch
import torch.nn as nn

from .tools import variance_scaling_init_linear
from omegaconf import DictConfig



class Subspace_Linear(nn.Module):

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

        self.input_size = input_size
        self.output_size = output_size
        self.use_biases = use_biases
        self.use_layer_norm = use_layer_norm
        self.init_scaling = init_scaling

        self.device = "cpu"
        self.n_anchors = 0

        self.linears = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        self.activation:nn.Module = instantiate_class(activation_cfg)
        self.dropout = nn.Dropout(dropout)
    
    def add_anchor(self) -> "Subspace_Linear":
        linear = nn.Linear(self.input_size,self.output_size,self.use_biases)
        layer_norm = nn.LayerNorm(self.output_size) if self.use_layer_norm else nn.Identity()
        if self.n_anchors == 0: variance_scaling_init_linear(linear,self.init_scaling)
        else:
            linear.weight.data.copy_(sum((1.0/self.n_anchors) * l.weight.data for l in self.linears))
            if self.use_biases: 
                linear.bias.data.copy_(sum((1.0/self.n_anchors) * l.bias.data for l in self.linears))
            if self.use_layer_norm:
                layer_norm.weight.data.copy_(sum((1.0/self.n_anchors) * l.weight.data for l in self.layer_norms))
                layer_norm.bias.data.copy_(sum((1.0/self.n_anchors) * l.bias.data for l in self.layer_norms))
        self.linears.append(linear.to(self.device))
        self.layer_norms.append(layer_norm.to(self.device))
        self.n_anchors += 1
        return self
    
    def remove_anchor(self) -> "Subspace_Linear":
        assert self.n_anchors > 0
        self.linears.pop(-1)
        self.layer_norms.pop(-1)
        self.n_anchors = self.n_anchors - 1
        return self
    
    def cosine_similarity(self) -> torch.Tensor:
        cosine_similarity = 0.0
        if self.n_anchors < 2: 
            return torch.tensor(cosine_similarity).to(self.device)
        for i in range(self.n_anchors):
            for j in range(i+1,self.n_anchors):
                cosine_similarity += torch.nn.functional.cosine_similarity(self.linears[i].weight.view(-1),self.linears[j].weight.view(-1),dim=0) ** 2
                if self.use_biases:
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.linears[i].bias.view(-1),self.linears[j].bias.view(-1),dim=0) ** 2
                if self.use_layer_norm:
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.layer_norms[i].weight.view(-1),self.layer_norms[j].weight.view(-1),dim=0) ** 2
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.layer_norms[i].bias.view(-1),self.layer_norms[j].bias.view(-1),dim=0) ** 2
        return 2 * cosine_similarity / (self.n_anchors * (self.n_anchors-1) )
    
    def forward(self,x:torch.Tensor,alpha:torch.Tensor=None) -> torch.Tensor:
        if alpha is None: alpha = torch.ones(self.n_anchors) / self.n_anchors
        alpha = alpha.to(self.device)
        o = sum([alpha[k] * self.layer_norms[k](self.linears[k](x)) for k in range(self.n_anchors)])
        o = self.activation(o)
        o = self.dropout(o)
        return o
    
    def to(self,device:torch.device) -> "Subspace_Linear":
        super().to(device)
        self.device = device
        self.linears = self.linears.to(device)
        self.layer_norms = self.layer_norms.to(device)
        self.activation = self.activation.to(device)
        self.dropout = self.dropout.to(device)
        return self
    
    def train(self,only_last:bool=False) -> "Subspace_Linear":
        # linears
        for linear in self.linears: linear.train()
        for param in self.linears.parameters(): param.requires_grad = True
        for name, param in self.linears.named_parameters(): param.requires_grad = True
        # layer norms
        for layer_norm in self.layer_norms: layer_norm.train()
        for param in self.layer_norms.parameters(): param.requires_grad = True
        for name, param in self.layer_norms.named_parameters(): param.requires_grad = True
        # activation
        self.activation.train()
        for param in self.activation.parameters(): param.requires_grad = True
        for name, param in self.activation.named_parameters(): param.requires_grad
        # dropout
        self.dropout.train()
        for param in self.dropout.parameters(): param.requires_grad = True
        for name, param in self.dropout.named_parameters(): param.requires_grad = True
        # if only_last
        if only_last:
            for i in range(self.n_anchors-1):
                # linears
                self.linears[i].eval()
                for param in self.linears[i].parameters(): param.requires_grad = False
                for name, param in self.linears[i].named_parameters(): param.requires_grad = False
                # layer norms
                self.layer_norms[i].eval()
                for param in self.layer_norms[i].parameters(): param.requires_grad = False
                for name, param in self.layer_norms[i].named_parameters(): param.requires_grad = False
        # return
        return self
    
    def eval(self) -> "Subspace_Linear":
        # linears
        for linear in self.linears: linear.eval()
        for param in self.linears.parameters(): param.requires_grad = False
        for name, param in self.linears.named_parameters(): param.requires_grad = False
        # layer norms
        for layer_norm in self.layer_norms: layer_norm.eval()
        for param in self.layer_norms.parameters(): param.requires_grad = False
        for name, param in self.layer_norms.named_parameters(): param.requires_grad = False
        # activation
        self.activation.eval()
        for param in self.activation.parameters(): param.requires_grad = False
        for name, param in self.activation.named_parameters(): param.requires_grad = False
        # dropout
        self.dropout.eval()
        for param in self.dropout.parameters(): param.requires_grad = False
        for name, param in self.dropout.named_parameters(): param.requires_grad = False
        # return
        return self



class Subspace_LinearBlock(nn.Module):

    def __init__(
        self,
        size:int,
        use_biases:bool=True,
        use_layer_norm:bool=True,
        activation_cfg:DictConfig={"classname":"torch.nn.ReLU"},
        dropout:float=0.0,
        init_scaling:float=0.1) -> None:
        
        super().__init__()

        self.size = size
        self.use_biases = use_biases
        self.use_layer_norm = use_layer_norm
        self.init_scaling = init_scaling

        self.device = "cpu"
        self.n_anchors = 0

        self.linears_1 = nn.ModuleList([])
        self.layer_norms_1 = nn.ModuleList([])
        self.activation_1:nn.Module = instantiate_class(activation_cfg)
        self.dropout_1 = nn.Dropout(dropout)

        self.linears_2 = nn.ModuleList([])
        self.layer_norms_2 = nn.ModuleList([])
        self.activation_2:nn.Module = instantiate_class(activation_cfg)
        self.dropout_2 = nn.Dropout(dropout)
    
    def add_anchor(self) -> "Subspace_Linear":
        # first block
        linear_1 = nn.Linear(self.size,self.size,self.use_biases)
        layer_norm_1 = nn.LayerNorm(self.size) if self.use_layer_norm else nn.Identity()
        if self.n_anchors == 0: variance_scaling_init_linear(linear_1,self.init_scaling)
        else:
            linear_1.weight.data.copy_(sum((1.0/self.n_anchors) * l.weight.data for l in self.linears_1))
            if self.use_biases: 
                linear_1.bias.data.copy_(sum((1.0/self.n_anchors) * l.bias.data for l in self.linears_1))
            if self.use_layer_norm:
                layer_norm_1.weight.data.copy_(sum((1.0/self.n_anchors) * l.weight.data for l in self.layer_norms_1))
                layer_norm_1.bias.data.copy_(sum((1.0/self.n_anchors) * l.bias.data for l in self.layer_norms_1))
        self.linears_1.append(linear_1.to(self.device))
        self.layer_norms_1.append(layer_norm_1.to(self.device))
        # second block
        linear_2 = nn.Linear(self.size,self.size,self.use_biases)
        layer_norm_2 = nn.LayerNorm(self.size) if self.use_layer_norm else nn.Identity()
        if self.n_anchors == 0: variance_scaling_init_linear(linear_2,self.init_scaling)
        else:
            linear_2.weight.data.copy_(sum((1.0/self.n_anchors) * l.weight.data for l in self.linears_2))
            if self.use_biases:
                linear_2.bias.data.copy_(sum((1.0/self.n_anchors) * l.bias.data for l in self.linears_2))
            if self.use_layer_norm:
                layer_norm_2.weight.data.copy_(sum((1.0/self.n_anchors) * l.weight.data for l in self.layer_norms_2))
                layer_norm_2.bias.data.copy_(sum((1.0/self.n_anchors) * l.bias.data for l in self.layer_norms_2))
        self.linears_2.append(linear_2.to(self.device))
        self.layer_norms_2.append(layer_norm_2.to(self.device))
        # increment number of anchors
        self.n_anchors += 1
        return self
    
    def remove_anchor(self) -> "Subspace_Linear":
        assert self.n_anchors > 0
        self.linears_1.pop(-1)
        self.layer_norms_1.pop(-1)
        self.linears_2.pop(-1)
        self.layer_norms_2.pop(-1)
        self.n_anchors = self.n_anchors - 1
        return self
    
    def cosine_similarity(self) -> torch.Tensor:
        cosine_similarity = 0.0
        if self.n_anchors < 2: 
            return torch.tensor(cosine_similarity).to(self.device)
        for i in range(self.n_anchors):
            for j in range(i+1,self.n_anchors):
                cosine_similarity += torch.nn.functional.cosine_similarity(self.linears_1[i].weight.view(-1),self.linears_1[j].weight.view(-1),dim=0) ** 2
                cosine_similarity += torch.nn.functional.cosine_similarity(self.linears_2[i].weight.view(-1),self.linears_2[j].weight.view(-1),dim=0) ** 2
                if self.use_biases:
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.linears_1[i].bias.view(-1),self.linears_1[j].bias.view(-1),dim=0) ** 2
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.linears_2[i].bias.view(-1),self.linears_2[j].bias.view(-1),dim=0) ** 2
                if self.use_layer_norm:
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.layer_norms_1[i].weight.view(-1),self.layer_norms_1[j].weight.view(-1),dim=0) ** 2
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.layer_norms_1[i].bias.view(-1),self.layer_norms_1[j].bias.view(-1),dim=0) ** 2
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.layer_norms_2[i].weight.view(-1),self.layer_norms_2[j].weight.view(-1),dim=0) ** 2
                    cosine_similarity += torch.nn.functional.cosine_similarity(self.layer_norms_2[i].bias.view(-1),self.layer_norms_2[j].bias.view(-1),dim=0) ** 2
        return 2 * cosine_similarity / (self.n_anchors * (self.n_anchors-1) )

    def forward(self,x:torch.Tensor,alpha:torch.Tensor=None) -> torch.Tensor:
        if alpha is None: alpha = torch.ones(self.n_anchors) / self.n_anchors
        alpha = alpha.to(self.device)
        o = sum([alpha[k] * self.layer_norms_1[k](self.linears_1[k](x)) for k in range(self.n_anchors)])
        o = self.activation_1(o)
        o = self.dropout_1(o)
        o = sum([alpha[k] * self.layer_norms_2[k](self.linears_2[k](o)) for k in range(self.n_anchors)])
        o = o + x
        o = self.activation_2(o)
        o = self.dropout_2(o)
        return o
    
    def to(self,device:torch.device) -> "Subspace_Linear":
        super().to(device)
        self.device = device
        self.linears_1 = self.linears_1.to(device)
        self.layer_norms_1 = self.layer_norms_1.to(device)
        self.activation_1 = self.activation_1.to(device)
        self.dropout_1 = self.dropout_1.to(device)
        self.linears_2 = self.linears_2.to(device)
        self.layer_norms_2 = self.layer_norms_2.to(device)
        self.activation_2 = self.activation_2.to(device)
        self.dropout_2 = self.dropout_2.to(device)
        return self
    
    def train(self,only_last:bool=False) -> "Subspace_Linear":
        # first block linears
        for linear in self.linears_1: linear.train()
        for param in self.linears_1.parameters(): param.requires_grad = True
        for name, param in self.linears_1.named_parameters(): param.requires_grad = True
        # first block layer norms
        for layer_norm in self.layer_norms_1: layer_norm.train()
        for param in self.layer_norms_1.parameters(): param.requires_grad = True
        for name, param in self.layer_norms_1.named_parameters(): param.requires_grad = True
        # first block activation
        self.activation_1.train()
        for param in self.activation_1.parameters(): param.requires_grad = True
        for name, param in self.activation_1.named_parameters(): param.requires_grad = True
        # first block dropout
        self.dropout_1.train()
        for param in self.dropout_1.parameters(): param.requires_grad = True
        for name, param in self.dropout_1.named_parameters(): param.requires_grad = True
        # second block linears
        for linear in self.linears_2: linear.train()
        for param in self.linears_2.parameters(): param.requires_grad = True
        for name, param in self.linears_2.named_parameters(): param.requires_grad = True
        # second block layer norms
        for layer_norm in self.layer_norms_2: layer_norm.train()
        for param in self.layer_norms_2.parameters(): param.requires_grad = True
        for name, param in self.layer_norms_2.named_parameters(): param.requires_grad = True
        # second block activation
        self.activation_2.train()
        for param in self.activation_2.parameters(): param.requires_grad = True
        for name, param in self.activation_2.named_parameters(): param.requires_grad = True
        # second block dropout
        self.dropout_2.train()
        for param in self.dropout_2.parameters(): param.requires_grad = True
        for name, param in self.dropout_2.named_parameters(): param.requires_grad = True
        # if only_last
        if only_last:
            for i in range(self.n_anchors-1):
                # first block linears
                self.linears_1[i].eval()
                for param in self.linears_1[i].parameters(): param.requires_grad = False
                for name, param in self.linears_1[i].named_parameters(): param.requires_grad = False
                # first block layer norms
                self.layer_norms_1[i].eval()
                for param in self.layer_norms_1[i].parameters(): param.requires_grad = False
                for name, param in self.layer_norms_1[i].named_parameters(): param.requires_grad = False
                # second block linears
                self.linears_2[i].eval()
                for param in self.linears_2[i].parameters(): param.requires_grad = False
                for name, param in self.linears_2[i].named_parameters(): param.requires_grad = False
                # second block layer norms
                self.layer_norms_2[i].eval()
                for param in self.layer_norms_2[i].parameters(): param.requires_grad = False
                for name, param in self.layer_norms_2[i].named_parameters(): param.requires_grad = False
        # return
        return self
    
    def eval(self) -> "Subspace_Linear":
        # first block linears
        for linear in self.linears_1: linear.eval()
        for param in self.linears_1.parameters(): param.requires_grad = False
        for name, param in self.linears_1.named_parameters(): param.requires_grad = False
        # first block layer norms
        for layer_norm in self.layer_norms_1: layer_norm.eval()
        for param in self.layer_norms_1.parameters(): param.requires_grad = False
        for name, param in self.layer_norms_1.named_parameters(): param.requires_grad = False
        # first block activation
        self.activation_1.eval()
        for param in self.activation_1.parameters(): param.requires_grad = False
        for name, param in self.activation_1.named_parameters(): param.requires_grad = False
        # first block dropout
        self.dropout_1.eval()
        for param in self.dropout_1.parameters(): param.requires_grad = False
        for name, param in self.dropout_1.named_parameters(): param.requires_grad = False
        # second block linears
        for linear in self.linears_2: linear.eval()
        for param in self.linears_2.parameters(): param.requires_grad = False
        for name, param in self.linears_2.named_parameters(): param.requires_grad = False
        # second block layer norms
        for layer_norm in self.layer_norms_2: layer_norm.eval()
        for param in self.layer_norms_2.parameters(): param.requires_grad = False
        for name, param in self.layer_norms_2.named_parameters(): param.requires_grad = False
        # second block activation
        self.activation_2.eval()
        for param in self.activation_2.parameters(): param.requires_grad = False
        for name, param in self.activation_2.named_parameters(): param.requires_grad = False
        # second block dropout
        self.dropout_2.eval()
        for param in self.dropout_2.parameters(): param.requires_grad = False
        for name, param in self.dropout_2.named_parameters(): param.requires_grad = False
        # return
        return self
