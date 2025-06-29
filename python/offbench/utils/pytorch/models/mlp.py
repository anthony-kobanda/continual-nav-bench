import torch
import torch.nn as nn

from itertools import chain
from offbench.utils.pytorch.utils import Linear, LinearBlock
from offbench.utils.pytorch.utils import Subspace_Linear, Subspace_LinearBlock
from omegaconf import DictConfig
from typing import Any, List, Dict, Generator, Tuple, Union



class MLP(nn.Module):

    def __init__(
        self,
        sizes:List[int],
        use_biases:bool=True,
        use_layer_norm:bool=True,
        layer_activation_cfg:DictConfig={"classname":"torch.nn.ReLU"},
        output_activation_cfg:DictConfig={"classname":"torch.nn.Identity"},
        dropout:float=0.0,
        init_scaling:float=0.1) -> None:
        
        super().__init__()

        assert len(sizes) >= 2

        self.layers = nn.ModuleList([])

        for i in range(len(sizes)-2):
            self.layers.append(Linear(
                sizes[i],
                sizes[i+1],
                use_biases,
                use_layer_norm,
                layer_activation_cfg,
                dropout,
                init_scaling
            ))
        self.layers.append(Linear(
            sizes[-2],
            sizes[-1],
            use_biases,
            False,
            output_activation_cfg,
            0.0,
            init_scaling
        ))
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train(self) -> "MLP":
        for layer in self.layers:
            layer.train()
        return self
    
    def eval(self) -> "MLP":
        for layer in self.layers:
            layer.eval()
        return self



class ResidualMLP(nn.Module):

    def __init__(
        self,
        sizes:List[int],
        use_biases:bool=True,
        use_layer_norm:bool=True,
        layer_activation_cfg:DictConfig={"classname":"torch.nn.ReLU"},
        output_activation_cfg:DictConfig={"classname":"torch.nn.Identity"},
        dropout:float=0.0,
        init_scaling:float=0.1) -> None:
        
        super().__init__()

        assert len(sizes) >= 2

        self.layers = nn.ModuleList([])

        for i in range(len(sizes)-2):
            if i == 0:
                self.layers.append(Linear(
                    sizes[i],
                    sizes[i+1],
                    use_biases,
                    use_layer_norm,
                    layer_activation_cfg,
                    dropout,
                    init_scaling
                ))
            else:
                assert sizes[i] == sizes[i+1]
                self.layers.append(LinearBlock(
                    sizes[i],
                    use_biases,
                    use_layer_norm,
                    layer_activation_cfg,
                    dropout,
                    init_scaling
                ))
        self.layers.append(Linear(
            sizes[-2],
            sizes[-1],
            use_biases,
            False,
            output_activation_cfg,
            0.0,
            init_scaling
        ))
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train(self) -> "MLP":
        for layer in self.layers:
            layer.train()
        return self
    
    def eval(self) -> "MLP":
        for layer in self.layers:
            layer.eval()
        return self



class Subspace_MLP(nn.Module):

    def __init__(
        self,
        sizes:List[int],
        use_biases:bool=True,
        use_layer_norm:bool=True,
        layer_activation_cfg:DictConfig={"classname":"torch.nn.ReLU"},
        output_activation_cfg:DictConfig={"classname":"torch.nn.Identity"},
        dropout:float=0.0,
        init_scaling:float=0.1) -> None:
        
        super().__init__()

        assert len(sizes) >= 2

        self.layers = nn.ModuleList([])

        for i in range(len(sizes)-2):
            self.layers.append(Subspace_Linear(
                sizes[i],
                sizes[i+1],
                use_biases,
                use_layer_norm,
                layer_activation_cfg,
                dropout,
                init_scaling
            ))
        self.layers.append(Subspace_Linear(
            sizes[-2],
            sizes[-1],
            use_biases,
            False,
            output_activation_cfg,
            0.0,
            init_scaling
        ))
    
    def add_anchor(self) -> "Subspace_MLP":
        for layer in self.layers:
            if isinstance(layer, Subspace_Linear):
                layer.add_anchor()
        return self
    
    def remove_anchor(self) -> "Subspace_MLP":
        for layer in self.layers:
            if isinstance(layer, Subspace_Linear):
                layer.remove_anchor()
        return self
    
    def cosine_similarity(self) -> torch.Tensor:
        cosine_similarity = 0.0
        for layer in self.layers:
            if isinstance(layer, Subspace_Linear):
                cosine_similarity += layer.cosine_similarity()
        return cosine_similarity / len(self.layers)
    
    def forward(self,x:torch.Tensor,alpha:torch.Tensor=None) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, Subspace_Linear):
                x = layer.forward(x,alpha)
        return x
    
    def train(self,only_last:bool=False) -> "Subspace_MLP":
        for layer in self.layers:
            if isinstance(layer, Subspace_Linear):
                layer.train(only_last)
        return self
    
    def eval(self) -> "Subspace_MLP":
        for layer in self.layers:
            if isinstance(layer, Subspace_Linear):
                layer.eval()
        return self



class Subspace_ResidualMLP(nn.Module):

    def __init__(
        self,
        sizes:List[int],
        use_biases:bool=True,
        use_layer_norm:bool=True,
        layer_activation_cfg:DictConfig={"classname":"torch.nn.ReLU"},
        output_activation_cfg:DictConfig={"classname":"torch.nn.Identity"},
        dropout:float=0.0,
        init_scaling:float=0.1) -> None:
        
        super().__init__()

        assert len(sizes) >= 2

        self.layers = nn.ModuleList([])

        for i in range(len(sizes)-2):
            if i == 0:
                self.layers.append(Subspace_Linear(
                    sizes[i],
                    sizes[i+1],
                    use_biases,
                    use_layer_norm,
                    layer_activation_cfg,
                    dropout,
                    init_scaling
                ))
            else:
                assert sizes[i] == sizes[i+1]
                self.layers.append(Subspace_LinearBlock(
                    sizes[i],
                    use_biases,
                    use_layer_norm,
                    layer_activation_cfg,
                    dropout,
                    init_scaling
                ))
        self.layers.append(Subspace_Linear(
            sizes[-2],
            sizes[-1],
            use_biases,
            False,
            output_activation_cfg,
            0.0,
            init_scaling
        ))
    
    def add_anchor(self) -> "Subspace_ResidualMLP":
        for layer in self.layers:
            if isinstance(layer, (Subspace_Linear, Subspace_LinearBlock)):
                layer.add_anchor()
        return self
    
    def remove_anchor(self) -> "Subspace_ResidualMLP":
        for layer in self.layers:
            if isinstance(layer, (Subspace_Linear, Subspace_LinearBlock)):
                layer.remove_anchor()
        return self
    
    def cosine_similarity(self) -> torch.Tensor:
        cosine_similarity = 0.0
        for layer in self.layers:
            if isinstance(layer, (Subspace_Linear, Subspace_LinearBlock)):
                cosine_similarity += layer.cosine_similarity()
        return cosine_similarity / len(self.layers)
    
    def forward(self,x:torch.Tensor,alpha:torch.Tensor=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x,alpha)
        return x
    
    def train(self,only_last:bool=False) -> "Subspace_ResidualMLP":
        for layer in self.layers:
            if isinstance(layer, (Subspace_Linear, Subspace_LinearBlock)):
                layer.train(only_last)
        return self
    
    def eval(self) -> "Subspace_ResidualMLP":
        for layer in self.layers:
            if isinstance(layer, (Subspace_Linear, Subspace_LinearBlock)):
                layer.eval()
        return self



class MLP_LIST(nn.Module):

    def __init__(self, mlps:List[Union[MLP, ResidualMLP]]) -> None:
        super().__init__()
        self.mlps = mlps
    
    def __getitem__(self,index:int) -> Union[MLP, ResidualMLP]:
        return self.mlps[index]
    
    def __iter__(self) -> Generator[Union[MLP, ResidualMLP], Any, None]:
        for mlp in self.mlps:
            yield mlp

    def __len__(self) -> int:
        return len(self.mlps)
    
    def append(self,mlp:Union[MLP, ResidualMLP]) -> "MLP_LIST":
        self.mlps.append(mlp)
        return self
    
    def extend(self,mlps:List[Union[MLP, ResidualMLP]]) -> "MLP_LIST":
        self.mlps.extend(mlps)
        return self
    
    def pop(self,index:int) -> Union[MLP, ResidualMLP]:
        return self.mlps.pop(index)
    
    def forward(self,x:torch.Tensor) -> List[torch.Tensor]:
        return [mlp(x) for mlp in self.mlps]
    
    def forward(self,index:int,x:torch.Tensor) -> torch.Tensor:
        return self.mlps[index](x)

    def train(self) -> "MLP_LIST":
        for i,mlp in enumerate(self.mlps):
            self.mlps[i] = mlp.train()
        return self
    
    def train_only(self,index:int) -> "MLP_LIST":
        for i,mlp in enumerate(self.mlps):
            if i == index: self.mlps[i] = mlp.train()
            else: self.mlps[i] = mlp.eval()
        return self
    
    def eval(self) -> "MLP_LIST":
        for i,mlp in enumerate(self.mlps):
            self.mlps[i] = mlp.eval()
        return self
    
    def to(self,device) -> "MLP_LIST":
        for i,mlp in enumerate(self.mlps):
            self.mlps[i] = mlp.to(device)
        return self

    def parameters(self,index:int=None) -> Generator[torch.nn.Parameter, None, None]:
        if index is None: return chain(*[mlp.parameters() for mlp in self.mlps])
        return self.mlps[index].parameters()
    
    def buffers(self,index:int=None) -> Generator[torch.Tensor, None, None]:
        if index is None: return chain(*[mlp.buffers() for mlp in self.mlps])
        return self.mlps[index].buffers()



class MLP_DICT(nn.Module):

    def __init__(self, mlps:Dict[Any,Union[MLP, ResidualMLP]]) -> None:
        super().__init__()
        self.mlps = mlps
    
    def __getitem__(self,key:Any) -> Union[MLP, ResidualMLP]:
        return self.mlps[key]
    
    def __iter__(self) -> Generator[Any, Any, None]:
        for key in self.mlps:
            yield key
    
    def items(self) -> Generator[Tuple[Any,Union[MLP, ResidualMLP]], Any, None]:
        for key in self.mlps:
            yield key, self.mlps[key]

    def __len__(self) -> int:
        return len(self.mlps)
    
    def __setitem__(self,key:Any,mlp:Union[MLP, ResidualMLP]) -> None:
        self.mlps[key] = mlp
        
    def __delitem__(self,key:Any) -> None:
        del self.mlps[key]
    
    def forward(self,x:torch.Tensor) -> Dict[Any,torch.Tensor]:
        return {key: mlp(x) for key,mlp in self.mlps.items()}
    
    def forward(self,key:Any,x:torch.Tensor) -> torch.Tensor:
        return self.mlps[key](x)

    def train(self) -> "MLP_DICT":
        for key in self.mlps:
            self.mlps[key] = self.mlps[key].train()
        return self
    
    def train_only(self,key:Any) -> "MLP_DICT":
        for k in self.mlps:
            if k == key: self.mlps[k] = self.mlps[k].train()
            else: self.mlps[k] = self.mlps[k].eval()
        return self
    
    def eval(self) -> "MLP_DICT":
        for key in self.mlps:
            self.mlps[key] = self.mlps[key].eval()
        return self
    
    def to(self,device) -> "MLP_DICT":
        for key in self.mlps:
            self.mlps[key] = self.mlps[key].to(device)
        return self
    
    def parameters(self,key:Any=None) -> Generator[torch.nn.Parameter, None, None]:
        if key is None: return chain(*[mlp.parameters() for mlp in self.mlps.values()])
        return self.mlps[key].parameters()
    
    def buffers(self,key:Any=None) -> Generator[torch.Tensor, None, None]:
        if key is None: return chain(*[mlp.buffers() for mlp in self.mlps.values()])
        return self.mlps[key].buffers()
