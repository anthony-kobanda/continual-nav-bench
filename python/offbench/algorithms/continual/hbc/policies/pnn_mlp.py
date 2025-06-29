import copy
import torch
import torch.nn as nn

from .utils import godot_goal_amazeville
from .utils import godot_goal_simpletown
from itertools import chain
from offbench.core.data import Frame
from offbench.core.policy import Policy
from offbench.utils.imports import get_class, get_arguments, instantiate_class
from offbench.utils.pytorch.utils.basic import Linear, LinearBlock
from offbench.utils.pytorch.utils.normalizer import AVAILABLE_NORMALIZERS, MinMax_Normalizer, Z_Normalizer
from omegaconf import DictConfig
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any, Dict, Iterator, List, Union



class MLP(Policy):
    
    def __init__(
        self,
        # general
        env:str,
        seed:int=None,
        # hbc specific
        waysteps:int=10,
        # all networks
        device:Union[torch.device,str]="cpu",
        normalizer:str="z",
        normalized_features:List[str]=[],
        # high policy network & optimizer
        high_dropout:float=0.1,
        high_init_scaling:float=0.1,
        high_hidden_sizes:List[int]=[256,256],
        high_optimizer_cfg:Union[DictConfig,Dict[str,Any]]={
            "classname": "torch.optim.Adam",
            "lr": 3e-4,
        },
        high_scheduler_cfg:Union[DictConfig,Dict[str,Any]]=None,
        # low policy network & optimizer
        low_dropout:float=0.1,
        low_init_scaling:float=0.1,
        low_hidden_sizes:List[int]=[256,256],
        low_optimizer_cfg:Union[DictConfig,Dict[str,Any]]={
            "classname": "torch.optim.Adam",
            "lr": 3e-4,
        },
        low_scheduler_cfg:Union[DictConfig,Dict[str,Any]]=None) -> None:

        super().__init__(seed=seed,device=device)

        # continual specific
        self._task_idx: int = None
        self._task_idxs: List[int] = []

        # for evaluation
        self._stochastic: bool = False

        # normalizers information
        self._normalizer: str = normalizer
        self._normalizers: Dict[str, Union[MinMax_Normalizer, Z_Normalizer]] = {}
        self._normalized_features: List[str] = normalized_features
        assert normalizer in AVAILABLE_NORMALIZERS, f"normalizer must be in {AVAILABLE_NORMALIZERS}"

        # hbc specific
        self._waysteps:int = waysteps

        self._env_name = env.split("-")[0]

        # high policy networks
        self._high_Ws = nn.ModuleDict({}).to(self._device)
        self._high_Us = nn.ModuleDict({}).to(self._device)
        self._high_Vs = nn.ModuleDict({}).to(self._device)
        self._high_As = nn.ModuleDict({}).to(self._device)
        self._high_acts_1 = nn.ModuleDict({}).to(self._device)
        self._high_acts_2 = nn.ModuleDict({}).to(self._device)
        # low policy networks
        self._low_Ws = nn.ModuleDict({}).to(self._device)
        self._low_Us = nn.ModuleDict({}).to(self._device)
        self._low_Vs = nn.ModuleDict({}).to(self._device)
        self._low_As = nn.ModuleDict({}).to(self._device)
        self._low_acts_1 = nn.ModuleDict({}).to(self._device)
        self._low_acts_2 = nn.ModuleDict({}).to(self._device)
        # general
        self._mlp_cfg: Dict[str, Any] = {
            "high_dropout": high_dropout,
            "high_init_scaling": high_init_scaling,
            "high_hidden_sizes": high_hidden_sizes,
            "low_dropout": low_dropout,
            "low_init_scaling": low_init_scaling,
            "low_hidden_sizes": low_hidden_sizes
        }

        # high_policy optimizer
        self._high_optimizer_cfg: Union[DictConfig,Dict[str,Any]] = high_optimizer_cfg
        self._high_scheduler_cfg: Union[DictConfig,Dict[str,Any]] = high_scheduler_cfg

        # low_policy optimizer
        self._low_optimizer_cfg: Union[DictConfig,Dict[str,Any]] = low_optimizer_cfg
        self._low_scheduler_cfg: Union[DictConfig,Dict[str,Any]] = low_scheduler_cfg
    
    def set_normalizers(self, normalize_values, **kwargs) -> "MLP":
        for k in self._normalized_features:
            if self._normalizer == "minmax": self._normalizers[k] = MinMax_Normalizer(normalize_values[k]["min"],normalize_values[k]["max"])
            elif self._normalizer == "z": self._normalizers[k] = Z_Normalizer(normalize_values[k]["mean"],normalize_values[k]["std"])
        return self

    def set_train_mode(self, task_idx:int, **kwargs) -> "Policy":

        self._task_idx = task_idx

        if not task_idx in self._task_idxs:

            self._task_idxs.append(task_idx)

            if self._env_name == "amazeville": module = godot_goal_amazeville
            elif self._env_name == "simpletown": module = godot_goal_simpletown
            else: raise ValueError(f"Unsupported environment : {self._env_name}")

            high_Ws: List[nn.Module] = []
            high_Us: List[nn.Module] = []
            high_Vs: List[nn.Module] = []
            high_As: List[nn.Parameter] = []
            high_acts_1: List[nn.Module] = []
            high_acts_2: List[nn.Module] = []

            low_Ws: List[nn.Module] = []
            low_Us: List[nn.Module] = []
            low_Vs: List[nn.Module] = []
            low_As: List[nn.Parameter] = []
            low_acts_1: List[nn.Module] = []
            low_acts_2: List[nn.Module] = []

            high_in_size = module.OBS_SIZE
            low_in_size = module.OBS_SIZE

            if len(self._task_idxs) == 1:
                
                ########
                # HIGH #
                ########

                # high hidden layers
                if len(self._mlp_cfg["high_hidden_sizes"]) > 0:

                    # high first layer
                    high_Ws.append(Linear(
                        input_size=high_in_size,
                        output_size=self._mlp_cfg["high_hidden_sizes"][0],
                        use_biases=True,
                        use_layer_norm=module.USE_LAYER_NORM,
                        activation_cfg={"classname": "torch.nn.Identity"},
                        dropout=0.0,
                        init_scaling=self._mlp_cfg["high_init_scaling"] 
                    ))
                    high_Us.append(None)
                    high_Vs.append(None)
                    high_As.append(None)
                    high_acts_1.append(None)
                    high_acts_2.append(nn.Sequential(instantiate_class(module.HIGH_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["high_dropout"])))

                    # high next layers
                    for i in range(1,len(self._mlp_cfg["high_hidden_sizes"])):

                        assert self._mlp_cfg["high_hidden_sizes"][i-1] == self._mlp_cfg["high_hidden_sizes"][i]
                        high_Ws.append(LinearBlock(
                            size=self._mlp_cfg["high_hidden_sizes"][i],
                            use_biases=True,
                            use_layer_norm=module.USE_LAYER_NORM,
                            activation_cfg=module.HIGH_LAYER_ACT_CFG,
                            dropout=self._mlp_cfg["high_dropout"],
                            init_scaling=self._mlp_cfg["high_init_scaling"]
                        ))
                        high_Us.append(None)
                        high_Vs.append(None)
                        high_As.append(None)
                        high_acts_1.append(None)
                        high_acts_2.append(nn.Identity())
                    
                    high_in_size = self._mlp_cfg["high_hidden_sizes"][-1]
                
                # high output layer
                high_Ws.append(Linear(
                    input_size=high_in_size,
                    output_size=2*module.GOAL_SIZE+1,
                    use_biases=True,
                    use_layer_norm=False,
                    activation_cfg={"classname": "torch.nn.Identity"},
                    dropout=0.0,
                    init_scaling=self._mlp_cfg["high_init_scaling"]
                ))
                high_Us.append(None)
                high_Vs.append(None)
                high_As.append(None)
                high_acts_1.append(None)
                high_acts_2.append(instantiate_class(module.HIGH_OUTPUT_ACT_CFG))

                #######
                # LOW #
                #######

                # low hidden layers
                if len(self._mlp_cfg["low_hidden_sizes"]) > 0:

                    # low first layer
                    low_Ws.append(Linear(
                        input_size=low_in_size,
                        output_size=self._mlp_cfg["low_hidden_sizes"][0],
                        use_biases=True,
                        use_layer_norm=module.USE_LAYER_NORM,
                        activation_cfg={"classname": "torch.nn.Identity"},
                        dropout=0.0,
                        init_scaling=self._mlp_cfg["low_init_scaling"] 
                    ))
                    low_Us.append(None)
                    low_Vs.append(None)
                    low_As.append(None)
                    low_acts_1.append(None)
                    low_acts_2.append(nn.Sequential(instantiate_class(module.LOW_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["low_dropout"])))

                    # low next layers
                    for i in range(1,len(self._mlp_cfg["low_hidden_sizes"])):

                        assert self._mlp_cfg["low_hidden_sizes"][i-1] == self._mlp_cfg["low_hidden_sizes"][i]
                        low_Ws.append(LinearBlock(
                            size=self._mlp_cfg["low_hidden_sizes"][i],
                            use_biases=True,
                            use_layer_norm=module.USE_LAYER_NORM,
                            activation_cfg=module.LOW_LAYER_ACT_CFG,
                            dropout=self._mlp_cfg["low_dropout"],
                            init_scaling=self._mlp_cfg["low_init_scaling"]
                        ))
                        low_Us.append(None)
                        low_Vs.append(None)
                        low_As.append(None)
                        low_acts_1.append(None)
                        low_acts_2.append(nn.Identity())
                    
                    low_in_size = self._mlp_cfg["low_hidden_sizes"][-1]
                
                # low output layer
                low_Ws.append(Linear(
                    input_size=low_in_size,
                    output_size=2*module.ACTION_SIZE,
                    use_biases=True,
                    use_layer_norm=False,
                    activation_cfg={"classname": "torch.nn.Identity"},
                    dropout=0.0,
                    init_scaling=self._mlp_cfg["low_init_scaling"]
                ))
                low_Us.append(None)
                low_Vs.append(None)
                low_As.append(None)
                low_acts_1.append(None)
                low_acts_2.append(instantiate_class(module.LOW_OUTPUT_ACT_CFG))
            
            else:

                ########
                # HIGH #
                ########

                # high hidden layers
                if len(self._mlp_cfg["high_hidden_sizes"]) > 0:
                    high_Ws.append(Linear(
                        input_size=high_in_size,
                        output_size=self._mlp_cfg["high_hidden_sizes"][0],
                        use_biases=True,
                        use_layer_norm=module.USE_LAYER_NORM,
                        activation_cfg={"classname": "torch.nn.Identity"},
                        dropout=0.0,
                        init_scaling=self._mlp_cfg["high_init_scaling"]
                    ))
                    high_Us.append(nn.Linear(self._mlp_cfg["high_hidden_sizes"][0],self._mlp_cfg["high_hidden_sizes"][0],False))
                    high_Vs.append(nn.Linear(high_in_size*self._task_idx,self._mlp_cfg["high_hidden_sizes"][0],False))
                    high_As.append(Parameter(torch.randn(high_in_size*self._task_idx)*self._mlp_cfg["high_init_scaling"]))
                    high_acts_1.append(nn.Sequential(instantiate_class(module.HIGH_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["high_dropout"])))
                    high_acts_2.append(nn.Sequential(instantiate_class(module.HIGH_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["high_dropout"])))

                    # high next layers
                    for i in range(1,len(self._mlp_cfg["high_hidden_sizes"])):

                        assert self._mlp_cfg["high_hidden_sizes"][i-1] == self._mlp_cfg["high_hidden_sizes"][i]
                        high_Ws.append(LinearBlock(
                            size=self._mlp_cfg["high_hidden_sizes"][i],
                            use_biases=True,
                            use_layer_norm=module.USE_LAYER_NORM,
                            activation_cfg=module.HIGH_LAYER_ACT_CFG,
                            dropout=self._mlp_cfg["high_dropout"],
                            init_scaling=self._mlp_cfg["high_init_scaling"]
                        ))
                        high_Us.append(nn.Linear(self._mlp_cfg["high_hidden_sizes"][i],self._mlp_cfg["high_hidden_sizes"][i],False))
                        high_Vs.append(nn.Linear(self._mlp_cfg["high_hidden_sizes"][i-1]*self._task_idx,self._mlp_cfg["high_hidden_sizes"][i],False))
                        high_As.append(Parameter(torch.randn(self._mlp_cfg["high_hidden_sizes"][i-1]*self._task_idx)*self._mlp_cfg["high_init_scaling"]))
                        high_acts_1.append(nn.Sequential(instantiate_class(module.HIGH_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["high_dropout"])))
                        high_acts_2.append(nn.Identity())
                    
                    high_in_size = self._mlp_cfg["high_hidden_sizes"][-1]
                
                # high output layer
                high_Ws.append(Linear(
                    input_size=high_in_size,
                    output_size=2*module.GOAL_SIZE+1,
                    use_biases=True,
                    use_layer_norm=False,
                    activation_cfg={"classname": "torch.nn.Identity"},
                    dropout=0.0,
                    init_scaling=self._mlp_cfg["high_init_scaling"]
                ))
                high_Us.append(nn.Linear(2*module.GOAL_SIZE+1,2*module.GOAL_SIZE+1,False))
                high_Vs.append(nn.Linear(high_in_size*self._task_idx,2*module.GOAL_SIZE+1,False))
                high_As.append(Parameter(torch.randn(high_in_size*self._task_idx)*self._mlp_cfg["high_init_scaling"]))
                high_acts_1.append(instantiate_class(module.HIGH_LAYER_ACT_CFG))
                high_acts_2.append(instantiate_class(module.HIGH_OUTPUT_ACT_CFG))
            
                ########
                # LOW #
                ########

                # low hidden layers
                if len(self._mlp_cfg["low_hidden_sizes"]) > 0:
                    low_Ws.append(Linear(
                        input_size=low_in_size,
                        output_size=self._mlp_cfg["low_hidden_sizes"][0],
                        use_biases=True,
                        use_layer_norm=module.USE_LAYER_NORM,
                        activation_cfg={"classname": "torch.nn.Identity"},
                        dropout=0.0,
                        init_scaling=self._mlp_cfg["low_init_scaling"]
                    ))
                    low_Us.append(nn.Linear(self._mlp_cfg["low_hidden_sizes"][0],self._mlp_cfg["low_hidden_sizes"][0],False))
                    low_Vs.append(nn.Linear(low_in_size*self._task_idx,self._mlp_cfg["low_hidden_sizes"][0],False))
                    low_As.append(Parameter(torch.randn(low_in_size*self._task_idx)*self._mlp_cfg["low_init_scaling"]))
                    low_acts_1.append(nn.Sequential(instantiate_class(module.LOW_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["low_dropout"])))
                    low_acts_2.append(nn.Sequential(instantiate_class(module.LOW_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["low_dropout"])))

                    # next layers
                    for i in range(1,len(self._mlp_cfg["low_hidden_sizes"])):

                        assert self._mlp_cfg["low_hidden_sizes"][i-1] == self._mlp_cfg["low_hidden_sizes"][i]
                        low_Ws.append(LinearBlock(
                            size=self._mlp_cfg["low_hidden_sizes"][i],
                            use_biases=True,
                            use_layer_norm=module.USE_LAYER_NORM,
                            activation_cfg=module.LOW_LAYER_ACT_CFG,
                            dropout=self._mlp_cfg["low_dropout"],
                            init_scaling=self._mlp_cfg["low_init_scaling"]
                        ))
                        low_Us.append(nn.Linear(self._mlp_cfg["low_hidden_sizes"][i],self._mlp_cfg["low_hidden_sizes"][i],False))
                        low_Vs.append(nn.Linear(self._mlp_cfg["low_hidden_sizes"][i-1]*self._task_idx,self._mlp_cfg["low_hidden_sizes"][i],False))
                        low_As.append(Parameter(torch.randn(self._mlp_cfg["low_hidden_sizes"][i-1]*self._task_idx)*self._mlp_cfg["low_init_scaling"]))
                        low_acts_1.append(nn.Sequential(instantiate_class(module.LOW_LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["low_dropout"])))
                        low_acts_2.append(nn.Identity())
                    
                    low_in_size = self._mlp_cfg["low_hidden_sizes"][-1]
                
                # output layer
                low_Ws.append(Linear(
                    input_size=low_in_size,
                    output_size=2*module.ACTION_SIZE,
                    use_biases=True,
                    use_layer_norm=False,
                    activation_cfg={"classname": "torch.nn.Identity"},
                    dropout=0.0,
                    init_scaling=self._mlp_cfg["low_init_scaling"]
                ))
                low_Us.append(nn.Linear(2*module.ACTION_SIZE,2*module.ACTION_SIZE,False))
                low_Vs.append(nn.Linear(low_in_size*self._task_idx,2*module.ACTION_SIZE,False,))
                low_As.append(Parameter(torch.randn(low_in_size*self._task_idx)*self._mlp_cfg["low_init_scaling"]))
                low_acts_1.append(instantiate_class(module.LOW_LAYER_ACT_CFG))
                low_acts_2.append(instantiate_class(module.LOW_OUTPUT_ACT_CFG))
            
            self._high_Ws[str(task_idx)] = nn.ModuleList(high_Ws).to(self._device)
            self._high_Us[str(task_idx)] = nn.ModuleList(high_Us).to(self._device)
            self._high_Vs[str(task_idx)] = nn.ModuleList(high_Vs).to(self._device)
            self._high_As[str(task_idx)] = nn.ParameterList(high_As).to(self._device)
            self._high_acts_1[str(task_idx)] = nn.ModuleList(high_acts_1).to(self._device)
            self._high_acts_2[str(task_idx)] = nn.ModuleList(high_acts_2).to(self._device)

            self._low_Ws[str(task_idx)] = nn.ModuleList(low_Ws).to(self._device)
            self._low_Us[str(task_idx)] = nn.ModuleList(low_Us).to(self._device)
            self._low_Vs[str(task_idx)] = nn.ModuleList(low_Vs).to(self._device)
            self._low_As[str(task_idx)] = nn.ParameterList(low_As).to(self._device)
            self._low_acts_1[str(task_idx)] = nn.ModuleList(low_acts_1).to(self._device)
            self._low_acts_2[str(task_idx)] = nn.ModuleList(low_acts_2).to(self._device)

            high_optimizer_class, high_optimizer_args = get_class(self._high_optimizer_cfg), get_arguments(self._high_optimizer_cfg)
            self._high_optimizer_mlp:Optimizer = high_optimizer_class(self.high_parameters(),**high_optimizer_args)
            self._high_scheduler_mlp:LRScheduler = None
            if not self._high_scheduler_cfg is None:
                high_scheduler_class, high_scheduler_args = get_class(self._high_scheduler_cfg), get_arguments(self._high_scheduler_cfg)
                self._high_scheduler_mlp = high_scheduler_class(self._high_optimizer_mlp,**high_scheduler_args)

            low_optimizer_class, low_optimizer_args = get_class(self._low_optimizer_cfg), get_arguments(self._low_optimizer_cfg)
            self._low_optimizer_mlp:Optimizer = low_optimizer_class(self.low_parameters(),**low_optimizer_args)
            self._low_scheduler_mlp:LRScheduler = None
            if not self._low_scheduler_cfg is None:
                low_scheduler_class, low_scheduler_args = get_class(self._low_scheduler_cfg), get_arguments(self._low_scheduler_cfg)
                self._low_scheduler_mlp = low_scheduler_class(self._low_optimizer_mlp,**low_scheduler_args)
        
        for idx in self._task_idxs:

            for param in self._high_Ws[str(idx)].parameters(): param.requires_grad = False
            for param in self._high_Us[str(idx)].parameters(): param.requires_grad = False
            for param in self._high_Vs[str(idx)].parameters(): param.requires_grad = False
            for param in self._high_As[str(idx)].parameters(): param.requires_grad = False
            for param in self._high_acts_1[str(idx)].parameters(): param.requires_grad = False
            for param in self._high_acts_2[str(idx)].parameters(): param.requires_grad = False
            
            for param in self._low_Ws[str(idx)].parameters(): param.requires_grad = False
            for param in self._low_Us[str(idx)].parameters(): param.requires_grad = False
            for param in self._low_Vs[str(idx)].parameters(): param.requires_grad = False
            for param in self._low_As[str(idx)].parameters(): param.requires_grad = False
            for param in self._low_acts_1[str(idx)].parameters(): param.requires_grad = False
            for param in self._low_acts_2[str(idx)].parameters(): param.requires_grad = False
        
        for param in self._high_Ws[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._high_Us[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._high_Vs[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._high_As[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._high_acts_1[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._high_acts_2[str(task_idx)].parameters(): param.requires_grad = True
        
        for param in self._low_Ws[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._low_Us[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._low_Vs[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._low_As[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._low_acts_1[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._low_acts_2[str(task_idx)].parameters(): param.requires_grad = True

        return self

    @torch.no_grad()
    def set_eval_mode(self, task_idx: int, stochastic: bool = False, **kwargs) -> "Policy":
        if task_idx in self._task_idxs: self._task_idx = task_idx
        else: self._task_idx = self._task_idxs[-1]
        self._high_Ws = self._high_Ws.eval()
        self._high_Us = self._high_Us.eval()
        self._high_Vs = self._high_Vs.eval()
        self._high_As = self._high_As.eval()
        self._high_acts_1 = self._high_acts_1.eval()
        self._high_acts_2 = self._high_acts_2.eval()
        self._low_Ws = self._low_Ws.eval()
        self._low_Us = self._low_Us.eval()
        self._low_Vs = self._low_Vs.eval()
        self._low_As = self._low_As.eval()
        self._low_acts_1 = self._low_acts_1.eval()
        self._low_acts_2 = self._low_acts_2.eval()
        self._stochastic = stochastic
        return self
    
    def _high_pnn_forward(self,inputs:torch.Tensor) -> torch.Tensor:
        
        idx = self._task_idx

        o_list:List[torch.Tensor] = [inputs for _ in range(idx+1)]

        for i in range(len(self._high_Ws[str(idx)])):

            new_o_list:List[torch.Tensor] = []

            for k in range(idx+1):
                o, current_input = 0.0, o_list[k]
                if k > 0:
                    prev_inputs = torch.cat(o_list[:k],dim=-1)
                    o = self._high_As[str(k)][i] * prev_inputs
                    o = self._high_Vs[str(k)][i](o)
                    o = self._high_acts_1[str(k)][i](o)
                    o = self._high_Us[str(k)][i](o)
                o += self._high_Ws[str(k)][i](current_input)
                o = self._high_acts_2[str(k)][i](o)
                new_o_list.append(o)

            o_list = new_o_list
        
        return o_list[self._task_idx]

    def _low_pnn_forward(self,inputs:torch.Tensor) -> torch.Tensor:

        idx = self._task_idx

        o_list:List[torch.Tensor] = [inputs for _ in range(idx+1)]

        for i in range(len(self._low_Ws[str(idx)])):

            new_o_list:List[torch.Tensor] = []

            for k in range(idx+1):
                o, current_input = 0.0, o_list[k]
                if k > 0:
                    prev_inputs = torch.cat(o_list[:k],dim=-1)
                    o = self._low_As[str(k)][i] * prev_inputs
                    o = self._low_Vs[str(k)][i](o)
                    o = self._low_acts_1[str(k)][i](o)
                    o = self._low_Us[str(k)][i](o)
                o += self._low_Ws[str(k)][i](current_input)
                o = self._low_acts_2[str(k)][i](o)
                new_o_list.append(o)

            o_list = new_o_list
        
        return o_list[self._task_idx]

    def _forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:

        idx = 0
        
        if isinstance(inputs,Frame): obs, pos, goal = inputs["observation"]["obs"], inputs["observation"]["pos"], inputs["observation"]["goal"]
        else: obs, pos, high_goal, low_goal = inputs["observation/obs"], inputs["observation/pos"], inputs["high_goal"], inputs["low_goal"]

        if "obs" in self._normalized_features: obs = self._normalizers["obs"](obs)
        if "pos" in self._normalized_features: pos = self._normalizers["pos"](pos)

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown

        if isinstance(inputs,Frame):

            if "goal" in self._normalized_features: goal = self._normalizers["goal"](goal)

            high_additional_features = module.additional_features(pos,goal)
            high_additional_features["obs"] = obs
            high_additional_features["pos"] = pos
            high_additional_features["goal"] = goal

            high_inputs = torch.cat([v for k,v in high_additional_features.items()],dim=-1)
            high_outputs = self._high_pnn_forward(high_inputs)

            waypoint_mean = high_outputs[:,:module.GOAL_SIZE]
            waypoint_log_std = high_outputs[:,module.GOAL_SIZE:2*module.GOAL_SIZE]
            waypoint_selection_probs = high_outputs[:,2*module.GOAL_SIZE:].sigmoid()

            waypoint_log_std = torch.clamp(module.LOW_STD_FACTOR * waypoint_log_std, module.LOW_LOG_STD_MIN, module.LOW_LOG_STD_MAX).exp()

            if self._stochastic: 
                waypoint = torch.normal(waypoint_mean,waypoint_log_std,generator=generator)
                waypoint_selection = torch.bernoulli(waypoint_selection_probs,generator=generator)

            else: 
                waypoint = waypoint_mean
                waypoint_selection = waypoint_selection_probs.round()
            
            waypoint = waypoint_selection * waypoint + (1 - waypoint_selection) * (goal - pos)

            low_goal = waypoint + pos

        else:

            if "goal" in self._normalized_features: high_goal = self._normalizers["goal"](high_goal)

            high_additional_features = module.additional_features(pos,high_goal)
            high_additional_features["obs"] = obs
            high_additional_features["pos"] = pos
            high_additional_features["goal"] = high_goal

            high_inputs = torch.cat([v for k,v in high_additional_features.items()],dim=-1)
            high_outputs = self._high_pnn_forward(high_inputs)
            
            waypoint_mean = high_outputs[:,:,:module.GOAL_SIZE]
            waypoint_log_std = high_outputs[:,:,module.GOAL_SIZE:2*module.GOAL_SIZE]
            waypoint_selection_probs = high_outputs[:,:,2*module.GOAL_SIZE:].sigmoid()

            waypoint_log_std = torch.clamp(module.LOW_STD_FACTOR * waypoint_log_std, module.LOW_LOG_STD_MIN, module.LOW_LOG_STD_MAX).exp()

            waypoint = low_goal - pos
        
        low_pos = pos * 0
        low_additional_features = module.additional_features(low_pos,waypoint)
        low_additional_features["obs"] = obs
        low_additional_features["pos"] = low_pos
        low_additional_features["waypoint"] = waypoint

        low_inputs = torch.cat([v for k,v in low_additional_features.items()],dim=-1)
        low_outputs = self._low_pnn_forward(low_inputs).chunk(2,dim=-1)
        
        return {
            "waypoint_mean": waypoint_mean,
            "waypoint_log_std": waypoint_log_std,
            "waypoint_selection_probs": waypoint_selection_probs,
            "low_mean": low_outputs[0],
            "low_log_std": low_outputs[1],
            "low_goal": low_goal
        }

    @torch.no_grad()
    def forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:
        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        outs = self._forward(inputs,generator,**kwargs)
        return module.forward(outs,self._stochastic,generator,**kwargs)

    @torch.no_grad()
    def parameters(self) -> Iterator[Parameter]:
        return chain(
            *[self._high_Ws[str_idx].parameters() for str_idx in self._high_Ws.keys()],
            *[self._high_Us[str_idx].parameters() for str_idx in self._high_Us.keys()],
            *[self._high_Vs[str_idx].parameters() for str_idx in self._high_Vs.keys()],
            *[self._high_As[str_idx].parameters() for str_idx in self._high_As.keys()],
            *[self._high_acts_1[str_idx].parameters() for str_idx in self._high_acts_1.keys()],
            *[self._high_acts_2[str_idx].parameters() for str_idx in self._high_acts_2.keys()],
            *[self._low_Ws[str_idx].parameters() for str_idx in self._low_Ws.keys()],
            *[self._low_Us[str_idx].parameters() for str_idx in self._low_Us.keys()],
            *[self._low_Vs[str_idx].parameters() for str_idx in self._low_Vs.keys()],
            *[self._low_As[str_idx].parameters() for str_idx in self._low_As.keys()],
            *[self._low_acts_1[str_idx].parameters() for str_idx in self._low_acts_1.keys()],
            *[self._low_acts_2[str_idx].parameters() for str_idx in self._low_acts_2.keys()]
        )
    
    @torch.no_grad()
    def high_parameters(self) -> Iterator[Parameter]:
        return chain(
            *[self._high_Ws[str_idx].parameters() for str_idx in self._high_Ws.keys()],
            *[self._high_Us[str_idx].parameters() for str_idx in self._high_Us.keys()],
            *[self._high_Vs[str_idx].parameters() for str_idx in self._high_Vs.keys()],
            *[self._high_As[str_idx].parameters() for str_idx in self._high_As.keys()],
            *[self._high_acts_1[str_idx].parameters() for str_idx in self._high_acts_1.keys()],
            *[self._high_acts_2[str_idx].parameters() for str_idx in self._high_acts_2.keys()]
        )
    
    @torch.no_grad()
    def low_parameters(self) -> Iterator[Parameter]:
        return chain(
            *[self._low_Ws[str_idx].parameters() for str_idx in self._low_Ws.keys()],
            *[self._low_Us[str_idx].parameters() for str_idx in self._low_Us.keys()],
            *[self._low_Vs[str_idx].parameters() for str_idx in self._low_Vs.keys()],
            *[self._low_As[str_idx].parameters() for str_idx in self._low_As.keys()],
            *[self._low_acts_1[str_idx].parameters() for str_idx in self._low_acts_1.keys()],
            *[self._low_acts_2[str_idx].parameters() for str_idx in self._low_acts_2.keys()]
        )

    @torch.no_grad()
    def inference_parameters(self) -> Iterator[Parameter]:
        return self.parameters()
    
    @torch.no_grad()
    def high_inference_parameters(self) -> Iterator[Parameter]:
        return self.high_parameters()

    @torch.no_grad()
    def low_inference_parameters(self) -> Iterator[Parameter]:
        return self.low_parameters()

    @torch.no_grad()
    def buffers(self) -> Iterator[torch.Tensor]:
        return chain(
            *[self._high_Ws[str_idx].buffers() for str_idx in self._high_Ws.keys()],
            *[self._high_Us[str_idx].buffers() for str_idx in self._high_Us.keys()],
            *[self._high_Vs[str_idx].buffers() for str_idx in self._high_Vs.keys()],
            *[self._high_As[str_idx].buffers() for str_idx in self._high_As.keys()],
            *[self._high_acts_1[str_idx].buffers() for str_idx in self._high_acts_1.keys()],
            *[self._high_acts_2[str_idx].buffers() for str_idx in self._high_acts_2.keys()],
            *[self._low_Ws[str_idx].buffers() for str_idx in self._low_Ws.keys()],
            *[self._low_Us[str_idx].buffers() for str_idx in self._low_Us.keys()],
            *[self._low_Vs[str_idx].buffers() for str_idx in self._low_Vs.keys()],
            *[self._low_As[str_idx].buffers() for str_idx in self._low_As.keys()],
            *[self._low_acts_1[str_idx].buffers() for str_idx in self._low_acts_1.keys()],
            *[self._low_acts_2[str_idx].buffers() for str_idx in self._low_acts_2.keys()]
        )
    
    @torch.no_grad()
    def high_buffers(self) -> Iterator[torch.Tensor]:
        return chain(
            *[self._high_Ws[str_idx].buffers() for str_idx in self._high_Ws.keys()],
            *[self._high_Us[str_idx].buffers() for str_idx in self._high_Us.keys()],
            *[self._high_Vs[str_idx].buffers() for str_idx in self._high_Vs.keys()],
            *[self._high_As[str_idx].buffers() for str_idx in self._high_As.keys()],
            *[self._high_acts_1[str_idx].buffers() for str_idx in self._high_acts_1.keys()],
            *[self._high_acts_2[str_idx].buffers() for str_idx in self._high_acts_2.keys()]
        )
    
    @torch.no_grad()
    def low_buffers(self) -> Iterator[torch.Tensor]:
        return chain(
            *[self._low_Ws[str_idx].buffers() for str_idx in self._low_Ws.keys()],
            *[self._low_Us[str_idx].buffers() for str_idx in self._low_Us.keys()],
            *[self._low_Vs[str_idx].buffers() for str_idx in self._low_Vs.keys()],
            *[self._low_As[str_idx].buffers() for str_idx in self._low_As.keys()],
            *[self._low_acts_1[str_idx].buffers() for str_idx in self._low_acts_1.keys()],
            *[self._low_acts_2[str_idx].buffers() for str_idx in self._low_acts_2.keys()]
        )

    @torch.no_grad()
    def inference_buffers(self) -> Iterator[torch.Tensor]:
        return self.buffers()
    
    @torch.no_grad()
    def high_inference_buffers(self) -> Iterator[torch.Tensor]:
        return self.high_buffers()
    
    @torch.no_grad()
    def low_inference_buffers(self) -> Iterator[torch.Tensor]:
        return self.low_buffers()
    
    def size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def high_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.high_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.high_buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def low_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.low_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.low_buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def inference_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.inference_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.inference_buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def high_inference_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.high_inference_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.high_inference_buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def low_inference_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.low_inference_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.low_inference_buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    @torch.no_grad()
    def to(self, device: Union[torch.device, str]) -> "MLP":
        super().to(device)
        self._normalizers = {k: v.to(device) for k,v in self._normalizers.items()}
        self._high_Ws = self._high_Ws.to(device)
        self._high_Us = self._high_Us.to(device)
        self._high_Vs = self._high_Vs.to(device)
        self._high_As = self._high_As.to(device)
        self._high_acts_1 = self._high_acts_1.to(device)
        self._high_acts_2 = self._high_acts_2.to(device)
        self._low_Ws = self._low_Ws.to(device)
        self._low_Us = self._low_Us.to(device)
        self._low_Vs = self._low_Vs.to(device)
        self._low_As = self._low_As.to(device)
        self._low_acts_1 = self._low_acts_1.to(device)
        self._low_acts_2 = self._low_acts_2.to(device)
        return self
    
    def update(
            self, 
            batch: Dict[str, torch.Tensor], 
            generator: torch.Generator, 
            mask: torch.Tensor,
            gradient_step: int,
            log_infos: bool = False,
            **kwargs) -> Dict[str, torch.Tensor]:

        # policy update
        ###############

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        outs = self._forward(batch,generator,**kwargs)
        losses = module.compute_loss(outs,batch,mask,log_infos=log_infos,**kwargs)

        # high policy update

        self._high_optimizer_mlp.zero_grad()
        losses["A1(Main)_high_policy_loss"].backward()
        self._high_optimizer_mlp.step()

        if not self._high_scheduler_mlp is None:
            try: self._high_scheduler_mlp.step()
            except: self._high_scheduler_mlp.step(gradient_step)
            if log_infos:
                losses["Z2(Infos)_high_learning_rate"] = torch.tensor(self._high_scheduler_mlp.get_last_lr())[0]
        
        # low policy update

        self._low_optimizer_mlp.zero_grad()
        losses["A1(Main)_low_policy_loss"].backward()
        self._low_optimizer_mlp.step()
        
        if not self._low_scheduler_mlp is None:
            try: self._low_scheduler_mlp.step()
            except: self._low_scheduler_mlp.step(gradient_step)
            if log_infos:
                losses["Z2(Infos)_low_learning_rate"] = torch.tensor(self._low_scheduler_mlp.get_last_lr())[0]

        # return losses
        ###############

        return losses
