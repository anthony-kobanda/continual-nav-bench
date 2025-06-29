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
        # all networks
        device:Union[torch.device,str]="cpu",
        normalizer:str="z",
        normalized_features:List[str]=[],
        # policy network & optimizer
        dropout:float=0.1,
        init_scaling:float=0.1,
        hidden_sizes:List[int]=[256,256],
        optimizer_cfg:Union[DictConfig,Dict[str,Any]]={
            "classname": "torch.optim.Adam",
            "lr": 3e-4,
        },
        scheduler_cfg:Union[DictConfig,Dict[str,Any]]=None) -> None:

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

        self._env_name = env.split("-")[0]

        # policy networks
        self._Ws = nn.ModuleDict({}).to(self._device)
        self._Us = nn.ModuleDict({}).to(self._device)
        self._Vs = nn.ModuleDict({}).to(self._device)
        self._As = nn.ModuleDict({}).to(self._device)
        self._acts_1 = nn.ModuleDict({}).to(self._device)
        self._acts_2 = nn.ModuleDict({}).to(self._device)
        self._mlp_cfg: Dict[str, Any] = {
            "dropout": dropout,
            "init_scaling": init_scaling,
            "hidden_sizes": hidden_sizes,
        }

        # policy optimizer
        self._optimizer_cfg: Union[DictConfig,Dict[str,Any]] = optimizer_cfg
        self._scheduler_cfg: Union[DictConfig,Dict[str,Any]] = scheduler_cfg
    
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

            Ws: List[nn.Module] = []
            Us: List[nn.Module] = []
            Vs: List[nn.Module] = []
            As: List[nn.Parameter] = []
            acts_1: List[nn.Module] = []
            acts_2: List[nn.Module] = []

            in_size = module.OBS_SIZE

            if len(self._task_idxs) == 1:

                # hidden layers
                if len(self._mlp_cfg["hidden_sizes"]) > 0:

                    # first layer
                    Ws.append(Linear(
                        input_size=in_size,
                        output_size=self._mlp_cfg["hidden_sizes"][0],
                        use_biases=True,
                        use_layer_norm=module.USE_LAYER_NORM,
                        activation_cfg={"classname": "torch.nn.Identity"},
                        dropout=0.0,
                        init_scaling=self._mlp_cfg["init_scaling"] 
                    ))
                    Us.append(None)
                    Vs.append(None)
                    As.append(None)
                    acts_1.append(None)
                    acts_2.append(nn.Sequential(instantiate_class(module.LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["dropout"])))

                    # next layers
                    for i in range(1,len(self._mlp_cfg["hidden_sizes"])):

                        if self._env_name in ["amazeville","simpletown","antmaze","pointmaze"]:
                            assert self._mlp_cfg["hidden_sizes"][i-1] == self._mlp_cfg["hidden_sizes"][i]
                            Ws.append(LinearBlock(
                                size=self._mlp_cfg["hidden_sizes"][i],
                                use_biases=True,
                                use_layer_norm=module.USE_LAYER_NORM,
                                activation_cfg=module.LAYER_ACT_CFG,
                                dropout=self._mlp_cfg["dropout"],
                                init_scaling=self._mlp_cfg["init_scaling"]
                            ))
                        
                        else:
                            Ws.append(Linear(
                                input_size=self._mlp_cfg["hidden_sizes"][i-1],
                                output_size=self._mlp_cfg["hidden_sizes"][i],
                                use_biases=True,
                                use_layer_norm=module.USE_LAYER_NORM,
                                activation_cfg=module.LAYER_ACT_CFG,
                                dropout=self._mlp_cfg["dropout"],
                                init_scaling=self._mlp_cfg["init_scaling"]
                            ))
                        
                        Us.append(None)
                        Vs.append(None)
                        As.append(None)
                        acts_1.append(None)
                        acts_2.append(nn.Identity())
                    
                    in_size = self._mlp_cfg["hidden_sizes"][-1]
                
                # output layer
                Ws.append(Linear(
                    input_size=in_size,
                    output_size=2*module.ACTION_SIZE,
                    use_biases=True,
                    use_layer_norm=False,
                    activation_cfg={"classname": "torch.nn.Identity"},
                    dropout=0.0,
                    init_scaling=self._mlp_cfg["init_scaling"]
                ))
                Us.append(None)
                Vs.append(None)
                As.append(None)
                acts_1.append(None)
                acts_2.append(instantiate_class(module.OUTPUT_ACT_CFG))
            
            else:

                # hidden layers
                if len(self._mlp_cfg["hidden_sizes"]) > 0:
                    Ws.append(Linear(
                        input_size=in_size,
                        output_size=self._mlp_cfg["hidden_sizes"][0],
                        use_biases=True,
                        use_layer_norm=module.USE_LAYER_NORM,
                        activation_cfg={"classname": "torch.nn.Identity"},
                        dropout=0.0,
                        init_scaling=self._mlp_cfg["init_scaling"]
                    ))
                    Us.append(nn.Linear(self._mlp_cfg["hidden_sizes"][0],self._mlp_cfg["hidden_sizes"][0],False))
                    Vs.append(nn.Linear(in_size*self._task_idx,self._mlp_cfg["hidden_sizes"][0],False))
                    As.append(Parameter(torch.randn(in_size*self._task_idx)*self._mlp_cfg["init_scaling"]))
                    acts_1.append(nn.Sequential(instantiate_class(module.LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["dropout"])))
                    acts_2.append(nn.Sequential(instantiate_class(module.LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["dropout"])))

                    # next layers
                    for i in range(1,len(self._mlp_cfg["hidden_sizes"])):

                        if self._env_name in ["amazeville","simpletown","antmaze","pointmaze"]:
                            assert self._mlp_cfg["hidden_sizes"][i-1] == self._mlp_cfg["hidden_sizes"][i]
                            Ws.append(LinearBlock(
                                size=self._mlp_cfg["hidden_sizes"][i],
                                use_biases=True,
                                use_layer_norm=module.USE_LAYER_NORM,
                                activation_cfg=module.LAYER_ACT_CFG,
                                dropout=self._mlp_cfg["dropout"],
                                init_scaling=self._mlp_cfg["init_scaling"]
                            ))

                        else:
                            Ws.append(Linear(
                                input_size=self._mlp_cfg["hidden_sizes"][i-1],
                                output_size=self._mlp_cfg["hidden_sizes"][i],
                                use_biases=True,
                                use_layer_norm=module.USE_LAYER_NORM,
                                activation_cfg=module.LAYER_ACT_CFG,
                                dropout=self._mlp_cfg["dropout"],
                                init_scaling=self._mlp_cfg["init_scaling"]
                            ))

                        Us.append(nn.Linear(self._mlp_cfg["hidden_sizes"][i],self._mlp_cfg["hidden_sizes"][i],False))
                        Vs.append(nn.Linear(self._mlp_cfg["hidden_sizes"][i-1]*self._task_idx,self._mlp_cfg["hidden_sizes"][i],False))
                        As.append(Parameter(torch.randn(self._mlp_cfg["hidden_sizes"][i-1]*self._task_idx)*self._mlp_cfg["init_scaling"]))
                        acts_1.append(nn.Sequential(instantiate_class(module.LAYER_ACT_CFG),nn.Dropout(self._mlp_cfg["dropout"])))
                        acts_2.append(nn.Identity())
                    
                    in_size = self._mlp_cfg["hidden_sizes"][-1]
                
                # output layer
                Ws.append(Linear(
                    input_size=in_size,
                    output_size=2*module.ACTION_SIZE,
                    use_biases=True,
                    use_layer_norm=False,
                    activation_cfg={"classname": "torch.nn.Identity"},
                    dropout=0.0,
                    init_scaling=self._mlp_cfg["init_scaling"]
                ))
                Us.append(nn.Linear(2*module.ACTION_SIZE,2*module.ACTION_SIZE,False))
                Vs.append(nn.Linear(in_size*self._task_idx,2*module.ACTION_SIZE,False,))
                As.append(Parameter(torch.randn(in_size*self._task_idx)*self._mlp_cfg["init_scaling"]))
                acts_1.append(instantiate_class(module.LAYER_ACT_CFG))
                acts_2.append(instantiate_class(module.OUTPUT_ACT_CFG))

            self._Ws[str(task_idx)] = nn.ModuleList(Ws).to(self._device)
            self._Us[str(task_idx)] = nn.ModuleList(Us).to(self._device)
            self._Vs[str(task_idx)] = nn.ModuleList(Vs).to(self._device)
            self._As[str(task_idx)] = nn.ParameterList(As).to(self._device)
            self._acts_1[str(task_idx)] = nn.ModuleList(acts_1).to(self._device)
            self._acts_2[str(task_idx)] = nn.ModuleList(acts_2).to(self._device)

            optimizer_class, optimizer_args = get_class(self._optimizer_cfg), get_arguments(self._optimizer_cfg)
            self._optimizer_mlp:Optimizer = optimizer_class(self.parameters(),**optimizer_args)
            self._scheduler_mlp:LRScheduler = None
            if not self._scheduler_cfg is None:
                scheduler_class, scheduler_args = get_class(self._scheduler_cfg), get_arguments(self._scheduler_cfg)
                self._scheduler_mlp = scheduler_class(self._optimizer_mlp,**scheduler_args)
        
        for idx in self._task_idxs:
            
            for param in self._Ws[str(idx)].parameters(): param.requires_grad = False
            for param in self._Us[str(idx)].parameters(): param.requires_grad = False
            for param in self._Vs[str(idx)].parameters(): param.requires_grad = False
            for param in self._As[str(idx)].parameters(): param.requires_grad = False
            for param in self._acts_1[str(idx)].parameters(): param.requires_grad = False
            for param in self._acts_2[str(idx)].parameters(): param.requires_grad = False
        
        for param in self._Ws[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._Us[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._Vs[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._As[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._acts_1[str(task_idx)].parameters(): param.requires_grad = True
        for param in self._acts_2[str(task_idx)].parameters(): param.requires_grad = True

        return self

    @torch.no_grad()
    def set_eval_mode(self, task_idx: int, stochastic: bool = False, **kwargs) -> "Policy":
        if task_idx in self._task_idxs: self._task_idx = task_idx
        else: self._task_idx = self._task_idxs[-1]
        self._Ws = self._Ws.eval()
        self._Us = self._Us.eval()
        self._Vs = self._Vs.eval()
        self._As = self._As.eval()
        self._acts_1 = self._acts_1.eval()
        self._acts_2 = self._acts_2.eval()
        self._stochastic = stochastic
        return self
    
    def _pnn_forward(self,inputs:torch.Tensor,evaluation:bool) -> torch.Tensor:
        
        idx = self._task_idx

        o_list:List[torch.Tensor] = [inputs for _ in range(idx+1)]

        for i in range(len(self._Ws[str(idx)])):

            new_o_list:List[torch.Tensor] = []

            for k in range(idx+1):
                o, current_input = 0.0, o_list[k]
                if k > 0:
                    prev_inputs = torch.cat(o_list[:k],dim=-1)
                    o = self._As[str(k)][i] * prev_inputs
                    o = self._Vs[str(k)][i](o)
                    o = self._acts_1[str(k)][i](o)
                    o = self._Us[str(k)][i](o)
                o += self._Ws[str(k)][i](current_input)
                o = self._acts_2[str(k)][i](o)
                new_o_list.append(o)

            o_list = new_o_list
        
        return o_list[self._task_idx]
    
    def _forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(inputs,Frame): inputs, evaluation = {k: inputs["observation"][k] for k in inputs["observation"]}, True
        else: inputs, evaluation = {k.split("/")[1]: inputs[k] for k in inputs if k.startswith("observation/")}, False
        for k in self._normalizers: inputs[k] = self._normalizers[k](inputs[k])
        if self._env_name in ["amazeville","simpletown","antmaze","pointmaze"]:
            if self._env_name == "amazeville": module = godot_goal_amazeville
            elif self._env_name == "simpletown": module = godot_goal_simpletown
            additional_features = module.additional_features(inputs["pos"],inputs["goal"])
            for k in additional_features: inputs[k] = additional_features[k]
        inputs = torch.cat([inputs[k] for k in inputs],dim=-1)
        outputs = self._pnn_forward(inputs,evaluation).chunk(2,dim=-1)
        return {"mean": outputs[0], "log_std": outputs[1]}

    @torch.no_grad()
    def forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:
        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        outs = self._forward(inputs,generator,**kwargs)
        return module.forward(outs,self._stochastic,generator,**kwargs)

    @torch.no_grad()
    def parameters(self) -> Iterator[Parameter]:
        return chain(
            *[self._Ws[str_idx].parameters() for str_idx in self._Ws.keys()],
            *[self._Us[str_idx].parameters() for str_idx in self._Us.keys()],
            *[self._Vs[str_idx].parameters() for str_idx in self._Vs.keys()],
            *[self._As[str_idx].parameters() for str_idx in self._As.keys()],
            *[self._acts_1[str_idx].parameters() for str_idx in self._acts_1.keys()],
            *[self._acts_2[str_idx].parameters() for str_idx in self._acts_2.keys()],
        )


    @torch.no_grad()
    def inference_parameters(self) -> Iterator[Parameter]:
        return self.parameters()

    @torch.no_grad()
    def buffers(self) -> Iterator[torch.Tensor]:
        return chain(
            *[self._Ws[str_idx].buffers() for str_idx in self._Ws.keys()],
            *[self._Us[str_idx].buffers() for str_idx in self._Us.keys()],
            *[self._Vs[str_idx].buffers() for str_idx in self._Vs.keys()],
            *[self._As[str_idx].buffers() for str_idx in self._As.keys()],
            *[self._acts_1[str_idx].buffers() for str_idx in self._acts_1.keys()],
            *[self._acts_2[str_idx].buffers() for str_idx in self._acts_2.keys()],
        )

    @torch.no_grad()
    def inference_buffers(self) -> Iterator[torch.Tensor]:
        return self.buffers()

    @torch.no_grad()
    def to(self, device: Union[torch.device, str]) -> "MLP":
        super().to(device)
        self._normalizers = {k: v.to(device) for k,v in self._normalizers.items()}
        self._Ws = self._Ws.to(self._device)
        self._Us = self._Us.to(self._device)
        self._Vs = self._Vs.to(self._device)
        self._As = self._As.to(self._device)
        self._acts_1 = self._acts_1.to(self._device)
        self._acts_2 = self._acts_2.to(self._device)
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

        self._optimizer_mlp.zero_grad()
        losses["A1(Main)_policy_loss"].backward()
        self._optimizer_mlp.step()

        if not self._scheduler_mlp is None:
            try: self._scheduler_mlp.step()
            except: self._scheduler_mlp.step(gradient_step)
            if log_infos:
                losses["Z2(Infos)_learning_rate"] = torch.tensor(self._scheduler_mlp.get_last_lr())[0]

        # return losses
        ###############

        return losses
