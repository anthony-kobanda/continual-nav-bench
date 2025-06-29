import copy
import numpy as np
import torch
import torch.nn as nn

from .utils import godot_goal_amazeville
from .utils import godot_goal_simpletown
from itertools import chain
from offbench.core.data import Frame, Sampler
from offbench.core.policy import Policy
from offbench.utils.imports import get_class, get_arguments
from offbench.utils.pytorch.models.mlp import MLP as BaseMLP
from offbench.utils.pytorch.models.mlp import Subspace_ResidualMLP as MLP_CLASS
from offbench.utils.pytorch.utils.normalizer import AVAILABLE_NORMALIZERS, MinMax_Normalizer, Z_Normalizer
from omegaconf import DictConfig
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from typing import Any, Dict, Iterator, List, Tuple, Union



class MLP(Policy):

    modes: List[str] = ["ll", "l2"]
    
    def __init__(
        self,
        # general
        env:str,
        seed:int=None,
        # continual specific
        high_eps:float=0.1,
        low_eps:float=0.1,
        mode:str="l2",
        n_alphas:int=1024,
        n_batches:int=100,
        high_cosine_lambda:float=1.0,
        low_cosine_lambda:float=1.0,
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
        # low policy network & optimizer
        low_dropout:float=0.1,
        low_init_scaling:float=0.1,
        low_hidden_sizes:List[int]=[256,256],
        # optimizer
        optimizer_cfg:Union[DictConfig,Dict[str,Any]]={
            "classname": "torch.optim.Adam",
            "lr": 3e-4,
        },
        scheduler_cfg:Union[DictConfig,Dict[str,Any]]=None) -> None:

        super().__init__(seed=seed,device=device)
        
        assert mode in MLP.modes, f"mode must be in {MLP.modes}"

        # continual specific
        self._high_eps: float = high_eps
        self._low_eps: float = low_eps
        self._mode: str = mode
        self._n_alphas: int = n_alphas
        self._n_batches: int = n_batches
        self._high_cosine_lambda: float = high_cosine_lambda
        self._low_cosine_lambda: float = low_cosine_lambda
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

        # policy networks
        self._high_mlp: MLP_CLASS = None
        self._low_mlp: MLP_CLASS = None
        self._mlp_cfg: Dict[str, Any] = {
            "high_dropout": high_dropout,
            "high_init_scaling": high_init_scaling,
            "high_hidden_sizes": high_hidden_sizes,
            "low_dropout": low_dropout,
            "low_init_scaling": low_init_scaling,
            "low_hidden_sizes": low_hidden_sizes,
        }

        # policy optimizer
        self._optimizer_cfg: Union[DictConfig,Dict[str,Any]] = optimizer_cfg
        self._scheduler_cfg: Union[DictConfig,Dict[str,Any]] = scheduler_cfg

        # high anchors
        self._high_n_anchors: int = 0
        self._high_anchor_mode: str = "prev" # pick from ["prev","curr","best","random"]
        self._high_best_prev_anchors: Dict[str, torch.Tensor] = {}
        self._high_best_prev_scores: Dict[str, float] = {}
        self._high_best_curr_anchors: Dict[str, torch.Tensor] = {}
        self._high_best_curr_scores: Dict[str, float] = {}
        self._high_best_anchors: Dict[str, torch.Tensor] = {}
        self._high_best_scores: Dict[str, float] = {}

        # low anchors
        self._low_n_anchors: int = 0
        self._low_anchor_mode: str = "prev" # pick from ["prev","curr","best","random"]
        self._low_best_prev_anchors: Dict[str, torch.Tensor] = {}
        self._low_best_prev_scores: Dict[str, float] = {}
        self._low_best_curr_anchors: Dict[str, torch.Tensor] = {}
        self._low_best_curr_scores: Dict[str, float] = {}
        self._low_best_anchors: Dict[str, torch.Tensor] = {}
        self._low_best_scores: Dict[str, float] = {}
    
    def set_normalizers(self, normalize_values, **kwargs) -> "MLP":
        for k in self._normalized_features:
            if self._normalizer == "minmax": self._normalizers[k] = MinMax_Normalizer(normalize_values[k]["min"],normalize_values[k]["max"])
            elif self._normalizer == "z": self._normalizers[k] = Z_Normalizer(normalize_values[k]["mean"],normalize_values[k]["std"])
        return self

    def set_train_mode(self, task_idx:int, **kwargs) -> "Policy":

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        else: raise ValueError(f"Unsupported environment : {self._env_name}")

        self._task_idx = task_idx

        if not task_idx in self._task_idxs:

            self._task_idxs.append(task_idx)

            if len(self._task_idxs) == 1:

                # high network
                high_mlp_sizes = [module.OBS_SIZE] + self._mlp_cfg["high_hidden_sizes"] + [2 * module.GOAL_SIZE + 1]
                high_mlp = MLP_CLASS(
                    sizes=high_mlp_sizes,
                    use_biases=True,
                    use_layer_norm=module.USE_LAYER_NORM,
                    layer_activation_cfg=module.HIGH_LAYER_ACT_CFG,
                    output_activation_cfg=module.HIGH_OUTPUT_ACT_CFG,
                    dropout=self._mlp_cfg["high_dropout"],
                    init_scaling=self._mlp_cfg["high_init_scaling"]
                ).to(self._device)
                self._high_mlp = high_mlp

                # low network
                low_mlp_sizes = [module.OBS_SIZE] + self._mlp_cfg["low_hidden_sizes"] + [2 * module.ACTION_SIZE]
                low_mlp = MLP_CLASS(
                    sizes=low_mlp_sizes,
                    use_biases=True,
                    use_layer_norm=module.USE_LAYER_NORM,
                    layer_activation_cfg=module.LOW_LAYER_ACT_CFG,
                    output_activation_cfg=module.LOW_OUTPUT_ACT_CFG,
                    dropout=self._mlp_cfg["low_dropout"],
                    init_scaling=self._mlp_cfg["low_init_scaling"]
                ).to(self._device)
                self._low_mlp = low_mlp

            # high anchors

            self._high_n_anchors += 1
            self._high_mlp = self._high_mlp.add_anchor()

            if self._high_n_anchors == 1:
            
                high_prev_alpha = None
                high_curr_alpha = torch.tensor([1.0]).to(self._device)
                high_best_alpha = torch.tensor([1.0]).to(self._device)
            
            else:
            
                for k in self._high_best_prev_anchors:
                    if not self._high_best_prev_anchors[k] is None: 
                        self._high_best_prev_anchors[k] = torch.cat([self._high_best_prev_anchors[k],torch.tensor([0.0]).to(self._device)],dim=0)
                    self._high_best_curr_anchors[k] = torch.cat([self._high_best_curr_anchors[k],torch.tensor([0.0]).to(self._device)],dim=0)
                    self._high_best_anchors[k] = torch.cat([self._high_best_anchors[k],torch.tensor([0.0]).to(self._device)],dim=0)
            
                n = self._high_n_anchors
                high_prev_alpha = torch.tensor([1.0/(n-1)]*(n-1)+[0.0]).to(self._device)
                high_curr_alpha = torch.tensor([1.0/n]*n).to(self._device)
                high_best_alpha = torch.tensor([1.0/n]*n).to(self._device)
            
            # high alpha optim
            self._high_curr_alpha_score = nn.Parameter(torch.zeros(self._high_n_anchors),requires_grad=True).to(self._device)
            
            self._high_best_prev_anchors[task_idx] = high_prev_alpha
            self._high_best_curr_anchors[task_idx] = high_curr_alpha
            self._high_best_anchors[task_idx] = high_best_alpha

            # low anchors

            self._low_n_anchors += 1
            self._low_mlp = self._low_mlp.add_anchor()
            
            if self._low_n_anchors == 1:

                low_prev_alpha = None
                low_curr_alpha = torch.tensor([1.0]).to(self._device)
                low_best_alpha = torch.tensor([1.0]).to(self._device)

            else:

                for k in self._low_best_prev_anchors:
                    if not self._low_best_prev_anchors[k] is None: 
                        self._low_best_prev_anchors[k] = torch.cat([self._low_best_prev_anchors[k],torch.tensor([0.0]).to(self._device)],dim=0)
                    self._low_best_curr_anchors[k] = torch.cat([self._low_best_curr_anchors[k],torch.tensor([0.0]).to(self._device)],dim=0)
                    self._low_best_anchors[k] = torch.cat([self._low_best_anchors[k],torch.tensor([0.0]).to(self._device)],dim=0)
                
                n = self._low_n_anchors
                low_prev_alpha = torch.tensor([1.0/(n-1)]*(n-1)+[0.0]).to(self._device)
                low_curr_alpha = torch.tensor([1.0/n]*n).to(self._device)
                low_best_alpha = torch.tensor([1.0/n]*n).to(self._device)
            
            # low alpha optim
            self._low_curr_alpha_score = nn.Parameter(torch.zeros(self._low_n_anchors),requires_grad=True).to(self._device)
            
            self._low_best_prev_anchors[task_idx] = low_prev_alpha
            self._low_best_curr_anchors[task_idx] = low_curr_alpha
            self._low_best_anchors[task_idx] = low_best_alpha

            # optimizer
            optimizer_class, optimizer_args = get_class(self._optimizer_cfg), get_arguments(self._optimizer_cfg)
            self._optimizer_mlp:Optimizer = optimizer_class(self.parameters(),**optimizer_args)
            self._scheduler_mlp:LRScheduler = None
            if not self._scheduler_cfg is None:
                scheduler_class, scheduler_args = get_class(self._scheduler_cfg), get_arguments(self._scheduler_cfg)
                self._scheduler_mlp = scheduler_class(self._optimizer_mlp,**scheduler_args)
        
        self._high_mlp = self._high_mlp.train(only_last=True)
        self._low_mlp = self._low_mlp.train(only_last=True)
        self._high_curr_alpha_score.requires_grad = True
        self._low_curr_alpha_score.requires_grad = True

        return self

    @torch.no_grad()
    def set_eval_mode(self, task_idx: int, stochastic: bool = False, final: bool = False, **kwargs) -> "Policy":
        if task_idx in self._task_idxs: self._task_idx = task_idx
        else: self._task_idx = self._task_idxs[-1]
        self._high_mlp = self._high_mlp.eval()
        self._low_mlp = self._low_mlp.eval()
        self._stochastic = stochastic
        return self
    
    def generate_alpha(
        self, 
        batch_size:int, 
        alpha_mode:str, 
        unsqueeze_last:bool=False,
        generator:torch.Generator=None) -> torch.Tensor:

        if alpha_mode == "prev": 
            high_alpha = self._high_best_prev_anchors[self._task_idx]
            high_alpha = high_alpha.unsqueeze(-1).repeat(1,batch_size)
            low_alpha = self._low_best_prev_anchors[self._task_idx]
            low_alpha = low_alpha.unsqueeze(-1).repeat(1,batch_size)

        elif alpha_mode == "curr": 
            high_alpha = self._high_best_curr_anchors[self._task_idx]
            high_alpha = high_alpha.unsqueeze(-1).repeat(1,batch_size)
            low_alpha = self._low_best_curr_anchors[self._task_idx]
            low_alpha = low_alpha.unsqueeze(-1).repeat(1,batch_size)

        elif alpha_mode == "best": 
            high_alpha = self._high_best_anchors[self._task_idx]
            high_alpha = high_alpha.unsqueeze(-1).repeat(1,batch_size)
            low_alpha = self._low_best_anchors[self._task_idx]
            low_alpha = low_alpha.unsqueeze(-1).repeat(1,batch_size)

        elif alpha_mode == "optim":
            # high
            high_score_mean = torch.tanh(self._high_curr_alpha_score).unsqueeze(-1).repeat(1,batch_size)
            high_score_std = torch.tensor([0.1] * self._high_n_anchors).to(self._device).unsqueeze(-1).repeat(1,batch_size)
            high_score_dist = torch.distributions.normal.Normal(high_score_mean,high_score_std)
            high_score = high_score_dist.rsample() * self._high_n_anchors
            high_alpha = torch.softmax(high_score,dim=0)
            # low
            low_score_mean = torch.tanh(self._low_curr_alpha_score).unsqueeze(-1).repeat(1,batch_size)
            low_score_std = torch.tensor([0.1] * self._low_n_anchors).to(self._device).unsqueeze(-1).repeat(1,batch_size)
            low_score_dist = torch.distributions.normal.Normal(low_score_mean,low_score_std)
            low_score = low_score_dist.rsample() * self._low_n_anchors
            low_alpha = torch.softmax(low_score,dim=0)

        elif alpha_mode == "random_curr": 
            # we simulate dirichlet sampling use stick breaking (fast and efficient)
            # https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
            # high
            hu = torch.rand(self._high_n_anchors-1,batch_size)
            hu = torch.sort(hu,dim=0)[0]
            hu = torch.cat([torch.zeros(1,batch_size),hu,torch.ones(1,batch_size)],dim=0)
            hu = hu[1:] - hu[:-1]
            high_alpha = hu.to(self._device)
            # low
            lu = torch.rand(self._low_n_anchors-1,batch_size)
            lu = torch.sort(lu,dim=0)[0]
            lu = torch.cat([torch.zeros(1,batch_size),lu,torch.ones(1,batch_size)],dim=0)
            lu = lu[1:] - lu[:-1]
            low_alpha = lu.to(self._device)
        
        elif alpha_mode == "random_prev":
            # high
            if self._high_n_anchors > 2:
                hu = torch.rand(self._high_n_anchors-2,batch_size)
                hu = torch.sort(hu,dim=0)[0]
                hu = torch.cat([torch.zeros(1,batch_size),hu,torch.ones(1,batch_size)],dim=0)
                hu = hu[1:] - hu[:-1]
                hu = torch.cat([hu,torch.zeros(1,batch_size)],dim=0)
            else: hu = torch.tensor([1.0,0.0]).unsqueeze(-1).repeat(1,batch_size)
            high_alpha = hu.to(self._device)
            # low
            if self._low_n_anchors > 2:
                lu = torch.rand(self._low_n_anchors-2,batch_size)
                lu = torch.sort(lu,dim=0)[0]
                lu = torch.cat([torch.zeros(1,batch_size),lu,torch.ones(1,batch_size)],dim=0)
                lu = lu[1:] - lu[:-1]
                lu = torch.cat([lu,torch.zeros(1,batch_size)],dim=0)
            else: lu = torch.tensor([1.0,0.0]).unsqueeze(-1).repeat(1,batch_size)
            low_alpha = lu.to(self._device)
        
        if unsqueeze_last: 
            high_alpha = high_alpha.unsqueeze(-1)
            low_alpha = low_alpha.unsqueeze(-1)

        # for frames  : alpha of size (self._n_anchors,batch_size,1)
        # for batches : alpha of size (self._n_anchors,batch_size,1,1)
        return high_alpha.unsqueeze(-1), low_alpha.unsqueeze(-1)

    def _forward(
        self, 
        inputs: Union[Frame, Dict[str, torch.Tensor]], 
        generator: torch.Generator,
        alpha_mode: str,
        alphas:Tuple[torch.Tensor,torch.Tensor]=None,
        **kwargs) -> Dict[str, torch.Tensor]:

        idx = 0

        if isinstance(inputs,Frame): obs, pos, goal, unsqueeze_last = inputs["observation"]["obs"], inputs["observation"]["pos"], inputs["observation"]["goal"], False
        else: obs, pos, high_goal, low_goal, unsqueeze_last = inputs["observation/obs"], inputs["observation/pos"], inputs["high_goal"], inputs["low_goal"], True

        if alphas is None:
            batch_size = obs.size(0)
            high_alpha, low_alpha = self.generate_alpha(batch_size,alpha_mode,unsqueeze_last=unsqueeze_last)
        
        else: high_alpha, low_alpha = alphas

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
            high_outputs = self._high_mlp.forward(high_inputs,high_alpha)

            waypoint_mean = high_outputs[...,:module.GOAL_SIZE]
            waypoint_log_std = high_outputs[...,module.GOAL_SIZE:2*module.GOAL_SIZE]
            waypoint_selection_probs = high_outputs[...,2*module.GOAL_SIZE:].sigmoid()

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
            high_outputs = self._high_mlp.forward(high_inputs,high_alpha)
            
            waypoint_mean = high_outputs[...,:module.GOAL_SIZE]
            waypoint_log_std = high_outputs[...,module.GOAL_SIZE:2*module.GOAL_SIZE]
            waypoint_selection_probs = high_outputs[...,2*module.GOAL_SIZE:].sigmoid()

            waypoint_log_std = torch.clamp(module.LOW_STD_FACTOR * waypoint_log_std, module.LOW_LOG_STD_MIN, module.LOW_LOG_STD_MAX).exp()

            waypoint = low_goal - pos
        
        low_pos = pos * 0
        low_additional_features = module.additional_features(low_pos,waypoint)
        low_additional_features["obs"] = obs
        low_additional_features["pos"] = low_pos
        low_additional_features["waypoint"] = waypoint

        low_inputs = torch.cat([v for k,v in low_additional_features.items()],dim=-1)
        low_outputs = self._low_mlp.forward(low_inputs,low_alpha).chunk(2,dim=-1)
        
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
        outs = self._forward(inputs,generator,alpha_mode="best",**kwargs)
        return module.forward(outs,self._stochastic,generator,**kwargs)

    def parameters(self) -> Iterator[Parameter]:
        l = [self._high_curr_alpha_score,self._low_curr_alpha_score]
        return chain(
            self._high_mlp.parameters(),
            self._low_mlp.parameters(),
            l
        )
    
    def high_parameters(self) -> Iterator[Parameter]:
        return self._high_mlp.parameters()
    
    def low_parameters(self) -> Iterator[Parameter]:
        return self._low_mlp.parameters()

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
            self._high_mlp.buffers(),
            self._low_mlp.buffers()
        )
    
    @torch.no_grad()
    def high_buffers(self) -> Iterator[torch.Tensor]:
        return self._high_mlp.buffers()
    
    @torch.no_grad()
    def low_buffers(self) -> Iterator[torch.Tensor]:
        return self._low_mlp.buffers()

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
        if len(self._task_idxs) > 0:
            self._high_mlp = self._high_mlp.to(self._device)
            self._low_mlp = self._low_mlp.to(self._device)
            self._high_curr_alpha_score = self._high_curr_alpha_score.to(self._device)
            self._low_curr_alpha_score = self._low_curr_alpha_score.to(self._device)
        for k in self._high_best_prev_anchors:
            if not self._high_best_prev_anchors[k] is None: self._high_best_prev_anchors[k] = self._high_best_prev_anchors[k].to(self._device)
            if not self._high_best_curr_anchors[k] is None: self._high_best_curr_anchors[k] = self._high_best_curr_anchors[k].to(self._device)
            if not self._high_best_anchors[k] is None: self._high_best_anchors[k] = self._high_best_anchors[k].to(self._device)
        for k in self._low_best_prev_anchors:
            if not self._low_best_prev_anchors[k] is None: self._low_best_prev_anchors[k] = self._low_best_prev_anchors[k].to(self._device)
            if not self._low_best_curr_anchors[k] is None: self._low_best_curr_anchors[k] = self._low_best_curr_anchors[k].to(self._device)
            if not self._low_best_anchors[k] is None: self._low_best_anchors[k] = self._low_best_anchors[k].to(self._device)
        return self
    
    def update(
            self, 
            batch: Dict[str, torch.Tensor], 
            generator: torch.Generator, 
            mask: torch.Tensor,
            gradient_step: int,
            log_infos: bool = False,
            **kwargs) -> Dict[str, torch.Tensor]:

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown

        #################
        # policy update #
        #################
        outs = self._forward(batch,generator,alpha_mode="optim",**kwargs)
        if log_infos: losses = module.compute_loss(outs,batch,mask,log_infos=log_infos,compute_l1=True,compute_l2=True)
        else: losses = module.compute_loss(outs,batch,mask,log_infos=log_infos)

        # high policy loss
        ##################

        # high policy loss

        high_total_loss = losses["A1(Main)_high_policy_loss"]

        # high cosine similarity loss

        losses["A1(Main)_high_cosine_loss"] = self._high_mlp.cosine_similarity().abs() * self._high_cosine_lambda

        high_total_loss += losses["A1(Main)_high_cosine_loss"]

        # low policy loss
        #################

        # low policy loss

        low_total_loss = losses["A1(Main)_low_policy_loss"]

        # low cosine similarity loss

        losses["A1(Main)_low_cosine_loss"] = self._low_mlp.cosine_similarity().abs() * self._low_cosine_lambda

        low_total_loss += losses["A1(Main)_low_cosine_loss"]
        
        # all update
        ############

        total_loss = high_total_loss + low_total_loss

        self._optimizer_mlp.zero_grad()
        total_loss.backward()
        self._optimizer_mlp.step()

        if not self._scheduler_mlp is None:
            try: self._scheduler_mlp.step()
            except: self._scheduler_mlp.step(gradient_step)
            if log_infos:
                losses["Z2(Infos)_alpha_learning_rate"] = torch.tensor(self._scheduler_mlp.get_last_lr())[0]

        # return losses
        ###############

        return losses
    
    def high_score_alphas(self,alphas:Tuple[torch.Tensor,torch.Tensor],sampler:Sampler,generator:torch.Generator) -> torch.Tensor:

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown

        # alphas of size (self._n_anchors)

        batch_size: int = None

        scores = 0.0
        
        for _ in range(self._n_batches):

            with torch.no_grad():
                batch = sampler.sample_batch()
                mask = (~batch["is_padding"]).float()
                mask_sum = mask.sum(-1,keepdim=True)
                if torch.any(mask_sum == 0):
                    raise ValueError("All steps are masked for some samples in the batch")
                mask = (mask / mask_sum).unsqueeze(-1)
            
            if batch_size is None: 
                batch_size = mask.size(0)
                high_alphas = alphas[0].unsqueeze(-1).repeat(1,batch_size).unsqueeze(-1).unsqueeze(-1)
                low_alphas = alphas[1].unsqueeze(-1).repeat(1,batch_size).unsqueeze(-1).unsqueeze(-1)
                alphas = (high_alphas,low_alphas)

            outs = self._forward(batch,generator,alpha_mode="none",alphas=alphas)

            if self._mode == "ll":

                losses = module.compute_loss(outs,batch,mask,log_infos=False)
                high_loss = losses["A1(Main)_high_policy_loss"]

                scores -= high_loss
            
            elif self._mode == "l1":

                losses = module.compute_loss(outs,batch,mask,log_infos=False,compute_l1=True)
                high_loss = losses["A2(Infos)_high_l1_loss"]
                selection_loss = losses["A2(Infos)_high_selection_l1_loss"]

                scores -= (high_loss + selection_loss)
            
            elif self._mode == "l2":

                losses = module.compute_loss(outs,batch,mask,log_infos=False,compute_l2=True)
                high_loss = losses["A2(Infos)_high_l2_loss"]
                selection_loss = losses["A2(Infos)_high_selection_l2_loss"]

                scores -= (high_loss + selection_loss)

        # scores of size (self._n_alphas)

        scores /= self._n_batches

        scores = scores.unsqueeze(-1)

        return scores
    
    def low_score_alphas(self,alphas:Tuple[torch.Tensor,torch.Tensor],sampler:Sampler,generator:torch.Generator) -> torch.Tensor:

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown

        # alphas of size (self._n_anchors)

        batch_size: int = None

        scores = 0.0
        
        for _ in range(self._n_batches):

            with torch.no_grad():
                batch = sampler.sample_batch()
                mask = (~batch["is_padding"]).float()
                mask_sum = mask.sum(-1,keepdim=True)
                if torch.any(mask_sum == 0):
                    raise ValueError("All steps are masked for some samples in the batch")
                mask = (mask / mask_sum).unsqueeze(-1)
            
            if batch_size is None: 
                batch_size = mask.size(0)
                high_alphas = alphas[0].unsqueeze(-1).repeat(1,batch_size).unsqueeze(-1).unsqueeze(-1)
                low_alphas = alphas[1].unsqueeze(-1).repeat(1,batch_size).unsqueeze(-1).unsqueeze(-1)
                alphas = (high_alphas,low_alphas)

            outs = self._forward(batch,generator,alpha_mode="none",alphas=alphas)

            if self._mode == "ll":

                losses = module.compute_loss(outs,batch,mask,log_infos=False)
                low_loss = losses["A1(Main)_low_policy_loss"]

                scores -= low_loss
            
            elif self._mode == "l1":

                losses = module.compute_loss(outs,batch,mask,log_infos=False,compute_l1=True)
                low_loss = losses["A2(Infos)_low_l1_loss"]

                scores -= low_loss
            
            elif self._mode == "l2":

                losses = module.compute_loss(outs,batch,mask,log_infos=False,compute_l2=True)
                low_loss = losses["A2(Infos)_low_l2_loss"]

                scores -= low_loss

        scores /= self._n_batches

        scores = scores.unsqueeze(-1)

        return scores
    
    def high_update_alphas(self,sampler:Sampler,final:bool,generator:torch.Generator,gradient_step:int,log_infos:bool) -> None:

        not_use_tqdm = not log_infos
        
        if self._high_n_anchors == 1:
            
            self._high_best_prev_anchors[self._task_idx] = None
            self._high_best_curr_anchors[self._task_idx] = torch.tensor([1.0]).to(self._device)
            self._high_best_anchors[self._task_idx] = torch.tensor([1.0]).to(self._device)

            alphas = (self._high_best_anchors[self._task_idx],self._low_best_anchors[self._task_idx])
            score = round(self.high_score_alphas(alphas,sampler,generator)[0].item(),3)
            self._high_best_prev_scores[self._task_idx] = None
            self._high_best_curr_scores[self._task_idx] = score
            self._high_best_scores[self._task_idx] = score
            
            # prints

            tqdm.write("")
            
            for task_idx in self._task_idxs:

                tqdm.write(f"///// [HIGH] Task {task_idx+1} @ {gradient_step} : ")
                tqdm.write(f"///// ///// Best Alpha : {self._high_best_anchors[task_idx]}")
                tqdm.write(f"///// ///// Best Score : {self._high_best_scores[task_idx]}")

                if task_idx == self._task_idx:

                    tqdm.write(f"///// ///// ///// Prev Alpha : {self._high_best_prev_anchors[task_idx]}")
                    tqdm.write(f"///// ///// ///// Prev Score : {self._high_best_prev_scores[task_idx]}")
                    tqdm.write(f"///// ///// ///// Curr Alpha : {self._high_best_curr_anchors[task_idx]}")
                    tqdm.write(f"///// ///// ///// Curr Score : {self._high_best_curr_scores[task_idx]}")
                    tqdm.write(f"///// ///// ///// Optim Alpha : {torch.softmax(torch.tanh(self._high_curr_alpha_score)*self._high_n_anchors,dim=0).data}")
                
                tqdm.write("")

            return None
        
        else:

            # best prev alpha

            if self._high_n_anchors == 2:
                
                if not self._task_idx in self._high_best_prev_scores:
                    self._high_best_prev_anchors[self._task_idx] = torch.tensor([1.0,0.0]).to(self._device)
                    alphas = (self._high_best_prev_anchors[self._task_idx],self._low_best_anchors[self._task_idx])
                    self._high_best_prev_scores[self._task_idx] = round(self.high_score_alphas(alphas,sampler,generator)[0].item(),3)
            
            else:
                
                best_prev_alpha_score = - np.inf
                best_prev_alpha = None
                if self._task_idx in self._high_best_prev_scores:
                    best_prev_alpha_score = self._high_best_prev_scores[self._task_idx]
                    best_prev_alpha = self._high_best_prev_anchors[self._task_idx]
                candidates = self.generate_alpha(self._n_alphas,"random_prev",False)
                for i in tqdm(range(self._n_alphas),disable=not_use_tqdm,desc=f"Sampling high best prev alphas @ {gradient_step}"):
                    candidate = candidates[0][:,i].squeeze(-1), candidates[1][:,i].squeeze(-1)
                    candidate_score = self.high_score_alphas(candidate,sampler,generator)[0].item()
                    if candidate_score > best_prev_alpha_score:
                        best_prev_alpha_score = candidate_score
                        best_prev_alpha = candidate[0]
                self._high_best_prev_anchors[self._task_idx] = best_prev_alpha
                self._high_best_prev_scores[self._task_idx] = round(best_prev_alpha_score,3)

            # best curr alpha

            best_curr_alpha_score = - np.inf
            best_curr_alpha = None
            if self._task_idx in self._high_best_curr_scores:
                best_curr_alpha_score = self._high_best_curr_scores[self._task_idx]
                best_curr_alpha = self._high_best_curr_anchors[self._task_idx]
            candidates = self.generate_alpha(self._n_alphas,"random_curr",False)
            for i in tqdm(range(self._n_alphas),disable=not_use_tqdm,desc=f"Sampling high best curr alphas @ {gradient_step}"):
                candidate = candidates[0][:,i].squeeze(-1), candidates[1][:,i].squeeze(-1)
                candidate_score = self.high_score_alphas(candidate,sampler,generator)[0].item()
                if candidate_score > best_curr_alpha_score:
                    best_curr_alpha_score = candidate_score
                    best_curr_alpha = candidate[0]
            self._high_best_curr_anchors[self._task_idx] = best_curr_alpha
            self._high_best_curr_scores[self._task_idx] = round(best_curr_alpha_score,3)

            # compute best alphas

            prev_score, curr_score = self._high_best_prev_scores[self._task_idx], self._high_best_curr_scores[self._task_idx]
            
            if prev_score <=0 and curr_score <=0:

                if ( 1 - self._high_eps) * prev_score >= curr_score: keep_prev = True
                else: keep_prev = False

            elif prev_score >= 0 and curr_score >= 0:

                if ( 1 + self._high_eps) * prev_score >= curr_score: keep_prev = True
                else: keep_prev = False
            
            elif prev_score <= 0 and curr_score >= 0: keep_prev = False

            elif prev_score >= 0 and curr_score <= 0: keep_prev = True

            # update best alphas

            if keep_prev:

                self._high_best_anchors[self._task_idx] = copy.deepcopy(self._high_best_prev_anchors[self._task_idx])
                self._high_best_scores[self._task_idx] = self._high_best_prev_scores[self._task_idx]
            
                # prints

                tqdm.write("")

                for task_idx in self._task_idxs:

                    tqdm.write(f"///// [HIGH] Task {task_idx+1} @ {gradient_step} : ")
                    tqdm.write(f"///// ///// Best Alpha : {self._high_best_anchors[task_idx]}")
                    tqdm.write(f"///// ///// Best Score : {self._high_best_scores[task_idx]}")

                    if task_idx == self._task_idx:

                        tqdm.write(f"///// ///// ///// Prev Alpha : {self._high_best_prev_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Prev Score : {self._high_best_prev_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Alpha : {self._high_best_curr_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Score : {self._high_best_curr_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Optim Alpha : {torch.softmax(torch.tanh(self._high_curr_alpha_score)*self._high_n_anchors,dim=0).data}")

                    tqdm.write("")

                if final:

                    self._high_best_curr_anchors[self._task_idx] = copy.deepcopy(self._high_best_prev_anchors[self._task_idx])
                    self._high_mlp.remove_anchor()
                    for k in self._high_best_prev_anchors:
                        if not self._high_best_prev_anchors[k] is None: self._high_best_prev_anchors[k] = self._high_best_prev_anchors[k][:-1]
                        if not self._high_best_curr_anchors[k] is None: self._high_best_curr_anchors[k] = self._high_best_curr_anchors[k][:-1]
                        if not self._high_best_anchors[k] is None: self._high_best_anchors[k] = self._high_best_anchors[k][:-1]
                    self._high_n_anchors -= 1
            
            else:

                self._high_best_anchors[self._task_idx] = copy.deepcopy(self._high_best_curr_anchors[self._task_idx])
                self._high_best_scores[self._task_idx] = self._high_best_curr_scores[self._task_idx]
            
                # prints

                tqdm.write("")

                for task_idx in self._task_idxs:

                    tqdm.write(f"///// [HIGH] Task {task_idx+1} @ {gradient_step} : ")
                    tqdm.write(f"///// ///// Best Alpha : {self._high_best_anchors[task_idx]}")
                    tqdm.write(f"///// ///// Best Score : {self._high_best_scores[task_idx]}")

                    if task_idx == self._task_idx:

                        tqdm.write(f"///// ///// ///// Prev Alpha : {self._high_best_prev_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Prev Score : {self._high_best_prev_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Alpha : {self._high_best_curr_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Score : {self._high_best_curr_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Optim Alpha : {torch.softmax(torch.tanh(self._high_curr_alpha_score)*self._high_n_anchors,dim=0).data}")

                    tqdm.write("")
                    
            return None
    
    def low_update_alphas(self,sampler:Sampler,final:bool,generator:torch.Generator,gradient_step:int,log_infos:bool) -> None:

        not_use_tqdm = not log_infos
        
        if self._low_n_anchors == 1:
            
            self._low_best_prev_anchors[self._task_idx] = None
            self._low_best_curr_anchors[self._task_idx] = torch.tensor([1.0]).to(self._device)
            self._low_best_anchors[self._task_idx] = torch.tensor([1.0]).to(self._device)

            alphas = (self._high_best_anchors[self._task_idx],self._low_best_anchors[self._task_idx])
            score = round(self.low_score_alphas(alphas,sampler,generator)[0].item(),3)
            self._low_best_prev_scores[self._task_idx] = None
            self._low_best_curr_scores[self._task_idx] = score
            self._low_best_scores[self._task_idx] = score
            
            # prints

            tqdm.write("")
            
            for task_idx in self._task_idxs:

                tqdm.write(f"///// [LOW] Task {task_idx+1} @ {gradient_step} : ")
                tqdm.write(f"///// ///// Best Alpha : {self._low_best_anchors[task_idx]}")
                tqdm.write(f"///// ///// Best Score : {self._low_best_scores[task_idx]}")

                if task_idx == self._task_idx:

                    tqdm.write(f"///// ///// ///// Prev Alpha : {self._low_best_prev_anchors[task_idx]}")
                    tqdm.write(f"///// ///// ///// Prev Score : {self._low_best_prev_scores[task_idx]}")
                    tqdm.write(f"///// ///// ///// Curr Alpha : {self._low_best_curr_anchors[task_idx]}")
                    tqdm.write(f"///// ///// ///// Curr Score : {self._low_best_curr_scores[task_idx]}")
                    tqdm.write(f"///// ///// ///// Optim Alpha : {torch.softmax(torch.tanh(self._low_curr_alpha_score)*self._low_n_anchors,dim=0).data}")
                
                tqdm.write("")

            return None
        
        else:

            # best prev alpha

            if self._low_n_anchors == 2:
                
                if not self._task_idx in self._low_best_prev_scores:
                    self._low_best_prev_anchors[self._task_idx] = torch.tensor([1.0,0.0]).to(self._device)
                    alphas = (self._high_best_anchors[self._task_idx],self._low_best_prev_anchors[self._task_idx])
                    self._low_best_prev_scores[self._task_idx] = round(self.low_score_alphas(alphas,sampler,generator)[0].item(),3)
            
            else:
                
                best_prev_alpha_score = - np.inf
                best_prev_alpha = None
                if self._task_idx in self._low_best_prev_scores:
                    best_prev_alpha_score = self._low_best_prev_scores[self._task_idx]
                    best_prev_alpha = self._low_best_prev_anchors[self._task_idx]
                candidates = self.generate_alpha(self._n_alphas,"random_prev",False)
                for i in tqdm(range(self._n_alphas),disable=not_use_tqdm,desc=f"Sampling low best prev alphas @ {gradient_step}"):
                    candidate = candidates[0][:,i].squeeze(-1), candidates[1][:,i].squeeze(-1)
                    candidate_score = self.low_score_alphas(candidate,sampler,generator)[0].item()
                    if candidate_score > best_prev_alpha_score:
                        best_prev_alpha_score = candidate_score
                        best_prev_alpha = candidate[1]
                self._low_best_prev_anchors[self._task_idx] = best_prev_alpha
                self._low_best_prev_scores[self._task_idx] = round(best_prev_alpha_score,3)

            # best curr alpha

            best_curr_alpha_score = - np.inf
            best_curr_alpha = None
            if self._task_idx in self._low_best_curr_scores:
                best_curr_alpha_score = self._low_best_curr_scores[self._task_idx]
                best_curr_alpha = self._low_best_curr_anchors[self._task_idx]
            candidates = self.generate_alpha(self._n_alphas,"random_curr",False)
            for i in tqdm(range(self._n_alphas),disable=not_use_tqdm,desc=f"Sampling best curr alphas @ {gradient_step}"):
                candidate = candidates[0][:,i].squeeze(-1), candidates[1][:,i].squeeze(-1)
                candidate_score = self.low_score_alphas(candidate,sampler,generator)[0].item()
                if candidate_score > best_curr_alpha_score:
                    best_curr_alpha_score = candidate_score
                    best_curr_alpha = candidate[1]
            self._low_best_curr_anchors[self._task_idx] = best_curr_alpha
            self._low_best_curr_scores[self._task_idx] = round(best_curr_alpha_score,3)

            # compute best alphas

            prev_score, curr_score = self._low_best_prev_scores[self._task_idx], self._low_best_curr_scores[self._task_idx]
            
            if prev_score <=0 and curr_score <=0:

                if ( 1 - self._low_eps) * prev_score >= curr_score: keep_prev = True
                else: keep_prev = False

            elif prev_score >= 0 and curr_score >= 0:

                if ( 1 + self._low_eps) * prev_score >= curr_score: keep_prev = True
                else: keep_prev = False
            
            elif prev_score <= 0 and curr_score >= 0: keep_prev = False

            elif prev_score >= 0 and curr_score <= 0: keep_prev = True

            # update best alphas

            if keep_prev:

                self._low_best_anchors[self._task_idx] = copy.deepcopy(self._low_best_prev_anchors[self._task_idx])
                self._low_best_scores[self._task_idx] = self._low_best_prev_scores[self._task_idx]
            
                # prints

                tqdm.write("")

                for task_idx in self._task_idxs:

                    tqdm.write(f"///// [LOW] Task {task_idx+1} @ {gradient_step} : ")
                    tqdm.write(f"///// ///// Best Alpha : {self._low_best_anchors[task_idx]}")
                    tqdm.write(f"///// ///// Best Score : {self._low_best_scores[task_idx]}")

                    if task_idx == self._task_idx:

                        tqdm.write(f"///// ///// ///// Prev Alpha : {self._low_best_prev_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Prev Score : {self._low_best_prev_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Alpha : {self._low_best_curr_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Score : {self._low_best_curr_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Optim Alpha : {torch.softmax(torch.tanh(self._low_curr_alpha_score)*self._low_n_anchors,dim=0).data}")

                    tqdm.write("")

                if final:

                    self._low_best_curr_anchors[self._task_idx] = copy.deepcopy(self._low_best_prev_anchors[self._task_idx])
                    self._low_mlp.remove_anchor()
                    for k in self._low_best_prev_anchors:
                        if not self._low_best_prev_anchors[k] is None: self._low_best_prev_anchors[k] = self._low_best_prev_anchors[k][:-1]
                        if not self._low_best_curr_anchors[k] is None: self._low_best_curr_anchors[k] = self._low_best_curr_anchors[k][:-1]
                        if not self._low_best_anchors[k] is None: self._low_best_anchors[k] = self._low_best_anchors[k][:-1]
                    self._low_n_anchors -= 1
            
            else:

                self._low_best_anchors[self._task_idx] = copy.deepcopy(self._low_best_curr_anchors[self._task_idx])
                self._low_best_scores[self._task_idx] = self._low_best_curr_scores[self._task_idx]
            
                # prints

                tqdm.write("")

                for task_idx in self._task_idxs:

                    tqdm.write(f"///// [LOW] Task {task_idx+1} @ {gradient_step} : ")
                    tqdm.write(f"///// ///// Best Alpha : {self._low_best_anchors[task_idx]}")
                    tqdm.write(f"///// ///// Best Score : {self._low_best_scores[task_idx]}")

                    if task_idx == self._task_idx:

                        tqdm.write(f"///// ///// ///// Prev Alpha : {self._low_best_prev_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Prev Score : {self._low_best_prev_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Alpha : {self._low_best_curr_anchors[task_idx]}")
                        tqdm.write(f"///// ///// ///// Curr Score : {self._low_best_curr_scores[task_idx]}")
                        tqdm.write(f"///// ///// ///// Optim Alpha : {torch.softmax(torch.tanh(self._low_curr_alpha_score)*self._low_n_anchors,dim=0).data}")

                    tqdm.write("")
                    
            return None

    def update_alphas(self,sampler:Sampler,final:bool,generator:torch.Generator,gradient_step:int,log_infos:bool) -> None:
        self.high_update_alphas(sampler,final,generator,gradient_step,log_infos)
        self.low_update_alphas(sampler,final,generator,gradient_step,log_infos)
