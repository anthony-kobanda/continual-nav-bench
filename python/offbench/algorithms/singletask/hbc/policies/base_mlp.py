import torch

from .utils import godot_goal_amazeville
from .utils import godot_goal_simpletown
from itertools import chain
from offbench.core.data import Frame
from offbench.core.policy import Policy
from offbench.utils.imports import get_class, get_arguments
from offbench.utils.pytorch.models.mlp import ResidualMLP as MLP_CLASS
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
        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        else: raise ValueError(f"Unsupported environment : {env}")

        # high policy network
        high_mlp_sizes = [module.OBS_SIZE] + high_hidden_sizes + [2 * module.GOAL_SIZE + 1]
        self._high_mlp = MLP_CLASS(
            sizes=high_mlp_sizes,
            use_biases=True,
            use_layer_norm=module.USE_LAYER_NORM,
            layer_activation_cfg=module.HIGH_LAYER_ACT_CFG,
            output_activation_cfg=module.HIGH_OUTPUT_ACT_CFG,
            dropout=high_dropout,
            init_scaling=high_init_scaling
        )

        # high_policy optimizer
        high_optimizer_class, high_optimizer_args = get_class(high_optimizer_cfg), get_arguments(high_optimizer_cfg)
        self._high_optimizer_mlp:Optimizer = high_optimizer_class(self._high_mlp.parameters(),**high_optimizer_args)
        self._high_scheduler_mlp:LRScheduler = None
        if not high_scheduler_cfg is None:
            high_scheduler_class, high_scheduler_args = get_class(high_scheduler_cfg), get_arguments(high_scheduler_cfg)
            self._high_scheduler_mlp = high_scheduler_class(self._high_optimizer_mlp,**high_scheduler_args)
        
        # low policy network
        low_mlp_sizes = [module.OBS_SIZE] + low_hidden_sizes + [2 * module.ACTION_SIZE]
        self._low_mlp = MLP_CLASS(
            sizes=low_mlp_sizes,
            use_biases=True,
            use_layer_norm=module.USE_LAYER_NORM,
            layer_activation_cfg=module.LOW_LAYER_ACT_CFG,
            output_activation_cfg=module.LOW_OUTPUT_ACT_CFG,
            dropout=low_dropout,
            init_scaling=low_init_scaling
        )

        # low_policy optimizer
        low_optimizer_class, low_optimizer_args = get_class(low_optimizer_cfg), get_arguments(low_optimizer_cfg)
        self._low_optimizer_mlp:Optimizer = low_optimizer_class(self._low_mlp.parameters(),**low_optimizer_args)
        self._low_scheduler_mlp:LRScheduler = None
        if not low_scheduler_cfg is None:
            low_scheduler_class, low_scheduler_args = get_class(low_scheduler_cfg), get_arguments(low_scheduler_cfg)
            self._low_scheduler_mlp = low_scheduler_class(self._low_optimizer_mlp,**low_scheduler_args)
    
    def set_normalizers(self, normalize_values, **kwargs) -> "MLP":
        for k in self._normalized_features:
            if self._normalizer == "minmax": self._normalizers[k] = MinMax_Normalizer(normalize_values[k]["min"],normalize_values[k]["max"])
            elif self._normalizer == "z": self._normalizers[k] = Z_Normalizer(normalize_values[k]["mean"],normalize_values[k]["std"])
        return self

    def set_train_mode(self, **kwargs) -> "Policy":
        self._high_mlp = self._high_mlp.train()
        self._low_mlp = self._low_mlp.train()
        return self

    @torch.no_grad()
    def set_eval_mode(self, stochastic: bool = False, **kwargs) -> "Policy":
        self._high_mlp = self._high_mlp.eval()
        self._low_mlp = self._low_mlp.eval()
        self._stochastic = stochastic
        return self

    def _forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:
        
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
            high_outputs = self._high_mlp.forward(high_inputs)

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
            high_outputs = self._high_mlp.forward(high_inputs)
            
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
        low_outputs = self._low_mlp.forward(low_inputs).chunk(2,dim=-1)
        
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
            self._high_mlp.parameters(),
            self._low_mlp.parameters()
        )

    @torch.no_grad()
    def inference_parameters(self) -> Iterator[Parameter]:
        return self.parameters()

    @torch.no_grad()
    def buffers(self) -> Iterator[torch.Tensor]:
        return chain(
            self._high_mlp.buffers(),
            self._low_mlp.buffers()
        )

    @torch.no_grad()
    def inference_buffers(self) -> Iterator[torch.Tensor]:
        return self.buffers()

    @torch.no_grad()
    def to(self, device: Union[torch.device, str]) -> "MLP":
        super().to(device)
        self._normalizers = {k: v.to(device) for k,v in self._normalizers.items()}
        self._high_mlp = self._high_mlp.to(self._device)
        self._low_mlp = self._low_mlp.to(self._device)
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
