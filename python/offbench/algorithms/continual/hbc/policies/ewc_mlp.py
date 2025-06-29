import copy
import torch

from .utils import godot_goal_amazeville
from .utils import godot_goal_simpletown
from itertools import chain
from offbench.core.data import Frame
from offbench.core.policy import Policy
from offbench.utils.imports import get_class, get_arguments
from offbench.utils.pytorch.models.mlp import MLP_DICT
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
        # continual specific
        ewc_lambda:float=1.0,
        ewc_update_freq:int=1,
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
        self._ewc_lambda: float = ewc_lambda
        self._ewc_update_freq: int = ewc_update_freq
        self._task_idx: int = None
        self._task_idxs: List[int] = []
        self._high_fisher_dict: Dict[int,Dict[str,torch.Tensor]] = {}
        self._high_optpar_dict: Dict[int,Dict[str,torch.Tensor]] = {}
        self._low_fisher_dict: Dict[int,Dict[str,torch.Tensor]] = {}
        self._low_optpar_dict: Dict[int,Dict[str,torch.Tensor]] = {}

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
        self._high_mlps: MLP_DICT = MLP_DICT({})
        self._low_mlps: MLP_DICT = MLP_DICT({})
        self._mlp_cfg: Dict[str, Any] = {
            "high_dropout": high_dropout,
            "high_init_scaling": high_init_scaling,
            "high_hidden_sizes": high_hidden_sizes,
            "low_dropout": low_dropout,
            "low_init_scaling": low_init_scaling,
            "low_hidden_sizes": low_hidden_sizes,
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

            if len(self._task_idxs) == 1:

                if self._env_name == "amazeville": module = godot_goal_amazeville
                elif self._env_name == "simpletown": module = godot_goal_simpletown
                else: raise ValueError(f"Unsupported environment : {self._env_name}")

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

                self._high_mlps[0] = high_mlp
                self._low_mlps[0] = low_mlp

            high_optimizer_class, high_optimizer_args = get_class(self._high_optimizer_cfg), get_arguments(self._high_optimizer_cfg)
            self._high_optimizer_mlp:Optimizer = high_optimizer_class(self._high_mlps.parameters(0),**high_optimizer_args)
            self._high_scheduler_mlp:LRScheduler = None
            if not self._high_scheduler_cfg is None:
                high_scheduler_class, high_scheduler_args = get_class(self._high_scheduler_cfg), get_arguments(self._high_scheduler_cfg)
                self._high_scheduler_mlp = high_scheduler_class(self._high_optimizer_mlp,**high_scheduler_args)

            low_optimizer_class, low_optimizer_args = get_class(self._low_optimizer_cfg), get_arguments(self._low_optimizer_cfg)
            self._low_optimizer_mlp:Optimizer = low_optimizer_class(self._low_mlps.parameters(0),**low_optimizer_args)
            self._low_scheduler_mlp:LRScheduler = None
            if not self._low_scheduler_cfg is None:
                low_scheduler_class, low_scheduler_args = get_class(self._low_scheduler_cfg), get_arguments(self._low_scheduler_cfg)
                self._low_scheduler_mlp = low_scheduler_class(self._low_optimizer_mlp,**low_scheduler_args)
        
        self._high_mlps = self._high_mlps.train()
        self._low_mlps = self._low_mlps.train()

        return self

    @torch.no_grad()
    def set_eval_mode(self, task_idx: int, stochastic: bool = False, **kwargs) -> "Policy":
        if task_idx in self._task_idxs: self._task_idx = task_idx
        else: self._task_idx = self._task_idxs[-1]
        self._high_mlps = self._high_mlps.eval()
        self._low_mlps = self._low_mlps.eval()
        self._stochastic = stochastic
        return self

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
            high_outputs = self._high_mlps.forward(idx,high_inputs)

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
            high_outputs = self._high_mlps.forward(idx,high_inputs)
            
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
        low_outputs = self._low_mlps.forward(idx,low_inputs).chunk(2,dim=-1)
        
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
            self._high_mlps.parameters(),
            self._low_mlps.parameters()
        )
    
    @torch.no_grad()
    def high_parameters(self) -> Iterator[Parameter]:
        return self._high_mlps.parameters()
    
    @torch.no_grad()
    def low_parameters(self) -> Iterator[Parameter]:
        return self._low_mlps.parameters()

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
            self._high_mlps.buffers(),
            self._low_mlps.buffers()
        )
    
    @torch.no_grad()
    def high_buffers(self) -> Iterator[torch.Tensor]:
        return self._high_mlps.buffers()
    
    @torch.no_grad()
    def low_buffers(self) -> Iterator[torch.Tensor]:
        return self._low_mlps.buffers()

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
        high_fisher_size = sum(sum(v0.nelement() * v0.element_size() for _,v0 in d.items()) for _,d in self._high_fisher_dict.items())
        high_optpar_size = sum(sum(v1.nelement() * v1.element_size() for _,v1 in d.items()) for _,d in self._high_optpar_dict.items())
        low_fisher_size = sum(sum(v0.nelement() * v0.element_size() for _,v0 in d.items()) for _,d in self._low_fisher_dict.items())
        low_optpar_size = sum(sum(v1.nelement() * v1.element_size() for _,v1 in d.items()) for _,d in self._low_optpar_dict.items())
        return (param_size + buffer_size + high_fisher_size + high_optpar_size + low_fisher_size + low_optpar_size) / (1024 ** 2)
    
    def high_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.high_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.high_buffers())
        high_fisher_size = sum(sum(v0.nelement() * v0.element_size() for _,v0 in d.items()) for _,d in self._high_fisher_dict.items())
        high_optpar_size = sum(sum(v1.nelement() * v1.element_size() for _,v1 in d.items()) for _,d in self._high_optpar_dict.items())
        return (param_size + buffer_size + high_fisher_size + high_optpar_size) / (1024 ** 2)
    
    def low_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.low_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.low_buffers())
        low_fisher_size = sum(sum(v0.nelement() * v0.element_size() for _,v0 in d.items()) for _,d in self._low_fisher_dict.items())
        low_optpar_size = sum(sum(v1.nelement() * v1.element_size() for _,v1 in d.items()) for _,d in self._low_optpar_dict.items())
        return (param_size + buffer_size + low_fisher_size + low_optpar_size) / (1024 ** 2)

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
        self._high_mlps = self._high_mlps.to(self._device)
        self._low_mlps = self._low_mlps.to(self._device)
        for k,d in self._high_fisher_dict.items(): self._high_fisher_dict[k] = {kk: v.to(device) for kk,v in d.items()}
        for k,d in self._high_optpar_dict.items(): self._high_optpar_dict[k] = {kk: v.to(device) for kk,v in d.items()}
        for k,d in self._low_fisher_dict.items(): self._low_fisher_dict[k] = {kk: v.to(device) for kk,v in d.items()}
        for k,d in self._low_optpar_dict.items(): self._low_optpar_dict[k] = {kk: v.to(device) for kk,v in d.items()}
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

        # high and low policy loss

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        outs = self._forward(batch,generator,**kwargs)
        losses = module.compute_loss(outs,batch,mask,log_infos=log_infos,**kwargs)

        # high ewc loss

        high_computed = False
        high_ewc_loss = 0.0
        for idx in self._task_idxs:
            if idx != self._task_idx:
                for high_name, high_param in self._high_mlps[0].named_parameters():
                    high_fisher = self._high_fisher_dict[idx][high_name]
                    high_optpar = self._high_optpar_dict[idx][high_name]
                    high_ewc_loss += (high_fisher * (high_param - high_optpar).pow(2)).sum() * self._ewc_lambda / 2
                high_computed = True
        
        if high_computed: losses["A2(Infos)_high_ewc_loss"] = high_ewc_loss
        else: losses["A2(Infos)_high_ewc_loss"] = torch.tensor(0.0)

        # high policy update
        
        if high_computed: high_total_loss = losses["A1(Main)_high_policy_loss"] + losses["A2(Infos)_high_ewc_loss"]
        else: high_total_loss = losses["A1(Main)_high_policy_loss"]

        self._high_optimizer_mlp.zero_grad()
        high_total_loss.backward()
        if gradient_step % self._ewc_update_freq == 0:
            self.high_update_fisher()
        self._high_optimizer_mlp.step()

        if not self._high_scheduler_mlp is None:
            try: self._high_scheduler_mlp.step()
            except: self._high_scheduler_mlp.step(gradient_step)
            if log_infos:
                losses["Z2(Infos)_high_learning_rate"] = torch.tensor(self._high_scheduler_mlp.get_last_lr())[0]
        
        # low ewc loss

        low_computed = False
        low_ewc_loss = 0.0
        for idx in self._task_idxs:
            if idx != self._task_idx:
                for low_name, low_param in self._low_mlps[0].named_parameters():
                    low_fisher = self._low_fisher_dict[idx][low_name]
                    low_optpar = self._low_optpar_dict[idx][low_name]
                    low_ewc_loss += (low_fisher * (low_param - low_optpar).pow(2)).sum() * self._ewc_lambda / 2
                low_computed = True
        
        if low_computed: losses["A2(Infos)_low_ewc_loss"] = low_ewc_loss
        else: losses["A2(Infos)_low_ewc_loss"] = torch.tensor(0.0)
        
        # low policy update
        
        if low_computed: low_total_loss = losses["A1(Main)_low_policy_loss"] + losses["A2(Infos)_low_ewc_loss"]
        else: low_total_loss = losses["A1(Main)_low_policy_loss"]

        self._low_optimizer_mlp.zero_grad()
        low_total_loss.backward()
        if gradient_step % self._ewc_update_freq == 0:
            self.low_update_fisher()
        self._low_optimizer_mlp.step()
        
        if not self._low_scheduler_mlp is None:
            try: self._low_scheduler_mlp.step()
            except: self._low_scheduler_mlp.step(gradient_step)
            if log_infos:
                losses["Z2(Infos)_low_learning_rate"] = torch.tensor(self._low_scheduler_mlp.get_last_lr())[0]

        # return losses
        ###############

        return losses

    def high_update_fisher(self) -> None:

        self._high_fisher_dict[self._task_idx] = {}
        self._high_optpar_dict[self._task_idx] = {}
        
        for high_name, high_param in self._high_mlps[0].named_parameters():
            self._high_fisher_dict[self._task_idx][high_name] = high_param.grad.data.clone().pow(2)
            self._high_optpar_dict[self._task_idx][high_name] = high_param.data.clone()

    def low_update_fisher(self) -> None:

        self._low_fisher_dict[self._task_idx] = {}
        self._low_optpar_dict[self._task_idx] = {}

        for low_name, low_param in self._low_mlps[0].named_parameters():
            self._low_fisher_dict[self._task_idx][low_name] = low_param.grad.data.clone().pow(2)
            self._low_optpar_dict[self._task_idx][low_name] = low_param.data.clone()
