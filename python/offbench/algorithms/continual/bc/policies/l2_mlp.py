import copy
import torch

from .utils import godot_goal_amazeville
from .utils import godot_goal_simpletown
from offbench.core.data import Frame
from offbench.core.policy import Policy
from offbench.utils.imports import get_class, get_arguments
from offbench.utils.pytorch.models.mlp import MLP_DICT
from offbench.utils.pytorch.models.mlp import MLP as SimpleMLP
from offbench.utils.pytorch.models.mlp import ResidualMLP as SimpleResidualMLP
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
        l2_lambda:float=1.0,
        l2_update_freq:int=1,
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
        self._l2_lambda: float = l2_lambda
        self._l2_update_freq: int = l2_update_freq
        self._task_idx: int = None
        self._task_idxs: List[int] = []
        self._optpar_dict: Dict[int,Dict[str,torch.Tensor]] = {}

        # for evaluation
        self._stochastic: bool = False

        # normalizers information
        self._normalizer: str = normalizer
        self._normalizers: Dict[str, Union[MinMax_Normalizer, Z_Normalizer]] = {}
        self._normalized_features: List[str] = normalized_features
        assert normalizer in AVAILABLE_NORMALIZERS, f"normalizer must be in {AVAILABLE_NORMALIZERS}"

        self._env_name = env.split("-")[0]

        # policy networks
        self._mlps: MLP_DICT = MLP_DICT({})
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

            if len(self._task_idxs) == 1:

                if self._env_name == "amazeville": module = godot_goal_amazeville
                elif self._env_name == "simpletown": module = godot_goal_simpletown
                else: raise ValueError(f"Unsupported environment : {self._env_name}")

                if self._env_name in ["amazeville","simpletown","antmaze","pointmaze"]: MLP_CLASS = SimpleResidualMLP
                elif self._env_name in ["ant","halfcheetah","hopper","walker2d"]: MLP_CLASS = SimpleMLP

                mlp_sizes = [module.OBS_SIZE] + self._mlp_cfg["hidden_sizes"] + [2 * module.ACTION_SIZE]
                mlp = MLP_CLASS(
                    sizes=mlp_sizes,
                    use_biases=True,
                    use_layer_norm=module.USE_LAYER_NORM,
                    layer_activation_cfg=module.LAYER_ACT_CFG,
                    output_activation_cfg=module.OUTPUT_ACT_CFG,
                    dropout=self._mlp_cfg["dropout"],
                    init_scaling=self._mlp_cfg["init_scaling"]
                ).to(self._device)

                self._mlps[0] = mlp

            optimizer_class, optimizer_args = get_class(self._optimizer_cfg), get_arguments(self._optimizer_cfg)
            self._optimizer_mlp:Optimizer = optimizer_class(self._mlps.parameters(0),**optimizer_args)
            self._scheduler_mlp:LRScheduler = None
            if not self._scheduler_cfg is None:
                scheduler_class, scheduler_args = get_class(self._scheduler_cfg), get_arguments(self._scheduler_cfg)
                self._scheduler_mlp = scheduler_class(self._optimizer_mlp,**scheduler_args)
        
        self._mlps = self._mlps.train()

        return self

    @torch.no_grad()
    def set_eval_mode(self, task_idx: int, stochastic: bool = False, **kwargs) -> "Policy":
        if task_idx in self._task_idxs: self._task_idx = task_idx
        else: self._task_idx = self._task_idxs[-1]
        self._mlps = self._mlps.eval()
        self._stochastic = stochastic
        return self

    def _forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(inputs,Frame): inputs = {k: inputs["observation"][k] for k in inputs["observation"]}
        else: inputs = {k.split("/")[1]: inputs[k] for k in inputs if k.startswith("observation/")}
        for k in self._normalizers: inputs[k] = self._normalizers[k](inputs[k])
        if self._env_name in ["amazeville","simpletown","antmaze","pointmaze"]:
            if self._env_name == "amazeville": module = godot_goal_amazeville
            elif self._env_name == "simpletown": module = godot_goal_simpletown
            additional_features = module.additional_features(inputs["pos"],inputs["goal"])
            for k in additional_features: inputs[k] = additional_features[k]
        inputs = torch.cat([inputs[k] for k in inputs],dim=-1)
        outputs = self._mlps.forward(0,inputs).chunk(2,dim=-1)
        return {"mean": outputs[0], "log_std": outputs[1]}

    @torch.no_grad()
    def forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:
        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        outs = self._forward(inputs,generator,**kwargs)
        return module.forward(outs,self._stochastic,generator,**kwargs)

    @torch.no_grad()
    def parameters(self) -> Iterator[Parameter]:
        return self._mlps.parameters()

    @torch.no_grad()
    def inference_parameters(self) -> Iterator[Parameter]:
        return self.parameters()

    @torch.no_grad()
    def buffers(self) -> Iterator[torch.Tensor]:
        return self._mlps.buffers()

    @torch.no_grad()
    def inference_buffers(self) -> Iterator[torch.Tensor]:
        return self.buffers()
    
    @torch.no_grad()
    def size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.buffers())
        optpar_size = sum(sum(v1.nelement() * v1.element_size() for _,v1 in d.items()) for _,d in self._optpar_dict.items())
        return (param_size + buffer_size + optpar_size) / (1024 ** 2)

    def inference_size(self) -> float:
        param_size = sum(param.nelement() * param.element_size() for param in self.inference_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.inference_buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    @torch.no_grad()
    def to(self, device: Union[torch.device, str]) -> "MLP":
        super().to(device)
        self._normalizers = {k: v.to(device) for k,v in self._normalizers.items()}
        self._mlps = self._mlps.to(self._device)
        for k,d in self._optpar_dict.items(): self._optpar_dict[k] = {kk: v.to(device) for kk,v in d.items()}
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

        # policy loss

        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        outs = self._forward(batch,generator,**kwargs)
        losses = module.compute_loss(outs,batch,mask,log_infos=log_infos,**kwargs)

        # l2 loss

        computed = False
        l2_loss = 0.0
        for idx in self._task_idxs:
            if idx != self._task_idx:
                for name, param in self._mlps[0].named_parameters():
                    optpar = self._optpar_dict[idx][name]
                    l2_loss += (param - optpar).pow(2).sum() * self._l2_lambda
                computed = True
                
        if computed: losses["A2(Infos)_l2_loss"] = l2_loss
        else: losses["A2(Infos)_l2_loss"] = torch.tensor(0.0)
        
        # policy update
        
        if computed: total_loss = losses["A1(Main)_policy_loss"] + losses["A2(Infos)_l2_loss"]
        else: total_loss = losses["A1(Main)_policy_loss"]

        self._optimizer_mlp.zero_grad()
        total_loss.backward()
        if gradient_step % self._l2_update_freq == 0:
            self.update_fisher()
        self._optimizer_mlp.step()

        if not self._scheduler_mlp is None:
            try: self._scheduler_mlp.step()
            except: self._scheduler_mlp.step(gradient_step)
            if log_infos:
                losses["Z2(Infos)_learning_rate"] = torch.tensor(self._scheduler_mlp.get_last_lr())[0]

        # return losses
        ###############

        return losses

    def update_fisher(self) -> None:

            self._optpar_dict[self._task_idx] = {}
            
            for name, param in self._mlps[0].named_parameters():
                self._optpar_dict[self._task_idx][name] = param.data.clone().detach()
