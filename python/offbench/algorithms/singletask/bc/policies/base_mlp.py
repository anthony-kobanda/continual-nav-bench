import torch

from .utils import godot_goal_amazeville
from .utils import godot_goal_simpletown
from offbench.core.data import Frame
from offbench.core.policy import Policy
from offbench.utils.imports import get_class, get_arguments
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

        # for evaluation
        self._stochastic: bool = False

        # normalizers information
        self._normalizer: str = normalizer
        self._normalizers: Dict[str, Union[MinMax_Normalizer, Z_Normalizer]] = {}
        self._normalized_features: List[str] = normalized_features
        assert normalizer in AVAILABLE_NORMALIZERS, f"normalizer must be in {AVAILABLE_NORMALIZERS}"

        self._env_name = env.split("-")[0]
        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        else: raise ValueError(f"Unsupported environment : {env}")

        if self._env_name in ["amazeville","simpletown","antmaze","pointmaze"]: MLP_CLASS = SimpleResidualMLP
        elif self._env_name in ["ant","halfcheetah","hopper","walker2d"]: MLP_CLASS = SimpleMLP
        else: raise ValueError(f"Unsupported environment : {env}")

        # policy network
        mlp_sizes = [module.OBS_SIZE] + hidden_sizes + [2 * module.ACTION_SIZE]
        self._mlp = MLP_CLASS(
            sizes=mlp_sizes,
            use_biases=True,
            use_layer_norm=module.USE_LAYER_NORM,
            layer_activation_cfg=module.LAYER_ACT_CFG,
            output_activation_cfg=module.OUTPUT_ACT_CFG,
            dropout=dropout,
            init_scaling=init_scaling
        )

        # policy optimizer
        optimizer_class, optimizer_args = get_class(optimizer_cfg), get_arguments(optimizer_cfg)
        self._optimizer_mlp:Optimizer = optimizer_class(self._mlp.parameters(),**optimizer_args)
        self._scheduler_mlp:LRScheduler = None
        if not scheduler_cfg is None:
            scheduler_class, scheduler_args = get_class(scheduler_cfg), get_arguments(scheduler_cfg)
            self._scheduler_mlp = scheduler_class(self._optimizer_mlp,**scheduler_args)
    
    def set_normalizers(self, normalize_values, **kwargs) -> "MLP":
        for k in self._normalized_features:
            if self._normalizer == "minmax": self._normalizers[k] = MinMax_Normalizer(normalize_values[k]["min"],normalize_values[k]["max"])
            elif self._normalizer == "z": self._normalizers[k] = Z_Normalizer(normalize_values[k]["mean"],normalize_values[k]["std"])
        return self

    def set_train_mode(self, **kwargs) -> "Policy":
        self._mlp = self._mlp.train()
        return self

    @torch.no_grad()
    def set_eval_mode(self, stochastic: bool = False, **kwargs) -> "Policy":
        self._mlp = self._mlp.eval()
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
        outputs = self._mlp.forward(inputs).chunk(2,dim=-1)
        return {"mean": outputs[0], "log_std": outputs[1]}

    @torch.no_grad()
    def forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: torch.Generator, **kwargs) -> Dict[str, torch.Tensor]:
        if self._env_name == "amazeville": module = godot_goal_amazeville
        elif self._env_name == "simpletown": module = godot_goal_simpletown
        outs = self._forward(inputs,generator,**kwargs)
        return module.forward(outs,self._stochastic,generator,**kwargs)

    @torch.no_grad()
    def parameters(self) -> Iterator[Parameter]:
        return self._mlp.parameters()

    @torch.no_grad()
    def inference_parameters(self) -> Iterator[Parameter]:
        return self.parameters()

    @torch.no_grad()
    def buffers(self) -> Iterator[torch.Tensor]:
        return self._mlp.buffers()

    @torch.no_grad()
    def inference_buffers(self) -> Iterator[torch.Tensor]:
        return self.buffers()

    @torch.no_grad()
    def to(self, device: Union[torch.device, str]) -> "MLP":
        super().to(device)
        self._normalizers = {k: v.to(device) for k,v in self._normalizers.items()}
        self._mlp = self._mlp.to(self._device)
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
