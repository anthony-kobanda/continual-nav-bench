import torch

from offbench.core.agent import Agent
from offbench.core.data import Episode
from offbench.envs.godot_goal.evaluate_agent import generate_godot_episodes
from omegaconf import DictConfig
from typing import Any, Dict, List, Union



def generate_episodes(
    agent:Agent,
    task:str,
    seed:int=None,
    device:Union[torch.device,str]="cpu",
    n_episodes:int=100,
    max_episode_steps:int=None,
    agent_reset_args:Union[DictConfig,Dict[str,Any]]=None,
    agent_eval_mode_args:Union[DictConfig,Dict[str,Any]]=None,
    verbose:bool=False,
    generate_description:str=None) -> List[Episode]:

    return generate_godot_episodes(
        agent,
        task,
        seed,
        device,
        n_episodes,
        max_episode_steps,
        agent_reset_args,
        agent_eval_mode_args,
        verbose,
    )
