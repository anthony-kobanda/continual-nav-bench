import torch

from .generate_episodes import generate_episodes
from offbench.core.agent import Agent
from offbench.core.data import Episode, EpisodesDB
from omegaconf import DictConfig
from typing import Any, Dict, List, Union



def evaluate_episode(
        task:str,
        episode:Episode,
        max_episode_steps:int=None,
        gamma:float=1.0) -> Dict[str,Union[float,int]]:

    assert len(episode) > 0, "The episode is empty."
    assert episode[0]["done"].size(0) == 1, "The batch size must be 1."

    results = {}

    _length = 0
    _undiscounted_return = 0
    _discounted_return = 0
    _success = 0

    for frame in episode:

        _length += 1
        _undiscounted_return += frame["reward"][0].item()
        _discounted_return += frame["reward"][0].item() * (gamma ** (_length - 1))
        _success += float(frame["done"][0].item())

        if frame["done"][0].item() or frame["truncated"][0].item(): break
        elif not (max_episode_steps is None) and _length >= max_episode_steps: break
    
    results["length"] = _length
    results["undiscounted_return"] = _undiscounted_return
    results["discounted_return"] = _discounted_return
    results["success"] = _success
    
    return results



def evaluate_episodes(
    task:str,
    episodes_db:Union[EpisodesDB,List[Episode]],
    max_episode_steps:int=None,
    gamma:float=1.0) -> Dict[str,Union[float,int]]:
    
    results = {}

    for episode in episodes_db:

        episode_results = evaluate_episode(task,episode,max_episode_steps,gamma)

        for k,v in episode_results.items():
            if not f"mean_{k}" in results: results[f"mean_{k}"] = 0.0
            results[f"mean_{k}"] += v
    
    for k in results.keys():
        results[k] /= len(episodes_db)

    return results



def evaluate_agent(
    agent: Agent,
    task: str,
    seed: int = None,
    device: Union[torch.device, str] = "cpu",
    n_episodes: int = 100,
    max_episode_steps: int = None,
    agent_reset_args: Union[DictConfig, Dict[str, Any]] = None,
    agent_eval_mode_args: Union[DictConfig, Dict[str, Any]] = None,
    gamma: float = 1.0,
    verbose: bool = False,
    generate_description: str = None) -> Dict[str, Union[float, int]]:

    episodes = generate_episodes(
        agent=agent,
        task=task,
        seed=seed,
        device=device,
        n_episodes=n_episodes,
        max_episode_steps=max_episode_steps,
        agent_reset_args=agent_reset_args,
        agent_eval_mode_args=agent_eval_mode_args,
        verbose=verbose,
        generate_description=generate_description)
    
    return evaluate_episodes(task,episodes,max_episode_steps,gamma)
