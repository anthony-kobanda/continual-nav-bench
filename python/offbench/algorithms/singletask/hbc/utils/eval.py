import time
import torch
import traceback

from typing import Any, Dict
from omegaconf import DictConfig
from offbench.core.agent import Agent
from offbench.core.logger import Logger
from offbench.utils.evaluate import evaluate_agent
from tqdm import tqdm



def launch_eval_agent(
    agent:Agent, pbar_train:tqdm, current_step:int, task_name:str, algo_cfg:DictConfig, logger:Logger, verbose:bool,
    eval_seed:int, eval_device:str, eval_n_episodes:int, eval_max_episode_steps:int, eval_gamma:float,
    eval_reset_args:Dict[str,Any], eval_mode_args:Dict[str,Any], train_reset_args:Dict[str,Any], train_mode_args:Dict[str,Any]) -> None:

    pbar_train.set_description(">>>>> Training paused...")
    pbar_train.refresh()
    
    try:
        
        start_eval = time.time()
        
        if current_step == 0: tqdm.write(">>>>> Evaluating initial agent...")
        else: tqdm.write(">>>>> Evaluating agent at gradient step {}...".format(current_step))

        eval_description = "{}Generating {} episodes for task {}...".format(" "*6,eval_n_episodes,task_name)
        with torch.no_grad():
            eval_results = evaluate_agent(
                agent=agent,
                task=task_name,
                seed=eval_seed,
                device=eval_device,
                n_episodes=eval_n_episodes,
                max_episode_steps=eval_max_episode_steps,
                agent_reset_args=eval_reset_args,
                agent_eval_mode_args=eval_mode_args,
                gamma=eval_gamma,
                verbose=verbose,
                generate_description=eval_description,
            )
        for k,v in eval_results.items(): logger.add_scalar(f"EVALUATION/{k}",v,current_step)
        tqdm.write(f"      Agent evaluated in {time.time()-start_eval:.2f} seconds\n")
    
    except Exception as e: 
        tqdm.write(f"Error during initial evaluation : {e}")
        traceback.print_exc()

    agent = agent.to(algo_cfg.train_device)
    agent.reset(algo_cfg.train_seed,**train_reset_args)
    agent.set_train_mode(**train_mode_args)

    pbar_train.set_description(">>>>> Training...")
    pbar_train.refresh()
