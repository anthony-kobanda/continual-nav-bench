import time
import torch

from .utils import save_agent, launch_eval_agent
from offbench.core.agent import Agent, AgentsDB
from offbench.core.data import EpisodesDB, Sampler, SamplerMulti
from offbench.core.algorithm import ContinualLearningAlgorithm
from offbench.core.logger import Logger
from offbench.utils.imports import get_class, get_arguments, instantiate_class
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from typing import Any, Dict, List, Union



# Measure inference time of the agent

def measure(
    agent: Agent,
    agent_id: str,
    task_idx: int,
    n_tasks: int,
    tasks_cfg: ListConfig,
    sampler_cfg: DictConfig,
    seed:int=None,
    n_inference_steps:int=10000,
    device:Union[str,torch.device]="cpu",
    train_reset_args:Union[Dict[str,Any],DictConfig]=None,
    train_mode_args:Union[Dict[str,Any],DictConfig]=None) -> float:

    task_cfg = tasks_cfg[task_idx]
    
    print("\n\tMeasuring for task : {}/{}".format(task_idx+1,n_tasks))
    print("\tMeasuring for task : {}".format(task_cfg.task_name))

    disable_tqdm = False
    
    #####################################
    # Creation of the agent to train... #
    #####################################
    agent = agent.to(device)
    train_reset_args:Dict[str,Any] = {} if train_reset_args is None else dict(train_reset_args)
    agent.reset(seed,**train_reset_args)
    train_mode_args:Dict[str,Any] = {} if train_mode_args is None else dict(train_mode_args)
    train_mode_args["task_idx"] = task_idx
    agent.set_train_mode(**train_mode_args)

    ####################
    # Episodes Dataset #
    ####################

    try:
        episodes_db:EpisodesDB = instantiate_class(task_cfg.episodes_db_cfg)
        sampler_class, sampler_args = get_class(sampler_cfg), get_arguments(sampler_cfg)
        sampler_args["episodes_db"] = episodes_db
        sampler:Sampler = sampler_class(**sampler_args)
    except:
        all_task_cfg = [tasks_cfg[i] for i in range(task_idx+1)]
        episodes_dbs:List[EpisodesDB] = [instantiate_class(_task_cfg.episodes_db_cfg) for _task_cfg in all_task_cfg]
        sampler_class, sampler_args = get_class(sampler_cfg), get_arguments(sampler_cfg)
        sampler_args["episodes_dbs"] = episodes_dbs
        sampler:SamplerMulti = sampler_class(**sampler_args)
    sampler.initialize(verbose=True,desc="\tInitializing sampler...")
    normalizer_values = sampler.normalizer_values()
    agent = agent.set_normalizers(normalizer_values)
    
    #############
    # MEASURING #
    #############

    total_duration:float = 0.0
    sampling_duration:float = 0.0

    with tqdm(range(n_inference_steps),desc="\t>>>>> Running...",disable=disable_tqdm) as pbar_train:
        
        for _ in pbar_train:

            # we perform a training step
            ############################

            start_batch = time.time()

            with torch.no_grad():
                batch = sampler.sample_batch()
                mask = (~batch["is_padding"]).float()
                mask_sum = mask.sum(-1,keepdim=True)
                if torch.any(mask_sum == 0):
                    raise ValueError("All steps are masked for some samples in the batch")
                mask = (mask / mask_sum).unsqueeze(-1)
            
            sampling_duration += time.time() - start_batch

            agent._policy._forward(batch,mask,alpha_mode="best")

            total_duration += time.time() - start_batch

    return (total_duration - sampling_duration) / n_inference_steps
