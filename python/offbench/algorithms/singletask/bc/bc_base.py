import time
import torch

from .utils import save_agent, launch_eval_agent
from offbench.core.agent import Agent, AgentsDB
from offbench.core.data import EpisodesDB, Sampler
from offbench.core.algorithm import SingleLearningAlgorithm
from offbench.core.logger import Logger
from offbench.utils.imports import get_class, get_arguments, instantiate_class
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Any, Dict



# (Base) Behavior Cloning Algorithm

class BC_Algorithm(SingleLearningAlgorithm):

    def run(
        algo_cfg: DictConfig,
        task_cfg: DictConfig,
        agent_cfg: DictConfig,
        agents_db: AgentsDB,
        logger: Logger,
        verbose: bool) -> Agent:

        print()
        print("#########################")
        print("# BC (Behavior Cloning) #")
        print("#########################")

        ############
        # DEFAULTS #
        ############
        disable_tqdm:bool = not verbose
        logging_frequency:int = int(algo_cfg.logging_frequency)
        log_infos:bool = algo_cfg.log_infos
        task_name:str = str(task_cfg.task_name)
        train_seed:int = int(algo_cfg.train_seed)
        train_device:str = str(algo_cfg.train_device)
        print("\n- TQDM Verbose is : {}".format(verbose))
        print("\n- Task name is : {}".format(task_name))

        #####################################
        # Creation of the agent to train... #
        #####################################
        agent:Agent = instantiate_class(agent_cfg).to(train_device)       
        train_reset_args:Dict[str,Any] = {} if algo_cfg.train_reset_args is None else dict(algo_cfg.train_reset_args)
        agent.reset(algo_cfg.train_seed,**train_reset_args)
        train_mode_args:Dict[str,Any] = {} if algo_cfg.train_mode_args is None else dict(algo_cfg.train_mode_args)
        agent.set_train_mode(**train_mode_args)
        print("\n- Agent ID is : {}\n".format(agent_cfg.agent_id))

        ####################
        # Episodes Dataset #
        ####################
        episodes_db:EpisodesDB = instantiate_class(task_cfg.episodes_db_cfg)
        sampler_class, sampler_args = get_class(algo_cfg.sampler_cfg), get_arguments(algo_cfg.sampler_cfg)
        sampler_args["episodes_db"] = episodes_db
        sampler:Sampler = sampler_class(**sampler_args)
        sampler.initialize(verbose=verbose,desc="Initializing sampler...")
        normalizer_values = sampler.normalizer_values()
        agent = agent.set_normalizers(normalizer_values)
        print(sampler)
        
        ##################
        # TRAINING SETUP #
        ##################
        train_save_every:int = algo_cfg.train_save_every
        train_gradient_steps:int = algo_cfg.train_gradient_steps
        print("\n- Training configuration :")
        print("\t* training seed is : {}".format(train_seed))
        print("\t* training device is : {}".format(train_device))
        print("\t* total number of gradient steps : {}".format(train_gradient_steps))
        print("\t* save agent every {} gradient steps".format(train_save_every))

        ########
        # EVAL #
        ########
        eval_agent:bool = algo_cfg.eval_agent
        eval_seed = int(algo_cfg.eval_seed)
        eval_device = str(algo_cfg.eval_device)
        eval_every = int(algo_cfg.eval_every)
        eval_n_episodes = int(algo_cfg.eval_n_episodes)
        eval_max_episode_steps = int(algo_cfg.eval_max_episode_steps)
        eval_gamma = float(algo_cfg.eval_gamma)
        eval_reset_args = {} if algo_cfg.eval_reset_args is None else dict(algo_cfg.eval_reset_args)
        eval_mode_args = {} if algo_cfg.eval_mode_args is None else dict(algo_cfg.eval_mode_args)
        print("\n- Evaluation configuration :")
        print("\t* evalutate agent is: {}".format(eval_agent))
        print("\t* evaluation seed is : {}".format(eval_seed))
        print("\t* evaluation device is : {}".format(eval_device))
        print("\t* evaluate agent every {} gradient steps".format(eval_every))
        print("\t* evaluate for {} episodes".format(eval_n_episodes))
        print("\t* evaluate with a maximum of {} steps per episode".format(eval_max_episode_steps))
        print("\t* evaluate with a discount factor of {}\n".format(eval_gamma))

        ############
        # TRAINING #
        ############

        # we will average over the last loggin_frequency batches
        total_batch_duration:float = 0.0
        total_losses:Dict[str,float] = {}
        n_batches:int = 1

        with tqdm(range(train_gradient_steps),desc=">>>>> Training...",disable=disable_tqdm) as pbar_train:
            
            for gradient_step in pbar_train:

                # saving initial agent (before training)
                ########################################
                
                if (gradient_step == 0):
                    save_agent(agent,agents_db,gradient_step,eval_mode_args)

                # evaluation of the initial agent
                #################################                
                
                if  (gradient_step == 0) and eval_agent:
                    launch_eval_agent(agent, pbar_train, gradient_step, task_name, algo_cfg,logger, verbose,
                        eval_seed, eval_device, eval_n_episodes, eval_max_episode_steps, eval_gamma, 
                        eval_reset_args, eval_mode_args, train_reset_args, train_mode_args)
                
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

                losses:Dict[str,torch.Tensor] = agent.update(batch=batch,mask=mask,gradient_step=gradient_step,log_infos=log_infos)

                # logging
                #########

                total_batch_duration += time.time() - start_batch

                if gradient_step == 0:
                    total_losses = {k: 0.0 for k in losses.keys()}

                for k,v in losses.items():
                    total_losses[k] += v.item()

                if (gradient_step == 0) or ((gradient_step + 1) % logging_frequency == 0) or ((gradient_step + 1) == train_gradient_steps):
                    for k,v in total_losses.items():
                        logger.add_scalar(f"TRAINING/{k}",v/n_batches,gradient_step+1)
                        total_losses[k] = 0.0
                    if log_infos:
                        logger.add_scalar(f"TRAINING/Z1(Infos)_batches_per_second",n_batches/total_batch_duration,gradient_step+1)
                        total_batch_duration = 0.0
                    n_batches = logging_frequency

                # saving the agent
                ##################
                
                if ((gradient_step + 1) % train_save_every == 0) or ((gradient_step + 1) == train_gradient_steps):
                    save_agent(agent,agents_db,gradient_step+1,eval_mode_args)
                
                # eventual evaluation of the agent
                ##################################

                if eval_agent and (((gradient_step + 1) % eval_every == 0) or ((gradient_step + 1) == train_gradient_steps)):
                    launch_eval_agent(agent, pbar_train, gradient_step + 1, task_name, algo_cfg,logger, verbose,
                        eval_seed, eval_device, eval_n_episodes, eval_max_episode_steps, eval_gamma, 
                        eval_reset_args, eval_mode_args, train_reset_args, train_mode_args)

        # logging size of the agent (inference and total in MB)
        logger.add_scalar(f"SIZE/inference_size",agent.inference_size(),0)
        logger.add_scalar(f"SIZE/total_size",agent.size(),0)

        return agent
