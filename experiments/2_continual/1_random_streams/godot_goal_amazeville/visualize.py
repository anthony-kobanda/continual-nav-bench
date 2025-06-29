import glfw
import hydra
import numpy as np
import os
import torch

from offbench.core.agent import Agent
from offbench.core.data import Frame
from offbench.data.agents_db.pytorch import PytorchAgentsDB
from offbench.data.streams.amazeville import AMAZEVILLE_STREAMS
from offbench.envs.godot_goal.visualize import visualize
from offbench.utils.paths import WORKSPACE_PATH
from omegaconf import DictConfig, ListConfig



@hydra.main(version_base="1.2")
def main(cfg:DictConfig) -> None:

    # create experiment folder (if it does not exist)
    #################################################

    experiment_folder_path = os.path.join(
        WORKSPACE_PATH,
        "experiments",
        "2_continual",
        "godot_goal",
        str(cfg.stream_name),
        str(cfg.agent_cfg.agent_id),
        f"seed_{cfg.seed}/"
    )

    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    
    # get task to visualize
    #######################

    visu_task_idx = int(cfg.algo_cfg.visu_task_idx)
    
    tasks_cfg:ListConfig = AMAZEVILLE_STREAMS[str(cfg.stream_name)]
    task_cfg:DictConfig = tasks_cfg[visu_task_idx]
    
    # create agent database (if it does not exist)
    ###############################################

    agents_db = PytorchAgentsDB(directory=os.path.join(experiment_folder_path,"agents_db_@_T{}".format(visu_task_idx+1)))

    agent = agents_db.get_last_stage(agent_id=str(cfg.agent_cfg.agent_id))

    task = task_cfg.task_name
    seed = cfg.algo_cfg.visu_seed
    device = cfg.algo_cfg.visu_device
    n_players = cfg.algo_cfg.visu_reset_args.batch_size
    n_episodes_per_player = cfg.algo_cfg.visu_n_episodes // n_players
    max_episode_steps = cfg.algo_cfg.visu_max_episode_steps
    agent_reset_args = {} if cfg.algo_cfg.visu_reset_args is None else dict(cfg.algo_cfg.visu_reset_args)
    agent_eval_mode_args = {} if cfg.algo_cfg.visu_mode_args is None else dict(cfg.algo_cfg.visu_mode_args)
    agent_eval_mode_args["task_idx"] = visu_task_idx

    visualize(
        agent=agent,
        task=task,
        seed=seed,
        device=device,
        n_players=n_players,
        n_episodes_per_player=n_episodes_per_player,
        max_episode_steps=max_episode_steps,
        agent_reset_args=agent_reset_args,
        agent_eval_mode_args=agent_eval_mode_args
    )



if __name__ == "__main__":
    main()

