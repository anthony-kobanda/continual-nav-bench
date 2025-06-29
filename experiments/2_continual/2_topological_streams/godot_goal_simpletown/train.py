import hydra
import os

from offbench.core.algorithm import ContinualLearningAlgorithm
from offbench.data.agents_db.pytorch import PytorchAgentsDB
from offbench.data.loggers import TensorBoardLogger
from offbench.data.streams.simpletown import SIMPLETOWN_STREAMS
from offbench.utils.imports import get_class
from offbench.utils.paths import WORKSPACE_PATH
from omegaconf import DictConfig, ListConfig, OmegaConf



@hydra.main(version_base="1.2")
def main(cfg:DictConfig) -> None:

    stream_name = str(cfg.stream_name)

    s = f"# Stream : {stream_name} #"
    print()
    print("#"*len(s))
    print("#{}#".format(" "*(len(s)-2)))
    print(s)
    print("#{}#".format(" "*(len(s)-2)))
    print("#"*len(s))
    print()
    
    tasks_cfg:ListConfig = SIMPLETOWN_STREAMS[stream_name]

    OmegaConf.update(cfg, "agent_cfg.policy_cfg.env", "simpletown")
    
    # create experiment folder (if it does not exist)
    #################################################

    experiment_folder_path = os.path.join(
        WORKSPACE_PATH,
        "experiments",
        "2_continual",
        "godot_goal",
        stream_name,
        str(cfg.agent_cfg.agent_id),
        f"seed_{cfg.seed}/"
    )

    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    
    # create logger
    ###############

    logger = TensorBoardLogger(
        directory=os.path.join(experiment_folder_path,"logs"),
        prefix=None,
        max_cache_size=1000
    )

    logger.log_params(cfg)

    # training
    ##########

    algorithm:ContinualLearningAlgorithm = get_class(cfg.algo_cfg.algorithm)

    n_tasks = len(tasks_cfg)

    print(tasks_cfg)

    start_from_task_idx = 0
    if "start_from_task_idx" in cfg.algo_cfg:
        start_from_task_idx = cfg.algo_cfg.start_from_task_idx
    
    end_after_task_idx = n_tasks
    if "end_after_task_idx" in cfg.algo_cfg:
        end_after_task_idx = cfg.algo_cfg.end_after_task_idx

    assert start_from_task_idx <= end_after_task_idx, f"start_from_task_idx ({start_from_task_idx}) must be less than end_after_task_idx ({end_after_task_idx})."

    for task_idx in range(n_tasks):

        if task_idx == 0: previous_agents_db:PytorchAgentsDB = None
        else: previous_agents_db = current_agents_db

        current_agents_db = PytorchAgentsDB(directory=os.path.join(experiment_folder_path,"agents_db_@_T{}".format(task_idx+1)))

        if task_idx >= start_from_task_idx:

            algorithm.run(
                algo_cfg=cfg.algo_cfg,
                task_idx=task_idx,
                tasks_cfg=tasks_cfg,
                agent_cfg=cfg.agent_cfg,
                previous_agents_db=previous_agents_db,
                current_agents_db=current_agents_db,
                logger=logger,
                verbose=cfg.verbose
            )
        
        if task_idx == end_after_task_idx: break

    logger.close()



if __name__ == "__main__":
    main()
