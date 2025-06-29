import hydra
import os

from offbench.core.algorithm import SingleLearningAlgorithm
from offbench.data.agents_db.pytorch import PytorchAgentsDB
from offbench.data.loggers import TensorBoardLogger
from offbench.utils.imports import get_class
from offbench.utils.paths import WORKSPACE_PATH
from omegaconf import DictConfig



@hydra.main(version_base="1.2")
def main(cfg:DictConfig) -> None:

    # create experiment folder (if it does not exist)
    #################################################

    experiment_folder_path = os.path.join(
        WORKSPACE_PATH,
        "experiments",
        "1_singletask",
        "godot_goal",
        str(cfg.task_cfg.task_name),
        str(cfg.agent_cfg.agent_id),
        f"seed_{cfg.seed}/"
    )

    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    
    # create agent database (if it does not exist)
    ###############################################

    agents_db = PytorchAgentsDB(directory=os.path.join(experiment_folder_path,"agents_db"))
    
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

    algorithm:SingleLearningAlgorithm = get_class(cfg.algo_cfg.algorithm)

    algorithm.run(
        algo_cfg=cfg.algo_cfg,
        task_cfg=cfg.task_cfg,
        agent_cfg=cfg.agent_cfg,
        agents_db=agents_db,
        logger=logger,
        verbose=cfg.verbose
    )

    logger.close()



if __name__ == "__main__":
    main()
