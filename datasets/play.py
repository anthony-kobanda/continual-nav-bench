import hydra
import os

from offbench.data.episodes_db.dummy import DummyEpisodesDB
from offbench.data.episodes_db.godot_goal import GodotGoalEpisodesDB
from offbench.envs.godot_goal.play import play
from offbench.utils.paths import DATASETS_PATH
from omegaconf import DictConfig



@hydra.main(version_base="1.2",config_path=".",config_name="play")
def main(cfg:DictConfig) -> None:

    # create dataset folder (if it does not exist)
    ##############################################

    task = str(cfg.task)

    if not cfg.dataset_path is None: directory = os.path.join(DATASETS_PATH,cfg.dataset_path)
    else: directory = None
    
    if cfg.save_episodes: episodes_db = GodotGoalEpisodesDB(env=task,directory=directory)    
    else: episodes_db = DummyEpisodesDB()
    
    play(
        task=task,
        episodes_db=episodes_db,
        seed=cfg.seed,
        exe_type=cfg.exe_type,
        max_episodes_db_len=cfg.max_episodes_db_len,
        use_websocket_pytorch=False,
        max_episode_steps=cfg.max_episode_steps
    )



if __name__ == "__main__":
    main()
