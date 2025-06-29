import copy

from omegaconf import ListConfig
from typing import Dict



# DEFAULT TASK CONFIG
#####################

__default_task_cfg = {
    "task_name": "antmaze-medium-normal",
    "episodes_db_cfg": {
        "classname": "offbench.data.episodes_db.GodotGoalEpisodesDB",
        "env": "amazeville-maze_1-high",
    }
}



# RANDOM STREAM 1
#################

__random_task_names_1 = [
    "amazeville-maze_2-low",
    "amazeville-maze_4-high",
    "amazeville-maze_4-low",
    "amazeville-maze_4-high",
    "amazeville-maze_1-high",
    "amazeville-maze_4-low",
    "amazeville-maze_1-high",
    "amazeville-maze_3-high"
]

__random_stream_1_cfg = []
for task_name in __random_task_names_1:
    task_cfg = copy.deepcopy(__default_task_cfg)
    task_cfg["task_name"] = task_name
    task_cfg["episodes_db_cfg"]["env"] = task_name
    __random_stream_1_cfg.append(task_cfg)

RANDOM_STREAM_1_CFG = ListConfig(__random_stream_1_cfg)



# RANDOM STREAM 2
#################

__random_task_names_2 = [
    "amazeville-maze_3-high",
    "amazeville-maze_2-high",
    "amazeville-maze_2-low",
    "amazeville-maze_3-low",
    "amazeville-maze_2-low",
    "amazeville-maze_4-low",
    "amazeville-maze_4-high",
    "amazeville-maze_1-high"
]

__random_stream_2_cfg = []
for task_name in __random_task_names_2:
    task_cfg = copy.deepcopy(__default_task_cfg)
    task_cfg["task_name"] = task_name
    task_cfg["episodes_db_cfg"]["env"] = task_name
    __random_stream_2_cfg.append(task_cfg)

RANDOM_STREAM_2_CFG = ListConfig(__random_stream_2_cfg)



# TOPOLOGICAL STREAM 1
######################

__topological_task_names_1 = [
    "amazeville-maze_2-high",
    "amazeville-maze_4-high",
    "amazeville-maze_4-high",
    "amazeville-maze_2-high",
    "amazeville-maze_4-high",
    "amazeville-maze_3-high",
    "amazeville-maze_4-high",
    "amazeville-maze_1-high"
]

__topological_stream_1_cfg = []
for task_name in __topological_task_names_1:
    task_cfg = copy.deepcopy(__default_task_cfg)
    task_cfg["task_name"] = task_name
    task_cfg["episodes_db_cfg"]["env"] = task_name
    __topological_stream_1_cfg.append(task_cfg)

TOPOLOGICAL_STREAM_1_CFG = ListConfig(__topological_stream_1_cfg)



# TOPOLOGICAL STREAM 2
######################

__topological_task_names_2 = [
    "amazeville-maze_1-low",
    "amazeville-maze_1-low",
    "amazeville-maze_4-low",
    "amazeville-maze_3-low",
    "amazeville-maze_1-low",
    "amazeville-maze_2-low",
    "amazeville-maze_3-low",
    "amazeville-maze_2-low"
]

__topological_stream_2_cfg = []
for task_name in __topological_task_names_2:
    task_cfg = copy.deepcopy(__default_task_cfg)
    task_cfg["task_name"] = task_name
    task_cfg["episodes_db_cfg"]["env"] = task_name
    __topological_stream_2_cfg.append(task_cfg)

TOPOLOGICAL_STREAM_2_CFG = ListConfig(__topological_stream_2_cfg)



# ALL STREAMS
#############


AMAZEVILLE_STREAMS: Dict[str, ListConfig] = {
    "amazeville_random_1": RANDOM_STREAM_1_CFG,
    "amazeville_random_2": RANDOM_STREAM_2_CFG,
    "amazeville_topological_1": TOPOLOGICAL_STREAM_1_CFG,
    "amazeville_topological_2": TOPOLOGICAL_STREAM_2_CFG
}
