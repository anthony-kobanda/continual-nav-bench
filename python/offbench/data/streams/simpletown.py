import copy

from omegaconf import ListConfig
from typing import Dict



# DEFAULT TASK CONFIG
#####################

__default_task_cfg = {
    "task_name": "antmaze-medium-normal",
    "episodes_db_cfg": {
        "classname": "offbench.data.episodes_db.GodotGoalEpisodesDB",
        "env": "simpletown-maze_0",
    }
}



# TOPOLOGICAL STREAM 1
######################

__topological_task_names_1 = [
    "simpletown-maze_0",
    "simpletown-maze_3",
    "simpletown-maze_0",
    "simpletown-maze_2",
    "simpletown-maze_4",
    "simpletown-maze_0",
    "simpletown-maze_7",
    "simpletown-maze_2",
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
    "simpletown-maze_0",
    "simpletown-maze_7",
    "simpletown-maze_4",
    "simpletown-maze_7",
    "simpletown-maze_7",
    "simpletown-maze_3",
    "simpletown-maze_3",
    "simpletown-maze_1"
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


SIMPLETOWN_STREAMS: Dict[str, ListConfig] = {
    "simpletown_topological_1": TOPOLOGICAL_STREAM_1_CFG,
    "simpletown_topological_2": TOPOLOGICAL_STREAM_2_CFG,
}
