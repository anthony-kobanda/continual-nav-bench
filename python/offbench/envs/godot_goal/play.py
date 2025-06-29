import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

import filelock
import pickle
import random as rd
import shutil
import socket
import subprocess
import tempfile
import time
import torch
import torch.multiprocessing as mp

from .application import start_godot_server
from offbench.core.agent import Agent
from offbench.core.data import Episode
from offbench.data.agents_db.pytorch import PytorchAgentsDB
from offbench.data.episodes_db.godot_goal import GodotGoalEpisodesDB
from offbench.utils.data import unbatch_episode, convert_cfg
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Union



CURRENT_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_FOLDER_PATH = os.path.join(CURRENT_FOLDER_PATH,"tmp")

UNAVAILABLE_PORTS_PATH = os.path.join(CURRENT_FOLDER_PATH,"unavailable_ports.pickle")
PORT_LOCK_PATH = os.path.join(CURRENT_FOLDER_PATH,"port_lock.lock")
CFG_LOCK_PATH = os.path.join(CURRENT_FOLDER_PATH,"cfg_lock.lock")



def load_unavailable_ports() -> List[int]:
    """
    Load the list of unavailable ports from a pickle file, 
    or create an empty list if the file doesn't exist.
    """
    if not os.path.exists(UNAVAILABLE_PORTS_PATH): return []
    with open(UNAVAILABLE_PORTS_PATH,'rb') as f:
        return pickle.load(f)



def save_unavailable_ports(ports:List[int]) -> None:
    """
    Save the list of unavailable ports to a pickle file.
    """
    with open(UNAVAILABLE_PORTS_PATH, 'wb') as f:
        pickle.dump(ports, f)



def find_port_custom(host:str="localhost",start_port:int=8000) -> int:
    """
    Find an available port starting from a given port number.
    """
    # load the unavailable ports list
    with filelock.FileLock(PORT_LOCK_PATH):
        used_ports = load_unavailable_ports()
        port = start_port + rd.randint(a=0,b=100)
        max_port = 65535
        while port <= max_port:
            if port not in used_ports:
                with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                    if s.connect_ex((host,port)) != 0:
                        used_ports.append(port)
                        save_unavailable_ports(used_ports)
                        return port
            port += 1
    raise RuntimeError("No available ports found.")



def release_port(port: int) -> None:
    """
    Release a port so that it can be used again.
    """
    with filelock.FileLock(PORT_LOCK_PATH):
        used_ports = load_unavailable_ports()
        if port in used_ports:
            used_ports.remove(port)
            save_unavailable_ports(used_ports)



def play(
    task:str,
    episodes_db:GodotGoalEpisodesDB,
    seed:int=None,
    exe_type:str="windows",
    max_episodes_db_len:int=None,
    use_websocket_pytorch:bool=False,
    max_episode_steps:int=None) -> GodotGoalEpisodesDB:
    
    #####################
    # FEW SANITY CHECKS #
    #####################

    # env, maze, mode
    #################

    env = task.split("-")[0]
    maze = task.split("-")[1]
    mode:str = None if len(task.split("-")) < 3 else task.split("-")[2]

    assert env in ["amazeville", "simpletown"], "Unknown environment. Available ones are : amazeville, simpletown"

    if maze == "amazeville": assert maze in [f"maze_{i}" for i in range(1,5)], "Unknown maze. Available ones are : maze_1, maze_2, maze_3, maze_4"
    if maze == "simpletown": assert maze in [f"town_{i}" for i in range(0,8)], "Unknown town. Available ones are : town_1, town_2, town_3, town_4, town_5, town_6, town_7"

    if not (mode is None): assert mode in ["high", "low"], "Unknown mode. Available ones are : high, low"

    assert not (max_episodes_db_len is None), "max_episodes_db_len must be provided."
    print("\nCurrently {} episodes in the {} database, and max {} are allowed.".format(len(episodes_db),task,max_episodes_db_len))

    n_episodes = max_episodes_db_len - len(episodes_db)
    if n_episodes <= 0: 
        print("No need to generate more episodes.")
        return episodes_db

    print("Generating max {} episodes ...".format(n_episodes))

    # other parameters
    ##################

    assert not max_episode_steps is None, "max_episode_steps must be provided."

    ##############################
    # Getting the evaluation cfg #
    ##############################

    current_folder_path = os.path.dirname(os.path.realpath(__file__))
    cfg:DictConfig = OmegaConf.load(current_folder_path+"/play.yaml")

    # Updating port value #
    #######################

    host = str(cfg.application.host)
    port = find_port_custom(host=host,start_port=cfg.application.port)
    OmegaConf.update(cfg,"application.port",port)

    # update map_name
    #################

    map_name = env + "_" + str(maze.split("_")[1])
    if not (mode is None): map_name += "_" + mode
    OmegaConf.update(cfg,"application.tracking_configuration.config.decorators.config.value.map_name",map_name)

    # update sampling infos
    #######################

    OmegaConf.update(cfg,"application.tracking_configuration.config.decorators.config.value.seed",seed)
    OmegaConf.update(cfg,"application.tracking_configuration.config.decorators.config.value.max_episode_steps",max_episode_steps)
    OmegaConf.update(cfg,"application.tracking_configuration.config.decorators.config.value.n_episodes_per_player",n_episodes)

    # update exe according to os
    ############################

    assert exe_type in ["windows","ubuntu_2004","ubuntu_2204"], "Unknown exe_type. Available ones are : windows, ubuntu_2004, ubuntu_2204"

    if exe_type == "windows": exe_path = current_folder_path + "/executables/windows.exe"
    if exe_type == "ubuntu_2004": exe_path = current_folder_path + "/executables/ubuntu2004.x86_64"
    if exe_type == "ubuntu_2204": exe_path = current_folder_path + "/executables/ubuntu2204.x86_64"
    assert os.path.exists(exe_path), "Executable does not exist..."

    OmegaConf.update(cfg,"run.exe",exe_path)
    OmegaConf.update(cfg,"run.headless",False)

    #############################
    # Getting episodes database #
    #############################
    
    session_id_to_db:Dict[str,GodotGoalEpisodesDB] = {f"player_{0}": episodes_db}

    ####################
    # Running Services #
    ####################

    mp.set_start_method("spawn",force=True)
    process = mp.Process(
        target=start_godot_server,
        args=(
            host,
            port,
            session_id_to_db,
            cfg.application,
            None,
            None,
            None,
            None,
            use_websocket_pytorch,
            False
        )
    )          
    process.daemon = True

    # need to wait a few seconds for the server to start
    print("\n>> Starting Godot server...")
    process.start()
    time.sleep(10.0)
    print(">> ... Server started !")

    exe = str(cfg.run.exe)
    exe += " --drainc_host={}".format(str(cfg.application.host))
    exe += " --drainc_port={}".format(str(cfg.application.port))
    exe_process = subprocess.Popen(
        exe.split(" "),
        stdout=subprocess.DEVNULL, # suppress standard output
        stderr=subprocess.DEVNULL  # suppress standard error
    )
    print(">> Running executable : {}".format(exe))
    print("\n##############################################################################")
    print("# PRESS ECHAP TO QUIT ( /!\ IT WILL GENERATE THE CORRESPONDING EPISODE /!\ ) #")
    print("##############################################################################\n")

    # waiting for the episodes to be generated
    ##########################################

    start = time.time()

    n_generated_episodes = 0
    current_episodes_db_len = len(episodes_db)
    previous_episodes_db_len = current_episodes_db_len

    while n_generated_episodes < n_episodes:

        current_episodes_db_len = len(episodes_db)
        if current_episodes_db_len > previous_episodes_db_len:
            n_generated_episodes += current_episodes_db_len - previous_episodes_db_len
            print(">> {}/{} episodes generated ...".format(n_generated_episodes,n_episodes))
            previous_episodes_db_len = current_episodes_db_len
        
        if not (exe_process is None):
            if not (exe_process.poll() is None):
                break
        
        time.sleep(1.0)

    time.sleep(2.0)
    current_episodes_db_len = len(episodes_db)
    n_generated_episodes += current_episodes_db_len - previous_episodes_db_len
    print(">> {}/{} episodes generated in {} seconds ...\n".format(n_generated_episodes,n_episodes,round(time.time()-start,2)))

    print(">> Killing Godot process (WAIT !) ...")
    exe_process.kill()
    process.kill()
    time.sleep(10.0)
    print(">> Godot process killed !")

    ######################
    # returning episodes #
    ######################

    print(">> Releasing port {} ...".format(port))
    release_port(port)    
    print(">> Port {} released !\n".format(port))

    return episodes_db
