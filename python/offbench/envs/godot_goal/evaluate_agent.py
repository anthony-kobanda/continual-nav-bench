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
        port = start_port + rd.randint(a=0,b=1000)
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



def generate_godot_episodes(
    agent:Agent,
    task:str,
    seed:int=None,
    device:Union[torch.device,str]="cpu",
    n_episodes:int=100,
    max_episode_steps:int=None,
    agent_reset_args:Union[DictConfig,Dict[str,Any]]=None,
    agent_eval_mode_args:Union[DictConfig,Dict[str,Any]]=None,
    verbose:bool=False) -> List[Episode]:
    
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

    print("{}Generating {} episodes for task {}...".format(" "*6,n_episodes,task))

    # other parameters
    ##################

    batch_size = int(agent_reset_args["batch_size"])
    if env == "amazeville": max_episode_steps = 300
    else: max_episode_steps = 100

    ##############################
    # Getting the evaluation cfg #
    ##############################

    with filelock.FileLock(CFG_LOCK_PATH):
        cfg:DictConfig = OmegaConf.load(CURRENT_FOLDER_PATH+"/evaluate_agent.yaml")

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
    OmegaConf.update(cfg,"application.tracking_configuration.config.decorators.config.value.n_players",batch_size)
    OmegaConf.update(cfg,"application.tracking_configuration.config.decorators.config.value.max_episode_steps",max_episode_steps)

    assert n_episodes % batch_size == 0, "n_episodes must be a multiple of batch_size."
    n_episodes_per_player = (n_episodes // batch_size)
    OmegaConf.update(cfg,"application.tracking_configuration.config.decorators.config.value.n_episodes_per_player",n_episodes_per_player)

    # update exe according to os
    ############################

    assert "exe_type" in agent_reset_args, "exe_type not found in agent_reset_args."
    exe_type = str(agent_reset_args["exe_type"])
    assert exe_type in ["windows","ubuntu_2004","ubuntu_2204"], "Unknown exe_type. Available ones are : windows, ubuntu_2004, ubuntu_2204"

    if exe_type == "windows": exe_path = CURRENT_FOLDER_PATH + "/executables/windows.exe"
    if exe_type == "ubuntu_2004": exe_path = CURRENT_FOLDER_PATH + "/executables/ubuntu2004.x86_64"
    if exe_type == "ubuntu_2204": exe_path = CURRENT_FOLDER_PATH + "/executables/ubuntu2204.x86_64"
    assert os.path.exists(exe_path), "Executable does not exist..."

    OmegaConf.update(cfg,"run.exe",exe_path)
    OmegaConf.update(cfg,"run.headless",True)

    ############################################
    # Getting the agents and episodes database #
    ############################################

    if not os.path.exists(TMP_FOLDER_PATH):
        os.makedirs(TMP_FOLDER_PATH)
    
    with tempfile.TemporaryDirectory(dir=TMP_FOLDER_PATH,prefix="tmp_",suffix="_port"+str(port)) as tmp_folder_path:
    
        agents_db_directory = os.path.join(tmp_folder_path,"agents_db")
        os.makedirs(agents_db_directory,exist_ok=True)
        print("{}>> Temporarily storing the agent in {}".format(" "*6,agents_db_directory))
        agents_db = PytorchAgentsDB(agents_db_directory)
        agents_db.add_agent(agent_id="agent",agent=agent,agent_stage=0)

        agent_reset_args = convert_cfg(agent_reset_args)
        agent_reset_args["seed"] = seed

        episodes_db_directory = os.path.join(tmp_folder_path,"episodes_db")
        os.makedirs(episodes_db_directory,exist_ok=True)
        print("{}>> Temporarily storing evaluation episodes in {}".format(" "*6,episodes_db_directory))
        episodes_db = GodotGoalEpisodesDB(episodes_db_directory)
        
        session_id_to_db:Dict[str,GodotGoalEpisodesDB] = {f"player_{i}": episodes_db for i in range(batch_size)}

        ####################
        # Running Services #
        ####################

        # we use a server to run the executable, and we communicate with it using a websocket

        use_websocket_pytorch = False
        if "use_websocket_pytorch" in agent_reset_args: use_websocket_pytorch = agent_reset_args["use_websocket_pytorch"]

        debug = False
        if "debug" in agent_reset_args: debug = agent_reset_args["debug"]

        mp.set_start_method("spawn",force=True)
        process = mp.Process(
            target=start_godot_server,
            args=(
                host,
                port,
                session_id_to_db,
                cfg.application,
                agents_db,
                device,
                agent_reset_args,
                agent_eval_mode_args,
                use_websocket_pytorch,
                debug
            )
        )          
        process.daemon = True

        # need to wait a few seconds for the server to start
        print("{}>> Starting Godot server...".format(" "*6))
        process.start()
        time.sleep(15.0)
        print("{}>> ... Server started !".format(" "*6))

        exe = str(cfg.run.exe)
        exe += " --headless"
        exe += " --drainc_host={}".format(str(cfg.application.host))
        exe += " --drainc_port={}".format(str(cfg.application.port))
        exe_process = subprocess.Popen(
            exe.split(" "),
            stdout=subprocess.DEVNULL, # suppress standard output
            stderr=subprocess.DEVNULL  # suppress standard error
        )
        print("{}>> Running executable : {}".format(" "*6,exe))

        # waiting for the episodes to be generated
        ##########################################

        start = time.time()

        current_episodes_db_len = 0
        previous_episodes_db_len = 0
        n_repetitions = 0

        while current_episodes_db_len < n_episodes:
            
            time.sleep(15.0)
            
            previous_episodes_db_len = current_episodes_db_len
            current_episodes_db_len = len(episodes_db)

            if verbose: print("{}>> len(evaluation_episodes_db) = {} ...".format(" "*9,current_episodes_db_len))

            if current_episodes_db_len == previous_episodes_db_len: n_repetitions += 1
            else: n_repetitions = 0

            if n_repetitions > 8: # more than 2 minutes without new episodes ...

                print("{}>> No new episodes generated for the last 2 minutes...".format(" "*6))
                print("{}>> Restarting the evaluation process...".format(" "*6))
                
                print("{}>> Killing Godot process...".format(" "*6))
                exe_process.kill()
                process.kill()
                time.sleep(10.0)
                print("{}>> Godot process killed !".format(" "*6))

                print("{}>> Removing temporary directories...".format(" "*6))
                shutil.rmtree(agents_db_directory)
                shutil.rmtree(episodes_db_directory)
                
                print("{}>> Releasing port {}...".format(" "*6,port))
                release_port(port)  

                return generate_godot_episodes(
                    agent=agent,
                    task=task,
                    seed=seed,
                    device=device,
                    n_episodes=n_episodes,
                    max_episode_steps=max_episode_steps,
                    agent_reset_args=agent_reset_args,
                    agent_eval_mode_args=agent_eval_mode_args,
                    verbose=verbose
                )

        print("{}>> Killing Godot process (WAIT !) ...".format(" "*6))
        exe_process.kill()
        process.kill()
        time.sleep(10.0)
        print("{}>> Godot process killed !".format(" "*6))

        ######################
        # returning episodes #
        ######################

        episodes = []
        for episode in episodes_db:
            episodes += unbatch_episode(episode)

        print("{}>> {} episodes sampled in {} seconds".format(" "*6,len(episodes),round(time.time()-start,2)))

        print("{}>> Removing temporary directories...".format(" "*6))
        shutil.rmtree(agents_db_directory)
        shutil.rmtree(episodes_db_directory)

        print("{}>> Releasing port {}...".format(" "*6,port))
        release_port(port)    

        return episodes
