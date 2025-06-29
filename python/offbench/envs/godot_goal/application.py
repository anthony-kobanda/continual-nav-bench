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

import offbench.envs.godot_goal.backend.services
import copy
import logging
import time
import torch
import uvicorn

from .backend.application import ApplicationManager
from contextlib import contextmanager
from offbench.core.agent import Agent, AgentsDB
from offbench.core.data import Frame, Episode, EpisodesDB
from offbench.envs.godot_goal.backend.services import router
from offbench.utils.data import convert_cfg
from fastapi import FastAPI
from omegaconf import DictConfig
from typing import Any, Dict, Union



class GodotApplicationManager(ApplicationManager):

    def __init__(
        self,
        session_id_to_db:Dict[str,EpisodesDB],
        session_id_to_cfg:Dict[str,DictConfig],
        agents_db:AgentsDB,
        agents_device:Union[torch.device,str]="cpu",
        agents_reset_arguments:Dict[str,Any]=None,
        agents_eval_arguments:Dict[str,Any]=None) -> None:

        self._session_id_to_db = session_id_to_db
        self._session_id_to_cfg = session_id_to_cfg

        self._agents_db = agents_db
        self._agents_device = agents_device
        self._agents_reset_arguments = {} if agents_reset_arguments is None else agents_reset_arguments
        self._agents_eval_arguments = {} if agents_eval_arguments is None else agents_eval_arguments

        self._n_events:Dict[tuple[str,str,str],int] = {}
        self._agents:Dict[tuple[str,str,str],Agent] = {}
        self._frames:Dict[tuple[str,str,str],list[Frame]] = {}
        self._session_names:Dict[tuple[str,str],str] = {}

    def _save_episode(self,a:str,s:str,e:str,frames:list[Dict[str,torch.Tensor]]) -> None:
        assert e == "action"
        session_name = self._session_names[(a,s)]
        if session_name in self._session_id_to_db:
            frames = self._frames[(a,s,e)]
            episode = Episode(episode_id="___".join([a,s,e]))
            for frame in frames: episode.add_frame(frame)
            self._session_id_to_db[session_name].add_episode(episode)

    def application_start(self,name:str,infos:Dict[str,Any],debug:bool=False) -> Dict[str,str]:
        return {"application_id": f"AID_{time.time()}"}

    def application_end(self,debug:bool=False) -> None:
        return None

    def session_start(self,application_id:str,session_name:str,infos:Dict[str,Any],debug:bool=False) -> Dict[str,Union[str,DictConfig]]:
        session_id = f"SID_{time.time()}"
        self._session_names[(application_id,session_id)] = session_name
        configuration = self._session_id_to_cfg[session_name]
        return {"application_id": application_id, "session_id": session_id, "configuration": configuration}

    def session_end(self,application_id:str,session_id:str,debug:bool=False) -> None:
        for a, s, e in list(self._frames.keys()):
            if (a == application_id) and (s == session_id):
                self._save_episode(a, s, e, self._frames[(a, s, e)])
                del self._frames[(a, s, e)]
                if (a, s, e) in self._n_events: del self._n_events[(a, s, e)]
                if (a, s, e) in self._agents: del self._agents[(a, s, e)]
        return None

    def serve(self,application_id:str,session_id:str,decorator_name:str,event,debug:bool=False) -> Union[Dict[str,torch.Tensor],None]:

        if decorator_name == "action":

            session_name = str(self._session_names[(application_id,session_id)])
            decorator_type = str(self._session_id_to_cfg[session_name]["decorators"][decorator_name]["type"])

            if decorator_type.startswith("serve"):
                
                key = (application_id, session_id, decorator_name)

                if not key in self._n_events:

                    try:

                        self._n_events[key] = 0

                        agent = self._agents_db.get_last_stage("agent")
                        agent.to(self._agents_device)
                        agent.reset(**self._agents_reset_arguments)
                        agent.set_eval_mode(**self._agents_eval_arguments)

                        if debug: print("          >> [GAME DEBUG] AGENT RESET AND EVAL MODE SET for SESSION : ", session_name)
                        
                        self._agents[key] = copy.copy(agent)

                    except Exception as e: print("Exception in reset/eval_mode methods :",e); return None
                
                else: self._n_events[key] += 1
                
                # for obs (except pos and goal)
                N,M = event["sensor/raycasts"].size()
                raycasts = event["sensor/raycasts"].view(N*M)
                rotation = event["sensor/rotation"]
                velocity = event["sensor/velocity"]

                # resulting frame
                frame = Frame(
                    observation={
                        "obs": torch.cat([raycasts, rotation, velocity],dim=-1).float().unsqueeze(0), 
                        "goal": event["sensor/absolute_goal_position"].float().unsqueeze(0), 
                        "pos": event["sensor/position"].float().unsqueeze(0)
                    },
                    action=None,
                    reward=event["sensor/touch_goal"].float().unsqueeze(0)-1.0,
                    done=event["sensor/touch_goal"].bool().unsqueeze(0),
                    truncated=torch.tensor(False).unsqueeze(0),
                    timestep=event["sensor/idx_frame"].long().unsqueeze(0)
                ).to(self._agents_device)

                action:Dict[str,torch.Tensor] = self._agents[key].act(frame)["action"]
                
                return action
        
        return None

    def push(self,application_id:str,session_id:str,episode_id:str,frame:Dict[str,torch.Tensor],debug:bool=False) -> None:
        # if the episode does not exist, we create it
        if not (application_id,session_id,episode_id) in self._frames: 
            self._frames[(application_id, session_id, episode_id)] = []
        # for obs (except pos and goal)
        N,M = frame["sensor/raycasts"].size()
        raycasts = frame["sensor/raycasts"].view(N*M)
        rotation = frame["sensor/rotation"]
        velocity = frame["sensor/velocity"]
        # resulting frame
        _frame = Frame(
            observation={
                "obs": torch.cat([raycasts, rotation, velocity],dim=-1).float().unsqueeze(0),
                "goal": frame["sensor/absolute_goal_position"].float().unsqueeze(0), 
                "pos": frame["sensor/position"].float().unsqueeze(0)
            },
            action={
                "run": frame["action/run"].unsqueeze(0).unsqueeze(-1).bool(), 
                "jump": frame["action/jump"].unsqueeze(0).unsqueeze(-1).bool(), 
                "move_backwards": frame["action/move_backwards"].unsqueeze(0).unsqueeze(-1).float(), 
                "move_forwards": frame["action/move_forwards"].unsqueeze(0).unsqueeze(-1).float(), 
                "move_left": frame["action/move_left"].unsqueeze(0).unsqueeze(-1).float(), 
                "move_right": frame["action/move_right"].unsqueeze(0).unsqueeze(-1).float(), 
                "rotation": frame["action/rotation"].unsqueeze(0).unsqueeze(-1).float()
            },
            reward=frame["sensor/touch_goal"].float().unsqueeze(0)-1.0,
            done=frame["sensor/touch_goal"].bool().unsqueeze(0),
            truncated=torch.tensor(False).unsqueeze(0),
            timestep=frame["sensor/idx_frame"].long().unsqueeze(0)
        )
        # we add the frame to the episode
        self._frames[(application_id, session_id, episode_id)].append(_frame)



def start_godot_server(
        host:str,
        port:int,
        session_id_to_db:Dict[str,EpisodesDB],
        application_configuration:DictConfig,
        agents_db:AgentsDB,
        agents_device:Union[torch.device,str],
        agents_reset_arguments:Dict[str,Any],
        agents_eval_arguments:Dict[str,Any],
        use_websocket_pytorch:bool=False,
        debug:bool=False) -> None:
    
    application_configuration = convert_cfg(application_configuration)
    app = FastAPI()
    
    offbench.envs.godot_goal.backend.services.application_manager = GodotApplicationManager(
        session_id_to_db=session_id_to_db,
        session_id_to_cfg=application_configuration["tracking_configuration"],
        agents_db=agents_db,
        agents_device=agents_device,
        agents_reset_arguments=agents_reset_arguments,
        agents_eval_arguments=agents_eval_arguments,
    )
    offbench.envs.godot_goal.backend.services.use_websocket_pytorch = use_websocket_pytorch
    offbench.envs.godot_goal.backend.services.debug = debug

    app.include_router(router)
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    del uvicorn_log_config["loggers"]["uvicorn"]
    logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
    uvicorn.run(app=app,host=host,port=port,log_config=None)
