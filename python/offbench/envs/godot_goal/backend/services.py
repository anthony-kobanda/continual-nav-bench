import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

import copy
import io
import json
import time
import torch

from .application import ApplicationManager
from .tools import event_to_pytorch
from .types import GameStartEvent, GameEndEvent, SessionStartEvent, SessionEndEvent, TrackingEvent
from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import Any, Dict, Union



router = APIRouter()

application_manager : ApplicationManager = None
use_websocket_pytorch : bool = False
debug : bool = False



def serialize(event:Any) -> str:
    return json.dumps(event)



def deserialize(msg:Union[bytes,bytearray,str]) -> Any:
    return json.loads(msg)



@router.post("/start_game")
async def start_game(event:GameStartEvent) -> Dict[str,Any]:
    t = str(time.time())
    event = copy.deepcopy(event)
    event.received_timestamp = t
    results = application_manager.application_start(event.game_name,event.infos,debug)
    if debug: print("          >> [GAME DEBUG] STARTING NEW GAME : ", results)
    return {"msg": "game_started", **results}



@router.post("/end_game")
async def end_game(event:GameEndEvent) -> Dict[str,Any]:
    return {"msg": "game_ended"}



@router.post("/start_session")
async def start_session(event:SessionStartEvent) -> Dict[str,Any]:
    results = application_manager.session_start(event.application_id,event.session_name,event.infos,debug)
    if debug: print("          >> [GAME DEBUG] STARTING SESSION :", results)
    return {"msg": "session_started", **results}



@router.post("/end_session")
async def end_session(event:SessionEndEvent,background_tasks:BackgroundTasks) -> Dict[str,str]:
    if debug: print("          >> [GAME DEBUG] ENDING SESSION : ", event)
    background_tasks.add_task(application_manager.session_end,event.application_id,event.session_id,debug)
    return {"msg": "session_ended"}



@router.websocket("/event")
async def data(websocket: WebSocket):

    await websocket.accept()

    if debug: print("          >> [GAME DEBUG] [WebSocket] / Event connection accepted")

    try:
        
        while True:

            if use_websocket_pytorch:
                data = await websocket.receive_bytes()
                buffer = io.BytesIO(data)
                event:TrackingEvent = torch.load(buffer)
            
            else:
                data = await websocket.receive_text()
                event:TrackingEvent = deserialize(data)
            
            application_id : str = event.pop("application_id")
            session_id : str = event.pop("session_id")
            name : str = event["event_name"]
            event : Dict[str,Any] = event["event"]

            tevent = event_to_pytorch(event)
            new_event = application_manager.serve(application_id,session_id,name,tevent,debug)

            if not (new_event is None):

                if use_websocket_pytorch:
                    buffer = io.BytesIO()
                    torch.save(new_event,buffer)
                    await websocket.send_bytes(buffer.getvalue())
                
                else:
                    new_event_s = serialize(new_event)
                    await websocket.send_text(new_event_s)
                
                event[name] = new_event
                tevent = event_to_pytorch(event)
                application_manager.push(application_id,session_id,name,tevent,debug)

            else:

                application_manager.push(application_id,session_id,name,tevent,debug)

                if use_websocket_pytorch:
                    buffer = io.BytesIO()
                    torch.save({}, buffer)
                    await websocket.send_bytes(buffer.getvalue())
                
                else:
                    await websocket.send_text("{}")

    except WebSocketDisconnect:        
        if debug: print("          >> [GAME DEBUG] [WebSocket] / Client disconnected")
        else: pass
