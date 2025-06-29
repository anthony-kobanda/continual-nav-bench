from pydantic import BaseModel
from typing import Any, Dict

##################
# GameStartEvent #
##################

class GameStartEvent(BaseModel):
    game_name : str
    infos : Dict[str,Any] = {}
    received_timestamp : str = None

################
# GameEndEvent #
################

class GameEndEvent(BaseModel):
    application_id : str
    received_timestamp : str = None

#####################
# SessionStartEvent #
#####################

class SessionStartEvent(BaseModel):
    application_id : str
    session_name : str
    infos : Dict[str,Any] = {}
    received_timestamp : str = None

###################
# SessionEndEvent #
###################

class SessionEndEvent(BaseModel):
    application_id : str
    session_id : str
    received_timestamp : str = None

#################
# TrackingEvent #
#################

class TrackingEvent(BaseModel):
    application_id : str
    session_id : str
    event_name: str
    event : Dict[str,Any] = {}

################
# ServingEvent #
################

class ServingEvent(BaseModel):
    application_id : str
    session_id : str
    decorator_name : str
    inputs : Dict[str,Any] = {}
