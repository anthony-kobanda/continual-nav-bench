import torch

from abc import ABC, abstractmethod
from typing import Any, Dict, Union



class ApplicationManager(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def application_start(self,name:str,infos:Dict[str,Any],debug:bool=False) -> Dict[str,Any]:
        raise NotImplementedError

    @abstractmethod
    def application_end(self,debug:bool=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def session_start(self,application_id:str,session_name:str,infos:Dict[str,Any],debug:bool=False) -> Dict[str,Any]:
        raise NotImplementedError

    @abstractmethod
    def session_end(self,application_id:str,session_id:str,debug:bool=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def serve(self,application_id:str,session_id:str,decorator_name:str,event:Union[Dict[str,torch.Tensor],None],debug:bool=False) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def push(self,application_id:str,session_id:str,episode_id:str,event:Dict[str,torch.Tensor],debug:bool=False) -> Any:
        raise NotImplementedError
