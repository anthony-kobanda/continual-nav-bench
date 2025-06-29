import importlib
import sys

from omegaconf import DictConfig
from typing import Any, Dict, Type, Union



def dynamic_import(name:str,fallback:bool=True) -> Any:

    try:

        if ":" in name: package_name, object_name = name.rsplit(":", 1)
        elif fallback: package_name, object_name = name.rsplit(".", 1)
        else: package_name, object_name = name, None

        sys.path.append(".")
        package = importlib.import_module(package_name)

        if object_name is None: return package
        
        for object_key in object_name.split("."):
            package = getattr(package, object_key)
        
        return package

    except AttributeError as e: 
        raise AttributeError(f"Object '{object_name}' could not be found in module '{package_name}'.") from e

    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Module '{package_name}' could not be found.") from e



def get_class(arguments:Union[Dict,DictConfig]) -> Type:
    return dynamic_import(arguments["classname"])



def get_arguments(arguments:Union[dict,DictConfig],**kwargs) -> dict:
    arguments = dict(arguments)
    arguments.pop("classname",None)
    arguments.update(kwargs)
    return arguments



def instantiate_class(arguments:Union[Dict,DictConfig],**kwargs) -> Any:
    return get_class(arguments)(**get_arguments(arguments,**kwargs))
