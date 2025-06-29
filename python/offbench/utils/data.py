from offbench.core.data import Frame, Episode
from omegaconf import DictConfig, ListConfig
from typing import Any, Dict, List, Union



def convert_cfg(data:Union[ListConfig,List,DictConfig,Dict,Any]) -> Union[List,Dict,Any]:
    """
    
    Convert a DictConfig or ListConfig object to a standard dictionary or list.

    Args:
        data (Union[ListConfig,List,DictConfig,Dict,Any]): The data to convert.
    
    Returns:
        Union[List,Dict,Any]: The converted data.
    """
    if isinstance(data,(ListConfig,List)): return [convert_cfg(e) for e in data]
    elif isinstance(data,(DictConfig,Dict)): return {k: convert_cfg(v) for k,v in data.items()}
    else: return data



def flatten_dict(d:Dict[str,Any],parent_key:str='',sep:str='/') -> Dict[str,Any]:
    """
    Flatten a nested dictionary.

    Args:
        d (Dict[str,Any]): The dictionary to flatten.
        parent_key (str): The parent key. Defaults to ''.
        sep (str): The separator for the keys. Defaults to '/'.
    
    Returns:
        Dict[str, Any]: The flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items



def unflatten_dict(d: Dict[str, Any], sep: str = '/') -> Dict[str, Any]:
    """
    Unflatten a dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to unflatten.
        sep (str): The separator for the keys. Defaults to '/'.
    
    Returns:
        Dict[str, Any]: The unflattened dictionary.
    """
    result_dict = {}
    for k, v in d.items():
        keys = k.split(sep)
        d = result_dict
        for key in keys[:-1]:
            d = d.setdefault(key,{})
        d[keys[-1]] = v
    return result_dict



def unbatch_frame(frame:Frame) -> List[Frame]:
    """
    Unbatch a `Frame`.

    Args:
        frame (Frame): The frame to unbatch.

    Returns:
        (List[Frame]): The unbatched frames.
    """
    data = frame.to_dict()
    batch_size = data["done"].size(0)
    return [Frame.from_dict({key: value[i].unsqueeze(0) for key, value in data.items()}) for i in range(batch_size)]



def unbatch_episode(episode:Episode) -> List[Episode]:
    """
    Unbatch an `Episode`.

    Args:
        episode (Episode): The episode to unbatch.

    Returns:
        (List[Episode]): The unbatched episodes.
    """
    data = episode.to_dict()
    batch_size = data["done"].size(0)
    return [Episode.from_dict({key: value[i].unsqueeze(0) for key, value in data.items()}) for i in range(batch_size)]
