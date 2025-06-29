import base64
import io
import numpy
import torch

from PIL import Image
from torchvision import transforms
from typing import Any, Dict



to_tensor = transforms.ToTensor()



def convert_dict_to_tensor(event:Dict[str,Any]) -> torch.Tensor:
    if event["dtype"] == "bool": t = torch.tensor(event["data"]).bool()
    elif event["dtype"] == "double": t = torch.tensor(event["data"]).float()
    elif event["dtype"] == "float": t = torch.tensor(event["data"]).float()
    elif event["dtype"] == "int": t = torch.tensor(event["data"]).long()
    else: raise Exception(f"Invalid `dtype` {event['dtype']}. Available ones are : `int`, `float`, `double`, `bool`")
    return t.reshape(*event["shape"]) if len(event["shape"]) > 1 else t



def event_to_pytorch(event:Dict[str,Any]) -> Dict[str,torch.Tensor]:
    results:Dict[str,torch.Tensor] = {}
    for k,v in event.items():
        if isinstance(v,Dict):
            if "shape" in v and "data" in v and "dtype" in v:
                results[k] = convert_dict_to_tensor(v)
            else:
                r = event_to_pytorch(v)
                for _k,_v in r.items():
                    results[k+"/"+_k] = _v
        elif isinstance(v,torch.Tensor): pass
        elif isinstance(v,int): results[k] = torch.tensor(v).long()
        elif isinstance(v,float): results[k] = torch.tensor(v).float()
        elif isinstance(v,numpy.ndarray): results[k] = torch.from_numpy(v)
        elif isinstance(v,str):
            if k.startswith("image"):
                decoded_bytes = base64.b64decode(v)
                byte_array = bytearray(decoded_bytes)
                image_file = io.BytesIO(byte_array)
                image = Image.open(image_file)
                v = to_tensor(image)
                results[k] = v
            else: pass
    return results
