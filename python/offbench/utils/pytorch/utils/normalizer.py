import torch

from typing import Union



AVAILABLE_NORMALIZERS = ["minmax", "z"]



class MinMax_Normalizer:

    def __init__(
        self,
        min_tensor: torch.Tensor,
        max_tensor: torch.Tensor) -> None:
        assert torch.all(min_tensor != max_tensor), "min_tensor and max_tensor must be different"
        self.min_tensor = min_tensor
        self.diff = max_tensor - min_tensor
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_tensor) / self.diff
    
    def to(self,device: Union[str, torch.device]) -> "MinMax_Normalizer":
        self.min_tensor = self.min_tensor.to(device)
        self.diff = self.diff.to(device)
        return self



class Z_Normalizer:

    def __init__(
        self,
        mean_tensor: torch.Tensor,
        std_tensor: torch.Tensor) -> None:
        assert torch.all(std_tensor != 0), "std_tensor must be non-zero"
        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_tensor) / self.std_tensor
    
    def to(self,device: Union[str, torch.device]) -> "Z_Normalizer":
        self.mean_tensor = self.mean_tensor.to(device)
        self.std_tensor = self.std_tensor.to(device)
        return self
