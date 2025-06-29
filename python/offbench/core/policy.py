import numpy as np
import random as rd
import torch
import torch.nn as nn

from .data import Frame
from abc import abstractmethod
from torch.nn.parameter import Parameter
from typing import Dict, Iterator, Optional, Union



class Policy(nn.Module):

    """
    Base class for all policies.

    Args:
        seed (Optional[int]): Random seed for initialization. Defaults to None.
        device (Union[torch.device, str]): The device to use ('cpu', 'cuda'). Defaults to 'cpu'.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu") -> None:

        super(Policy, self).__init__()

        self._seed: int = seed
        self._device: Union[torch.device, str] = device

        self.seeding(self._seed)
        
    def seeding(self, seed: Optional[int] = None) -> None:
        """
        Seeds the random number generators.

        Args:
            seed (Optional[int]): Optional seed value to reproduce experiments. Defaults to None.
        
        Returns:
            (None): Nothing.
        """
        self._seed = seed
        if not (self._seed is None): 
            np.random.seed(self._seed)
            rd.seed(self._seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(self._seed)
        else:
            np.random.seed()
            rd.seed()
            torch.manual_seed(rd.randint(0,10000000))
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(rd.randint(0,10000000))
    
    @abstractmethod
    def set_normalizers(self, normalize_values: Dict[str, torch.Tensor], **kwargs) -> "Policy":
        """
        Sets the normalizers for the policy.

        Args:
            normalize_values (Dict[str, torch.Tensor]): The values to normalize the input data.
        """
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, **kwargs) -> "Policy":
        """
        Resets the model's seed and parameters.

        Args:
            seed (Optional[int]): New seed for random number generators. Defaults to None.

        Returns:
            (Policy): The instance of the model after reset.
        """
        self.seeding(seed)
        return self

    @abstractmethod
    def set_train_mode(self, **kwargs) -> "Policy":
        """
        Prepares the model for training.

        Returns:
            (Policy): The model instance ready for training.
        """
        raise NotImplementedError

    @abstractmethod
    def set_eval_mode(self, **kwargs) -> "Policy":
        """
        Prepares the model for evaluation.

        Returns:
            (Policy): The model instance ready for evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def forward(self, inputs: Union[Frame, Dict[str, torch.Tensor]], generator: Optional[torch.Generator] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the policy network.

        Args:
            inputs (Dict[str, torch.Tensor]): Input data for the policy network.
            generator (Optional[torch.Generator]): Random number generator for stochastic operations. Defaults to None.

        Returns:
            (Dict[str, torch.Tensor]): The output of the policy network.
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over all model parameters.

        Returns:
            (Iterator[nn.Parameter]): An iterator over the model parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def inference_parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over model parameters used for inference.

        Returns:
            (Iterator[nn.Parameter]): An iterator over the model parameters used for inference.
        """
        raise NotImplementedError

    @abstractmethod
    def buffers(self) -> Iterator[torch.Tensor]:
        """
        Returns an iterator over all model buffers.

        Returns:
            (Iterator[torch.Tensor]): An iterator over the model buffers.
        """
        raise NotImplementedError

    @abstractmethod
    def inference_buffers(self) -> Iterator[torch.Tensor]:
        """
        Returns an iterator over model buffers used for inference.

        Returns:
            (Iterator[torch.Tensor]): An iterator over the model buffers used for inference.
        """
        raise NotImplementedError

    def size(self) -> float:
        """
        Calculates the total size of the model's parameters and buffers in megabytes (MB).

        Returns:
            (float): The total size of the model in MB.
        """
        param_size = sum(param.nelement() * param.element_size() for param in self.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def inference_size(self) -> float:
        """
        Calculates the size of the model's parameters and buffers used for inference in megabytes (MB).

        Returns:
            (float): The size of the model for inference in MB.
        """
        param_size = sum(param.nelement() * param.element_size() for param in self.inference_parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.inference_buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def to(self, device: Union[torch.device, str]) -> "Policy":
        """
        Moves the model to a specified device.

        Args:
            device (Union[torch.device, str]): The target device.

        Returns:
            (Policy): The model instance after being moved to the specified device.
        """
        self._device = device
        super().to(self._device)
        return self
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor], generator: Optional[torch.Generator] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Computes the loss(es) for the policy.

        Args:
            batch (Dict[str, torch.Tensor]): The batch of data to compute the loss.
            generator (Optional[torch.Generator]): Random number generator for stochastic operations. Defaults to None.

        Returns:
            (torch.Tensor): The computed loss.
        """
        raise NotImplementedError
