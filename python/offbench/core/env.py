import numpy as np
import random as rd
import torch

from .data import Frame
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional



class Env(ABC):

    """
    `Env` is a framework allowing us to build PyTorch based environments to easily 
    have `Agent` instances interacting with them. During every step of an session,
    an agent will have access to a batch of informations and will have to provide
    an action for each instance of the environment in the batch. It may be useful 
    for fast data generation (if the environment allows parallelization).
    If the environment is not parallelizable, no matter, independently of the set
    batch size, there will be only one instance of the environment.

    Args:
        seed (Optional[int]): Optional seed value to reproduce experiments.
        batch_size (int): The number of instances in the batch. Default to 1.
        autoreset (bool): Whether to automatically reset the environment.
        max_episode_steps (Optional[int]): The maximum number of steps we can perform.
        device (Union[torch.device, str]): The device used by PyTorch.
    """

    def __init__(
            self,
            seed: Optional[int] = None,
            batch_size: int = 1,
            autoreset: bool = True,
            max_episode_steps: Optional[int] = None,
            device: Union[torch.device, str] = "cpu") -> None:
        
        # general parameters
        self._seed: Optional[int] = seed
        self._batch_size: int = batch_size
        self._autoreset: bool = autoreset
        self._max_episode_steps: Optional[int] = max_episode_steps
        self._device: Union[torch.device, str] = device

        # batch data used to generate frames
        self._batch_obs: Dict[str, torch.Tensor] = {}
        self._batch_reward: torch.Tensor = None
        self._batch_done: torch.Tensor = None
        self._batch_truncated: torch.Tensor = None
        self._batch_timestep: torch.Tensor = None
        
        # seeding and tp device
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

    @torch.no_grad()
    def _observe(self) -> Frame:
        """
        Returns a `Frame` with torch.Tensor data describing the current state.

        Returns:
            (Frame): The current state of the environment.
        """
        return Frame(
            observation=self._batch_obs,
            action=None,
            reward=self._batch_reward,
            done=self._batch_done,
            truncated=self._batch_truncated,
            timestep=self._batch_timestep
        )

    @abstractmethod
    def _initialize_observations(self) -> Dict[str, torch.Tensor]:
        """
        Initialize the batch of observations. This method should be overridden
        in the subclass to define specific observation initialization.

        Returns:
            (Dict[str, torch.Tensor]): Initialized batch of observations.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_action(self) -> Frame:
        """
        Returns a `Frame` with torch.Tensor data describing a random action.

        Returns:
            (Frame): A random action.
        """
        raise NotImplementedError

    @torch.no_grad()
    def reset(self, seed: Optional[int] = None, **kwargs) -> Frame:
        
        """
        Resets the `Env`, starts the counter of steps, 
        and then returns its current state.

        Args:
            seed (Optional[int]): Optional seed value to reproduce experiments.

        Returns:
            (Frame): The current state of the environment.
        """
        
        # seeding
        self.seeding(seed)

        # Initialize rewards, done flags, truncated flags, and timesteps
        self._batch_reward = torch.zeros(self._batch_size, device=self._device, dtype=torch.float32)
        self._batch_done = torch.zeros(self._batch_size, device=self._device, dtype=torch.bool)
        self._batch_truncated = torch.zeros(self._batch_size, device=self._device, dtype=torch.bool)
        self._batch_timestep = torch.zeros(self._batch_size, device=self._device, dtype=torch.long)

        # Initialize observations (to be defined in the specific environment)
        self._batch_obs = self._initialize_observations()

        return self._observe()

    @abstractmethod
    def step(self, action_frame: Frame) -> Frame:
        """
        Proceed to one environment step, using action 
        information provided by a `Frame`.
        Then it returns the current state of the `Env`.

        Args:
            action_frame (Frame): The action used to modify the environment state.

        Returns:
            (Frame): The current state of the environment.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Closes the Env and erases all data of the current state.
        """
        self._batch_obs = {}
        self._batch_reward = None
        self._batch_done = None
        self._batch_truncated = None
        self._batch_timestep = None

    def to(self, device: Union[torch.device, str]) -> 'Env':
        """
        Update the torch.device used and all the torch.Tensor data accordingly.

        Args:
            device (Union[torch.device, str]): The device to consider for PyTorch.

        Returns:
            (Env): The updated environment.
        """
        self._device = device        
        self._batch_obs = {key: value.to(self._device) for key, value in self._batch_obs.items()}    
        if self._batch_reward is not None: self._batch_reward = self._batch_reward.to(self._device)        
        if self._batch_done is not None: self._batch_done = self._batch_done.to(self._device)        
        if self._batch_truncated is not None: self._batch_truncated = self._batch_truncated.to(self._device)        
        if self._batch_timestep is not None: self._batch_timestep = self._batch_timestep.to(self._device)        
        return self
