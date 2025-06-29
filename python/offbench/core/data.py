import numpy as np
import random as rd
import torch
import uuid

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Dict, Generator, Iterator, List, NoReturn, Optional, Tuple, Union



def flatten_dict(dictionary: Dict[str, Any], parent_key: str = '', separator: str = '/') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        dictionary (Dict[str, Any]): The dictionary to flatten.
        parent_key (str): The parent key. Defaults to ''.
        separator (str): The separator for the keys. Defaults to '/'.

    Returns:
        flattened_dictionary (Dict[str, Any]): The flattened dictionary.
    """
    # stack dictionaries and their parent keys (to avoid using recursion)
    stack: List[Tuple[Dict[str, Any], str]] = [(dictionary, parent_key)]
    flattened_dictionary: Dict[str, Any] = {}
    while stack:
        current_dictionary, current_key = stack.pop()
        for key, value in current_dictionary.items():
            new_key = f"{current_key}{separator}{key}" if current_key else key
            if isinstance(value, dict): stack.append((value, new_key))
            else: flattened_dictionary[new_key] = value
    return flattened_dictionary



def unflatten_dict(dictionary: Dict[str, Any], separator: str = '/') -> Dict[str, Any]:
    """
    Unflatten a dictionary into a nested one.

    Args:
        dictionary (Dict[str, Any]): The dictionary to unflatten.
        separator (str): The separator for the keys. Defaults to '/'.

    Returns:
        unflattened_dictionary (Dict[str, Any]): The unflattened dictionary.
    """
    unflattened_dictionary = {}
    for key, value in dictionary.items():
        keys = key.split(separator)
        current_dict = unflattened_dictionary
        for other_key in keys[:-1]:
            # use setdefault to avoid overwriting existing dictionaries
            current_dict = current_dict.setdefault(other_key, {})
        current_dict[keys[-1]] = value
    return unflattened_dictionary



class ImmutableMapping(MutableMapping):

    """
    An abstract class that defines read-only and immutable mappings.
    """
    
    __slots__ = []
    
    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} object is immutable.")

    def __delitem__(self, key: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} object is immutable.")



class Frame(ImmutableMapping):

    """
    A data structure that describes the state of an `Episode` at a given timestep.
    It is similar to a dictionary but does not provide any write access.

    The dimension of the tensors in the frame is `(batch_size, tensor_size)`.

    Args:
        observation (Dict[str, torch.Tensor]): The observation/state at the current time step.
        action (Dict[str, torch.Tensor]): The action taken by the agent.
        reward (torch.Tensor): The reward received after taking the action.
        done (torch.Tensor): Indicates whether the episode has ended (terminal state).
        truncated (torch.Tensor): Indicates whether the episode has been truncated (maximum number of steps reached).
        timestep (torch.Tensor): The current timestep in the episode.
    """

    __slots__ = ["_data", "_flattened_cache"]

    def __init__(
        self,
        observation: Dict[str, torch.Tensor], 
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor, 
        done: torch.Tensor, 
        truncated: torch.Tensor, 
        timestep: torch.Tensor) -> None:

        self._data: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "timestep": timestep
        }

        self._flattened_cache: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = None

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self) -> str:
        return f"Frame({', '.join([f'{key}={value}' for key, value in self._data.items()])})"

    def items(self) -> Generator[tuple[str, Any], Any, None]:
        for key in self:
            yield key, self._data[key]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        if self._flattened_cache is None:
            self._flattened_cache = flatten_dict(self._data)
        return flatten_dict(self._data)
    
    @staticmethod
    def from_dict(data: Dict[str, torch.Tensor]) -> "Frame":
        return Frame(**unflatten_dict(data))

    def to(self, device: Union[str, torch.device]) -> "Frame":
        for key, value in self._data.items():
            if isinstance(value, torch.Tensor):
                self._data[key] = value.to(device)
            elif isinstance(value, dict):
                self._data[key] = {sub_key: sub_value.to(device) for sub_key, sub_value in value.items()}
        return self



class Episode:

    """
    A class representing an episode, which is a sequence of frames in a reinforcement learning environment.

    The dimension of the tensors in the frame is `(batch_size, sequence_length, tensor_size)`.

    Args:
        episode_id (str): The unique identifier of the episode, if None a random UUID is generated. Defaults to None.
    """

    __slots__ = ["_episode_id", "_empty", "_data"]
    
    def __init__(self, episode_id: str = None) -> None:

        self._episode_id: str = episode_id if episode_id else str(uuid.uuid4())
        self._empty: bool = True
        self._data: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {
            "observation": None,
            "action": None,
            "reward": None,
            "done": None,
            "truncated": None,
            "timestep": None
        }

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @torch.no_grad()
    def __getitem__(self, t: int) -> Frame:
        if self._empty:
            raise IndexError("Episode is empty.")
        frame_dict = {}
        for key, value in self._data.items():
            if isinstance(value, torch.Tensor):
                frame_dict[key] = value[:, t]
            elif isinstance(value, dict):
                frame_dict[key] = {sub_key: sub_value[:, t] for sub_key, sub_value in value.items()}
        return Frame(**frame_dict)

    @torch.no_grad()
    def __setitem__(self, t: int, frame: Frame) -> None:
        assert not self._empty, "Episode is empty."
        assert t < len(self), "Index out of range. Episode length is {len(self)} but got {t} as index."
        for key, value in frame.items():
            if isinstance(value, torch.Tensor):
                self._data[key][:, t] = value
            elif isinstance(value, dict):
                for sub_key,sub_value in value.items():
                    self._data[key][sub_key][:, t] = sub_value

    def __iter__(self) -> Iterator[Frame]:
        for t in range(len(self)):
            yield self[t]

    def __len__(self) -> int:
        return 0 if self._empty else self._data["done"].size(1)
    
    def __repr__(self) -> str:
        return f"Episode(episode_id={self._episode_id}," + ", ".join([f"{key}={value}" for key, value in self._data.items()]) + ")"

    def get_id(self) -> str:
        return self._episode_id

    @torch.no_grad()
    def add_frame(self, frame: Frame) -> None:
        if self._empty:
            self._empty = False
            for key,value in frame.items():
                if isinstance(value, torch.Tensor):
                    self._data[key] = value.unsqueeze(1)
                elif isinstance(value, dict):
                    self._data[key] = {sub_key: sub_value.unsqueeze(1) for sub_key, sub_value in value.items()}
        else:
            for key,value in frame.items():
                if isinstance(value, torch.Tensor):
                    self._data[key] = torch.cat((self._data[key], value.unsqueeze(1)), dim=1)
                elif isinstance(value, dict):
                    self._data[key] = {sub_key: torch.cat((self._data[key][sub_key], sub_value.unsqueeze(1)), dim=1) for sub_key, sub_value in value.items()}

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return flatten_dict(self._data)
    
    def from_dict(data: Dict[str, torch.Tensor], episode_id: str = None) -> "Episode":
        episode = Episode(episode_id)
        episode._data = unflatten_dict(data)
        episode._empty = False
        return episode

    def to(self, device: Union[str, torch.device]) -> "Episode":
        for key,value in self._data.items():
            if isinstance(value, torch.Tensor):
                self._data[key] = value.to(device)
            elif isinstance(value, dict):
                self._data[key] = {sub_key: sub_value.to(device) for sub_key, sub_value in value.items()}
        return self



class EpisodesDB(ABC):

    """
    Abstract base class for a database of episodes.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []

    @abstractmethod
    def __contains__(self, episode_id: str) -> bool:
        """
        Check if an episode is in the database.

        Args:
            episode_id (str): The ID of the episode.
        
        Returns:
            (bool): True if the episode is in the database, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, episode_id: Union[str,Tuple[str,Any]]) -> Union[Episode,dict[str,torch.Tensor]]:
        """
        Get an episode from the database given its ID.

        Args:
            episode_id (Union[str,Tuple[str,Any]]): The ID of the episode.

        Returns:
            (Union[Episode,dict[str,torch.Tensor]]): The episode given its ID (eventually as a dictionary of tensors).
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Episode]:
        """
        Iterate over the episodes in the database.

        Returns:
            (Iterator[Episode]): An iterator over the episodes in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of episodes in the database.

        Returns:
            (int): The number of episodes in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Get a string representation of the database.

        Returns:
            (str): A string representation of the database.
        """
        raise NotImplementedError

    @abstractmethod
    def add_episode(self, episode: Episode) -> None:
        """
        Add an episode to the database.

        Args:
            episode (Episode): The episode to add.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_episode(self, episode_id: str) -> None:
        """
        Delete an episode from the database.

        Args:
            episode_id (str): The ID of the episode to delete.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def pop(self, episode_id: str) -> Episode:
        """
        Pop an episode from the database given its ID and return it.

        Args:
            episode_id (str): The ID of the episode to pop.
        
        Returns:
            (Episode): The popped episode.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ids(self) -> List[str]:
        """
        Get the IDs of the episodes in the database.

        Returns:
            (List[str]): The IDs of the episodes in the database.
        """
        raise NotImplementedError



class Sampler(ABC):

    """
    Abstract base class for a sampler of frames (or episodes) from a database of episodes.

    Args:
        episode_db (EpisodesDB): The database of episodes.
        n_episodes (int): The number of episodes to sample. Defaults to None.
        batch_size (int): The batch size. Defaults to 1.
        context_size (int): The context size. Defaults to 0.
        padding_size_begin (int): The padding size at the beginning of the episodes. Defaults to 0.
        padding_size_end (int): The padding size at the end of the episodes. Defaults to 0.
        padding_value_begin (float): The padding value at the beginning of the episodes. Defaults to 0.
        padding_value_end (float): The padding value at the end of the episodes. Defaults to 0.
        seed (Optional[int]): The seed for the random number generators. Defaults to None.
        reward_scalew (float): The reward scale weight. Defaults to 1.0.
        reward_scale_b (float): The reward scale bias. Defaults to 0.0.
        device (Union[str, torch.device]): The device to use. Defaults to "cpu".
    """
    
    def __init__(
        self,
        episodes_db: EpisodesDB,
        n_episodes: int = None,
        batch_size: int = 1,
        context_size: int = 0,
        padding_size_begin: int = 0,
        padding_size_end: int = 0,
        padding_value_begin: float = 0,
        padding_value_end: float = 0,
        reward_scale_w: float = 1.0,
        reward_scale_b: float = 0.0,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu") -> None:

        # general parameters

        assert padding_size_begin >= context_size, "Padding size at the beginning of the episodes must be greater than or equal to the context size."

        self._episodes_db: EpisodesDB = episodes_db
        self._n_episodes: int = n_episodes
        self._batch_size: int = batch_size
        self._context_size: int = context_size
        self._padding_size_begin: int = padding_size_begin
        self._padding_size_end: int = padding_size_end
        self._padding_value_begin: float = padding_value_begin
        self._padding_value_end: float = padding_value_end
        self._reward_scale_w: float = reward_scale_w
        self._reward_scale_b: float = reward_scale_b
        self._seed: Optional[int] = seed
        self._device: Union[str, torch.device] = device

        # seeding

        self._seeding(self._seed)

        # episode ids

        if self._n_episodes is None: self._n_episodes = len(self._episodes_db)
        assert self._n_episodes <= len(self._episodes_db), "Number of episodes to sample is greater than the number of episodes in the database."
        self._episode_ids = rd.sample(self._episodes_db.get_ids(), self._n_episodes)
        self._episodes_lengths = {episode_id: len(self._episodes_db[episode_id]) for episode_id in self._episode_ids}

    def _seeding(self, seed: Optional[int]) -> None:

        self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)
            rd.seed(self._seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)
        
        else:
            np.random.seed()
            rd.seed()
            torch.manual_seed(rd.randint(0, 10000000))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rd.randint(0, 10000000))    
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the sampler.
        """
        raise NotImplementedError
    
    @abstractmethod
    def normalizer_values(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get the normalizer values.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_batch(self) -> Union[Dict[str, torch.Tensor], None]:
        """
        Sample a batch of frames (or episodes) from the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of elements in the dataset.
        """
        raise NotImplementedError



class SamplerMulti(ABC):

    """
    Abstract base class for a sampler of frames (or episodes) from multiple databases of episodes.

    Args:
        episodes_dbs (List[EpisodesDB]): The list of databases of episodes.
        n_episodes (int): The number of episodes to sample. Defaults to None.
        batch_size (int): The batch size. Defaults to 1.
        context_size (int): The context size. Defaults to 0.
        padding_size_begin (int): The padding size at the beginning of the episodes. Defaults to 0.
        padding_size_end (int): The padding size at the end of the episodes. Defaults to 0.
        padding_value_begin (float): The padding value at the beginning of the episodes. Defaults to 0.
        padding_value_end (float): The padding value at the end of the episodes. Defaults to 0.
        seed (Optional[int]): The seed for the random number generators. Defaults to None.
        reward_scalew (float): The reward scale weight. Defaults to 1.0.
        reward_scale_b (float): The reward scale bias. Defaults to 0.0.
        device (Union[str, torch.device]): The device to use. Defaults to "cpu".
    """

    def __init__(
        self,
        episodes_dbs: List[EpisodesDB],
        n_episodes: int = None,
        batch_size: int = 1,
        context_size: int = 0,
        padding_size_begin: int = 0,
        padding_size_end: int = 0,
        padding_value_begin: float = 0,
        padding_value_end: float = 0,
        reward_scale_w: float = 1.0,
        reward_scale_b: float = 0.0,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu") -> None:

        # general parameters

        assert padding_size_begin >= context_size, "Padding size at the beginning of the episodes must be greater than or equal to the context size."

        self._episodes_dbs: List[EpisodesDB] = episodes_dbs
        self._n_episodes: List[int] = [n_episodes] * len(self._episodes_dbs)
        self._batch_size: int = batch_size
        self._context_size: int = context_size
        self._padding_size_begin: int = padding_size_begin
        self._padding_size_end: int = padding_size_end
        self._padding_value_begin: float = padding_value_begin
        self._padding_value_end: float = padding_value_end
        self._reward_scale_w: float = reward_scale_w
        self._reward_scale_b: float = reward_scale_b
        self._seed: Optional[int] = seed
        self._device: Union[str, torch.device] = device

        # seeding

        self._seeding(self._seed)

        # episode ids

        if not self._n_episodes[0] is None: 
            self._n_episodes = [len(episodes_db) for episodes_db in self._episodes_dbs]
        
        for i,episodes_db in enumerate(self._episodes_dbs):
            assert self._n_episodes[i] <= len(episodes_db), "Number of episodes to sample is greater than the number of episodes in the database {}.".format(episodes_db)
        
        self._episodes_ids = [rd.sample(episodes_db.get_ids(), n_episodes) for episodes_db,n_episodes in zip(self._episodes_dbs,self._n_episodes)]
        self._episodes_lengths = [{episode_id: len(episodes_db[episode_id]) for episode_id in episodes_ids} for episodes_db,episodes_ids in zip(self._episodes_dbs,self._episodes_ids)]

    def _seeding(self, seed: Optional[int]) -> None:

        self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)
            rd.seed(self._seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)
        
        else:
            np.random.seed()
            rd.seed()
            torch.manual_seed(rd.randint(0, 10000000))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rd.randint(0, 10000000)) 
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the sampler.
        """
        raise NotImplementedError
    
    @abstractmethod
    def normalizer_values(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get the normalizer values.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_batch(self) -> Union[Dict[str, torch.Tensor], None]:
        """
        Sample a batch of frames (or episodes) from the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of elements in the dataset.
        """
        raise NotImplementedError