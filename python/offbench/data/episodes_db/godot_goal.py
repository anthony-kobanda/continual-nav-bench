import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import torch

from filelock import FileLock
from offbench.core.data import Episode, EpisodesDB
from offbench.utils.paths import DATASETS_PATH
from typing import Any, Iterator, List, Tuple, Union



CURRENT_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))



class GodotGoalEpisodesDB(EpisodesDB):

    """
    A database of episodes stored as PyTorch files.

    Args:
        env (str): The environment name.
    """

    def __init__(self, env: str, directory: str = None) -> None:

        self._env = env
        if not directory is None: self._directory = directory
        else: self._directory = os.path.join(DATASETS_PATH, "godot", *self._env.split("-"))

        # check if directory exists
        if not os.path.exists(os.path.join(CURRENT_FOLDER_PATH,"locks")):
            os.makedirs(os.path.join(CURRENT_FOLDER_PATH,"locks"))

        lock = FileLock(os.path.join(CURRENT_FOLDER_PATH,"locks",f"{env}.lock"))
        with lock:
            if not os.path.exists(self._directory): 
                os.makedirs(self._directory)

    def __contains__(self, episode_id: str) -> bool:
        """
        Returns True if the episode_id is in the database.

        Args:
            episode_id (str): The episode ID.
        
        Returns:
            (bool): True if the episode_id is in the database.
        """
        return episode_id in self.get_ids()

    def __getitem__(self, episode_id: Union[str,Tuple[str,Any]]) -> Union[Episode,dict[str,torch.Tensor]]:
        """
        Returns the episode with the given episode_id.

        Args:
            episode_id (str): The episode ID.

        Returns:
            (Episode): The episode with the given episode_id.
        """
        if isinstance(episode_id, str):
            filename = os.path.join(self._directory, episode_id + ".pt")
            episode_dict = torch.load(filename)
            episode_dict["reward"][:,-1] = 0.0
            return Episode.from_dict(episode_dict, episode_id)
        else:
            filename = os.path.join(self._directory, episode_id[0] + ".pt")
            episode_dict = torch.load(filename)
            episode_dict["reward"][:,-1] = 0.0
            return episode_dict

    def __iter__(self) -> Iterator[Episode]:
        """
        Returns an iterator over the episodes in the database.

        Returns:
            (Iterator[Episode]): An iterator over the episodes in the database.
        """
        ids = self.get_ids()
        for episode_id in ids:
            yield self[episode_id]

    def __len__(self) -> int:
        """
        Returns the number of episode_ids in the database.

        Returns:
            (int): The number of episode_ids in the database.
        """
        return len(self.get_ids())

    def __repr__(self) -> str:
        """
        Returns a string representation of the database.

        Returns:
            (str): A string representation of the database.
        """
        return f"GodotGoalEpisodesDB(env={self._env}) with {len(self)} episodes."

    def add_episode(self, episode: Episode) -> None:
        """
        Adds an episode to the database.

        Args:
            episode (Episode): The episode to add.

        Returns:
            None
        """
        episode_id, ids = episode.get_id(), self.get_ids()
        assert episode_id not in ids
        lock = FileLock(os.path.join(self._directory, "lock.lock"))
        with lock:
            filename = os.path.join(self._directory, episode_id + ".pt")
            torch.save(episode.to_dict(), filename)
        return None

    def delete_episode(self, episode_id: str) -> None:
        """
        Deletes an episode from the database.

        Args:
            episode_id (str): The episode ID.
        
        Returns:
            None
        """
        assert episode_id in self.get_ids()
        lock = FileLock(os.path.join(self._directory, "lock.lock"))
        with lock:
            filename = os.path.join(self._directory, episode_id + ".pt")
            os.remove(filename)
        return None

    def pop(self, episode_id: str) -> Episode:
        """
        Deletes an episode from the database and returns it.

        Args:
            episode_id (str): The episode ID.

        Returns:
            episode (Episode): The deleted episode.
        """
        episode = self[episode_id]
        lock = FileLock(os.path.join(self._directory, "lock.lock"))
        with lock:
            filename = os.path.join(self._directory, episode_id + ".pt")
            os.remove(filename)
        return episode

    def get_ids(self) -> List[str]:
        """
        Returns the list of episode_ids in the database.

        Returns:
            (List[str]): The episode_ids in the database.
        """
        return [".".join(f.split(".")[:-1]) for f in os.listdir(self._directory) if f.endswith(".pt")]
