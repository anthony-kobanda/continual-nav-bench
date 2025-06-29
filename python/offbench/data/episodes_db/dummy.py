import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from offbench.core.data import Episode, EpisodesDB
from typing import Iterator, List



class DummyEpisodesDB(EpisodesDB):

    """
    A database of episodes used for testing, debugging, or when no database is needed.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []

    def __contains__(self, episode_id: str) -> bool:
        """
        Returns True if the episode_id is in the database.

        Args:
            episode_id (str): The episode ID.
        
        Returns:
            (bool) True if the episode_id is in the database.
        """
        return episode_id in self._ids

    def __getitem__(self, episode_id: str) -> Episode:
        """
        Returns the episode with the given episode_id.

        Args:
            episode_id (str): The episode ID.

        Returns:
            (Episode) The episode with the given episode_id.
        """
        return Episode()

    def __iter__(self) -> Iterator[Episode]:
        """
        Returns an iterator over the episodes in the database.

        Returns:
            (Iterator[Episode]) An iterator over the episodes in the database.
        """
        for episode_id in self._ids:
            yield self[episode_id]

    def __len__(self) -> int:
        """
        Returns the number of episode_ids in the database.

        Returns:
            (int) The number of episode_ids in the database.
        """
        return len(self._ids)

    def __repr__(self) -> str:
        """
        Returns a string representation of the database.

        Returns:
            (str) A string representation of the database.
        """
        return f"DummyEpisodesDB() with {len(self)} episodes."

    def add_episode(self, episode: Episode, episode_id: str) -> None:
        """
        Adds an episode to the database.

        Args:
            episode (Episode): The episode to add.

        Returns:
            None
        """
        assert episode_id not in self._ids
        self._ids.append(episode_id)

    def delete_episode(self, episode_id: str) -> None:
        """
        Deletes an episode from the database.

        Args:
            episode_id (str): The episode ID.
        
        Returns:
            None
        """
        assert episode_id in self._ids
        self._ids.remove(episode_id)

    def pop(self, episode_id: str) -> Episode:
        """
        Deletes an episode from the database and returns it.

        Args:
            episode_id (str): The episode ID.

        Returns:
            (Episode) The deleted episode.
        """
        assert episode_id in self._ids
        self._ids.remove(episode_id)
        return Episode()

    def get_ids(self) -> List[str]:
        """
        Returns the episode_ids in the database.

        Returns:
            (List[str]) The episode_ids in the database.
        """
        return self._ids
