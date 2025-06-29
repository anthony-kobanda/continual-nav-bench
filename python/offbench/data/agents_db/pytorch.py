import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")

import os
import torch

from offbench.core.agent import Agent, AgentsDB
from filelock import FileLock
from typing import List



CURRENT_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))
LOCK_FILE_PATH = os.path.join(CURRENT_FOLDER_PATH, "locks", "create_db.lock")



class PytorchAgentsDB(AgentsDB):
    
    """
    A database of agents stored as PyTorch tensors.

    Args:
        directory (str): The directory where the agents are stored.
    """
    
    def __init__(self, directory: str) -> None:
        self._directory = directory
        with FileLock(LOCK_FILE_PATH):
            if not os.path.exists(self._directory):
                os.makedirs(self._directory)

    def __contains__(self, agent_id: str) -> bool:
        """
        Returns True if the agent_id is in the database.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (bool): True if the agent_id is in the database.
        """
        return len(self.stages(agent_id)) > 0

    def __len__(self) -> int:
        """
        Returns the number of agent_ids in the database.

        Returns:
            (int): The number of agent_ids in the database.s
        """
        return len(self.get_ids())

    def __repr__(self) -> str:
        """
        Returns a string representation of the database.

        Returns:
            (str): A string representation of the database.
        """
        return f"PytorchAgentsDB(directory={self._directory}) with {len(self)} agents: {self.get_ids()}"

    def add_agent(self, agent: Agent, agent_id: str, agent_stage: int) -> None:
        """
        Adds an agent to the database.

        Args:
            agent (Agent): The agent to add.
            agent_id (str): The agent ID.
            agent_stage (int): The agent stage.
        
        Returns:
            (None): Nothing.
        """
        assert "___" not in agent_id
        assert agent_stage >= 0
        with FileLock(os.path.join(self._directory, "lock.lock")):
            filename = os.path.join(self._directory, f"{agent_id}___{agent_stage}.pt")
            # if agent_stage already exists for agent_id, it is overwritten
            torch.save(agent, filename)

    def delete_agent(self, agent_id: str, agent_stage: int = None) -> None:
        """
        Deletes an agent from the database.

        Args:
            agent_id (str): The agent ID.
            agent_stage (int): The agent stage (if None, all stages are deleted). Default to None.

        Returns:
            (None): Nothing.
        """
        stages = [agent_stage] if not agent_stage is None else self.stages(agent_id)
        with FileLock(os.path.join(self._directory, "lock.lock")):
            for stage in stages:
                filename = os.path.join(self._directory, f"{agent_id}___{stage}.pt")
                if not os.path.exists(filename): pass
                else: os.remove(filename)

    def get_ids(self) -> List[str]:
        """
        Returns the list of agent IDs in the database.

        Returns:
            (List[str]): The list of agent IDs in the database.
        """
        with FileLock(os.path.join(self._directory, "lock.lock")):
            ids = list(set([f.split("___")[0] for f in os.listdir(self._directory) if f.endswith(".pt")]))
        return ids

    def get(self, agent_id: str, agent_stage: int) -> Agent:
        """
        Returns the agent with the specified ID and stage.

        Args:
            agent_id (str): The agent ID.
            agent_stage (int): The agent stage.
        
        Returns:
            (Agent): The agent with the specified ID and stage.
        """
        with FileLock(os.path.join(self._directory, "lock.lock")):
            filename = os.path.join(self._directory, f"{agent_id}___{agent_stage}.pt")
            if not os.path.exists(filename): agent = None
            else: agent = torch.load(filename)
        if agent is None: raise ValueError(f"Agent {agent_id} at stage {agent_stage} not found.")
        return agent

    def get_first_stage(self, agent_id: str) -> Agent:
        """
        Returns the first agent of the specified ID.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (Agent): The first agent of the specified ID.
        """
        assert agent_id in self.get_ids()
        stage = min(self.stages(agent_id))
        return self.get(agent_id, stage)

    def get_last_stage(self, agent_id: str) -> Agent:
        """
        Returns the last agent of the specified ID.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (Agent): The last agent of the specified ID.
        """
        assert agent_id in self.get_ids(), f"Agent {agent_id} not found."
        stage = max(self.stages(agent_id))
        return self.get(agent_id, stage)

    def n_stages(self, agent_id: str) -> int:
        """
        Returns the number of stages of the specified agent.

        Args:
            agent_id (str): The agent ID.
        
        Returns:
            (int): The number of stages of the specified agent.
        """
        with FileLock(os.path.join(self._directory, "lock.lock")):
            n_stages = len([f for f in os.listdir(self._directory) if f.startswith(f"{agent_id}___") and f.endswith(".pt")])
        return n_stages
    
    def stages(self, agent_id: str) -> List[int]:
        """
        Returns the stages of the specified agent.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (List[int]): The stages of the specified agent
        """
        with FileLock(os.path.join(self._directory, "lock.lock")):
            stages = [int(f.split("___")[1].replace(".pt", "")) for f in os.listdir(self._directory) if f.startswith(f"{agent_id}___") and f.endswith(".pt")]
        return stages
