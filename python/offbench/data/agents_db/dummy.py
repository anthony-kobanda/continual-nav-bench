import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")

from offbench.core.agent import Agent, AgentsDB
from typing import Dict, List



class DummyAgentsDB(AgentsDB):
    
    """
    A database of agents used for testing, debugging, or when no database is needed.
    """

    def __init__(self) -> None:
        self._ids: Dict[str, List[int]] = {}

    def __contains__(self, agent_id: str) -> bool:
        """
        Returns True if the agent_id is in the database.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (bool) True if the agent_id is in the database.
        """
        return agent_id in self._ids

    def __len__(self) -> int:
        """
        Returns the number of agent_ids in the database.

        Returns:
            (int) The number of agent_ids in the database.s
        """
        return len(self.get_ids())

    def __repr__(self) -> str:
        """
        Returns a string representation of the database.

        Returns:
            (str) A string representation of the database.
        """
        return f"DummyAgentsDB() with {len(self)} agents: {self.get_ids()}"

    def add_agent(self, agent: Agent, agent_id: str, agent_stage: int) -> None:
        """
        Adds an agent to the database.

        Args:
            agent (Agent): The agent to add.
            agent_id (str): The agent ID.
            agent_stage (int): The agent stage.
        
        Returns:
            None
        """
        if not agent_id in self._ids: self._ids[agent_id] = []
        if not agent_stage in self._ids[agent_id]: self._ids[agent_id].append(agent_stage)

    def delete_agent(self, agent_id: str, agent_stage: int = None) -> None:
        """
        Deletes an agent from the database.

        Args:
            agent_id (str): The agent ID.
            agent_stage (int): The agent stage (if None, all stages are deleted). Default to None.

        Returns:
            None
        """
        if agent_id in self._ids:
            stages = [agent_stage] if not agent_stage is None else self.stages(agent_id)
            for stage in stages:
                self._ids[agent_id].remove(stage)
            if len(self._ids[agent_id]) == 0: del self._ids[agent_id]

    def get_ids(self) -> List[str]:
        """
        Returns the list of agent IDs in the database.

        Returns:
            (List[str]) The list of agent IDs in the database.
        """
        return list(self._ids.keys())

    def get(self, agent_id: str, agent_stage: int) -> Agent:
        """
        Returns the agent with the specified ID and stage.

        Args:
            agent_id (str): The agent ID.
            agent_stage (int): The agent stage.
        
        Returns:
            (Agent) The agent with the specified ID and stage.
        """
        if not agent_id in self.get_ids(): raise ValueError(f"Agent ID {agent_id} not in the database.")
        elif not agent_stage in self.stages(agent_id): raise ValueError(f"Agent ID {agent_id} does not have stage {agent_stage}.")
        else: return None

    def get_first(self, agent_id: str) -> Agent:
        """
        Returns the first agent of the specified ID.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (Agent) The first agent of the specified ID.
        """
        assert agent_id in self.get_ids()
        stage = min(self.stages(agent_id))
        return self.get(agent_id, stage)

    def get_last(self, agent_id: str) -> Agent:
        """
        Returns the last agent of the specified ID.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (Agent) The last agent of the specified ID.
        """
        assert agent_id in self.get_ids()
        stage = max(self.stages(agent_id))
        return self.get(agent_id, stage)

    def n_stages(self, agent_id: str) -> int:
        """
        Returns the number of stages of the specified agent.

        Args:
            agent_id (str): The agent ID.
        
        Returns:
            (int) The number of stages of the specified agent.
        """
        if not agent_id in self.get_ids(): return 0
        return len(self.stages(agent_id))
    
    def stages(self, agent_id: str) -> List[int]:
        """
        Returns the stages of the specified agent.

        Args:
            agent_id (str): The agent ID.

        Returns:
            (List[int]) The stages of the specified agent
        """
        if not agent_id in self.get_ids(): return []
        return self._ids[agent_id]
