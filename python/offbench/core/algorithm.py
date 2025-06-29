from .agent import Agent, AgentsDB
from .logger import Logger

from abc import ABC, abstractmethod
from omegaconf import DictConfig, ListConfig



class SingleLearningAlgorithm(ABC):

    """
    Abstract base class to define a learning algorithm for a single task.
    """

    @abstractmethod
    def run(
        algo_cfg: DictConfig,
        task_cfg: DictConfig,
        agent_cfg: DictConfig,
        agents_db: AgentsDB,
        logger: Logger,
        verbose: bool) -> Agent:
        """
        Runs the training algorithm.

        Args:
            algo_cfg (DictConfig): Configuration for the learning algorithm.
            task_cfg (DictConfig): Configuration for the task.
            agent_cfg (DictConfig): Configuration for the agent.
            agents_db (AgentsDB): A database to store agents.
            logger (Logger): A logger to log information.
            verbose (bool): Whether to print detailed information during training.

        Returns:
            (Agent): The trained agent.
        """
        raise NotImplementedError



class ContinualLearningAlgorithm(ABC):

    """
    Abstract base class to define a continual learning algorithm for an identified single task (over a sequence of tasks).
    """

    @abstractmethod
    def run(
        algo_cfg: DictConfig,
        task_idx: int,
        tasks_cfg: ListConfig,
        agent_cfg: DictConfig,
        previous_agents_db: AgentsDB,
        current_agents_db: AgentsDB,
        logger: Logger,
        verbose: bool) -> Agent:
        """
        Runs the training algorithm.

        Args:
            algo_cfg (DictConfig): Configuration for the learning algorithm.
            task_idx (int): The index of the current task.
            tasks_cfg (ListConfig): Configuration for the tasks.
            agent_cfg (DictConfig): Configuration for the agent.
            previous_agents_db (AgentsDB): A database to store agents from the previous task.
            current_agents_db (AgentsDB): A database to store agents for the current task.
            logger (Logger): A logger to log information.
            verbose (bool): Whether to print detailed information during training.

        Returns:
            (Agent): The trained agent.
        """
        raise NotImplementedError
