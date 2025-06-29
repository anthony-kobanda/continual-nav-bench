import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Union



class Logger(ABC):

    """
    An abstract class for a logger.
    """

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the logger.

        Args:
            params (Dict[str, Any]): The parameters to log.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError
    
    @abstractmethod
    def add_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """
        Add a histogram to the logger.

        Args:
            tag (str): The tag for the histogram.
            values (np.ndarray): The values to log as a histogram. Typically, this is a flattened array of numerical data.
            step (int): The step for the histogram.
            
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def add_image(self,tag:str,image:np.ndarray,step:int) -> None:
        """
        Add an image to the logger.

        Args:
            tag (str): The tag for the image.
            step (int): The step for the image.
            image (np.ndarray): The image to add.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def add_scalar(self,tag:str,value:Union[int,float],step:int) -> None:
        """
        Add a scalar to the logger.

        Args:
            tag (str): The tag for the scalar.
            step (int): The step for the scalar.
            value (Union[int,float]): The value to add.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError
    
    @abstractmethod
    def add_text(self,tag:str,text:str,step:int) -> None:
        """
        Add text to the logger.

        Args:
            tag (str): The tag for the text.
            step (int): The step for the text.
            text (str): The text to add.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the logger.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError
    