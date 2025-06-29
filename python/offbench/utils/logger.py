import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent
from typing import List, Tuple



def extract_metrics_from_event_file(event_file_path:str) -> List[Tuple[str,int,float]]:
    """
    Extracts metrics from a TensorBoard event file.
    
    Args:
        event_file_path (str): The path to the TensorBoard event file.
    
    Returns:
        (List[Tuple[str,int,float]]): A list of tuples containing the tag, step, and value of each metric.
    """
    event_acc, metrics = EventAccumulator(event_file_path), []
    event_acc.Reload()
    for tag in event_acc.Tags()['scalars']:
        event:ScalarEvent
        for event in event_acc.Scalars(tag):
            metrics.append((tag, event.step, event.value))
    return metrics
