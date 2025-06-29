import copy
import time

from typing import Any, Dict
from offbench.core.agent import Agent, AgentsDB
from tqdm import tqdm



def save_agent(agent:Agent,agents_db:AgentsDB,current_step:int,eval_mode_args:Dict[str,Any]) -> None:
    start_save = time.time()
    if current_step == 0: tqdm.write(">>>>> Saving initial agent...")
    else: tqdm.write(">>>>> Saving agent at gradient step {}...".format(current_step))
    _agent = copy.deepcopy(agent)
    _agent = _agent.to("cpu")
    _agent = _agent.set_eval_mode(**eval_mode_args)
    agents_db.add_agent(_agent,_agent.get_id(),current_step)
    tqdm.write(f"      Agent saved in {time.time()-start_save:.2f} seconds\n")
