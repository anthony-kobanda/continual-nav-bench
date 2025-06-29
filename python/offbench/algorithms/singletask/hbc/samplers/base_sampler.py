import numpy as np
import torch

from offbench.core.data import EpisodesDB, Sampler
from tqdm import tqdm
from typing import Dict, List, Optional, Union



class HBC_Base_Sampler(Sampler):

    """
    Sampler for Hierarchical Behavioural Cloning (HBC) Algorithm.

    Args:
        episode_db (EpisodesDB): The database of episodes.
        waysteps (int): The number of waysteps (sub-goal step distance).
        percentage_episodes: (int): The percentage of episodes to use. Defaults to 1.
        batch_size (int): The batch size. Defaults to 1.
        context_size (int): The context size. Defaults to 0.
        padding_size_begin (int): The padding size at the beginning of the episodes. Defaults to 0.
        padding_size_end (int): The padding size at the end of the episodes. Defaults to 0.
        padding_value_begin (float): The padding value at the beginning of the episodes. Defaults to 0.
        padding_value_end (float): The padding value at the end of the episodes. Defaults to 0.
        reward_scale_w (float): The reward scale weight. Defaults to 1.
        reward_scale_b (float): The reward scale bias. Defaults to 0.
        seed (Optional[int]): The seed for the random number generators. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cpu".
    """
    
    def __init__(
        self,
        episodes_db: EpisodesDB,
        waysteps: int,
        percentage_episodes: float = 1.0,
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

        n_episodes = int(len(episodes_db) * percentage_episodes)
        
        super().__init__(
            episodes_db=episodes_db,
            n_episodes=n_episodes,
            batch_size=batch_size,
            context_size=context_size,
            padding_size_begin=padding_size_begin,
            padding_size_end=padding_size_end,
            padding_value_begin=padding_value_begin,
            padding_value_end=padding_value_end,
            reward_scale_w=reward_scale_w,
            reward_scale_b=reward_scale_b,
            seed=seed,
            device=device
        )

        # dataset

        self._dataset: Dict[str, np.ndarray] = None
        self._n_frames: int = None

        # HBC specific

        self._end_idxs: np.ndarray = None
        self._waysteps = waysteps     
    
    def __repr__(self) -> str:
        repr = ""
        repr += "\n- Sampler configuration :"
        repr += "\n\t* episodes_db : {}".format(self._episodes_db)
        repr += "\n\t* waysteps : {}".format(self._waysteps)
        repr += "\n\t* using : {} episodes ({} %)".format(self._n_episodes,self._n_episodes/len(self._episodes_db)*100)
        repr += "\n\t* dataset size : {}".format(len(self))
        repr += "\n\t* batch size : {}".format(self._batch_size)
        repr += "\n\t* context size : {}".format(self._context_size)
        repr += "\n\t* padding size begin : {}".format(self._padding_size_begin)
        repr += "\n\t* padding size end : {}".format(self._padding_size_end)
        repr += "\n\t* padding value begin : {}".format(self._padding_value_begin)
        repr += "\n\t* padding value end : {}".format(self._padding_value_end)
        repr += "\n\t* reward scale weight : {}".format(self._reward_scale_w)
        repr += "\n\t* reward scale bias : {}".format(self._reward_scale_b)
        repr += "\n\t* seed : {}".format(self._seed)
        repr += "\n\t* device : {}".format(self._device)
        return repr

    @torch.no_grad()
    def _get_episode(self, episode_id: str) -> Dict[str, torch.Tensor]:

        # get episode from database

        episode_dict:Dict[str, torch.Tensor] = self._episodes_db[(episode_id,0)]

        # reward scaling
        episode_dict["reward"] = self._reward_scale_w * episode_dict["reward"] + self._reward_scale_b
        
        # padding

        for k, v in episode_dict.items():
            
            v_contiguous = []

            # begin
            if self._padding_size_begin > 0: 
                v_contiguous.append(torch.full((1, self._padding_size_begin, *v.shape[2:]), self._padding_value_begin, dtype=v.dtype, device=v.device))
            # middle
            v_contiguous.append(v)
            # end
            if self._padding_size_end > 0:
                v_contiguous.append(torch.full((1, self._padding_size_end, *v.shape[2:]), self._padding_value_end, dtype=v.dtype, device=v.device))
            
            episode_dict[k] = torch.cat(v_contiguous, dim=1).contiguous()
        
        # padding mask
        
        is_padding = torch.zeros(1, self._padding_size_begin + self._episodes_lengths[episode_id] + self._padding_size_end, dtype=torch.bool)
        if self._padding_size_begin > 0: is_padding[:,:self._padding_size_begin] = True
        if self._padding_size_end > 0: is_padding[:,-self._padding_size_end:] = True
        episode_dict["is_padding"] = is_padding

        return episode_dict
    
    @torch.no_grad()
    def initialize(self, verbose: bool = False, desc: str = "Initializing sampler...", **kwargs) -> None:

        disable_tqdm = not verbose

        dataset: Dict[str, List[np.ndarray]] = {}

        total = 0

        for episode_id in tqdm(self._episode_ids, disable=disable_tqdm, desc=desc):
            
            episode_dict = self._get_episode(episode_id)
            episode_dict = {k: v.cpu().numpy() for k,v in episode_dict.items()}

            length = self._episodes_lengths[episode_id]
            end_idx = np.array([total + length - 1 for _ in range(length)])
            if self._end_idxs is None: self._end_idxs = end_idx
            else: self._end_idxs = np.concatenate([self._end_idxs, end_idx],axis=0)
            total += self._episodes_lengths[episode_id]
            
            frames = {k: [episode_dict[k][:,t:t+self._context_size+1] for t in range(self._episodes_lengths[episode_id])] for k in episode_dict.keys()}
            frames = {k: np.concatenate(v,axis=0) for k,v in frames.items()}

            if not dataset: dataset = {k: [] for k in frames.keys()}
            for k,v in frames.items():
                dataset[k].append(v)
        
        self._dataset = {k: np.concatenate(v,axis=0) for k,v in dataset.items()}
        self._n_frames = len(self._dataset["is_padding"])
    
    @torch.no_grad()
    def normalizer_values(self) -> Dict[str, Dict[str, torch.Tensor]]:
        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for k,v in self._dataset.items():
            if k.startswith("observation/"):
                tensor_v = torch.as_tensor(v, device=self._device).view(-1, v.shape[-1])
                results[k.split("/")[1]] = {
                    "mean": tensor_v.mean(dim=0),
                    "std": tensor_v.std(dim=0) + 1e-6,
                    "min": tensor_v.min(dim=0),
                    "max": tensor_v.max(dim=0)
                }
        return results
    
    @torch.no_grad()
    def sample_batch(self) -> Union[Dict[str, torch.Tensor], None]:

        """
        Sample a batch of frames (or episodes) from the database.

        Returns:
            
        """

        idxs = np.random.randint(0, self._n_frames, self._batch_size)

        # high
        ######

        high_goal = torch.as_tensor(self._dataset["observation/goal"][idxs])

        # low
        #####

        sub_goal_idxs = idxs + self._waysteps
        sub_goal_idxs += np.random.randint(-2, 3, self._batch_size)
        sub_goal_idxs = np.clip(sub_goal_idxs, idxs, self._end_idxs[idxs])
        low_goal = self._dataset["observation/pos"][sub_goal_idxs][:,-self._padding_size_end-1]
        low_goal = np.repeat(low_goal[:,np.newaxis], self._context_size + 1, axis=1)

        # relabelling
        #############

        batch = {k: torch.as_tensor(v[idxs], device=self._device) for k,v in self._dataset.items()}

        batch["high_goal"] = high_goal
        batch["low_goal"] = low_goal

        batch["high_done"] = batch["done"]
        batch["low_done"] = torch.norm(batch["observation/pos"] - batch["low_goal"], dim=-1) < 1e-3

        return batch

    def __len__(self) -> int:
        """
        Return the total number of elements in the dataset.
        """
        return self._n_frames
