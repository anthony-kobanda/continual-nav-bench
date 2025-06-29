# **Play \& Generate Trajectories**

___
___

We suppose you have followed the instruction provided in the installation pages ([Windows (11)](./installation/WINDOWS.md), [WSL2 (Ubuntu 22.04)](./installation/WSL.md), or [Linux (Ubuntu 22.04)](./installation/LINUX.md)).

When playing, you may have to change the executable type depending on your configuration.
We only provide executables for *Windows*, *Ubuntu 20.04*, and *Ubuntu 22.04*.

In the following we will consider the Windows framework.

The datasets of episodes are available [here](https://drive.google.com/drive/folders/1QHzGofKymDIkoN1_4FlPwLlJ0pFwilUo?usp=sharing).  

## **1. Initialization**

The yaml file `./play.yaml` alongside this `README.md` file contains all the only few parameters to tune in order to play and to generate trajectories.

- First you should select the ***task*** you want to consider :

    <img src="../assets/readme/mazes.png" alt="Mazes Trajectories"/>

    When it comes to naming the task, given the order provided in the above figure, the *SimpleTown* tasks are named `simpletown-maze_i` with `i` between `0` and `7`. For *AmazeVille*, the naming convention of the datasets are `amazeville-maze_i-mode` with `i` between `1` and `4`, and `mode` in `{ low , high }`.

- Then comes ***save_episodes***, a boolean to tune according to whether or not you want to save the episodes you generate.

- The ***dataset_path*** parameter defines the name of the datasets. If set to `None` and *save_episodes* is `True` the generated episodes will automatically fill the default datasets.

- The ***seed*** parameter controls the randomize process of start and goal positions generation.

- The ***verbose*** allows to print some information in the terminal regarding the godot process.

- ***n_episodes*** is the maximum number of episodes to generate during the run of the executable.

- ***max_episode_steps*** is the maximum number of timesteps allowed during an episode.

- ***max_episodes_db_len*** is the maximum number of episodes allowed in the dataset. If set to `None` we can generate as much episodes we want. Else, the executable will stop (and even not be launched) if the number of episodes in the given dataset is reached.

- Finally, the **exe_type** defines the type of the executable to consider. The available ones are : `windows`, `ubuntu_2004`, `ubuntu_2204`.

___
___
