# **Launch a Training and Monitor Results**

___
___

We suppose you have followed the instruction provided in the installation pages ([Windows (11)](./installation/WINDOWS.md), [WSL2 (Ubuntu 22.04)](./installation/WSL.md), or [Linux (Ubuntu 22.04)](./installation/LINUX.md)).

When visualizing or playing, you may have to change the executable type depending on your configuration.
We only provide executables for *Windows*, *Ubuntu 20.04*, and *Ubuntu 22.04*.

In the following we will consider the Windows framework.

## **1. Initialization**

Activate the conda environment :

```powershell
cd C:/Users/<YOU>/workspace_continual/
conda activate offbench
```

When launching trainings, the experiment folders containing the logs and models will be set to `C:/Users/<YOU>/workspace_continual/experiments/`.

Then, a first simple way to track the training and evaluation results would be through tensorboard (installed with the *continual-navbench* library) :

```powershell
tensorboard --logdir .\experiments\
```

## **2. Training**

To generate trajectories by yourself see this [tutorial](../datasets/README.md).

The configuration folder, containing the yaml configuration files, are contained in the sub-folders of `C:/Users/<YOU>/workspace_continual/continual-nav-bench/experiments/`.

A leaf configuration folder contains three files :

- `analysis.ipynb` ;
- `training.py` ;
- `visualize.py`.

And some folder depending on the os, with one yaml file per algorithm :

- `windows` ;
- `wsl`.

All yaml files have straightforwardly named parameters and/or explanation of their utility in the python script they are called.

Moreover, all trained models are available [here](https://drive.google.com/drive/folders/1QHzGofKymDIkoN1_4FlPwLlJ0pFwilUo?usp=sharing).

### **1.A. Single Task Training \& Visualization**

Given the name of the yaml file and the os to consider, use the following command to launch a training ( e.g. for the GCBC algorithm and some custom parameters ) :

```powershell
python .\experiments\1_singletask\godot_goal_amazeville\train.py  --config-path=windows --config-name=gcbc_base_mlp algo_cfg.log_infos=true seed=100
```

Then to visualize the latest saved agent :

```powershell
python .\experiments\1_singletask\godot_goal_amazeville\visualize.py  --config-path=windows --config-name=gcbc_base_mlp seed=100
```

### **1.B. Continual Training \& Visualization**

Similarly, you can launch training on a stream of tasks defined in a given yaml configuration with the command (the default streams are defined here ``C:/Users/<YOU>/workspace_continual/continual-nav-bench/python/offbench/data/streams/`) :

```powershell
python .\experiments\2_continual\1_random_streams\godot_goal_amazeville\train.py  --config-path=windows --config-name=hgcbc_scratch_1_mlp algo_cfg.log_infos=true seed=100 stream_name=amazeville_random_1
```

Then to visualize the latest saved agent at a given task idx ( starting at 0 ) :

```powershell
python .\experiments\2_continual\1_random_streams\godot_goal_amazeville\visualize.py  --config-path=windows --config-name=hgcbc_scratch_1_mlp seed=100  stream_name=amazeville_random_1 algo_cfg.visu_task_idx=2
```

## **2. Monitor Training**

The `analysis.ipynb` files contained scripts to generate the metrics for the given task, or stream of tasks, for the different yaml files pre-configured. If you add your own method, or set of parameters, you may have to update these files.

___
___
