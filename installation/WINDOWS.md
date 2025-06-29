# **Installation** ( [Windows 11](https://news.microsoft.com/fr-ca/2021/06/29/presentation-de-windows-11/) )

___
___

It is advised to create a workspace in your home directory ( `C:/Users/<YOU>/` ) that will contain all needed repository and data :

( `git clone` may not work depending on the os configuration, you would have to directly copy the folder from the main branch )

```powershell
cd C:/Users/<YOU>/
mkdir -p workspace_continual
```

From now on, it is supposed we have such a workspace ( `C:/Users/<YOU>/workspace_continual` ).

As we use [Hydra](https://hydra.cc/docs/intro/) it is advised to verify (and to modify) the paths considered in the `.yaml` configurations files.

## **1. Python**

### **1.A. Python Installation**

Follow the instructions in the provided links to install the latest version of either [Anaconda](https://docs.anaconda.com/free/anaconda/install/windows/), [Miniconda](https://docs.anaconda.com/free/miniconda/index.html), or [Miniforge](https://github.com/conda-forge/miniforge).

Here is how to simply do it with **Miniforge** :

- Download and execute the [Windows installer](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe).

- Following the prompts, select *"Create start menu shortcuts"* and *"Add Miniforge3 to my PATH environment variable"*.

To complete the installation, make sure to refresh all terminals opened during the process.

### **1.B. Conda Channels Setup**

Create a `.condarc` file in your home directory ( `C:/Users/<YOU>/` ) and edit it with the following content :

```text
channels:
    - conda-forge
```

Add additional channels for package installation :

```powershell
conda config --env --add channels nvidia
conda config --env --add channels pytorch
```

And update conda :

```powershell
conda update -n base -c conda-forge conda
```

Here is a command to show the channels :

```powershell
conda config --show channels
```

The output should be :

```bash
channels:
  - conda-forge
  - pytorch
  - nvidia
```

### **1.C. Conda Environment Setup**

It is advised to create a workspace containing all needed repository and data :

And to create a new environment :

```powershell
conda create -n offbench python=3.10.14 -y
conda activate offbench
conda install -c conda-forge pip
conda install -c nvidia cuda-toolkit=12.1
python -m pip install --upgrade pip
```

From now on, it is supposed we are working within such an environment (`offbench`).

## **2. Install Current Library**

Follow these commands :

```powershell
cd C:/Users/<YOU>/
cd ./workspace_continual/
git clone https://github.com/anosubcog9438/continual-nav-bench.git
```

## **3. Required Packages & Libraries**

### **3.A. Requirements**

Follow these commands :

```powershell
cd C:/Users/<YOU>/
cd ./workspace_continual/continual-nav-bench/python/
pip install -r requirements.txt
pip install -e .
```

### **3.B. Pytorch**

Run this command (to install the CUDA 12.1 version) :

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## **4. Datasets**

Follow the [README](../README.md) instructions.

___
___
