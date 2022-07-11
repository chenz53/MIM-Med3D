## Table of Contents
- [Introduction](#introduction)
- [Install](#install)
- [Replication of Experiments](#replication)

## Introduction
A self-supervised learning framework for internal usage (ImagingAI)

## Install
Install `conda` (Recommended) & Setup the Python environment
```bash
# Conda Installation
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
$ bash miniconda.sh -b -p $HOME/miniconda
$ source "$HOME/miniconda/etc/profile.d/conda.sh"
$ hash -r
$ conda config --set always_yes yes --set changeps1 no
$ conda update -q conda
$ conda info -a
# Create the environment
$ conda create -n <name> python=3.8
# Install Pytorch (careful with the cuda version)
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# Clone the repo and setup
$ git clone https://github.web.bms.com/chenz53/bmssl.git
$ cd bmssl
$ python3 -m pip install -r requirements.txt
$ python3 setup.py develop
```

## Replication of Experiments
First, configure the yaml files in `code/configs/ssl/` and set the experimental settings, then run
```Python
sh slurm_train.sh
```

In `slurm_train.sh`, the `wrap` command stands for the command using `train.sh`, the command will be in the following style:
```Python
sh train.sh code/experiments/(path/to/your/experiment/main.py) code/configs/(path/to/your/experiment/config)
```