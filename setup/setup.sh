#!/bin/bash

# Setup Environment
env=${1:-jax}

pkgs=/pkgs/anaconda3/bin
envs=${HOME}/condaenvs
channel=intel
requirements=requirements.txt


# Setup conda
mkdir -p ${envs}


[[ -f ${pkgs} || -d ${pkgs} ]] && export PATH=${pkgs}:${PATH}

which conda

conda create -p ${envs}/${env}

export PYTHONPATH=${envs}/${env}:$PYTHONPATH


source ~/.bashrc

conda activate ${envs}/${env}
source activate ${envs}/${env}
# source ${HOME}/env/${env}.env


# Install packages
conda install --file ${requirements} --channel ${channel}



