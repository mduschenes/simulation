#!/bin/bash

# Environment
env=${1:-env}
exe=${2:-conda}
requirements=${3:-requirements.txt}
device=${4:-cpu}
envs=${5:-${HOME}/conda/envs}

# Setup environment
mkdir -p ${envs}
conda deactivate
conda config --remove envs_dirs ${envs} &>/dev/null 2>&1
conda config --append envs_dirs ${envs} &>/dev/null 2>&1
conda remove --name ${env} --all
conda create --prefix ${envs}/${env}

# Activate environment
conda activate ${env}

# Install environment
options=()
options+=(--channel conda-forge)
options+=(--file ${requirements})

conda install ${options[@]}

# Test environment
pytest -rA -W ignore::DeprecationWarning test.py
rm -rf __pycache__ .pytest_cache