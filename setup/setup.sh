#!/bin/bash

# Environment
env=${1:-env}
requirements=${2:-requirements.txt}
envs=${3:-${HOME}/.conda/envs}
device=${4:-cpu}
exe=${5:-conda}

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