#!/bin/bash

# Environment
env=${1:-env}
requirements=${2:-requirements.txt}
envs=${3:-${HOME}/conda/envs}
device=${4:-cpu}
exe=${5:-conda}

# Load modules
if [[ ${device} == "cpu" ]]
then
	modules=(python)
elif [[ ${device} == "gpu" ]]
then
	modules=(python cuda)
else
	modules=(python)
fi

module purge
module load ${modules[@]}


# Setup environment
mkdir -p ${envs}
deactivate &>/dev/null 2>&1
rm -rf ${envs}/${env}
virtualenv --no-download ${envs}/${env}

# Activate environment
source ${envs}/${env}/bin/activate

# Install environment
options=()
options+=(--no-index)
options+=(-r ${requirements})

pip install ${options[@]}

# Test environment
pytest -rA -W ignore::DeprecationWarning test.py
rm -rf __pycache__ .pytest_cache