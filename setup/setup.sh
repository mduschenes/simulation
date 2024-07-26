#!/bin/bash

# Environment
env=${1:-env}

# Paths
envs=${HOME}/env

# Requirements
requirements=${2:-requirements.txt}
device=${3:-cpu}

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


# Setup paths
mkdir -p ${envs}

# Setup environment
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