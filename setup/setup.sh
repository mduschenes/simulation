#!/bin/bash

# Environment
env=${1:-env}
requirements=${2:-requirements.txt}
architecture=${3:-cpu}
envs=${4:-${HOME}/envs}

# Load modules
if [[ ${architecture} == "cpu" ]]
then
	modules=(python)
elif [[ ${architecture} == "gpu" ]]
then
	modules=(python/3.11.5 cuda/12.2)
else
	modules=(python)
fi

module purge &>/dev/null 2>&1
module load ${modules[@]}


# Setup environment
dir=$(dirname ${env})
env=$(basename ${env})
if [[ ! ${dir} == . ]];then envs=${dir};fi

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