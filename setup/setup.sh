#!/usr/bin/env bash


env=${1:-env}
requirements=${2:-requirements.txt}
envs=${3:-${HOME}/envs}
# modules=(${4:-cuda-12.3})
modules=(${4:-})

module purge
module load python cuda cudnn
mkdir -p ${envs}
deactivate
rm -rf ${envs}/${env}
virtualenv --no-download ${envs}/${env}
source ${envs}/${env}/bin/activate
pip install --no-index --requirement requirements.txt
pytest -rA -W ignore::DeprecationWarning test.py