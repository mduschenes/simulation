#!/usr/bin/env bash

env=${1:-env}
requirements=${2:-requirements.txt}
envs=${3:-${HOME}/.conda/envs}
# modules=(${4:-cuda-12.3})
modules=(${4:-cuda11.8+cudnn8.9.6})

rm -rf ${requirements}.tmp.*
awk -v requirements="${requirements}" -v RS= '{print > (requirements".tmp." NR "")}' ${requirements}
requirements=(${requirements}.tmp.*)

mkdir -p ${envs}
conda deactivate
conda remove --name ${env} --all
conda create --prefix ${envs}/${env}

module purge
module load ${modules[@]}
conda activate ${env}
conda install --channel conda-forge --file ${requirements[0]}

module purge
module load ${modules[@]}

# export LD_LIBRARY_PATH=/pkgs/cuda-12.3/lib64:/pkgs/cudnn-11.x-v8.9.2/lib:$LD_LIBRARY_PATH

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/pkgs/cuda-12.3

# export PATH=/pkgs/cuda-12.3/lib64:/pkgs/cudnn-9.2-v7.3.1:$PATH

export LD_LIBRARY_PATH=/pkgs/cuda-11.8/lib64:/pkgs/cuda-11.8/extras/CUPTI/lib64:/pkgs/cudnn-11.x-v8.9.2/lib:$LD_LIBRARY_PATH

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/pkgs/cuda-11.8

export PATH=/pkgs/cuda-11.8:/pkgs/cuda-11.8:/pkgs/cudnn-11.x-v8.9.2:$PATH

conda activate ${env}
# conda install --channel conda-forge --file ${requirements[1]}
pip uninstall --yes $(pip list | grep -e jax -e nvidia -e cuda -e ml-dtypes -e opt-einsum | awk '{ print $1 }')
pip cache purge
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --requirement ${requirements[1]}

pytest -rA -W ignore::DeprecationWarning test.py

rm -rf ${requirements[@]}

