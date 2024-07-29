#!/usr/bin/env bash

mkdir -p ${HOME}/conda/envs
conda deactivate
conda remove --name env --all
conda create --prefix ${HOME}/conda/envs/env
conda activate env
conda install --channel conda-forge --file requirements.txt
pytest -rA -W ignore::DeprecationWarning test.py


# export LD_LIBRARY_PATH=/scratch/ssd001/pkgs/cuda-XXXX/lib64:/scratch/ssd001/pkgs/cudnn-XXXX/lib64:$LD_LIBRARY_PATH

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/scratch/ssd001/pkgs/cuda-XXXX

# export PATH=/scratch/ssd001/pkgs/cuda-XXXX/bin:$PATH