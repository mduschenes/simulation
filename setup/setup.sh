#!/usr/bin/env bash

module purge
module load cuda11.8+cudnn8.9.6
mkdir -p /h/mduschen/.conda/envs
conda deactivate
conda remove --name env --all
conda create --prefix /h/mduschen/.conda/envs/env
conda activate env
conda install --channel conda-forge --file requirements.txt.tmp.1

module purge
module load cuda12.3
mkdir -p /h/mduschen/.conda/envs
conda deactivate
conda activate env
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --requirement requirements.txt.tmp.2
pytest -rA -W ignore::DeprecationWarning test.py


# export LD_LIBRARY_PATH=/scratch/ssd001/pkgs/cuda-XXXX/lib64:/scratch/ssd001/pkgs/cudnn-XXXX/lib64:$LD_LIBRARY_PATH

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/scratch/ssd001/pkgs/cuda-XXXX

# export PATH=/scratch/ssd001/pkgs/cuda-XXXX/bin:$PATH