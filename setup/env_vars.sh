#!/bin/bash

envs=${HOME}/env
env=jax

export LD_LIBRARY_PATH=${envs}/${env}/lib:$LD_LIBRARY_PATH

#export XLA_FLAGS=--xla_gpu_cuda_data_dir=/scratch/ssd001/pkgs/cuda-11.3

#export PATH=/scratch/ssd001/pkgs/cuda-11.3/bin:$PATH

# export LD_LIBRARY_PATH=/pkgs/cuda-11.1/lib64:/pkgs/cudnn-11.4-v8.2.4.15/lib64:$LD_LIBRARY_PATH

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/pkgs/cuda-11.1

# export PATH=/pkgs/cuda-11.1/bin:$PATH

