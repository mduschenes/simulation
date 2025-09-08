#!/usr/bin/env bash

env=${1:-tensor}
requirements=${2:-requirements.txt}
envs=${3:-${HOME}/conda/envs}
modules=(${4:-})
channel=${5:-conda-forge}
test=${6:-test.py}


if [ ! -z ${modules} ]
then
	module purge
	module load ${modules[@]}
fi

mkdir -p ${envs}

conda deactivate
conda remove --name ${env} --all
conda create --prefix ${envs}/${env}
conda activate ${env}

conda install --channel ${channel} --file ${requirements}

pytest -rA -W ignore::DeprecationWarning ${test}


# export LD_LIBRARY_PATH=/scratch/ssd001/pkgs/cuda-XXXX/lib64:/scratch/ssd001/pkgs/cudnn-XXXX/lib64:$LD_LIBRARY_PATH

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/scratch/ssd001/pkgs/cuda-XXXX

# export PATH=/scratch/ssd001/pkgs/cuda-XXXX/bin:$PATH