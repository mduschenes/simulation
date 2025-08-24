#!/bin/bash
export SLURM_JOB_NAME=
export SLURM_JOB_ID=
export SLURM_ARRAY_JOB_ID=
export SLURM_ARRAY_TASK_ID=
export SLURM_ARRAY_TASK_MIN=
export SLURM_ARRAY_TASK_MAX=
export SLURM_ARRAY_TASK_STEP=
export SLURM_ARRAY_TASK_COUNT=
export SLURM_ARRAY_TASK_SLICE=
export SLURM_ARRAY_TASK_SIZE=1
sbatch -J job --export=JOB_CWD=.,JOB_ENV=tensor,JOB_MODULES="python hdf5 cuda cudnn",JOB_SRC=~/code/tensor/src,JOB_CMD=process.py,JOB_PROCESSES=1,JOB_ARGS="--data=**/data.hdf5 --plots=plot.json --processes=process.json --pwd=. --cwd=. --verbose",NUMPY_BACKEND=jax < preprocess.slurm
