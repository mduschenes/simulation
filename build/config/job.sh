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
sbatch -J job --export=JOB_CWD=.,JOB_ENV=tensor,JOB_SRC=~/code/tensor/src,JOB_CMD=train.py,JOB_PROCESSES=1,JOB_ARGS=settings.json,NUMPY_BACKEND=jax < job.slurm
