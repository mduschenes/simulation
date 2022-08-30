#!/bin/bash

##SBATCH --account=mduschenes
#SBATCH --job-name=output
#SBATCH --output=%x.%A.stdout
#SBATCH --error=%x.%A.stderr

#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

##SBATCH --array=0-99:1%100

##SBATCH --dependency=afterany:

#SBATCH --parsable

# Variables
JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}
JOB_ARRAY=${SLURM_ARRAY_TASK_ID:-"."}

JOB_SRC=${1:-${JOB_SRC}}
JOB_CMD=${2:-${JOB_CMD}}
shift 2;
JOB_ARGS=${@:-${JOB_ARGS}}


# Directory
cd ${JOB_ARRAY}


# Environment
module purge &>/dev/null 2>&1
# module load jax0.2.24-cuda11.0-python3.8_jupyter &>/dev/null 2>&1

env=jax
source ${HOME}/env/${env}.env &>/dev/null 2>&1
source activate ${HOME}/env/${env} &>/dev/null 2>&1


# Command
# echo "${PWD} \(${JOB_ID},${JOB_ARRAY}\) : ${JOB_SRC}/${JOB_CMD} ${JOB_ARGS[@]}"
${JOB_SRC}/${JOB_CMD} ${JOB_ARGS[@]}