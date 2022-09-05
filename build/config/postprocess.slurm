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
export JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
export JOB_ARRAY=${SLURM_ARRAY_TASK_ID}
export JOB_CWD=${JOB_ARRAY:-"."}

export JOB_ENV=${1:-${JOB_ENV}}
export JOB_SRC=${2:-${JOB_SRC}}
export JOB_CMD=${3:-${JOB_CMD}}
shift 3;
export JOB_ARGS=${@:-${JOB_ARGS}}


# Directory
export JOB_CWD=$(echo ${JOB_CWD/#\~/${HOME}} | envsubst)
cd ${JOB_CWD}

# Environment
export JOB_SRC=$(echo ${JOB_SRC/#\~/${HOME}} | envsubst)
module purge &>/dev/null 2>&1
conda activate ${JOB_ENV} &>/dev/null 2>&1
# source activate ${JOB_ENV} &>/dev/null 2>&1

# Command
CMD=(${JOB_SRC}/${JOB_CMD} ${JOB_ARGS[@]})

CMD=($(echo ${CMD[@]/#\~/${HOME}} | envsubst))

${CMD[@]}