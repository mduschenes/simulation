#!/bin/bash

##SBATCH --account=mduschenes
#SBATCH --job-name=submit
#SBATCH --output=%x.%j.stdout
#SBATCH --error=%x.%j.stderr

#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

##SBATCH --array=0-1:4%100

##SBATCH --dependency=afterany:

##SBATCH --signal=B:USRSIG@60
##SBATCH --open-mode=append

##--nodelist=cpu[0-10]
##--exclude=cpu[0-10]

#SBATCH --parsable
#SBATCH --get-user-env
#SBATCH --cpus-per-task=2


# Variables
export JOB_ARGS=(${1:-${JOB_ARGS:-settings.json}})
export JOB_ENV=${2:-${JOB_ENV:-env}}
export JOB_CMD=${3:-${JOB_CMD:-${HOME}/code/tensor/build/main.py}}

# Environment
module purge &>/dev/null 2>&1
conda activate ${JOB_ENV} &>/dev/null 2>&1
# source activate ${JOB_ENV} &>/dev/null 2>&1

# Command
EXE=()
EXE+=(${JOB_CMD})
EXE+=(${JOB_ARGS[@]})

# Run
echo $(pwd) ${EXE[@]}
${EXE[@]}