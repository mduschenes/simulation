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
export JOB_SETTINGS=${1:-${JOB_SETTINGS:-settings.json}}
export JOB_ENV=${2:-${JOB_ENV:-env}}
export JOB_CMD=${3:-${JOB_CMD:-${HOME}/code/tensor/build/main.py}}

# Environment
module purge &>/dev/null 2>&1
conda activate ${JOB_ENV} &>/dev/null 2>&1
# source activate ${JOB_ENV} &>/dev/null 2>&1

# Command
CMD=()
CMD+=(${JOB_CMD} ${JOB_SETTINGS})

EXE=()
EXE+=(${CMD[@]})

# Run
echo $(pwd) ${EXE[@]}
${EXE[@]}

# Status
# squeue -h -o "%V  %N  %u  %a  %j  %P  %M  %D  %C  %R  %T" -j${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
# sacct -j ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
scontrol show job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} &>/dev/null 2>&1

# Queue
# trap 'QUEUE' SIGUSRSIG

# Cleanup
FILE=${SLURM_JOB_NAME}.${SLURM_JOB_ID}.stdout
if [[ ! -s ${FILE} ]]
then
	rm -rf ${FILE}
fi 

FILE=${SLURM_JOB_NAME}.${SLURM_JOB_ID}.stderr
if [[ ! -s ${FILE} ]]
then
	rm -rf ${FILE}
fi