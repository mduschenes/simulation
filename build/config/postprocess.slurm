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


# Run Parallel tasks


# Function
function RUN(){

	# echo ${1}

	TASK=$(realpath ${1:-"."})
	shift 1;
	CMDS=${@}
	CWD=$(pwd)

	echo $(pwd) $(date '+%Y-%m-%d %H:%M:%S')

	cd ${TASK}

	${CMDS[@]}

	echo $(pwd) $(date '+%Y-%m-%d %H:%M:%S')

	cd ${CWD}

}
export -f RUN

# Job
export JOB_NAME=${SLURM_JOB_NAME}
export JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
export JOB_TASK=${SLURM_ARRAY_TASK_ID}

# Variables
export JOB_CWD=${1:-${JOB_CWD}}
export JOB_ENV=${2:-${JOB_ENV}}
export JOB_SRC=${3:-${JOB_SRC}}
export JOB_CMD=${4:-${JOB_CMD}}
export JOB_PROCESSES=${5:-1}
shift 5;
export JOB_ARGS=${@:-${JOB_ARGS}}

# Directory
if [[ -z ${JOB_CWD} ]]
then
	export JOB_CWD=($(seq ${SLURM_ARRAY_TASK_ID} $(( ${SLURM_ARRAY_TASK_ID} + ${SLURM_ARRAY_TASK_STEP} - 1 )) ))
else
	export JOB_CWD=($(echo ${JOB_CWD[@]/\~/"${HOME}"} | envsubst))
	export JOB_CWD=(${JOB_CWD[@]:-"."})
fi
export JOB_SRC=$(echo ${JOB_SRC/\~/"${HOME}"} | envsubst)

# Environment
module purge &>/dev/null 2>&1
conda activate ${JOB_ENV} &>/dev/null 2>&1
# source activate ${JOB_ENV} &>/dev/null 2>&1

# Command
CMD=(${JOB_SRC}/${JOB_CMD} ${JOB_ARGS[@]})
CMD=($(echo ${CMD[@]/\~/"${HOME}"} | envsubst))

# Processes
TASKS=(${JOB_CWD[@]})
JOBS=${JOB_PROCESSES}

# Executable
EXE=()
EXE+=(parallel)
EXE+=(--jobs ${JOBS} --env _ --linebuffer --tmpdir ${PWD} --compress)
# EXE+=(--resume-failed --joblog ${JOB_NAME:-job}.${JOB_ID:-id}.${JOB_TASK:-task}.parallel)
EXE+=(RUN {} ${CMD[@]} :::)

# Run
${EXE[@]} ${TASKS[@]}