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


# Function
function RUN(){

	TASK=$(realpath ${1:-"."})
	shift 1;
	CMDS=${@}
	CWD=$(pwd)

	echo $(pwd) $(date '+%Y-%m-%d %H:%M:%S')

	mkdir -p ${TASK}
	cd ${TASK}

	${CMDS[@]}

	# sleep 1
	echo $(pwd) $(date '+%Y-%m-%d %H:%M:%S')

	cd ${CWD}

}

# Job
export JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
export JOB_TASK=${SLURM_ARRAY_TASK_ID}

# Variables
export JOB_CWD=${1:-${JOB_CWD:-${JOB_TASK:-"."}}}
export JOB_ENV=${2:-${JOB_ENV}}
export JOB_SRC=${3:-${JOB_SRC}}
export JOB_CMD=${4:-${JOB_CMD}}
export JOB_PARALLELISM=${5:-"serial"}
shift 5;
export JOB_ARGS=${@:-${JOB_ARGS}}

# Directory
export JOB_CWD=($(echo ${JOB_CWD[@]/\~/"${HOME}"} | envsubst))
export JOB_CWD=(${JOB_CWD[@]:-"."})
export JOB_SRC=$(echo ${JOB_SRC/\~/"${HOME}"} | envsubst)

# Environment
# module purge &>/dev/null 2>&1
# conda activate ${JOB_ENV} &>/dev/null 2>&1
# source activate ${JOB_ENV} &>/dev/null 2>&1

# Command
CMD=(${JOB_SRC}/${JOB_CMD} ${JOB_ARGS[@]})
CMD=($(echo ${CMD[@]/\~/"${HOME}"} | envsubst))

# Processes
PROCESSES=()
CWD=${JOB_PWD}
TASKS=(${JOB_CWD[@]})
PARALLELISM=${JOB_PARALLELISM}
for TASK in ${TASKS[@]}
do
	if [[ "${PARALLELISM}" == "parallel" ]]
	then
		RUN ${TASK} ${CMD[@]} &
	elif [[ "${PARALLELISM}" == "serial" ]]
	then
		RUN ${TASK} ${CMD[@]}
	else
		RUN ${TASK} ${CMD[@]}
	fi

	PROCESSES+=($!)
	
done

wait ${PROCESSES[@]}

