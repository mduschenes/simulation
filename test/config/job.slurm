#!/usr/bin/env bash

##SBATCH --account=def-rgmelko
#SBATCH --job-name=job
#SBATCH --output=%x.%A.stdout
#SBATCH --error=%x.%A.stderr

##SBATCH --partition=cpu
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
##SBATCH --gpus-per-task=2

#SBATCH --array=0-0:1%100

##SBATCH --dependency=afterany:

#SBATCH --signal=B:SIGUSR1@60

#SBATCH --parsable
#SBATCH --get-user-env

# Run Parallel tasks

# Job
JOB_NAME=${SLURM_JOB_NAME}
JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
JOB_TASK=${SLURM_ARRAY_TASK_ID}
JOB_MIN=${SLURM_ARRAY_TASK_MIN}
JOB_MAX=${SLURM_ARRAY_TASK_MAX}
JOB_STEP=${SLURM_ARRAY_TASK_STEP}
JOB_COUNT=${SLURM_ARRAY_TASK_COUNT}
JOB_SLICE=${SLURM_ARRAY_TASK_SLICE}
JOB_SIZE=${SLURM_ARRAY_TASK_SIZE}
JOB_LOCAL=${SLURM_ARRAY_TASK_LOCAL}
JOB_DATA=${SLURM_ARRAY_TASK_DATA}

# Variables
JOB_CWD="${1:-${JOB_CWD}}"
JOB_ENV="${2:-${JOB_ENV}}"
JOB_MODULES=(${3:-${JOB_MODULES[@]}})
JOB_SRC="${4:-${JOB_SRC}}"
JOB_CMD="${5:-${JOB_CMD}}"
JOB_PROCESSES="${6:-${JOB_PROCESSES}}"
JOB_ENVIRONMENT="${7:-${JOB_ENVIRONMENT}}"
JOB_CLEANUP="${8:-${JOB_CLEANUP}}"
shift 8;
JOB_ARGS=${@:-${JOB_ARGS[@]}}

JOB_PATH=$(pwd)
JOB_DATA=$(echo ${JOB_DATA/\~/"${HOME}"} | envsubst)
JOB_TMP=${JOB_PATH}/tmp

# Functions
if [ ! -z ${JOB_CLEANUP} ]
then
function CLEANUP(){
	for path in ${JOB_CLEANUP[@]}
	do
		if [[ -f ${path} ]]
		then
			rm ${path}
		fi
	done
}
else
function CLEANUP(){
	for path in ${JOB_CLEANUP[@]}
	do
		if [[ -f ${path} ]]
		then
			rm ${path}
		fi
	done
}
fi
export -f CLEANUP

function QUEUE(){

	job=${BASH_SOURCE[0]}
	id=${SLURM_ARRAY_JOB_ID}

	# scontrol requeue ${id}
	# sbatch --export=ALL < ${job}
}
export -f QUEUE

if ([ ! -z ${JOB_LOCAL} ]) || ([ -z ${JOB_SLICE} ] || [ -z ${JOB_SIZE} ]) || ([ ! -z ${JOB_CWD} ] && [ -z ${JOB_TASK} ])
then
	function RUN(){

		trap 'CLEANUP' SIGUSR1

		TASK=${1:-"."}
		shift 1;	
		CMDS=()
		CMDS+=(${@})

		CWD=$(realpath ${TASK})	
		PWD=$(pwd)
		cd ${CWD}


		echo $(pwd) ${TASK} $(date '+%Y-%m-%d %H:%M:%S')


		${CMDS[@]}


		echo ${TASK} $(date '+%Y-%m-%d %H:%M:%S')


		cd ${PWD}

	}

else

	function RUN(){

		trap 'CLEANUP' SIGUSR1

		TASK=${1:-"."}
		shift 1;	
		CMDS=()
		CMDS+=(${@})


		echo $(pwd) ${TASK} $(date '+%Y-%m-%d %H:%M:%S')


		TASK=${TASK} ${CMDS[@]} ${TASK}


		echo ${TASK} $(date '+%Y-%m-%d %H:%M:%S')

	}
fi
export -f RUN


# Directory
if ([ ! -z ${JOB_SLICE} ] && [ ! -z ${JOB_SIZE} ]) && ([ -z ${JOB_CWD} ] || [ ! -z ${JOB_TASK} ])
then
	START=$(( ${JOB_TASK}*${JOB_SLICE} ))
	END=$(( (${JOB_TASK}+1)*${JOB_SLICE} ))
	if [ ${START} -lt 0 ];then START=0;fi; START=$(( ${START} ));
	if [ ${END} -gt ${JOB_SIZE} ];then END=${JOB_SIZE};fi; END=$(( ${END} - 1 ));
	JOB_CWD=($(seq ${START} ${END}))
else
	JOB_CWD=($(echo ${JOB_CWD[@]/\~/"${HOME}"} | envsubst))
	JOB_CWD=(${JOB_CWD[@]:-"."})
fi
JOB_SRC=$(echo ${JOB_SRC/\~/"${HOME}"} | envsubst)

JOB_ARGS=(${JOB_ARGS[@]//\"})


# Environment
JOB_ENV=${JOB_ENV//"${HOME}/envs/"}
JOB_MODULES=(${JOB_MODULES[@]//\"})
module purge &>/dev/null 2>&1
module load ${JOB_MODULES[@]} &>/dev/null 2>&1
# conda activate ${JOB_ENV} &>/dev/null 2>&1
# source activate ${JOB_ENV} &>/dev/null 2>&1
source ${HOME}/envs/${JOB_ENV}/bin/activate &>/dev/null 2>&1


# Command
CMD=()
if ([ ! -z ${JOB_SRC} ] && [ ! -z ${JOB_CMD} ])
then
	CMD+=(${JOB_SRC}/${JOB_CMD})
fi
CMD+=(${JOB_ARGS[@]})
CMD=($(echo ${CMD[@]/\~/"${HOME}"} | envsubst))


# Processes
TASKS=(${JOB_CWD[@]})
JOBS=${JOB_PROCESSES:-1}


# Executable
EXE=()
EXE+=(parallel)
EXE+=(--jobs ${JOBS} --env _ --linebuffer --tmpdir ${JOB_TMP}) #--resume-failed --joblog ${JOB_NAME:-job}.${JOB_ID:-id}.${JOB_TASK:-task}.parallel)
[ ! -z ${JOB_ENVIRONMENT} ] && EXE+=(${JOB_ENVIRONMENT}='$(({%} - 1))')
EXE+=(RUN)
EXE+=({})
EXE+=(${CMD[@]})
EXE+=(:::)

# Trap
trap 'CLEANUP' SIGUSR1
# trap 'QUEUE' SIGUSR1


# Run

if [ ! -z ${JOB_DATA} ]
then
	mkdir -p ${JOB_DATA}
	cd ${JOB_DATA}
else
	mkdir -p ${JOB_PATH}
	cd ${JOB_PATH}
fi
mkdir -p ${JOB_TMP}


echo $(pwd) ${EXE[@]} ${TASKS[@]}


${EXE[@]} ${TASKS[@]}


cd ${JOB_PATH}


# Cleanup
# rm -rf ${JOB_TMP}


# Status
# squeue -h -o "%V  %N  %u  %a  %j  %P  %M  %D  %C  %R  %T" -j${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
# sacct -j ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
scontrol show job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} &>/dev/null 2>&1