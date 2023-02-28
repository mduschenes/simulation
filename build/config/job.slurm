#!/bin/bash

##SBATCH --account=mduschenes
#SBATCH --job-name=output
#SBATCH --output=%x.%A.stdout
#SBATCH --error=%x.%A.stderr

#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --array=0-1:4%100

##SBATCH --dependency=afterany:

#SBATCH --parsable
#SBATCH --get-user-env
#SBATCH --cpus-per-task=1


# Run Parallel tasks


# Function
function RUN(){

	TASK=$(realpath ${1:-"."})
	shift 1;
	CMDS=${@}
	CWD=$(pwd)

	echo $(pwd) ${TASK} $(date '+%Y-%m-%d %H:%M:%S')

	cd ${TASK}

	${CMDS[@]}

	cd ${CWD}

	echo $(pwd) ${TASK} $(date '+%Y-%m-%d %H:%M:%S')

}
export -f RUN

# Job
export JOB_NAME=${SLURM_JOB_NAME}
export JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
export JOB_TASK=${SLURM_ARRAY_TASK_ID}
export JOB_MIN=${SLURM_ARRAY_TASK_MIN}
export JOB_MAX=${SLURM_ARRAY_TASK_MAX}
export JOB_STEP=${SLURM_ARRAY_TASK_STEP}
export JOB_COUNT=${SLURM_ARRAY_TASK_COUNT}
export JOB_SLICE=${SLURM_ARRAY_TASK_SLICE}
export JOB_SIZE=${SLURM_ARRAY_TASK_SIZE}


# Variables
export JOB_CWD=${1:-${JOB_CWD}}
export JOB_ENV=${2:-${JOB_ENV}}
export JOB_SRC=${3:-${JOB_SRC}}
export JOB_CMD=${4:-${JOB_CMD}}
export JOB_PROCESSES=${5:-${JOB_PROCESSES}}
shift 5;
export JOB_ARGS=${@:-${JOB_ARGS}}

# Directory
if ([ ! -z ${JOB_SLICE} ] && [ ! -z ${JOB_SIZE} ]) && ([ -z ${JOB_CWD} ] || [ ! -z ${JOB_TASK} ])
then
	START=$(( ${JOB_TASK}*${JOB_SLICE} ))
	END=$(( (${JOB_TASK}+1)*${JOB_SLICE} ))
	if [ ${START} -lt 0 ];then START=0;fi; START=$(( ${START} ));
	if [ ${END} -gt ${JOB_SIZE} ];then END=${JOB_SIZE};fi; END=$(( ${END} - 1 ));
	export JOB_CWD=($(seq ${START} ${END}))
else
	export JOB_CWD=($(echo ${JOB_CWD[@]/\~/"${HOME}"} | envsubst))
	export JOB_CWD=(${JOB_CWD[@]:-"."})
fi
export JOB_SRC=$(echo ${JOB_SRC/\~/"${HOME}"} | envsubst)

# Environment
# source ~/.bashrc &>/dev/null 2>&1
module purge &>/dev/null 2>&1
# module load python/3.9.10 nccl_2.8.3-1+cuda11.0 cuda-11.3
conda activate ${JOB_ENV} &>/dev/null 2>&1
# source activate ${JOB_ENV} &>/dev/null 2>&1

# Command
CMD=(${JOB_SRC}/${JOB_CMD} ${JOB_ARGS[@]})
CMD=($(echo ${CMD[@]/\~/"${HOME}"} | envsubst))

# Processes
TASKS=(${JOB_CWD[@]})
JOBS=${JOB_PROCESSES:-1}

# Executable
EXE=()
EXE+=(parallel)
EXE+=(--jobs ${JOBS} --env _ --linebuffer --tmpdir ${PWD} --compress)
# EXE+=(--resume-failed --joblog ${JOB_NAME:-job}.${JOB_ID:-id}.${JOB_TASK:-task}.parallel)
EXE+=(RUN {} ${CMD[@]} :::)

# Run
echo $(pwd) ${EXE[@]} ${TASKS[@]}
${EXE[@]} ${TASKS[@]}