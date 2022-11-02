#!/bin/bash

function RUN(){
	TASK=${1}
	shift
	CMD=${@}

	CWD=$(pwd)

	mkdir -p ${TASK}
	cp ${ARGS[@]} ${TASK}/

	cd ${TASK}

	${CMD[@]}

	cd ${CWD}

}

# Variables
SRC=${1}
EXE=${2}
ARGS=${3}
INTERFACE=${4:-parallel}
shift 4
TASKS=${@}

# Directory
CWD=$(pwd)

# Command
CMD=(${SRC}/${EXE} ${ARGS[@]})
CMD=($(echo ${CMD[@]/\~/"${HOME}"} | envsubst))

# Processes
PROCESSES=()
for TASK in ${TASKS[@]}
do
	

	if [[ "${INTERFACE}" == "parallel" ]]
	then
		RUN ${TASK} ${CMD[@]} &
	elif [[ "${INTERFACE}" == "serial" ]]
	then
		RUN ${TASK} ${CMD[@]}
	else
		RUN ${TASK} ${CMD[@]}
	fi

	PROCESSES+=($!)
	
	cd ${CWD}
done

wait ${PROCESSES[@]}