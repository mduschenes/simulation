#!/bin/bash

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
	mkdir -p ${TASK}
	cp ${ARGS[@]} ${TASK}/

	cd ${TASK}

	if [[ "${INTERFACE}" == "parallel" ]]
	then
		${CMD[@]} &
	elif [[ "${INTERFACE}" == "serial" ]]
	then
		${CMD[@]}
	else
		${CMD[@]}
	fi

	PROCESSES+=($!)
	
	cd ${CWD}
done

wait ${PROCESSES[@]}