#!/bin/bash

# Environment

env=
requirements=requirements.txt
architecture=cpu
installer=conda
sources=()
envs=${HOME}/envs
pkgs=${HOME}/conda

run=

function usage(){

	variable=(e r a i v p)
	variables=(env requirements architecture installer envs pkgs)

	file=$(basename ${0})
	directory=$(dirname ${0})

	echo Usage: ${file}
	for i in ${!variables[@]}
	do
		case ${variables[$i]} in
			env) string="environment name: <env>" ;;
			requirements) string="requirements file: requirements.txt" ;;
			architecture) string="architecture type: cpu|gpu" ;;
			installer) string="installer type: pip|conda" ;;
			sources) string="sources for requirements <channel>,<url>" ;;
			envs) string="path to environments: \${HOME}/envs" ;;
			pkgs) string="path to root packages: \${HOME}/conda" ;;
			dry-run) string="dry-run boolean" ;;
			help) string="man page" ;;
			*) string=${!variables[$i]};;
		esac
		printf "\t%s | %-16s %s\n" "-${variables[$i]:0:1}" "--${variables[$i]}"  "${string[@]}"
	done
}

if [[ $# -eq 0 ]]
then
	usage
	exit
fi

eval set -- "$(getopt --options eraisvpdh --longoptions env:,requirements:,architecture:,installer:,sources:,envs:,pkgs:,dry-run,help, -- "$@")"
while [[ $# -gt 0 ]]; do case ${1} in 
-e|--env) env=${2}; shift 2;; 
-r|--requirements) requirements=${2}; shift 2;; 
-a|--architecture) architecture=${2}; shift 2;;
-i|--installer) installer=${2}; shift 2;;
-s|--sources) sources+=(${2}); shift 2;;
-v|--envs) envs=${2}; shift 2;;
-p|--pkgs) pkgs=${2}; shift 2;;
-d|--dry-run) run=true; shift;;
-h|--help) usage; exit; shift;; 
--) shift;break;; *);;
esac done

env=${env:-${1}}

if [[ -z ${env} ]]
then
	usage
	exit
fi

if [[ ! -z ${run} ]]
then
	run=echo
fi

# Setup environment
dir=$(dirname ${env})
env=$(basename ${env})
if [[ ! ${dir} == . ]];then envs=${dir};fi

case ${installer} in
	pip)
		case ${architecture} in
			gpu)
				modules=(python cuda cudnn)
				${run} module purge
				${run} module load ${modules[@]}
				;;
			cpu)
				modules=(python)
				${run} module purge
				${run} module load ${modules[@]}
				;;
			*)
				modules=(python)
				${run} module purge
				${run} module load ${modules[@]}
			;;
		esac

		${run} mkdir -p ${envs}
		${run} deactivate
		${run} rm -rf ${envs}/${env}
		${run} pip install --upgrade pip --no-index
		${run} virtualenv --no-download ${envs}/${env}

		${run} source ${envs}/${env}/bin/activate

		rm -rf ${requirements}.tmp.*
		awk -v requirements="${requirements}" -v RS= '{print > (requirements".tmp." NR "")}' ${requirements}
		requirements=(${requirements}.tmp.*)
		for requirement in ${requirements[@]}
		do
			options=()
			options+=(--no-index)
			if [[ ! -z ${sources[@]} ]]
			then
				options+=(--find-links ${sources[@]})
			fi

			if [[ ! -z ${requirement} ]]
			then
				options+=(--requirement ${requirement})
			fi

			${run} pip install ${options[@]}
		done
		rm -rf ${requirements[@]}
		;;
	conda)
		${run} mkdir -p ${envs}
		${run} conda deactivate
		${run} conda config --remove envs_dirs ${envs}
		${run} conda config --append envs_dirs ${envs}
		${run} conda remove --name ${env} --all
		${run} conda create --prefix ${envs}/${env}

		${run} conda activate ${env}

		rm -rf ${requirements}.tmp.*
		awk -v requirements="${requirements}" -v RS= '{print > (requirements".tmp." NR "")}' ${requirements}
		requirements=(${requirements}.tmp.*)
		for requirement in ${requirements[@]}
		do
			options=()
			if [[ ! -z ${sources[@]} ]]
			then
				for src in ${sources[@]}
				do
					options+=(--channel ${src})
				done
			fi
			if [[ ! -z ${requirement} ]]
			then
				options+=(--file ${requirement})
			fi

			${run} conda install ${options[@]}
		done
		rm -rf ${requirements[@]}
		;;
	*)
		;;
esac


# Test environment
test=test.py
if [[ -f ${test} ]]
then
	options=(-rA -W ignore::DeprecationWarning)
	${run} pytest ${options[@]} ${test}
fi


# # Setup environment
# env=${1:-env}
# default=base
# envs=${HOME}/envs
# pkgs=${HOME}/conda

# if [[ ! -d ${envs}/${env} ]];then env=${default};fi

# case ${installer} in
# 	pip)
# 		case ${architecture} in
# 			gpu)
# 				modules=(python cuda cudnn)
# 				module purge
# 				module load ${modules[@]}
# 				;;
# 			cpu)
# 				modules=(python)
# 				module purge
# 				module load ${modules[@]}
# 				;;
# 			*)
# 				modules=(python)
# 				module purge
# 				module load ${modules[@]}
# 			;;
# 		esac
# 		source ${envs}/${env}/bin/activate
# 		;;
# 	conda)
# 		conda activate ${env}
# 		;;
# 	*)
# 		;;
# esac

# # Setup variables
# case ${architecture} in
# 	gpu)
# 		variables=(PATH LD_LIBRARY_PATH PYTHONPATH)
# 		strings=("${envs}/${env}/bin" "${envs}/${env}/lib" "${envs}/${env}")
# 	;;
# 	cpu)
# 		variables=()
# 		strings=()
# 	;;
# 	*)
# 		variables=()
# 		strings=()
# 	;;
# esac

# for i in ${!variables[@]}
# do
# 	variable=${variables[${i}]}
# 	string=${strings[${i}]}
# 	if [[ -z ${!variable} ]]
# 	then
# 		export ${variable}=${string}
# 	elif [[ -z $(grep : <<< ${!variable}) ]] && [[ -z $(grep ${string} <<< ${!variable}) ]]
# 	then
# 		export ${variable}=${string}:${!variable}		
# 	elif [[ ! -z $(grep : <<< ${!variable}) ]] && [[ -z $(grep ${string}: <<< ${!variable}) ]]
# 	then
# 		export ${variable}=${string}:${!variable}
# 	fi
# done


# # Setup flags
# case ${architecture} in
# 	gpu)
# 		variables=(XLA_FLAGS)
# 		strings=("--xla_gpu_cuda_data_dir=${CUDA_HOME}")
# 	;;
# 	cpu)
# 		variables=()
# 		strings=()
# 	;;
# 	*)
# 		variables=()
# 		strings=()
# 	;;
# esac

# for i in ${!variables[@]}
# do
# 	variable=${variables[${i}]}
# 	string=${strings[${i}]}
# 	if [[ -z ${!variable} ]]
# 	then
# 		export ${variable}=${string}
# 	elif [[ -z $(grep " " <<< ${!variable}) ]] && [[ -z $(grep "${string}" <<< ${!variable}) ]]
# 	then
# 		export ${variable}="${string} ${!variable}"
# 	elif [[ ! -z $(grep " " <<< ${!variable}) ]] && [[ -z $(grep "${string} " <<< ${!variable}) ]]
# 	then
# 		export ${variable}="${string} ${!variable}"
# 	fi
# done
