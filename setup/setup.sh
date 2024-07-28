#!/bin/bash

# Environment

env=
requirements=requirements.txt
architecture=cpu
environment=conda
installer=()
modules=()
sources=()
type=
envs=${HOME}/conda/envs
pkgs=${HOME}/conda/envs

run=

program=${0}

delimiter=,


# Functions
function join() {
  local IFS="${1}"
  shift
  echo "$*"
}

function split() {
  delimiter="${1}"
  shift
  array=(${@//${delimiter}/ })
  echo "${array[@]}"
}

function unique() {
	echo $(printf "%s\n" ${@} | cat -n | sort -k2 -k1n  | uniq -f1 | sort -nk1,1 | cut -f2-)
	return
}


# Usage
function usage() {

	variable=(e r a n i m s t v p d h)
	variables=(\
		env requirements \
		architecture environment installer \
		modules sources \
		type \
		envs pkgs \
		dry-run help)

	file=$(basename ${0})
	directory=$(dirname ${0})

	echo Usage: ${file}
	for i in ${!variables[@]}
	do
		case ${variables[$i]} in
			env) string="environment name: <env>" ;;
			requirements) string="requirements file: <requirements.txt>" ;;
			architecture) string="architecture type: cpu|gpu" ;;
			environment) string="environment manager: virtualenv|conda" ;;
			installer) string="installer program: pip|conda" ;;
			modules) string="modules for installation: <modules>" ;;
			sources) string="sources for requirements: <channel>,<url>" ;;
			type) string="type of installation: install|upgrade" ;;			
			envs) string="path to environments: <path>" ;;
			pkgs) string="path to packages: <path>" ;;
			dry-run) string="dry-run flag" ;;
			help) string="help flag" ;;
			*) string=${!variables[$i]};;
		esac
		printf "\t%s | %-16s %s\n" "-${variable[$i]:0:1}" "--${variables[$i]}"  "${string[@]}"
	done
	
	printf "Examples:\n\n"
	
	name="Compute Canada Cluster"
	string=(
		${program} \
		--env env --requirements requirements.txt \
		--architecture gpu --environment pip --installer pip \
		--modules python,cuda,cudnn --sources "conda-forge,https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" \
		--type install \
		--env \${HOME}/envs --pkgs \${HOME}/envs/\${env}
		)
	echo "${name}:"
	echo ${string[@]}
	echo

	name="Vector Institute Cluster"
	string=(
		${program} \
		--env env --requirements requirements.txt \
		--architecture gpu --environment conda --installer conda,pip \
		--modules cuda11.8+cudnn8.9.6 --sources "--no-index" \
		--type install \
		--env \${HOME}/.conda/envs --pkgs /pkgs/anaconda3 
		)
	echo "${name}:"
	echo ${string[@]}
	echo
}

if [[ $# -eq 0 ]]
then
	usage
	exit
fi

# Arguments

eval set -- "$(getopt --options eranimstvpdh --longoptions env:,requirements:,architecture:,environment:,installer:,modules:,sources:,type:,envs:,pkgs:,dry-run,help, -- "$@")"
while [[ $# -gt 0 ]]; do case ${1} in 
	-e|--env) env=${2}; shift 2;; 
	-r|--requirements) requirements=${2}; shift 2;; 
	-a|--architecture) architecture=${2}; shift 2;;
	-n|--environment) environment=${2}; shift 2;;
	-i|--installer) installer+=($(split ${delimiter} "${2}")); shift 2;;
	-m|--modules) modules+=($(split ${delimiter} "${2}")); shift 2;;
	-s|--sources) sources+=($(split ${delimiter} "${2}")); shift 2;;
	-t|--type) type=${2}; shift 2;;
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

# Recursive installation
number=$(awk 'NF {p=1} p' ${requirements} | tac | awk 'NF {p=1} p' | tac | sed '/^$/N;/^\n$/D' | grep -c "^$")
if [[ ${number} >  0 ]]
then

	rm -rf ${requirements}.tmp.*
	awk -v requirements="${requirements}" -v RS= '{print > (requirements".tmp." NR "")}' ${requirements}
	requirements=(${requirements}.tmp.*)

	for i in ${!requirements[@]}
	do

		env=${env}

		requirement=${requirements[${i}]}

		arch=${architecture}

		if [[ ${#installer[@]} -eq ${#requirements[@]} ]]
		then
			install=${installer[${i}]}
		else
			install=${installer}
		fi

		environment=${environment}

		if [[ ${i} == 0 ]]
		then
			if [[ ${#type[@]} -eq ${#requirements[@]} ]]
			then
				types=${type[${i}]}
			else
				types=${type}
			fi
		else
			types=upgrade
		fi

		module="$(join ${delimiter} "${modules[@]}")"

		if [[ ${#sources[@]} -eq ${#requirements[@]} ]]
		then
			source="$(join ${delimiter} ${sources[${i}]})"
		else
			source="$(join ${delimiter} ${sources[@]})"
		fi

		envs=${envs}
	
		pkgs=${pkgs}

		options=()
		if [[ ${run} ]]
		then
			options+=(--dry-run)
		fi

		cmd=()
		cmd+=(${program})
		if [[ ! -z ${env} ]]
		then
			cmd+=(--env "${env}")
		fi
		if [[ ! -z ${requirement} ]]
		then
			cmd+=(--requirements "${requirement}")
		fi
		if [[ ! -z ${arch} ]]
		then
			cmd+=(--architecture "${arch}")
		fi
		if [[ ! -z ${environment} ]]
		then
			cmd+=(--environment "${environment}")
		fi
		if [[ ! -z ${install} ]]
		then
			cmd+=(--installer "${install}")
		fi
		if [[ ! -z ${module} ]]
		then
			cmd+=(--modules "${module}")
		fi
		if [[ ! -z ${source} ]]
		then
			cmd+=(--sources "${source}")
		fi
		if [[ ! -z ${types} ]]
		then
			cmd+=(--type "${types}")
		fi
		if [[ ! -z ${envs} ]]
		then
			cmd+=(--envs "${envs}")
		fi	
		if [[ ! -z ${pkgs} ]]
		then
			cmd+=(--pkgs "${pkgs}")
		fi	
		if [[ ! -z ${options[@]} ]]
		then
			cmd+=(${options[@]})
		fi			
		echo ${cmd[@]}
		echo		
		${cmd[@]}
		echo
		echo
	done
	
	rm -rf ${requirements[@]}

	exit
fi


# Setup installer
if [[ -z ${installer} ]]
then
	case ${environment} in
		virtualenv)
			installer=pip;;
		conda)
			installer=conda;;
		*)
			installer=;;
	esac
fi


# Setup modules
case ${architecture} in
	gpu)
		case ${installer} in 
			pip)
				modules+=();;
			conda)
				modules+=();;
			*)
				modules+=();;
		esac
		;;
	cpu)
		case ${installer} in 
			pip)
				modules+=();;
			conda)
				modules+=();;
			*)
				modules+=();;
		esac
		;;
	*)
		
	;;
esac
modules=($(unique ${modules[@]}))
${run} module purge
${run} module load ${modules[@]}


# Setup environment
dir=$(dirname ${env})
env=$(basename ${env})
if [[ ! ${dir} == . ]];then envs=${dir};fi

case ${environment} in
	virtualenv)
		${run} mkdir -p ${envs}
		${run} deactivate

		case ${type} in
			install)
				${run} rm -rf ${envs}/${env}
				${run} ${environment} --no-download ${envs}/${env}
				;;
			upgrade)
				;;
			*)
				;;
		esac
		${run} source ${envs}/${env}/bin/activate
		;;
	conda)
		${run} mkdir -p ${envs}
		${run} ${environment} deactivate
	
		case ${type} in
			install)
				;;
			upgrade)
				;;
			*)
				;;
		esac
		${run} ${environment} activate ${env}
		;;
esac


# Install environment
case ${installer} in
	pip)
		options=()
		for source in ${sources[@]}
		do
			if [[ ! -z ${source} ]]
			then
				case ${source} in
					--no-index)
						options+=(${source})
					;;
					*)	
						options+=(--find-links ${source})
					;;
				esac
			fi
		done
		options+=(--requirement ${requirements[${i}]})

		${run} ${installer} install ${options[@]}
		;;
	
	conda)
		options=()
		for source in ${sources[@]}
		do
			if [[ ! -z ${source} ]]
			then
				options+=(--channel ${source})
			fi
		done
		options+=(--file ${requirements[${i}]})

		${run} ${installer} install ${options[@]}
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
# envs=${HOME}/.conda/envs
# pkgs=/pkgs/anaconda3

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
