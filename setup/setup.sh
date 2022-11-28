#!/bin/bash

# Setup variables

# Environment Name
env=${1:-jax}

# Install Type ["install","reinstall","uninstall"]
install=${2:-"reinstall"}

# Silent yes to all commands ["yes","no"]
yes=${3}

# pkgs=${HOME}/miniconda3
# envs=${HOME}/miniconda/envs

# pkgs=/pkgs/anaconda3
envs=${HOME}/env
version=3.8

channels=(intel conda-forge)
requirements=requirements.txt

# Load python
module load python/${version}

# Setup paths
mkdir -p ${envs}

if [[ -z $(grep ${pkgs}/bin <<< ${PATH}) ]] && ( [[ -f ${pkgs}/bin ]] || [[ -d ${pkgs}/bin ]] )
then
	export PATH=${PATH}
fi

if [[ -z $(grep ${pkgs}/lib <<< ${PATH}) ]] && ( [[ -f ${pkgs}/lib ]] || [[ -d ${pkgs}/lib ]] )
then
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
fi

if [[ -z $(grep ${envs}/${env} <<< ${PYTHONPATH}) ]] && ( [[ -f ${envs}/${env} ]] || [[ -d ${envs}/${env} ]] )
then
	export PYTHONPATH=${envs}/${env}:$PYTHONPATH
fi


# Setup environment
if [ "${install}" == "reinstall" ]
then
	deactivate &>/dev/null 2>&1
	
	rm -rf ${envs}

	virtualenv --no-download ${envs}/${env}

elif [ "${install}" == "uninstall" ]
then
	deactivate &>/dev/null 2>&1

	rm -rf ${envs}/env
fi

# Activate environment
# conda activate ${env}
module load python/${version}
source ${envs}/${env}/bin/activate

pip -V

# Install packages

# Get line-break separated groups of requirements to install individually
awk -v requirements="${requirements}" -v RS= '{print > (requirements".tmp." NR "")}' ${requirements}
requirements=(${requirements}.tmp.*)

# Get installation options
options=()
options+=(--no-index)

# Install packages
for file in ${requirements[@]}
do
	pip install -r ${file} ${options[@]}
	options=()

done

rm ${requirements[@]} 

# conda update --all

# Install rmate (from https://stackoverflow.com/questions/37458814/how-to-open-remote-files-in-sublime-text-3)
# path=${HOME}/.local/bin/rmate
# url=\https://raw.github.com/aurora/rmate/master/rmate

# wget -O ${path} ${url}
# chmod a+x ${path}

# Install latex packages (from http://physino.xyz/learning/2021/12/22/use-LaTeX-on-a-Linux-HPC-cluster/)
# if [ "${install}" == "reinstall" ]
# do
# 	var=TEXMFHOME
# 	default=${HOME}/texmf
# 	pkgs=(physics)

# 	path=$(kpsewhich --var-value ${var})
# 	path=${path:-${default}}

# 	mkdir -p ${path}

# 	export ${var}=${path}

# 	pwd=${PWD}

# 	cd ${path}

# 	texhash . # or mktexlsr .

# 	tlmgr --init-usertree

# 	tlmgr --usermode install ${pkgs[@]}

# 	cd ${pwd}
# done


# Copy .bashrc
# source=.bashrc
# destination=${HOME}

# cp ${source} ${destination}