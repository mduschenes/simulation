#!/bin/bash

# Setup variables

# Environment Name
env=${1:-jax}

# Install Type ["install","reinstall","uninstall"]
install=${2:-"reinstall"}

# Silent yes to all commands ["yes","no"]
yes=${3}

pkgs=/pkgs/anaconda3
pkgs=${HOME}/miniconda3
envs=${HOME}/env
envs=${HOME}/miniconda/envs
channels=(intel conda-forge)
requirements=requirements.txt

# Setup paths
mkdir -p ${envs}

if [[ -z $(grep ${pkgs}/bin <<< ${PATH}) ]] && ( [[ -f ${pkgs}/bin ]] || [[ -d ${pkgs}/bin ]] )
then
	export PATH=${pkgs}/bin:${PATH}
fi

if [[ -z $(grep ${pkgs}/lib <<< ${PATH}) ]] && ( [[ -f ${pkgs}/lib ]] || [[ -d ${pkgs}/lib ]] )
then
	export LD_LIBRARY_PATH=${pkgs}/lib:${LD_LIBRARY_PATH}
fi

if [[ -z $(grep ${envs}/${env} <<< ${PYTHONPATH}) ]] && ( [[ -f ${envs}/${env} ]] || [[ -d ${envs}/${env} ]] )
then
	export PYTHONPATH=${envs}/${env}:$PYTHONPATH
fi


# Setup conda
__conda_setup="$('${pkgs}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${pkgs}/etc/profile.d/conda.sh" ]; then
        . "${pkgs}/etc/profile.d/conda.sh"
    else
        export PATH="${pkgs}/bin:$PATH"
    fi
fi
unset __conda_setup


# Setup environment
if [ "${install}" == "reinstall" ]
then
	conda deactivate
	
	conda config --remove envs_dirs ${envs} &>/dev/null 2>&1
	conda config --append envs_dirs ${envs} &>/dev/null 2>&1

	conda remove --name ${env} --all

	conda create --prefix ${envs}/${env}

elif [ "${install}" == "uninstall" ]
then
	conda deactivate

	conda remove --name ${env} --all
fi

# Activate environment
# conda activate ${env}
source activate ${envs}/${env}

# Install packages

# Get line-break separated groups of requirements to install individually
awk -v requirements="${requirements}" -v RS= '{print > (requirements".tmp." NR "")}' ${requirements}
requirements=(${requirements}.tmp.*)

# Get installation options
options=()
for channel in ${channels[@]}
do 
	conda config --remove channels ${channel} &>/dev/null 2>&1
	conda config --append channels ${channel} &>/dev/null 2>&1
	options+=("--channel" ${channel})
done

if [ ${yes} == "yes" ]
then
	options+=("--yes")
fi

# Install packages
for file in ${requirements[@]}
do
	conda install --file ${file} ${options[@]}
done

rm ${requirements[@]} 

# conda update --all

