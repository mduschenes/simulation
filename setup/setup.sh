#!/bin/bash

# Setup variables
env=${1:-jax}


pkgs=/pkgs/anaconda3/bin
envs=${HOME}/env
channels=(intel conda-forge)
requirements=requirements.txt

reinstall=true

# Setup paths
mkdir -p ${envs}
if [[ -z $(grep ${pkgs} <<< ${PATH}) ]] && ( [[ -f ${pkgs} ]] || [[ -d ${pkgs} ]] )
then
	export PATH=${pkgs}:${PATH}
fi

if [[ -z $(grep ${envs}/${env} <<< ${PYTHONPATH}) ]] && ( [[ -f ${envs}/${env} ]] || [[ -d ${envs}/${env} ]] )
then
	export PYTHONPATH=${envs}/${env}:$PYTHONPATH
fi


conda config --remove envs_dirs ${envs} &>/dev/null 2>&1
conda config --append envs_dirs ${envs} &>/dev/null 2>&1


# Setup conda
__conda_setup="$('/pkgs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/pkgs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/pkgs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/pkgs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup


# Setup environment

if ${reinstall}
then
	conda deactivate
	conda remove --yes --name ${env} --all
	conda create --yes --prefix ${envs}/${env}
fi

# Activate environment
# conda activate ${env}
source activate ${envs}/${env}

# Install packages
awk -v requirements="${requirements}" -v RS= '{print > (requirements".tmp." NR "")}' ${requirements}
requirements=(${requirements}.tmp.*)

for channel in ${channels[@]}
do 
	conda config --remove channels ${channel} &>/dev/null 2>&1
	conda config --append channels ${channel} &>/dev/null 2>&1
	options+=("--channel" ${channel})
done

# Install packages
for file in ${requirements[@]}
do
	conda install --yes --file ${file} ${options[@]}
done

rm ${requirements[@]} 

# conda update --all

