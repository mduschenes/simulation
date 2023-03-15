#!/bin/bash

# Setup variables

# Environment Name
env=${1:-jax}

# Install Type ["install","reinstall","uninstall"]
install=${2:-"reinstall"}

# Type of env (env,intel)
type=${3:-intel}

# Silent yes to all commands ["yes","no"]
yes=${4:-no}


# Paths
pkgs=${HOME}/miniconda3
envs=${HOME}/miniconda3/envs
env_vars=env_vars.sh

# pkgs=/pkgs/anaconda3
# envs=${HOME}/env

mkdir -p ${envs}

if [ ${type} == "intel" ]
then
	channels=(intel conda-forge)
	requirements=requirements.txt
elif [ ${type} == "env" ]
then
	channels=(conda-forge)
	requirements=requirements.txt	
else
	channels=(intel conda-forge)
	requirements=requirements.txt	
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

# Setup activation scripts
source=${env_vars}
destination=${envs}/${env}/etc/conda/activate.d

mkdir -p ${destination}

cp ${source} ${destination}/

sed -i "s%env=.*%env=${env}%g" ${destination}/${env_vars}
source ${destination}/${env_vars}


# Setup environment variables

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


# Activate environment
# conda activate ${env}
source activate ${envs}/${env}

# Install packages

# Get line-break separated groups of requirements to install individually
rm -rf ${requirements}.tmp.*
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

if [ ! -z ${yes} ] && [ ${yes} == "yes" ]
then
	options+=("--yes")
fi
# options+=(--strict-channel-priority)
options+=()

# Install packages
for file in ${requirements[@]}
do
	conda install --file ${file} ${options[@]}
done

rm ${requirements[@]} 

# Pip packages
# packages=(gnureadline)
# pip install ${packages[@]}

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

# Allow larger linux stack space for cli arguments
# cache=16384
# ulimit -s ${cache}



# Multiple Cpus on Single Task
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=8
#parallel --jobs ${SLURM_CPUS_PER_TASK}

# Single Cpu on Multiple Tasks
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=8
##SBATCH --cpus-per-task=1
#parallel --jobs ${SLURM_NTASKS} srun --nodes 1 --ntasks 1 ${CMD} ::: ${ARGS}