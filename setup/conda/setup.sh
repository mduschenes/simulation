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

pkgs=/pkgs/anaconda3
envs=${HOME}/env


# Setup paths
mkdir -p ${envs}


if [[ -z $(grep ${pkgs}/bin <<< ${PATH}) ]] && ( [[ -f ${pkgs}/bin ]] || [[ -d ${pkgs}/bin ]] )
then
        export PATH=${pkgs}/bin:${envs}/${env}/bin:${PATH}
fi

if [[ -z $(grep ${pkgs}/lib <<< ${PATH}) ]] && ( [[ -f ${pkgs}/lib ]] || [[ -d ${pkgs}/lib ]] )
then
        export LD_LIBRARY_PATH=${envs}/${env}/lib:${pkgs}/lib:${LD_LIBRARY_PATH}
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


# Add activation scripts to conda
source=conda
destination=${envs}/env/etc/conda/activate.d

cp ${scripts}/* ${destination}/



# Activate environment
conda activate ${env}
#source activate ${envs}/${env}


# Install jax
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Setup paths
mkdir -p ${envs}


if [[ -z $(grep ${pkgs}/bin <<< ${PATH}) ]] && ( [[ -f ${pkgs}/bin ]] || [[ -d ${pkgs}/bin ]] )
then
        export PATH=${pkgs}/bin:${envs}/${env}/bin:${PATH}
fi

if [[ -z $(grep ${pkgs}/lib <<< ${PATH}) ]] && ( [[ -f ${pkgs}/lib ]] || [[ -d ${pkgs}/lib ]] )
then
        export LD_LIBRARY_PATH=${envs}/${env}/lib:${pkgs}/lib:${LD_LIBRARY_PATH}
fi

if [[ -z $(grep ${envs}/${env} <<< ${PYTHONPATH}) ]] && ( [[ -f ${envs}/${env} ]] || [[ -d ${envs}/${env} ]] )
then
        export PYTHONPATH=${envs}/${env}:$PYTHONPATH
fi

