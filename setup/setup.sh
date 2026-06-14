#!/usr/bin/env bash

env=${1:-env}
requirements=${2:-requirements.txt}
path=${3:-$(realpath ..)}
envs=${4:-${HOME}/conda/envs}
modules=(${5:-})
channel=${6:-conda-forge}
test=${7:-test.py}

if [ ! -z ${modules} ]
then
	module purge
	module load ${modules[@]}
fi

mkdir -p ${envs}

source ${envs}/../etc/profile.d/conda.sh

for i in {1..5}; do conda deactivate; done
conda info --envs

conda remove --name ${env} --all

conda create --prefix ${envs}/${env}

conda activate ${env}
conda info --envs

conda install --channel ${channel} --file ${requirements}

conda activate ${env}
conda info --envs

pth=${envs}/${env}/lib/python*/site-packages/env.${env}.pth
echo ${path} > ${pth}

pytest -rA -W ignore::DeprecationWarning ${test}