#!/usr/bin/env bash

#SBATCH --account=def-rgmelko
#SBATCH --job-name=distribution
#SBATCH --output=data/%x.%A.stdout
#SBATCH --error=data/%x.%A.stderr

##SBATCH --partition=cpu
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
##SBATCH --gpus-per-task=2

##SBATCH --dependency=afterany:

#SBATCH --signal=B:SIGUSR1@60

#SBATCH --parsable
#SBATCH --get-user-env

module purge &>/dev/null 2>&1
module load python hdf5 &>/dev/null 2>&1
source ${HOME}/envs/test/bin/activate &>/dev/null 2>&1

path=/home/${USER}/files/uw/research/code/tensor/code/test/distribution_test.py
# path=/home/${USER}/code/tensor/test/distribution_test.py

python ${path}
