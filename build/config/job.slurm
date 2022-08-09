#!/bin/bash

##SBATCH --account=mduschenes
#SBATCH --job-name=results
#SBATCH --output=data/%x.%A.stdout
#SBATCH --error=data/%x.%A.stderr

#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

##SBATCH --array=0:100%100

##SBATCH --dependency=afterany:

#SBATCH --parsable


module purge &>/dev/null 2>&1
# module load jax0.2.24-cuda11.0-python3.8_jupyter &>/dev/null 2>&1


env=jax
source ${HOME}/env/${env}.env &>/dev/null 2>&1
source activate ${HOME}/condaenvs/${env} &>/dev/null 2>&1

CMD=
shift 1;
ARGS=

${CMD} ${ARGS[@]}