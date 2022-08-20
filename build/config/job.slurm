#!/bin/bash

##SBATCH --account=mduschenes
#SBATCH --job-name=output
#SBATCH --output=%x.%A.stdout
#SBATCH --error=%x.%A.stderr

#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

##SBATCH --array=0:100%100

##SBATCH --dependency=afterany:

#SBATCH --parsable



SLURM_SUBMIT_DIR='.'
SLURM_ARRAY_TASK_ID='0'

cwd=${SLURM_SUBMIT_DIR:-$PWD}
task=${SLURM_ARRAY_TASK_ID:-}

if [ ! -z $task ]
then
	cwd=$cwd/$task
fi

cd $cwd

echo $(pwd)
exit

module purge &>/dev/null 2>&1
# module load jax0.2.24-cuda11.0-python3.8_jupyter &>/dev/null 2>&1


env=jax
source ${HOME}/env/${env}.env &>/dev/null 2>&1
source activate ${HOME}/condaenvs/${env} &>/dev/null 2>&1


JOB_CMD=${1:-${JOB_CMD}}
shift 1;
JOB_ARGS=${@:-${JOB_ARGS}}

${JOB_CMD} ${JOB_ARGS[@]}
