#!/bin/bash
set -x

directory=${HOME}/code/tensor/build
settings=${1:-settings.json}
env=${2:-env}
cmd=${3:-${directory}/main.py}
job=${4:-${directory}/config/submit.slurm}

sbatch --export=JOB_SETTINGS=${settings},JOB_ENV=${env},JOB_CMD=${cmd} < ${job}