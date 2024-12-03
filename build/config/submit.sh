#!/bin/bash
settings=${1:-settings.json}
env=${2:-env}
cmd=${3:-${HOME}/code/tensor/build/main.py}

sbatch --export=JOB_SETTINGS=${settings},JOB_ENV=${env},JOB_CMD=${cmd} < submit.slurm
