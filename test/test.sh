#!/bin/bash
export SLURM_VAR=10
export SLURM_FOO=HIIII
./job.slurm . mkl ~/files/uw/research/code/simulation/code/src train.py 1 settings.json
