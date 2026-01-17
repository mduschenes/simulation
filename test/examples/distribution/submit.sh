#!/usr/bin/env bash

job=${1:-job.slurm}

sbatch < ${job}