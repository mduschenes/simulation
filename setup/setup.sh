#!/usr/bin/env bash

module purge
module load python cuda cudnn
mkdir -p /home/mduschen/envs
deactivate
rm -rf /home/mduschen/envs/env
virtualenv --no-download /home/mduschen/envs/env
source /home/mduschen/envs/env/bin/activate
pip install --no-index --requirement requirements.txt
pytest -rA -W ignore::DeprecationWarning test.py