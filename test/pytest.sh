#!/bin/bash

files=(${@:-*_test.py})
options=("-s" "-rA" "--durations=0" "-W ignore::DeprecationWarning")
# options=("-s" "-rA" "--durations=0" "-k dump")

cmd=()
cmd+=("pytest")
cmd+=(${files[@]})
cmd+=(${options[@]})


${cmd[@]}
# ${cmd[@]} -k="main"
# ${cmd[@]} -k="data"