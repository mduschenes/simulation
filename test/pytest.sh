#!/bin/bash

files=(${@:-*_test.py})
options=("-s" "-rA" "--durations=0" "-W ignore::DeprecationWarning" "-W ignore::UserWarning" "--show-capture=no")
# options=("-s" "-rA" "--durations=0" "-k dump")

cmd=()
# cmd+=("conda" "run" "-n" "jax")
cmd+=("python" -m)
cmd+=("pytest")
cmd+=(${files[@]})
cmd+=(${options[@]})


${cmd[@]}
# ${cmd[@]} -k="main"
# ${cmd[@]} -k="grad"
# ${cmd[@]} -k="data"