#!/bin/bash

exe=${1}
input=${2:-stats.profile}
output=${3:-pdf}

./${exe}

gprof2dot -f pstats ${input} | dot -T${output} -o ${input}.${output}

# snakeviz ${input}