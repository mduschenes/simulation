#!/usr/bin/env bash

run=${1}

path=../../../../notes/paper

folder=~/scratch/probability/distribution/plot
file="plot.distribution.N.*.pdf"



exe=main.py
options=()

[[ ${run} ]] && ./${exe} ${options[@]}




folders=figures
options=(-rfv)

cp ${options[@]} ${folder}/${file} ${path}/figures



cd ${path}

file=main
ext=pdf

options=(-quiet)

latexmk ${options[@]} ${file} &>/dev/null
latexmk -c ${options[@]} ${file} &>/dev/null

qpdfview ${file}.${ext} --unique & >/dev/null 2>&1

cd -