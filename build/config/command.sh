#!/bin/bash
cluster=0

options="${@:-settings.json}"


if [[ ${cluster} == 1 ]]
then
	module purge

	module load jax0.2.24-cuda11.0-python3.8_jupyter

	# module load python-anaconda3/2019.10
	#pip install msgpack qtconsole pydot pydot-ng graphviz networkx python-igraph SALib configparser html2text pdf2image --user -q
	# pip install natsort jsonpickle xarray  --user -q
else
	:
fi
#echo PWD: $(pwd)




if [[ ${cluster} == 0 ]]
then
	src="/home/matt/files/uw/research/projects/simulation/code/src"
	exe="train.py"
else
	src="~/code/simulation/code/src"
	exe="train.py"	
fi



src=$(echo ${src} | sed "s%\~%${HOME}%g")
src="$(realpath ${src})"
exe="${src}/${exe}"
cmd=()

cmd+=("${exe}")
cmd+=(${options[@]})

${cmd[@]}