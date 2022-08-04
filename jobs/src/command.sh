#!/bin/bash
cluster=1

options="${@:-settings.prm}"


if [[ ${cluster} == 1 ]]
then
	module purge

	module load python-anaconda3/2019.10
	#pip install msgpack qtconsole pydot pydot-ng graphviz networkx python-igraph SALib configparser html2text pdf2image --user -q
	pip install natsort jsonpickle xarray  --user -q
fi
#echo PWD: $(pwd)




if [[ ${cluster} == 0 ]]
then
	src="/home/matt/files/um/code/mechanochem-ml-code/workflows/graphtheory/examples/microstructures"
else
	src="~/code/mechanochem-ml-code/workflows/graphtheory/examples/microstructures"
fi



src=$(echo ${src} | sed "s%\~%${HOME}%g")
src="$(realpath ${src})"
exe="${src}/main.py"
cmd=()

cmd+=("${exe}")
cmd+=(${options[@]})

${cmd[@]}

