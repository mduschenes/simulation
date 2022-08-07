#!/bin/bash
cluster=0

options="${@:-settings.json}"


if [[ ${cluster} == 1 ]]
then
	module purge

	module load python-anaconda3/2019.10
	#pip install msgpack qtconsole pydot pydot-ng graphviz networkx python-igraph SALib configparser html2text pdf2image --user -q
	# pip install natsort jsonpickle xarray  --user -q
else
	:
fi
#echo PWD: $(pwd)




if [[ ${cluster} == 0 ]]
then
	src="/home/matt/files/uw/research/projects/simulation/code/build"
	exe="main.py"
else
	src="~/code/simulation/code/build"
	exe="main.py"	
fi



src=$(echo ${src} | sed "s%\~%${HOME}%g")
src="$(realpath ${src})"
exe="${src}/${exe}"
cmd=()

cmd+=("${exe}")
cmd+=(${options[@]})

${cmd[@]}

