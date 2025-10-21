#!/usr/bin/env bash

source ~/.bashrc

path=${1}
type=${2:-log}
indices=(${3:-0 4 3 2 1})

options=()
process=()

for index in ${indices[@]}
do
	case ${index} in
		0)
			strings=(${index} ${index})
			;;
		*)
			strings=(1e-${index} ${index})
			;;
	esac

	case ${type} in
		log)
			options=()
			process=()

			options+=(
				-i \
				-e "s/\(\"noise.parameters\"\:\ \)\[.*\]/\1[${strings[0]}]/" \
				-e "s/\(\"fig.savefig.fname\"\:\).*/\1\"M.noise.${strings[1]}\",/" \
				${path}/process.json
			)

			process+=(pdfmerge)
			name=${path}/plot/plot.sample.array.M.noise.parameters.M.noise
			string="$(echo ${indices[@]} | sed 's/ /./g')"; process+=("${name}.${string}.pdf");
			for i in ${indices[@]}
			do
				process+=("${name}.${i}.pdf");
			done
			;;
		scale)
			options=()
			process=()

			options+=(
				-i \
				-e "s/\(\"noise.parameters\"\:\ \)\[.*\]/\1[${strings[0]}]/" \
				-e "s/\(\"fig.savefig.fname\"\:\).*/\1\"scale.M.noise.${strings[1]}\",/" \
				${path}/process.json
			)

			process+=(pdfmerge)
			name=${path}/plot/plot.sample.array.M.noise.parameters.scale.M.noise
			string="$(echo ${indices[@]} | sed 's/ /./g')"; process+=("${name}.${string}.pdf");
			for i in ${indices[@]}
			do
				process+=("${name}.${i}.pdf");
			done
			;;
		*)
			;;
	esac

	continue

	if [[ ! -s ${options} ]]
	then
		sed "${options[@]}"

		exe=./process.py
		args=(${path})

		${exe} ${args[@]}
	fi

done


if [[ ! -s ${process} ]]
then
	${process[@]}
fi

# from src.utils import array,flatten,is_naninf
# from src.io import load,dump,exists

# if not exists(path):
# 	merge(data,path,*args,**kwargs)

# options = dict(wrapper='df',verbose=True)
# data = load(path,**options)

# print(data.shape,[*data.columns])

# keys = {key:['%s'%(key),'%s.error'%(key)] for key in ['sample.array.information','sample.state.information']}
# by = ['N','M','noise.parameters']
# options = dict(as_index=False,dropna=False)
# def func(data):
# 	data = array([*flatten(data)])
# 	return data

# data = data.groupby(by=by,**options)

# for groups in data.groups:

# 	print(dict(zip(by,groups)))

# 	group = data.get_group(groups)

# 	for key in keys:
# 		for i in keys[key]:
# 			value = func(group[i])
# 			print(i,i[is_naninf(i)].shape)
# 			print(i)
# 			print()