#!/usr/bin/env bash

source ~/.bashrc

path=${1}
type=${2:-log}
indices=(${3:-0 4 3 2 1})
string="$(echo ${indices[@]} | sed 's/ /./g')"
case ${type} in
	log)
		label=""
		;;
	scale)
		label="scale."
		;;
	*)
		label=""
		;;
esac

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

	options=()

	options+=(
		-i \
		-e "s/\(\"noise.parameters\"\:\ \)\[.*\]/\1[${strings[0]}]/" \
		-e "s/\(\"fig.savefig.fname\"\:\).*/\1\"${label}M.noise.${strings[1]}\",/" \
		${path}/process.json
	)

	if [[ ! -s ${options} ]]
	then
		sed "${options[@]}"

		exe=./process.py
		args=(${path})

		echo ${exe} ${args[@]}
	fi

done


process=()

for variable in array state
do
	process+=(${path}/plot/plot.sample.${variable}.M.noise.parameters.${label}M.noise)
done

if [[ ! -s ${process} ]]
then

	for name in ${process[@]}
	do
		cmd=()
		cmd+=(pdfmerge)
		cmd+=("${name}.${string}.pdf")
		for i in ${indices[@]}
		do
			cmd+=("${name}.${i}.pdf");
		done
		echo ${cmd[@]}
	done
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
