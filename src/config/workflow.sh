#!/bin/bash

path=${HOME}/scratch/scaling
hosts=(vector beluga cedar)

declare -A settings strings patterns
settings=(
	[N]="4;6;8"
	[seed]="135792468;246813579"
	[M]="1,2,4,6,8,10,12,14,16,18;20,25,30,35,40,45,50"
	[noise.parameters]="1e-4,1e-3,2.5e-3,5e-3,7.5e-3,1e-2,2e-2,3e-2;4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1,5e-1;1.5e-1,2e-1,2.5e-1,3e-1,3.5e-1,4e-1,4.5e-1"
	)
strings=(
	[N]="4;6;8"
	[seed]="135792468;246813579"
	[M]="1.18;20.50"
	[noise.parameters]="1e4.3e2;4e2.5e1;15e1.45e1"
	)
patterns=(
	[N]="module.N;model.N"
	[seed]="seed.seed"
	[M]="module.M"
	[noise.parameters]="model.data.noise.parameters"
	)
format="scaling.M._M_.N._N_.noise._noise.parameters_.seed._seed_"
paths="jobs.job.cwd;jobs.postprocess.cwd"
delimiter=";"
separator=","



function split() {
  delimiter="${1}"
  shift;
  array=(${@//"${delimiter}"/ })
  echo "${array[@]}"
}

function join() {
	local delimiter="${1-}" array=${2-}
	if shift 2; then
		printf %s "${array}" "${@/#/${delimiter}}"
	fi
}

function permute() {
	delimiter="${1-}"
	shift;
	declare -n temporary=$1

	exe=python
	flags=()
	cmd=$(echo "
		import sys;
		from itertools import product;
		strings=sys.argv[1:];
		delimiter='${delimiter}';
		strings = [string.split(delimiter) for string in strings];
		print('${IFS}'.join([delimiter.join(string) for string in product(*strings)]));
		" | tr -d "\t" | tr -d "\n")

	${exe} ${flags[@]} -c "${cmd}" ${temporary[@]}
}



function setup(){
	cwd=${1:-${path}/data}
	pwd=${2:-${HOME}/mnt/_host_/${path}}
	dir=${3:-${HOME}/mnt/_host_/code/tensor/build}
	data=(${4:-data.tmp.hdf5})
	config=(${4:-settings.json})

	names=(${!strings[@]})
	permutations=($(permute "${delimiter}" strings))

	for index in ${!permutations[@]}
	do
		permutation=($(split "${delimiter}" ${permutations[${index}]}))
		name=${format}

		for i in ${!permutation[@]}
		do
			name=${name%%_${names[$i]}_*}${permutation[$i]}${name##*_${names[$i]}_}
		done

		host=
		for string in ${hosts[@]}
		do
			tmp=${pwd%%_host_*}${string}${pwd##*_host_}/${name}
			if [ -d ${tmp} ]
			then
				host=${string}
				break
			fi
		done

		if [ -z ${host} ]
		then 
			continue
		fi

		obj=${cwd}/${name}
		if [[ ! -d ${obj} ]]
		then
			mkdir -v -p ${obj}
		fi

		obj=${cwd}/${name}/${host}
		if [[ ! -f ${obj} ]]
		then
			echo "-> ${obj}"
			touch ${obj}
		fi

		for file in ${data[@]}
		do
			source=${pwd%%_host_*}${host}${pwd##*_host_}/${name}/${file}
			destination=${cwd}/${name}/${file}
			if [[ -f ${source} ]] && [[ ! -f ${destination} ]]
			then
				cp -v ${source} ${destination}
			fi
		done

		for file in ${config[@]}
		do
			source=${dir%%_host_*}${host}${dir##*_host_}/${file}

			tmp="$(grep "${pattern}" ${source} | tail -1 | awk -F/ '{print $NF}' | tr -d '[],"')"

			destination=${cwd}/${tmp}/${file}

			if [[ ${tmp} == ${name} ]] && [[ -f ${source} ]] && [[ ! -f ${destination} ]]
			then
				cp -v ${source} ${destination}
			fi
		done
	done
}

function submit(){
	config=${1:-settings.json}
	exe=${2:-main.py}
	flags=(${3})
	env=(${4:-"NUMPY_BACKEND=numpy"})

	names=(${!settings[@]})

	_settings_=($(permute "${delimiter}" settings))
	_strings_=($(permute "${delimiter}" strings))

	for index in ${!_settings_[@]}
	do
		setting=($(split "${delimiter}" ${_settings_[${index}]}))
		string=($(split "${delimiter}" ${_strings_[${index}]}))
		name=${format}

		for i in ${!names[@]}
		do
			name=${name%%_${names[$i]}_*}${string[$i]}${name##*_${names[$i]}_}
		done

		if [[ ! -d ${path}/${name} ]]
		then

			for pattern in $(split "${delimiter}" ${paths})
			do
				echo sed -i "s%\"${pattern}\"[^:]*:.*[^,]\(,*\)%\"${pattern}\":[\"${path}/${name}\"]\1%g" ${config}
			done

			for i in ${!names[@]}
			do
				for pattern in $(split "${delimiter}" ${patterns[${names[$i]}]})
				do
					echo sed -i "s%\"${pattern}\"[^:]*:.*[^,]\(,*\)%\"${pattern}\":[${setting[$i]}]\1%g" ${config}
				done
			done
		fi

		if [[ -d ${path}/${name} ]] && [[ -z "$(squeue --format="%.100j" -u ${USER} | tr " " | grep ${name})" ]]
		then
			echo Running Job: ${name}
			break	
		elif [[ -d ${path}/${name} ]]
		then
			echo Existing Job: ${name}
			echo queue ${path}/${name}
			break	
		elif [[ ! -d ${path}/${name} ]]
		then
			echo New Job: ${name}
			echo ${env[@]} ${exe} ${flags[@]} ${config}
			break
		else
			continue
		fi

	done

}

function queue(){
	
	cwd=${1:-.}
	name="${2:-"*"}"

	script="job.slurm"
	exe="job.sh"
	ext="stderr"
	pattern="#SBATCH --array="
	options="%100"
	delimiter=,


	pwd=$(pwd)

	cd ${cwd}


	jobs=($(find . -maxdepth 1 -mindepth 1 -type f  -name "${name}\.[0-9]*\.[0-9]*\.${ext}" | sort -n | tail -1 |  awk '{print $NF}' | sed "s:.*\.\([^\.]*\)\.\([^\.]\).*${ext}:\1:"))
	
	errors=()

	for job in ${jobs[@]}
	do
		files=($(ls ${name}.${job}*${ext}))
		string=
		for file in ${files[@]}
		do
			if [[ -s ${file} ]]
			then
				string=${job}
				tmp=$(echo ${file} | sed "s:.*\.\([^\.]*\)\.\([^\.]*\)\.${ext}:\2:")
				if [[ ! ${tmp} == ${file} ]]
				then
					errors+=(${tmp})
				fi
			fi
		done
		if [[ ! -z ${string} ]]
		then
			echo Job: ${string}
		fi
	done
	errors=($(printf "%s\n" ${errors[@]} | sort -n))

	file=${script}

	line=$(( $(grep -n "${pattern}" ${file} | tail -1 | cut -f1 -d:) +1 ))	
	
	if [[ ! -z ${jobs} ]] && [[ ! -z ${errors} ]] && [[ ! $(grep "^${pattern}$(join "${delimiter}" ${errors[@]})${options}" ${file}) ]]
	then
		sed -i "s%^${pattern}%#${pattern}%g" ${file}
		sed -i "${line}i ${pattern}$(join "${delimiter}" ${errors[@]})${options}" ${file}
	fi

	if [[ ! -z ${jobs} ]] && [[ ! -z ${errors} ]]
	then
		echo Jobs: ${errors[@]}
		job=$(./${exe} | tail -1)
	else
		job=
	fi


		script="postprocess.slurm"
		exe="postprocess.sh"
		pattern="#SBATCH --dependency="
		options=
		delimiter=,

	file=${script}

	line=$(( $(grep -n "${pattern}" ${file} | tail -1 | cut -f1 -d:) +1 ))	
	
	if [[ ! -z ${job} ]] && [[ ! $(grep "^${pattern}$(join "${delimiter}" ${job})${options}" ${file}) ]]
	then
		sed -i "s%^${pattern}%#${pattern}%g" ${file}
		sed -i "${line}i ${pattern}$(join "${delimiter}" ${job})${options}" ${file}
	fi

	if [[ ! -z ${job} ]]
	then
		job=$(./${exe} | tail -1)
	else
		job=
	fi


	cd ${pwd}

}



func=${1:-setup}

shift;

if [ ! -z ${func} ]
then
	${func} ${@}
fi