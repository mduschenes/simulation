#!/bin/bash


function setup(){
	cwd=${1:-scratch/povm/scaling/data}
	pwd=${2:-${HOME}/mnt/_HOST_/scratch/scaling}
	config=${3:-${HOME}/mnt/_HOST_/code/tensor/build}
	files=(${4:-data.tmp.hdf5})
	settings=(${4:-settings.json})
	names=${5:-scaling.M._M_.N._N_.noise._noise_.seed._seed_}
	patterns=${6:-jobs.job.cwd}

	hosts=(vector beluga cedar)
	N=(4 6 8)
	seed=(135792468 246813579)
	M=("1.18" "20.50")
	noise=("1e4.3e2" "4e2.5e1" "15e1.45e1")

	for n in ${N[@]}
	do
		for s in ${seed[@]}
		do
			for m in ${M[@]}
			do
				for g in ${noise[@]}
				do
					
					name=${names}
					name=${name%%_M_*}${m}${name##*_M_}
					name=${name%%_N_*}${n}${name##*_N_}
					name=${name%%_noise_*}${g}${name##*_noise_}
					name=${name%%_seed_*}${s}${name##*_seed_}

					pattern=${patterns}


					host=
					for string in ${hosts[@]}
					do
						tmp=${pwd%%_HOST_*}${string}${pwd##*_HOST_}/${name}
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


					path=${cwd}/${name}
					if [[ ! -d ${path} ]]
					then
						mkdir -p ${path}
					fi

					path=${cwd}/${name}/${host}
					if [[ ! -f ${path} ]]
					then
						touch -v ${path}
					fi

					for file in ${files[@]}
					do
						source=${pwd%%_HOST_*}${host}${pwd##*_HOST_}/${name}/${file}
						destination=${cwd}/${name}/${file}
						if [[ -f ${source} ]] && [[ ! -f ${destination} ]]
						then
							cp -v ${source} ${destination}
						fi
					done

					for setting in ${settings[@]}
					do
						source=${config%%_HOST_*}${host}${config##*_HOST_}/${setting}

						tmp="$(grep "${pattern}" ${source} | tail -1 | awk -F/ '{print $NF}' | tr -d '[],"')"

						destination=${cwd}/${tmp}/${setting}

						if [[ ${tmp} == ${name} ]] && [[ -f ${source} ]] && [[ ! -f ${destination} ]]
						then
							cp -v ${source} ${destination}
						fi
					done

				done
			done
		done
	done
}


func=${1:-setup}

shift;


if [ ! -z ${func} ]
then
	${func} ${@}
fi