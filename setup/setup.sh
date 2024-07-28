#!/usr/bin/env bash

sources=(shell .ssh)
destinations=(${HOME} ${HOME}/.ssh)

for i in ${!sources[@]}
do
	source=${sources[${i}]}
	destination=${destinations[${i}]}

	files=($(find ${source} -type f ))
	for file in ${files[@]}
	do
		source=$(realpath $(dirname ${file}))
		file=$(basename ${file})

		if [ -L ${destination}/${file} ]
		then
			rm -rfv ${destination}/${file}
		elif [ -f ${destination}/${file} ]
		then
			continue
		elif [ -d ${destination}/${file} ]
		then		
			continue
		fi

		echo ${source}/${file} "->" ${destination}/${file}

		ln -s ${source}/${file} ${destination}/${file}

	done

done
