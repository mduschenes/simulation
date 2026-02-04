#!/usr/bin/bash -i

source ~/.bashrc tensor

path=${1}

name=${2:-all}

case ${name} in
	test)
		types=(stats)
		indices=("4 6")
		;;
	exit)
		types=(sample)
		indices=("1")
		types=(stats)
		indices=("6")
		;;
	sample)
		types=(sample)
		indices=("1")
		;;
	stats)
		types=(stats)
		indices=("4 6")
		;;
	all|*)
		types=(stats sample)
		indices=("4 6" "1")
		;;
esac

for count in ${!types[@]}
do

	type=${types[${count}]}

	indexes=(${indices[${count}]})

	folder=${path}
	file=${folder}/process.json
	bkp=${file}.bkp
	ext=json

	echo Process: ${type} ${indexes[@]} ${path}

	for number in ${!indexes[@]}
	do

		index=${indexes[${number}]}

		options=()

		case ${type} in
			sample)
				options+=(
					-i \
					-e "s/\(\"load\":\) [^{]*,$/\1 1,/" \
					-e "s/\(\"dump\":\) [^{]*,$/\1 0,/" \
					-e "s/\(\"stats.array.state.M.noise.parameters.N\":\) [^{]*,$/\1 0,/" \
					-e "s/\(\"sample.array.M.noise.parameters.N\":\) [^{]*,$/\1 1,/" \
					-e "s/\(\"sample.state.M.noise.parameters.N\":\) [^{]*,$/\1 0,/" \
					-e "s/\(\"fig.savefig.fname\": \"[^\"]*\)\",/\1.${index}\",/" \
					-e "s/\(size\":\) 45/\1 240/" \
				)
				case ${index} in
					0)
						options+=(
							-e "s/\(\"noise.parameters\":\ \)\[.*\]/\1[${index}]/" \
						)
						;;
					*)
						options+=(
							-e "s/\(\"noise.parameters\":\ \)\[.*\]/\1[1e-${index}]/" \
						)
						;;
				esac
				;;
			stats)
				options+=(
					-i \
					-e "s/\(\"load\":\) [^{]*,$/\1 1,/" \
					-e "s/\(\"dump\":\) [^{]*,$/\1 0,/" \
					-e "s/\(\"stats.array.state.M.noise.parameters.N\":\) [^{]*,$/\1 1,/" \
					-e "s/\(\"sample.array.M.noise.parameters.N\":\) [^{]*,$/\1 0,/" \
					-e "s/\(\"sample.state.M.noise.parameters.N\":\) [^{]*,$/\1 0,/" \
					-e "s/\(\"fig.savefig.fname\": \"[^\"]*\)\",/\1.${index}\",/" \
					-e "s/\(\"N\":\ \)null\(.*\)/\1[${index}]\2/" \
					-e "s/\(\"ax.set_title.label\":\).*/\1 null,/" \
					-e "s/\(size\":\) 45/\1 120/" \
				)
				case ${index} in
					4)
						options+=(
							-e "s/\(\"style.share.ax.legend\":\).*,/\1 true,/" \
						)
						;;
					*)
						options+=(
							-e "s/\(\"style.share.ax.legend\":\).*,/\1 false,/" \
						)
						;;
				esac
				;;
			log)
				options+=(
					-i \
					-e "s/\(\"fig.savefig.fname\": \"[^\"]*\)\",/\1.${index}\",/" \
					)
				case ${index} in
					0)
						options+=(
							-e "s/\(\"noise.parameters\":\ \)\[.*\]/\1[${index}]/" \
						)
						;;
					*)
						options+=(
							-e "s/\(\"noise.parameters\":\ \)\[.*\]/\1[1e-${index}]/" \
						)
						;;
				esac
				;;
			scale)
				options+=(
					-i \
					-e "s/\(\"fig.savefig.fname\": \"[^\"]*\)\",/\1.${index}\",/" \
					)
				case ${index} in
					0)
						options+=(
							-e "s/\(\"noise.parameters\":\ \)\[.*\]/\1[${index}]/" \
						)
						;;
					*)
						options+=(
							-e "s/\(\"noise.parameters\":\ \)\[.*\]/\1[1e-${index}]/" \
						)
						;;
				esac
				;;
			*)
				options+=()
				;;
		esac

		if [[ ! -s ${options} ]]
		then

			cp ${file} ${bkp}

			sed "${options[@]}" ${file}

			[[ ${name} == exit ]] && exit 0

			exe=./process.py
			args=(${path})

			${exe} ${args[@]}

			mv ${bkp} ${file}

		fi

	done


	processes=()

	case ${type} in
		sample|stats)

			for number in ${!indexes[@]}
			do
				index=${indexes[${number}]}

				folder=${path}/plot
				file=
				ext=pdf

				folders=../../notes/paper/figures
				files=()
				exts=pdf

				options=(-rfv)

				case ${type} in
					sample)
						file=plot.sample.array.M.noise.parameters.N.tetrad
						case ${index} in
							*)
								for i in 0 2 1
								do
									files+=(${file}.${i})
								done
								;;
							*)
								;;
						esac
						;;
					stats)
						file=plot.stats.array.state.M.noise.parameters.N.tetrad
						case ${index} in
							4)
								for i in 4
								do
									files+=(${file}.${i})
								done
								;;
							*)
								for i in 6 8 10
								do
									files+=(${file}.${i})
								done
								;;
							*)
								;;
						esac
						;;
					*)
						continue
						;;
				esac

				for i in ${files[@]}
				do
					cp ${options[@]} ${folder}/${file}.${index}.${ext} ${folders}/${i}.${exts}
				done

			done
			;;
		log)
			for variable in array state
			do
				folder=${path}/plot
				file=plot.sample.${variable}.M.noise.parameters.M.noise
				ext=pdf
				processes+=(${folder}/${file})

				string="$(echo ${indexes[@]} | sed 's/ /./g')"

			done

			if [[ ! -s ${processes} ]]
			then

				for process in ${processes[@]}
				do
					cmd=()
					cmd+=(pdfmerge)
					cmd+=("${process}.${string}.${ext}")
					for i in ${indexes[@]}
					do
						cmd+=("${process}.${i}.${ext}");
					done
					${cmd[@]}
				done
			fi
			;;
		scale)
			for variable in array state
			do
				folder=${path}/plot
				file=plot.sample.${variable}.M.noise.parameters.scale.M.noise
				ext=pdf
				processes+=(${folder}/${file})

				string="$(echo ${indexes[@]} | sed 's/ /./g')"

			done

			if [[ ! -s ${processes} ]]
			then

				for process in ${processes[@]}
				do
					cmd=()
					cmd+=(pdfmerge)
					cmd+=("${process}.${string}.${ext}")
					for i in ${indexes[@]}
					do
						cmd+=("${process}.${i}.${ext}");
					done
					${cmd[@]}
				done
			fi
			;;
		*)
			;;
	esac

done


for count in ${!types[@]}
do

	type=${types[${count}]}

	case ${type} in
		sample|stats)

			folder=../../notes/paper
			file=main.tex
			ext=tex

			options=(--options quiet)

			cd ${folder}

			complatex ${options[@]} ${file}

			cd -

			break
			;;
		*)
			;;
	esac
done



