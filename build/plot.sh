#!/usr/bin/env bash

path=${1}

name=${2:-all}

device=${3:-local}

case ${name} in
	test)
		types=()
		indices=()
		;;
	exit)
		types=()
		indices=()
		;;
	sample)
		types=(sample)
		indices=("0 4 3 2 1")
		;;
	stats)
		types=(stats)
		indices=("4 6 8 10")
		;;
	all)
		types=(stats sample)
		indices=("4 6 8 10" "0 4 3 2 1")
		;;
	*)
		types=()
		indices=()
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

	for string in tetrad pauli
	do
		[[ ${path} =~ .*${string}.* ]] && break
		string=
	done

	[[ -f ${bkp} ]] && mv ${bkp} ${file}

	echo Process: ${path} ::: ${string} ${type} ${indexes[@]}

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
					-e "s/\(\"fig.savefig.fname\":\).*,/\1 \"${string}.${index}\",/" \
					-e "s/\(size\":\) 45/\1 240/" \
					-e "s/\(\"M\":\) null/\1 [2,8,32]/" \
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

				case ${string} in
					tetrad)
						options+=(
							-e "s/\(\"ax.bar.plots\":\).*,/\1 \"errorbar\",/" \
						)
						;;
					pauli)
						options+=(
							-e "s/\(\"ax.bar.plots\":\).*,/\1 false,/" \
						)
						;;
					*)
						options+=()
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

				case ${string} in
					tetrad)
						options+=(
						)
						;;
					pauli)
						options+=(
						)
						;;
					*)
						options+=(
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

			[[ ${name} == exit ]] && exit

			exe=./process.py
			args=(${path})

			${exe} ${args[@]}

			mv ${bkp} ${file}

		fi

	done

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
						file=plot.sample.array.M.noise.parameters.N.${string}
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
						file=plot.stats.array.state.M.noise.parameters.N.${string}
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

				case ${device} in
					local)
						for i in ${files[@]}
						do
							cp ${options[@]} ${folder}/${file}.${index}.${ext} ${folders}/${i}.${exts}
						done
						;;
					slurm)
						;;
					*)
						;;
				esac
			done
			;;
		log|scale)

			processes=()
			for variable in array state
			do
				folder=${path}/plot
				file=plot.sample.${variable}.M.noise.parameters.M.noise
				ext=pdf
				processes+=(${folder}/${file})

				strings="$(echo ${indexes[@]} | sed 's/ /./g')"

			done

			if [[ ! -s ${processes} ]]
			then
				case ${device} in
					local)
						for process in ${processes[@]}
						do
							options=()
							for i in ${indexes[@]}
							do
								options+=("${process}.${i}.${ext}");
							done
							pdftk ${options[@]} cat output ${process}.${strings}.${ext}
						done
						;;
					slurm)
						;;
					*)
						;;
				esac
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
			file=main
			ext=tex

			options=()

			case ${device} in
				local)
					cd ${folder}

					latexmk ${options[@]} ${file} &>/dev/null
					latexmk -c ${options[@]} ${file} &>/dev/null

					cd -

					break
				;;
				slurm)
					;;
				*)
					;;
			esac
			;;
		*)
			;;
	esac
done