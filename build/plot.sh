#!/usr/bin/env bash

path=${1}

name=${2:-all}

device=${3:-${HOSTNAME}}

case ${name} in
	test)
		types=()
		indices=()
		strings=("")
		;;
	exit)
		types=()
		indices=()
		strings=("")
		;;
	process)
		types=(process stats sample)
		indices=("null" "4 6 8 10" "0 4 3 2 1")
		strings=(1.0 0.5 0.25)
		;;
	sample)
		types=(sample)
		indices=("0 4 3 2 1")
		strings=("")
		;;
	stats)
		types=(stats)
		indices=("4 6 8 10")
		strings=()
		;;
	all)
		types=(stats sample)
		indices=("4 6 8 10" "0 4 3 2 1")
		strings=("")
		;;
	*)
		types=()
		indices=()
		strings=("")
		;;
esac

case ${device} in
	local)
		if [[ ${path} =~ ~/mnt/.* ]]
		then
			device=cluster
		else
			device=${device}
		fi
		;;
	*)
		device=slurm
		;;
esac

for counter in ${!strings[@]}
do

	string=${strings[${counter}]}

	for count in ${!types[@]}
	do

		type=${types[${count}]}

		indexes=(${indices[${count}]})

		folder=${path}
		files=(process.json plot.json)
		bkp=bkp
		ext=json

		for variable in tetrad pauli
		do
			[[ ${path} =~ .*${variable}.* ]] && break
			variable=
		done

		for file in ${files[@]}
		do
			[[ -f ${path}/${file}.${bkp} ]] && mv ${path}/${file}.${bkp} ${path}/${file}
		done

		echo Process: ${path} ::: ${variable} ${type} ${indexes[@]}

		for number in ${!indexes[@]}
		do

			index=${indexes[${number}]}

			options=()
			settings=()

			case ${type} in
				process)
					options+=(
						-i \
						-e "s/\(\"load\":\) [^{]*,$/\1 0,/" \
						-e "s/\(\"dump\":\) [^{]*,$/\1 1,/" \
						-e "s/\(\"plot\":\) [0|1],$/\1 0,/" \
						-e "s/\(\"stats.array.state.M.noise.parameters.N\":\) [^{]*,$/\1 1,/" \
						-e "s/\(\"sample.array.M.noise.parameters.N\":\) [^{]*,$/\1 1,/" \
						-e "s/\(\"sample.state.M.noise.parameters.N\":\) [^{]*,$/\1 0,/" \
					)
					settings+=(
						-i \
						-e "s/\({\"sample\":\) [^}]*}/\1 ${string}}/" \
					)
					;;
				sample)
					options+=(
						-i \
						-e "s/\(\"load\":\) [^{]*,$/\1 1,/" \
						-e "s/\(\"dump\":\) [^{]*,$/\1 0,/" \
						-e "s/\(\"stats.array.state.M.noise.parameters.N\":\) [^{]*,$/\1 0,/" \
						-e "s/\(\"sample.array.M.noise.parameters.N\":\) [^{]*,$/\1 1,/" \
						-e "s/\(\"sample.state.M.noise.parameters.N\":\) [^{]*,$/\1 0,/" \
						-e "s/\(\"fig.savefig.fname\":\).*,/\1 \"${variable}.${index}\",/" \
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

					case ${variable} in
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

					settings+=()

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

					case ${variable} in
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

					settings+=()

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

					settings+=()

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

					settings+=()

					;;
				*)
					options+=()
					settings+=()
					;;
			esac

			case ${device} in

				local|slurm)

					for file in ${files[@]}
					do
						cp ${path}/${file} ${path}/${file}.${bkp}

						flags=()
						case ${file} in
							process.json)
								flags+=("${options[@]}")
								;;
							plot.json)
								flags+=("${settings[@]}")
								;;
							*)
								continue
								;;
						esac

						[[ -z "${flags[@]}" ]] && continue

						sed "${flags[@]}" ${path}/${file}

					done

					[[ ${name} == exit ]] && exit

					exe=./process.py
					args=(${path})

					${exe} ${args[@]}

					for file in ${files[@]}
					do
						mv ${path}/${file}.${bkp} ${path}/${file}
					done
					;;
				*)
					;;
			esac

		done

		case ${type} in
			process|sample|stats)

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
						process)
							files=()
							;;
						sample|stats)
							file=plot.${type}.array.M.noise.parameters.N.${variable}
							case ${index} in
								*)
									files+=(${file}.${index})
									;;
							esac
							;;
						*)
							continue
							;;
					esac

					case ${device} in
						local|cluster)
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

					data="$(echo ${indexes[@]} | sed 's/ /./g')"

				done

				if [[ ! -z ${processes} ]]
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
								pdftk ${options[@]} cat output ${process}.${data}.${ext}
							done
							;;
						cluster)
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
			process)
				folder=${path}/sample
				file=plot
				ext=json

				files=data
				exts=hdf5

				options=(-rfv)


				data=$(grep -o "\"sample\": [0-9.0-9]*" ${path}/${file}.${ext} | head -n1 | awk '{ print $2 }')

				if [[ ! -z ${data} ]] && [[ -f ${path}/${files}.${exts} ]]
				then
					mkdir -p ${folder}
					cp ${options[@]} ${path}/${files}.${exts} ${folder}/${files}.${data}.${exts}
				fi

				;;
			sample|stats)

				folder=../../notes/paper
				file=main
				ext=pdf

				options=()

				case ${device} in
					local|cluster)
						cd ${folder}

						latexmk ${options[@]} ${file} &>/dev/null
						latexmk -c ${options[@]} ${file} &>/dev/null

						qpdfview ${file}.${ext} --unique & >/dev/null 2>&1

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
done