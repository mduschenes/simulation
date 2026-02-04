#!/usr/bin/bash -i

source ~/.bashrc tensor

path=${1}
types=(${2:-log})
types=(${2:-stats})
indices=(${3:-0 4 3 2 1})
indices=(${3:-4 6})
string="$(echo ${indices[@]} | sed 's/ /./g')"


for type in ${types[@]}
do

	case ${type} in
		log)
			label=""
			;;
		scale)
			label="scale."
			;;
		stats)
			label=""
			;;
		*)
			label=""
			;;
	esac



	for number in ${!indices[@]}
	do
		index=${indices[${number}]}

		case ${type} in
			log|scale)
				case ${index} in
					0)
						strings=(${index} ${index})
						;;
					*)
						strings=(1e-${index} ${index})
						;;
				esac
				;;
			stats)
				case ${index} in
					0)
						strings=(${index} ${index})
						;;
					*)
						strings=(${index} ${index})
						;;
				esac
				;;
			*)
				string=""
				;;
		esac


		folder=${path}
		file=${folder}/process.json
		bkp=${file}.bkp
		ext=json

		options=()


		case ${type} in
			log|scale)

				options+=(
					-i \
					-e "s/\(\"fig.savefig.fname\"\:\).*/\1\"${label}${strings[1]}\",/" \
					-e "s/\(\"noise.parameters\"\:\ \)\[.*\]/\1[${strings[0]}]/" \
				)
				;;
			stats)

				options+=(
					-i \
					-e "s/\(\"fig.savefig.fname\"\:\"tetrad[^.]*\)[^\"]*\",/\1.${label}${strings[1]}\",/" \
					-e "s/\(\"N\"\:\ \)null\(.*\)/\1[${strings[0]}]\2/" \
					-e "s/\(\"ax.set_title.label\": \).*/\1 null,/" \
					-e "s/\(size\":\) 45/\1 120/" \
				)

				case ${number} in
					0)
						options+=(
							-e "s/\(\"style.share.ax.legend\"\).*,/\1: true,/" \
						)
						;;
					*)
						options+=(
							-e "s/\(\"style.share.ax.legend\"\).*,/\1: false,/" \
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

			exe=./process.py
			args=(${path})

			${exe} ${args[@]}

			mv ${bkp} ${file}

		fi

	done


	process=()

	case ${type} in
		log|scale)
			for variable in array state
			do
				folder=${path}/plot
				file=plot.sample.${variable}.M.noise.parameters.${label}M.noise
				ext=pdf
				process+=(${folder}/${file})
			done

			if [[ ! -s ${process} ]]
			then

				for name in ${process[@]}
				do
					cmd=()
					cmd+=(pdfmerge)
					cmd+=("${name}.${string}.${ext}")
					for i in ${indices[@]}
					do
						cmd+=("${name}.${i}.${ext}");
					done
					${cmd[@]}
				done
			fi
			;;
		stats)

			for number in ${!indices[@]}
			do
				index=${indices[${number}]}

				folder=${path}/plot
				file=plot.stats.array.state.M.noise.parameters.N.tetrad
				ext=pdf

				folders=../../notes/paper/figures
				files=()
				exts=pdf

				options=(-rfv)

				case ${number} in
					0)
						files+=(${file}.${index})
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

				for i in ${files[@]}
				do
					cp ${options[@]} ${folder}/${file}.${index}.${ext} ${folders}/${i}.${exts}
				done

			done


			folder=../../notes/paper
			file=main.tex
			ext=tex

			options=(--options quiet)

			cd ${folder}

			complatex ${options[@]} ${file}

			cd -

			;;
		*)
			;;
	esac

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

done