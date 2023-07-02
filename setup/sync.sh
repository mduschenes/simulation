#!/bin/bash


# rclone backup
home="h/mduschen"
scripts="${home}/backup"

folder=${1:-"/scratch/ssd004/scratch/mduschen"}
local=$(realpath ${folder})
remote="${2:-"mnt"}:/data/uw/simulation/vectorv" #$(basename ${folder})"
dryrun=${3:-0}
local_remote=${4:-1}
remote_local=${5:-0}


log="${scripts}/log.txt"
filter="${scripts}/filter.txt"
options=("--verbose" --progress "--filter-from ${filter}" "--log-file ${log}" --fast-list --transfers=40 --checkers=40 --tpslimit=10 --onedrive-chunk-size=10M)

if [[ $dryrun -eq 1 ]]
then
	options+=("--dry-run")
fi

cmd_sync_local_remote=("rclone" "sync" "${local}" "${remote}" ${options[@]}) 
cmd_copy_remote_local=("rclone" "copy" "${remote}" "${local}" ${options[@]})

# Only Keep Log Data Yearly
dates=(m d H M)
values=(1 1 0 0)
bool=1

for i in ${!dates[@]}
do
	d=$(echo $(date +"%${dates[$i]}") | sed 's/^0*//')
	if ! [[ $d -eq ${values[$i]} ]]
	then
		bool=0
		break
	fi
done

if [[ $bool == 1 ]]
then
	# echo REMOVE log
	rm $log
	touch $log
else
	# echo KEEP log
	:
	# rm $log
	# touch $log
fi

# Sync and Copy

echo ${local_remote}

if [[ ${local_remote} -eq 1 ]] || [[ ${remote_local} -eq 1 ]]
then
	echo "***Start sync and copy: $(date +"%Y-%m-%d %H:%M:%S")***" | tee -a $log
	echo Local: ${local}
	echo Remote: ${remote}
	echo
fi

if [[ ${local_remote} -eq 1 ]]
then
	echo "Start sync local -> remote: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a $log
	echo "${cmd_sync_local_remote[@]}" | tee -a $log
	${cmd_sync_local_remote[@]}	
	# timeout 25m /usr/bin/flock -n /tmp/fcj.lockfile $cmd_copy_remote_local
	echo "Done  sync: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a $log
	echo | tee -a $log
fi

if [[ ${remote_local} -eq 1 ]]
then
	echo "Start copy remote -> local: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a $log
	echo "${cmd_copy_remote_local[@]}" | tee -a $log
	${cmd_copy_remote_local[@]} 
	# timeout 25m /usr/bin/flock -n /tmp/fcj.lockfile $cmd_copy_remote_local
	echo "Done copy: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a $log
	echo | tee -a $log
fi


if [[ ${local_remote} -eq 1 ]] || [[ ${remote_local} -eq 1 ]]
then
	echo "***Done sync and copy: $(date +"%Y-%m-%d %H:%M:%S")***" | tee -a $log
	echo | tee -a $log
	echo | tee -a $log
fi
