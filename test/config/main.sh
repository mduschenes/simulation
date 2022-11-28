#!/bin/bash

args=(${@})

echo HELLO WORLD ${#args[@]}
sleep 1
for i in ${!args[@]}
do
	arg=${args[${i}]}
	if [[ -f ${arg} ]]
	then
		echo $i ${arg}
		jq . ${arg}
		echo
	fi
done
sleep 1
echo GOODBYE WORLD ${#args[@]}