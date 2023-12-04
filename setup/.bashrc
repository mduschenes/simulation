#!/bin/bash


# Path
add_to_path() {
  for ARG in "$@"
  do
    if [ -d "$ARG" ] && [[ ":$PATH:" != *":$ARG:"* ]]; then
        PATH="${PATH:+"$PATH:"}$ARG"
    fi
  done
  export PATH="${PATH}"
}
add_to_path ~/.local/bin


# Aliases
alias breset="source ~/.bashrc"
alias lsh="ls -lht"
alias subl="rmate"
alias gis="git status"
alias bjob="squeue | grep ${USER}"
alias ijob='bjob | awk "{ print \$1 }" '
alias njob="bjob | wc -l"
alias scr="/scratch/gobi3/${USER}"

# SSH
# keys=($(find ~/.ssh -type f -regextype egrep -regex '.*/id_[^.]+$'))
# eval `keychain --quiet --eval ${keys[@]}`
{ eval "$(ssh-agent -s)"; } &>/dev/null
find ~/.ssh -type f -regextype egrep -regex '.*/id_[^.]+$' | xargs ssh-add {} &>/dev/null;

# Functions

function join {
	local delimiter=${1-} array=${2-}
	if shift 2; then
		printf %s "${array}" "${@/#/${delimiter}}"
	fi
}

# Git Add, Commit, Push Changes
function gips(){
	msg="${@}";

	git status;

	if [[ "${msg}" == "" ]]
	then
		return 0
	fi

	git commit -a -m "${msg}";
	git push;
	return 0
}

# Git Pull, Checkout, Merge, Push Changes
function gims(){
	branch=${1:-master}
	current=$(git name-rev --name-only HEAD)

	git status;

	git pull
	git checkout ${branch}
	git pull
	git checkout ${current}
	git merge ${branch}
	git push

	return 0
}

# Git Pull, Checkout, Copy
function gics(){
	branch=${1:-master}
	shift 1;
	files=(${@})
	current=$(git name-rev --name-only HEAD)

	git status;
	git pull

	git checkout ${branch}
	git pull
	
	git checkout ${current}
	
	for file in ${files[@]}
	do
		echo "git show ${branch}:./${file} > ${file}"
		git show ${branch}:./${file} > ${file}
	done

	return 0
}

function bkill(){
	jobs=(${@})

	if [ -z "${jobs}" ]
	then
		jobs=$(ijob | sed "s#\(.*\)_.*#\1#g")
	else
		for i in ${!jobs[@]}
		do
			jobs[$i]=$(bjob | grep ${jobs[$i]} | awk '{print $1}' | sed "s#\(.*\)_.*#\1#g")
		done
	fi
	
	jobs=($(echo "${jobs[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

	for job in ${jobs[@]}
	do
		scancel ${job}
	done

	return 0
}

function bint(){
	time=${1:-01:00:00}
	mem=${2:-15G}
	partition=${3:-cpu}
	srun --nodes=1 --ntasks-per-node=1 --time=${time} --mem=${mem} --pty bash -i
	return 0
}


function balloc(){
	time=${1:-01:00:00}
	mem=${2:-15G}
	partition=${3:-cpu}
	salloc --nodes=1 --ntasks-per-node=1 --time=${time} --mem=${mem} --partition=${partition}
	return 0
}


function bctl(){
	job=${1}
	time=${2}
	mem=${3}

	if [ -z ${job} ]
	then
		return 0
	fi

	if [ ! -z ${time} ]
	then
		scontrol update job=${job} TimeLimit=${time}
	fi

	if [ ! -z ${time} ]
	then
		scontrol update job=${job} MinMemoryNode=${mem}
	fi

	return 0
}

function berr(){
	jobs=(${@})
	script="job.slurm"
	exe="job.sh"
	name="output."
	ext="stderr"
	pattern="#SBATCH --array="
	options=":1%100"
	
	errors=()
	if [[ ${#jobs[@]} -eq 0 ]]
	then
		jobs=($(ls -t ${name}*${ext} | head -1 | sed "s:${name}\([^\.]*\)\.\([^\.]\).*${ext}:\1:"))
	fi

	for job in ${jobs[@]}
	do
		echo Job: ${job}
		files=($(ls ${name}${job}*${ext}))
		echo Files: ${files[@]}
		for file in ${files[@]}
		do
			echo File: ${file} 
			if [[ -s ${file} ]]
			then
				errors+=($(echo ${file} | sed "s:${name}\([^\.]*\)\.\([^\.]\).*${ext}:\2:"))
			fi
		done
	done
	errors=($(printf "%s\n" ${errors[@]} | sort -n))

	file=${script}

	line=$(( $(grep -n "${pattern}" ${file} | tail -1 | cut -f1 -d:) +1 ))	
	
	if [[ ! $(grep "^${pattern}$(join , ${errors[@]})${options}" ${file}) ]]
	then
		sed -i "s%^${pattern}%#${pattern}%g" ${file}
		sed -i "${line}i ${pattern}$(join , ${errors[@]})${options}" ${file}
	fi
	
	return 0
}

function breq(){
	jobs=(${@})
	script="job.slurm"
	exe="job.sh"
	name="output."
	ext="tmp"
	
	file=${script}.${ext}
	cp ${script} ${file}
	
	pattern="#SBATCH --array="
	options="#${pattern}"
	sed -i "s%^${pattern}%${options}%g" ${file}

	pattern="TASKS=.*"
	options="TASKS=(${jobs[@]})"
	sed -i "s%^${pattern}%${options}%g" ${file}

	file=${exe}.${ext}
	cp ${exe} ${file}

	pattern="< ${script}"
	options="< ${script}.${ext}"
	sed -i "s%${pattern}%${options}%g" ${file}

	return 0
}

function search(){
	files=(${@})

	patterns=('"M": [0-9]' '"ndim": 3' '"instance": [0-9]')
	lines=(0 -1 0)
	buffers=(1 1 1)

	patterns=("M:" "noise :" "instance:" "f(x)")
	lines=(0 0 0 0)
	buffers=(1 1 1 1)

	for i in ${!files[@]}
	do
		file=${files[${i}]}
		
		if [ ! ${#patterns[@]} -eq 0 ]
		then
			echo ${file}
		fi

		for j in ${!patterns[@]}
		do
			pattern="${patterns[${j}]}"
			line=${lines[${j}]}
			buffer=${buffers[${j}]}
			
			options=()
			pipe=()

			if [[ ${line} -lt 0 ]]
			then
				options+=(-B$((-${line})))
				pipe+=(head -n$((${buffer})))
			else
				options+=(-A$((${line})))
				pipe+=(tail -n$((${buffer})))
			fi

			if [ ! ${#pipe[@]} -eq 0 ]
			then
				grep ${options[@]} "${pattern}" ${file} | ${pipe[@]}
			else
				grep ${options[@]} "${pattern}" ${file}
			fi				
		done

		if [ ! ${#patterns[@]} -eq 0 ]
		then
			echo
		fi

	done

	return 0
}

function catls(){
	files=(${@})
	files=($(ls ${files[@]} 2> /dev/null | sort -V))
	for file in ${files[@]}
	do
		echo ${file}
		cat ${file}
		echo
	done

}

function idkeys(){
	encryption=${1:-ed25519}
	shift 1;
	hosts=(${@:-$(grep -P "^Host ([^*]+)$" $HOME/.ssh/config | sed 's/Host //')})
	for host in ${hosts[@]}
	do
		file=~/.ssh/id_${encryption}_${host}
		private=${file}
		public=${file}.pub
		if [ -f ${file} ]
		then
			continue
		fi
		echo ${host}
		ssh-add -D ${public} ${private}
		ssh-keygen -t ${encryption} -f ${file} -C "${host}" -N ""
		ssh-copy-id -i ${public} ${host}
		echo
	done
}


# Conda

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/pkgs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/pkgs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/pkgs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/pkgs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# <<< conda initialize <<<

export -f conda
export -f __conda_activate
export -f __conda_reactivate
export -f __conda_hashr	
export -f __add_sys_prefix_to_path


# Activate environment
env=jax
conda activate ${env}
# source activate ${HOME}/env/${env}
# source ${HOME}/env/${env}.env


# History Setup
export HISTCONTROL=ignoreboth:erasedups


# Color Setup

# # set a fancy prompt (non-color, unless we know we "want" color)
# case "$TERM" in
#     xterm-color|*-256color) color_prompt=yes;;
# esac

# # uncomment for a colored prompt, if the terminal has the capability; turned
# # off by default to not distract the user: the focus in a terminal window
# # should be on the output of commands, not on the prompt
# #force_color_prompt=yes

# if [ -n "$force_color_prompt" ]; then
#     if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
# 	# We have color support; assume it's compliant with Ecma-48
# 	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
# 	# a case would tend to support setf rather than setaf.)
# 	color_prompt=yes
#     else
# 	color_prompt=
#     fi
# fi

# if [ "$color_prompt" = yes ]; then
#     PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
# else    
#     PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
# fi
# unset color_prompt force_color_prompt

# # If this is an xterm set the title to user@host:dir
# case "$TERM" in
# xterm*|rxvt*)
#     PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
#     PROMPT_COMMAND='echo -ne "${USER}@${HOSTNAME}:${PWD/#$HOME/~}"'
#     ;;
# *)
# #    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
#     ;;
# esac

# # Prompt
PS1='${debian_chroot:+($debian_chroot)}[\t]\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\$ '
# #PROMPT_COMMAND="echo -ne \"\033]0;${PWD}\007\"; $PROMPT_COMMAND" ###*/
unset PROMPT_COMMAND
PROMPT_COMMAND='echo -ne """\033]0;${USER/"${HOME}"/\~}@${HOSTNAME%%.*}:${PWD}\007"""'



# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'
parallel --record-env