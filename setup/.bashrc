#!/bin/bash


# Path
export PATH="$PATH:~/.local/bin"


# Aliases
alias breset="source ~/.bashrc"
alias lsh="ls -lht"
alias subl="rmate"
alias mlab="matlab -nodisplay -nojvm -batch"
alias gis="git status"
alias bjob="squeue | grep ${USER}"
alias ijob='bjob | awk "{ print \$1 }" '
alias njob="bjob | wc -l"

# SSH
{
	eval $(ssh-agent -s);
	ssh-add ~/.ssh/id_25519;
} &>/dev/null 2>&1


# Functions

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

function bkill(){
	job=${1}
	if [ -z "${job}" ]
	then
		jobs=$(ijob | sed "s#\(.*\)_.*#\1#g")
	else
		jobs=($(bjob | grep ${job} | awk '{print $1}' | sed "s#\(.*\)_.*#\1#g"))
	fi
	for job in ${jobs[@]}
	do
		scancel ${job}
	done
	return 0
}

function bint(){
	time=${1:-02:00:00}
	mem=${2:-15G}
	partition=${3:-cpu}
	account=${4:-def-carrasqu}
	srun --nodes=1 --ntasks-per-node=1 --time=${time} --mem=${mem} --account=${account} --pty bash -i
	return 0
}


function balloc(){
	time=${1:-01:00:00}
	mem=${2:-15G}
	partition=${3:-cpu}
	account=${4:-def-carrasqu}	
	salloc --nodes=1 --ntasks-per-node=1 --time=${time} --mem=${mem} --partition=${partition} --account=${account}
	return 0
}


function catls(){
	files=(${@})
	files=($(ls ${files[@]} | sort -V))
	for file in ${files[@]}
	do
		echo ${file}
		cat ${file}
		echo
	done

}





# Conda

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/pkgs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/pkgs/anaconda3/etc/profile.d/conda.sh" ]; then
#         . "/pkgs/anaconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/pkgs/anaconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup

# # <<< conda initialize <<<

# export -f conda
# export -f __conda_activate
# export -f __conda_reactivate
# export -f __conda_hashr	
# export -f __add_sys_prefix_to_path


# Activate environment
env=jax
# conda activate ${env}
# source activate ${HOME}/env/${env}
# source ${HOME}/env/${env}.env
source ${HOME}/env/${env}/bin/activate

# History Setup
export HISTCONTROL=ignoreboth:erasedups


# Color Setup

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi
if [ "$color_prompt" = yes ]; then
    PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else    
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    # PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]\$ "
    # PROMPT_COMMAND='echo -ne "${USER}@${HOSTNAME}:${PWD/#$HOME/~}"'
    ;;
*)
   # PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
esac

PS1="[\t]\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\$ "
PROMPT_COMMAND='echo -ne "\033]0;${USER}@${HOSTNAME}: ${PWD/$HOME/~}\007"'


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
# Source global definitions
# if [ -f /etc/bashrc ]; then
# 	. /etc/bashrc
# fi
