#!/usr/bin/env python

# Import python modules
import os,sys,itertools,functools,datetime
from copy import deepcopy

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..','../..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.utils import argparser
from src.call import submit,command,call,cp,rm,ls,echo,sed,sleep,touch
from src.io import load,dump,glob,dirname

def test_touch(path=None,tol=None):
	path = '.tmp.tmp/tmp.sh'
	args = ['./job.slurm . mkl ~/files/uw/research/code/simulation/code/src train.py 1 settings.json']
	env = {'SLURM_VAR':10,"SLURM_FOO":"BAR"}
	mod = True

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	touch(path,*args,env=env,mod=mod,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)

	path = dirname(path)
	rm(path,execute=True)

	return

def test_cp(path=None,tol=None):
	source = '.tmp.tmp/tmp.sh'
	destination = '.tmp.tmp/new.sh'

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	cp(source,destination,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)

	path = source
	path = dirname(path)
	rm(path,execute=True)

	path = destination
	path = dirname(path)
	rm(path,execute=True)

	return


def test_rm(path=None,tol=None):
	path = '.tmp.tmp/tmp.sh'

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	touch(path,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	
	path = dirname(path)
	rm(path,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)

	return


def test_ls(path=None,tol=None):
	path = '.'
	args = ['-lht']

	process = None
	processes = None
	device = None
	execute = True
	verbose = False
	kwargs = {}


	returns = ls(path,*args,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)

	return

def test_echo(path=None,tol=None):
	args = ['Hello','World']

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	echo(*args,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return

def test_sed(path=None,tol=None):

	path = 'config/test.slurm'
	patterns = {'nodes=.*':'nodes=4'}
	default = '#SBATCH'

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	source = 'config/job.slurm'
	destination = path

	cp(source,destination,execute=True)
	sed(path,patterns,default,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	rm(destination,execute=True)
	return

def test_sleep(path=None,tol=None):
	pause = 3

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	sleep(pause,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return


def test_call(path=None,tol=None):

	file = 'text.txt'
	path = file
	args = ['Hello','World']
	touch(path,*args,execute=True)

	process = None
	device = None
	execute = True
	verbose = 'info'

	default = -1
	def wrapper(stdout,stderr,returncode,default=default):
		try:
			result = int(stdout)
		except:
			result = stdout
		return result


	pattern = '#SBATCH'
	path = 'config/test.slurm'

	args = []

	exe = ['awk']
	flags = []
	cmd = [' /%s/ {print FNR}'%(pattern),path]
	arg = [*exe,*flags,*cmd]
	args.append(arg)

	exe = ['tail']
	flags = ['--lines=1']
	cmd = []
	arg = [*exe,*flags,*cmd]
	args.append(arg)


	exe = ['cat']
	flags = ['<']
	cmd = [file]
	arg = [*exe,*flags,*cmd]
	args.append(arg)

	exe = ['grep']
	flags = []
	cmd = ['job']
	arg = [*exe,*flags,*cmd]
	args.append(arg)


	exe = ['sbatch']
	flags = ['--export=JOB_SRC=../../src,JOB_CMD=train.py,JOB_ARGS=settings.json','<']
	cmd = [path]

	arg = [*exe,*flags,*cmd]
	args.append(arg)

	source = 'config/job.slurm'
	destination = path

	cp(source,destination,execute=True)
	result = call(*args,wrapper=wrapper,process=process,device=device,execute=execute,verbose=verbose)
	rm(destination,execute=True)

	pause = 10
	sleep(pause,execute=True,verbose=True)

	path = file
	rm(path,execute=True)

	path = "output.*.std*"
	rm(path,execute=True,verbose=True)

	return


if __name__ == '__main__':
	path = 'config/settings.json'
	tol = 5e-8 

	test_touch(path,tol)
	# test_ls(path,tol)
	# test_call(path,tol)