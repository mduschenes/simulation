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
from src.call import submit,command,call,cp,rm,echo,sed,sleep
from src.io import load,dump

def test_cp(path=None):
	source = 'tmp.tmp'
	destination = 'tmp'

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	cp(source,destination,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)

	return


def test_rm(path=None):
	return
	path = 'tmp'

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	rm(path,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return

def test_echo(path=None):
	args = ['Hello','World']

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	echo(*args,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return

def test_sed(path=None):
	path = 'tmp.tmp'
	patterns = {'nodes=.*':'nodes=4'}
	default = '#SBATCH'

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	sed(path,patterns,default,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return

def test_sleep(path=None):
	pause = 3

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	sleep(pause,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return

def test_command(path=None):

	process = 'serial'
	process = 'parallel'
	processes = 4
	device = None
	execute = 1
	verbose = True

	exe = ['date']
	flags = []
	cmd = ["+%Y-%m-%d %H:%M:%S"]*10
	options = [';','sleep',1]
	args = []

	print(args)
	args = command(args,exe=exe,flags=flags,cmd=cmd,options=options,process=process,processes=processes,device=device,execute=execute,verbose=verbose)
	print(args)

	call(*args,execute=execute,verbose=verbose)

	return

def test_submit(path=None):
	default = {}
	settings = load(path,default=default)
	submit(**settings)
	return