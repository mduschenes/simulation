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

def test_cp():
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


def test_rm():
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

def test_echo():
	args = ['Hello','World']

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	echo(*args,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return

def test_sed():
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

def test_sleep():
	pause = 3

	process = None
	processes = None
	device = None
	execute = True
	verbose = True
	kwargs = {}

	sleep(pause,process=process,processes=processes,device=device,execute=execute,verbose=verbose,**kwargs)
	return
