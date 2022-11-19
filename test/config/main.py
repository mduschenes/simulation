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
from src.call import submit,command,call
from src.io import load,dump

def main(*args,**kwargs):
	path = kwargs['path']

	settings = load(path)

	# submit(**settings)


	path = None
	process = 'serial'
	process = 'parallel'
	processes = 4
	device = None
	execute = 1
	verbose = True

	exe = ['-k','touch']
	exe = ['date']
	flags = []
	cmd = ["+%Y-%m-%d %H:%M:%S"]*10
	options = [';sleep 1;']
	args = []

	print(args)
	args = command(args,exe=exe,flags=flags,cmd=cmd,options=options,process=process,processes=processes,device=device,execute=execute,verbose=verbose)
	print(args)

	call(*args,execute=execute,verbose=verbose)

	return

if __name__ == "__main__":

	args = argparser('path')

	main(*args,**args)

	
