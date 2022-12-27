#!/usr/bin/env python

# Import python modules
import os,sys

# Import User modules
ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = ['','..']
for PATH in PATHS:
	sys.path.append(os.path.abspath(os.path.join(ROOT,PATH)))

from src.io import join
from src.process import process

def main(args):

	kwargs = {
		'data':'**/data.hdf5',
		'settings':'config/plot.json',
		'hyperparameters':'config/process.json',
		'cwd':args[0] if len(args)>0 else None
	}

	kwargs.update({kwarg: join(arg,kwargs[kwarg]) if (kwargs[kwarg] is None or kwargs[kwarg].startswith('*')) and not arg.startswith("*") else arg 
		for arg,kwarg in zip(args,kwargs)})

	process(**kwargs)

	return

if __name__ == '__main__':
	main(sys.argv[1:])